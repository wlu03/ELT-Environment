from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from task_fixtures import write_bundle

from elt_environment.grading import GraderConfig, grade_artifact
from elt_environment.outcomes import DiscardGroup, ValidReward
from elt_environment.tasks import discover_tasks
from elt_environment.train import CLIConfig, check_run


class CommandGraderTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        write_bundle(self.root / "bundle")
        (self.task,) = discover_tasks(self.root / "bundle")

    def grade(self, program, *, deadline=10):
        script = self.root / "grader.py"
        script.write_text(program)
        config = GraderConfig(
            attempts_root=self.root / "attempts",
            command=(sys.executable, str(script)),
            grader_deadline_s=deadline,
        )
        outcome = grade_artifact(self.task, {"elt/main.tf": "terraform {}"}, config)
        self.assertEqual(list(config.attempts_root.iterdir()), [])
        return outcome

    def test_command_receives_artifact_and_returns_reward_without_release_metadata(self):
        outcome = self.grade("""import json, os, sys
from pathlib import Path
request = Path(sys.argv[1])
assert request.stat().st_mode & 0o777 == 0o600
payload = json.loads(request.read_text())
assert payload['task_id'] == 't'
assert payload['destination'] == 'snowflake'
assert payload['files'] == {'elt/main.tf': 'terraform {}'}
assert sorted(os.listdir(payload['release_dir'])) == ['private', 'public']
print(json.dumps({'label_valid': True, 'reward': .75, 'metrics': {'r_t': .5}}))
""")
        self.assertEqual(outcome, ValidReward(0.75, {"r_t": 0.5}))

    def test_faults_are_discarded_and_private_requests_are_cleaned(self):
        for program, code in (
            (
                'print(\'{"label_valid": false, "reason_code": "warehouse_down"}\')',
                "warehouse_down",
            ),
            ('print(\'{"label_valid": false, "reason_code": "private/path"}\')', "harness_fault"),
            ("print('{}')", "grader_response_invalid"),
            ("print('x' * 65537)", "grader_response_invalid"),
            ("raise SystemExit(1)", "grader_command_failed"),
        ):
            with self.subTest(code=code):
                self.assertEqual(self.grade(program), DiscardGroup(code))

    def test_timeout_kills_grader_and_cleans_request(self):
        self.assertEqual(
            self.grade("import time; time.sleep(10)", deadline=0.05), DiscardGroup("grader_timeout")
        )

    def test_changed_task_metadata_is_refused_before_running_grader(self):
        write_bundle(self.task.release_dir, family="changed")
        config = GraderConfig(self.root / "attempts", ("nonexistent-grader",))
        with self.assertLogs("elt_environment.grading", "ERROR"):
            outcome = grade_artifact(self.task, {}, config)
        self.assertEqual(outcome, DiscardGroup("grader_exception"))
        self.assertFalse(config.attempts_root.exists())

    def test_missing_command_is_rejected_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "grader command"):
            GraderConfig(self.root, ())
        for command in ("", "   "):
            with (
                self.subTest(command=command),
                self.assertRaisesRegex(ValueError, "grader_command"),
            ):
                check_run(CLIConfig(grader_command=command))


if __name__ == "__main__":
    unittest.main()
