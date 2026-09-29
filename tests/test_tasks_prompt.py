from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from elt_environment.artifact import format_artifact, parse_artifact
from elt_environment.prompt import (
    CREDENTIAL_PLACEHOLDER,
    EXAMPLE_REPLY,
    EXAMPLE_TASK_ID,
    BundleRedactionError,
    build_messages,
    bundle_files,
    redact_credentials,
)
from elt_environment.tasks import TaskRef, discover_tasks, split_by_family

CONFIG = """Airbyte:
  config:
    password: ''
    server_url: http://airbyte:80/api/public/v1/
aws_s3:
  AWS_ACCESS_KEY_ID: key-id-value
  AWS_SECRET_ACCESS_KEY: "secret value"
  AWS_ENDPOINT_URL: http://localstack:4566
postgres:
  config:
    host: elt-postgres
    password: pg-password-value  # set by the harness
    user: postgres
"""


class RedactionTests(unittest.TestCase):
    def test_credential_values_are_replaced_and_everything_else_is_kept(self) -> None:
        redacted = redact_credentials(CONFIG)
        for secret in ("key-id-value", "secret value", "pg-password-value"):
            self.assertNotIn(secret, redacted)
        document = yaml.safe_load(redacted)
        self.assertEqual(document["aws_s3"]["AWS_ACCESS_KEY_ID"], CREDENTIAL_PLACEHOLDER)
        self.assertEqual(document["aws_s3"]["AWS_SECRET_ACCESS_KEY"], CREDENTIAL_PLACEHOLDER)
        self.assertEqual(document["postgres"]["config"]["password"], CREDENTIAL_PLACEHOLDER)
        self.assertEqual(document["Airbyte"]["config"]["password"], "")
        self.assertEqual(document["postgres"]["config"]["user"], "postgres")
        self.assertEqual(document["aws_s3"]["AWS_ENDPOINT_URL"], "http://localstack:4566")

    def test_a_credential_the_line_rule_cannot_replace_refuses_the_bundle(self) -> None:
        for text in (
            "postgres:\n  password: |\n    multi\n    line\n",
            "sources:\n  - password: listed-value\n",
            "postgres: {password: inline-value}\n",
        ):
            with self.subTest(text=text), self.assertRaises(BundleRedactionError):
                redact_credentials(text)

TASKGEN = Path(__file__).resolve().parents[2] / "ELT-taskgen"
RELEASES = TASKGEN / "runs" / "batch50c_20260923" / "workspace" / "releases"
FIXTURE = TASKGEN / "tests" / "fixtures" / "semantic_gate" / "release"


@unittest.skipUnless(FIXTURE.is_dir(), "taskgen fixture release is not on disk")
class ExampleReplyTests(unittest.TestCase):
    def test_packaged_example_matches_the_canonical_render_of_the_fixture(self) -> None:
        from elt_taskgen.training import load_workspace_package
        from elt_taskgen.training.canonical import render_canonical_project

        canonical = render_canonical_project(
            load_workspace_package(FIXTURE, EXAMPLE_TASK_ID, verify=True)
        )
        example = parse_artifact(EXAMPLE_REPLY)
        self.assertEqual(example, {path: canonical[path] for path in example})
        self.assertEqual(
            set(example), {"main.tf", "dbt_project.yml", "models/sources.yml", "models/event_wide.sql"}
        )
        self.assertEqual(EXAMPLE_REPLY, format_artifact(example) + "\n")


def _fake(task_id: str, family: str) -> TaskRef:
    return TaskRef(Path("/nonexistent"), task_id, family, "snowflake")


class SplitTests(unittest.TestCase):
    def test_split_is_family_disjoint_and_seeded(self) -> None:
        tasks = [_fake(f"t{index}", f"f{index % 7}") for index in range(30)]
        train, evaluation = split_by_family(tasks, eval_families=2, seed=3)
        self.assertEqual(len(train) + len(evaluation), len(tasks))
        self.assertFalse({t.family for t in train} & {t.family for t in evaluation})
        self.assertEqual(len({t.family for t in evaluation}), 2)
        self.assertEqual(split_by_family(list(reversed(tasks)), 2, 3)[1], evaluation)
        self.assertNotEqual(split_by_family(tasks, 2, 4)[1], evaluation)


@unittest.skipUnless(RELEASES.is_dir(), "batch50c releases are not on disk")
class ReleaseAndPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = discover_tasks(RELEASES)
        cls.task = next(t for t in cls.tasks if t.task_id == "dlt__workable")

    def test_discovers_every_release(self) -> None:
        self.assertEqual(len(self.tasks), 50)
        self.assertEqual(self.task.family, "dlt__workable")
        self.assertEqual(self.task.destination, "snowflake")

    def test_bundle_reads_only_public_files_and_leaves_out_other_destinations(self) -> None:
        paths = [path for path, _ in bundle_files(self.task)]
        self.assertEqual(paths[:3], ["documentation/README.md", "config.yaml", "data_model.yaml"])
        self.assertIn("elt/main.tf", paths)
        self.assertIn("documentation/destination_snowflake.md", paths)
        for absent in (
            "check_job_status.py",
            "snowflake_credential.json",
            "documentation/destination_redshift.md",
            "documentation/destination_databricks.md",
            "documentation/databricks_authentication.md",
            "documentation/trigger_job.md",
        ):
            self.assertNotIn(absent, paths)
        self.assertFalse(any(path.startswith("destinations/") for path in paths))

    def test_prompt_is_deterministic_and_names_no_private_material(self) -> None:
        first, second = build_messages(self.task), build_messages(self.task)
        self.assertEqual(first, second)
        text = first[0]["content"] + first[1]["content"]
        for marker in ("answer_key", "private/", "/gold/", ".duckdb", "task_ir", str(RELEASES)):
            self.assertNotIn(marker, text)

    def test_no_task_sends_a_credential_value_to_the_model(self) -> None:
        checked = 0
        for task in self.tasks:
            original = yaml.safe_load((task.public_dir / "config.yaml").read_text(encoding="utf-8"))
            shown = yaml.safe_load(dict(bundle_files(task))["config.yaml"])
            fields = [("aws_s3", key) for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")]
            with self.subTest(task=task.task_id):
                for section, key in fields:
                    if (original.get(section) or {}).get(key):
                        self.assertEqual(shown[section][key], CREDENTIAL_PLACEHOLDER)
                        checked += 1
                if ((original.get("postgres") or {}).get("config") or {}).get("password"):
                    self.assertEqual(shown["postgres"]["config"]["password"], CREDENTIAL_PLACEHOLDER)
                    checked += 1
                self.assertEqual(
                    shown["snowflake"]["config"]["database"], original["snowflake"]["config"]["database"]
                )
        self.assertEqual(checked, 45 + 45 + 38)

    def test_system_prompt_carries_the_example_and_the_user_message_carries_the_task(self) -> None:
        system, user = build_messages(self.task)
        self.assertTrue(system["content"].endswith(EXAMPLE_REPLY))
        self.assertIn(EXAMPLE_TASK_ID, system["content"])
        self.assertNotIn(EXAMPLE_TASK_ID, user["content"])
        self.assertIn("dlt__workable", user["content"])
        self.assertFalse(any(task.task_id == EXAMPLE_TASK_ID for task in self.tasks))


if __name__ == "__main__":
    unittest.main()
