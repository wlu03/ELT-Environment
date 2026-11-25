from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from task_fixtures import write_bundle

from elt_environment.admission import admit_tasks
from elt_environment.runner import AgentConfig
from elt_environment.tasks import discover_tasks


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        public = write_bundle(self.root / "bundle")
        (self.task,) = discover_tasks(self.root / "bundle")
        self.installed = self.root / "inputs/t"
        shutil.copytree(public, self.installed)
        self.agent = AgentConfig(inputs_root=str(self.root / "inputs"))

    def test_bundle_is_admitted_without_release_metadata(self):
        admitted, excluded = admit_tasks([self.task], agent=self.agent)
        self.assertEqual([entry.task for entry in admitted], [self.task])
        self.assertEqual(excluded, [])

    def test_changed_family_is_excluded(self):
        write_bundle(self.task.release_dir, family="other")
        admitted, excluded = admit_tasks([self.task], agent=self.agent)
        self.assertEqual(admitted, [])
        self.assertEqual(excluded[0].reason, "input_invalid:ValueError")

    def test_changed_development_schema_is_excluded(self):
        (self.installed / "data_model.yaml").write_text("models: [unexpected]\n")
        admitted, excluded = admit_tasks([self.task], agent=self.agent)
        self.assertEqual(admitted, [])
        self.assertEqual(excluded[0].task_id, self.task.task_id)

    def test_private_task_ir_cannot_enter_agent_inputs(self):
        (self.installed / "task_ir.json").write_text("{}")
        admitted, excluded = admit_tasks([self.task], agent=self.agent)
        self.assertEqual(admitted, [])
        self.assertEqual(excluded[0].reason, "input_invalid:ValueError")


if __name__ == "__main__":
    unittest.main()
