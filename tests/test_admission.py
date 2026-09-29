from __future__ import annotations

import unittest
from pathlib import Path

from elt_environment.admission import admit_tasks
from elt_environment.env import cached_renderer
from elt_environment.tasks import discover_tasks
from elt_environment.train import CLIConfig, default_renderer_name

TASKGEN = Path(__file__).resolve().parents[2] / "ELT-taskgen"
RELEASES = TASKGEN / "runs" / "batch50c_20260923" / "workspace" / "releases"


@unittest.skipUnless(RELEASES.is_dir(), "batch50c releases are not on disk")
class AdmissionTests(unittest.TestCase):
    def test_every_batch50c_task_fits_the_default_model_and_reply_budget(self) -> None:
        config = CLIConfig()
        admitted, excluded = admit_tasks(
            discover_tasks(RELEASES),
            cached_renderer(config.model_name, default_renderer_name(config.model_name)),
            context_length=config.context_length,
            max_tokens=config.max_tokens,
            verify=False,
        )
        self.assertEqual(excluded, [])
        self.assertEqual(len(admitted), 50)


if __name__ == "__main__":
    unittest.main()
