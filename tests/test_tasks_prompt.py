from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from elt_environment.tasks import TaskRef, discover_tasks, split_by_family


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


class DiscoverTests(unittest.TestCase):
    def _release(self, root: Path, manifest: dict) -> None:
        (root / "release_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_a_task_without_a_family_or_destination_entry_is_refused(self) -> None:
        for manifest in (
            {"tasks": {"t": "0"}, "destinations": {"t": "snowflake"}},
            {"tasks": {"t": "0"}, "families": {"t": "f"}},
        ):
            with tempfile.TemporaryDirectory() as root, self.subTest(manifest=manifest):
                self._release(Path(root), manifest)
                with self.assertRaises(ValueError):
                    discover_tasks(Path(root))

    def test_family_and_destination_come_from_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self._release(
                Path(root),
                {"tasks": {"t": "0"}, "families": {"t": "f"}, "destinations": {"t": "databricks"}},
            )
            (task,) = discover_tasks(Path(root))
        self.assertEqual((task.task_id, task.family, task.destination), ("t", "f", "databricks"))



if __name__ == "__main__":
    unittest.main()
