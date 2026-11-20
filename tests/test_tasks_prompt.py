from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from task_fixtures import write_bundle

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
    def test_metadata_comes_from_task_ir_and_public_config(self) -> None:
        for semantic in (False, True):
            with tempfile.TemporaryDirectory() as directory, self.subTest(semantic=semantic):
                root = Path(directory)
                write_bundle(root, "t", "actual_family", "databricks", semantic=semantic)
                self.assertEqual(
                    discover_tasks(root), [TaskRef(root, "t", "actual_family", "databricks")]
                )

    def test_multiple_bundles_and_tasks_are_sorted_and_variants_are_not_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_bundle(root / "first", "z")
            write_bundle(root / "first", "a")
            write_bundle(root / "second", "m", destination="redshift")
            (root / "first/public/z__el").mkdir()
            (root / "first/public/z__t").mkdir()
            self.assertEqual([t.task_id for t in discover_tasks(root)], ["a", "m", "z"])

    def test_duplicate_task_ids_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_bundle(root / "first")
            write_bundle(root / "second")
            with self.assertRaisesRegex(ValueError, "more than one"):
                discover_tasks(root)

    def test_missing_or_invalid_metadata_is_refused(self) -> None:
        for metadata in (
            {},
            [],
            {"task_id": "other", "family_id": "f"},
            *({"task_id": "t", "family_id": value} for value in (None, "", " ", 7)),
        ):
            with tempfile.TemporaryDirectory() as directory, self.subTest(metadata=metadata):
                root = Path(directory)
                write_bundle(root)
                (root / "private/t/task_ir.json").write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    discover_tasks(root)

    def test_missing_or_ambiguous_destination_is_refused(self) -> None:
        for config in ("", "[]", "{}", "snowflake: {}\nredshift: {}"):
            with tempfile.TemporaryDirectory() as directory, self.subTest(config=config):
                root = Path(directory)
                public = write_bundle(root)
                (public / "config.yaml").write_text(config)
                with self.assertRaises((TypeError, ValueError)):
                    discover_tasks(root)

    def test_missing_task_ir_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_bundle(root)
            (root / "private/t/task_ir.json").unlink()
            with self.assertRaises(FileNotFoundError):
                discover_tasks(root)

    def test_symlinked_inputs_are_refused(self) -> None:
        for relative in (
            "public",
            "private",
            "private/t",
            "private/t/task_ir.json",
            "public/t/config.yaml",
            "public/t/schemas",
        ):
            with tempfile.TemporaryDirectory() as directory, self.subTest(path=relative):
                root = Path(directory)
                write_bundle(root / "bundle")
                original = root / "bundle" / relative
                moved = root / "moved"
                original.rename(moved)
                original.symlink_to(moved)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    discover_tasks(root / "bundle")

    def test_empty_root_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root, self.assertRaises(FileNotFoundError):
            discover_tasks(Path(root))


if __name__ == "__main__":
    unittest.main()
