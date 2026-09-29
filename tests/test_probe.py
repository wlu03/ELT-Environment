from __future__ import annotations

import unittest
from pathlib import Path

from elt_environment.probe import choose_tasks
from elt_environment.tasks import TaskRef

TRAIN = [
    TaskRef(Path("/nonexistent"), f"t{index}", f"f{index}", "snowflake") for index in range(10)
]


class ChooseTasksTests(unittest.TestCase):
    def test_a_count_spreads_over_the_training_tasks(self) -> None:
        self.assertEqual([t.task_id for t in choose_tasks(TRAIN, 3, None)], ["t0", "t3", "t6"])
        self.assertEqual(len(choose_tasks(TRAIN, 50, None)), 10)

    def test_named_tasks_are_returned_in_training_order(self) -> None:
        self.assertEqual([t.task_id for t in choose_tasks(TRAIN, 6, "t7,t2")], ["t2", "t7"])

    def test_a_zero_count_an_unknown_name_and_an_empty_choice_are_refused(self) -> None:
        for count, task_ids, train in ((0, None, TRAIN), (6, "t2,missing", TRAIN), (6, None, [])):
            with self.subTest(count=count, task_ids=task_ids), self.assertRaises(ValueError):
                choose_tasks(train, count, task_ids)


if __name__ == "__main__":
    unittest.main()
