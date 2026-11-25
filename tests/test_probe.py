from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from elt_environment.env import GroupRetryExhausted
from elt_environment.probe import ProbeConfig, choose_tasks, probe
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


class ProbeDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_worker_exit_and_startup_discard_separately(self):
        transition = SimpleNamespace(
            ac=SimpleNamespace(tokens=[], logprobs=[]), reward=0.75,
            metrics={"solver_exit_status/submitted (exit_format)": 1.0, "el_pass": 1.0},
        )
        trajectory = SimpleNamespace(transitions=[transition])
        renderer = Mock()
        renderer.parse_response.return_value = ({}, "stop")
        renderer.to_openai_message.return_value = {}
        with (
            TemporaryDirectory() as output,
            patch("elt_environment.probe.check_run"),
            patch("elt_environment.probe.cached_renderer", return_value=renderer),
            patch("elt_environment.probe.discover_tasks", return_value=TRAIN[:2]),
            patch("elt_environment.probe.admit_tasks",
                  return_value=([SimpleNamespace(task=task) for task in TRAIN[:2]], [])),
            patch("elt_environment.probe.tinker.ServiceClient") as service,
            patch("elt_environment.probe.do_group_rollout", side_effect=[
                SimpleNamespace(trajectories_G=[trajectory]),
                GroupRetryExhausted("startup failure", reason_codes=("swe_rex_relay_unavailable",)),
            ]),
        ):
            service.return_value.create_sampling_client_async = AsyncMock()
            summary = await probe(ProbeConfig(eval_families=0, out_dir=output, grader_command="grade"))

        self.assertEqual(summary["mean_reward"], 0.75)
        self.assertEqual(summary["rollouts"], 1)
        self.assertEqual(summary["groups_discarded"], 1)
        self.assertEqual(summary["discard_reasons_by_task"], {"t1": ["swe_rex_relay_unavailable"]})
        self.assertEqual(summary["solver_exit_status_by_task"], {"t0": ["submitted (exit_format)"]})
        self.assertEqual(summary["solver_exit_status_counts"], {"submitted (exit_format)": 1})


if __name__ == "__main__":
    unittest.main()
