from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from elt_taskgen.training import load_workspace_package
from elt_taskgen.training.canonical import render_canonical_project

from elt_environment.artifact import format_artifact
from elt_environment.env import EltGroupBuilder
from elt_environment.grading import GraderCleanupError, GraderConfig, grade_artifact
from elt_environment.outcomes import DiscardGroup, ValidReward
from elt_environment.tasks import TaskRef, discover_tasks
from elt_environment.train import DEFAULT_MODEL, default_renderer_name

TASKGEN = Path(__file__).resolve().parents[2] / "ELT-taskgen"
RELEASES = TASKGEN / "runs" / "batch50c_20260923" / "workspace" / "releases"
TASK_ID = "synsql__3d_motion_tracking_and_analysis__sensors_motion_data_cohorts"
MODEL = DEFAULT_MODEL
RENDERER = default_renderer_name(DEFAULT_MODEL)


@unittest.skipUnless(
    os.environ.get("ELT_ENVIRONMENT_GRADING_TESTS") == "1" and RELEASES.is_dir(),
    "set ELT_ENVIRONMENT_GRADING_TESTS=1 to run real dbt grading",
)
class GradingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.task = next(t for t in discover_tasks(RELEASES) if t.task_id == TASK_ID)
        package = load_workspace_package(cls.task.release_dir, TASK_ID, verify=False)
        cls.canonical = render_canonical_project(package)
        cls.scratch = tempfile.TemporaryDirectory()
        cls.grader = GraderConfig(taskgen_root=TASKGEN, attempts_root=Path(cls.scratch.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.scratch.cleanup()

    def test_canonical_artifact_scores_one_and_leaves_no_attempt_behind(self) -> None:
        outcome = grade_artifact(self.task, self.canonical, self.grader)
        self.assertIsInstance(outcome, ValidReward)
        self.assertEqual(outcome.value, 1.0)
        self.assertEqual(outcome.metrics["el_pass"], 1.0)
        self.assertEqual(outcome.metrics["phase_none"], 1.0)
        self.assertEqual(list((Path(self.scratch.name) / "episodes").iterdir()), [])

    def test_env_step_decodes_a_sampled_reply_and_grades_it(self) -> None:
        builder = EltGroupBuilder(
            task=self.task,
            group_size=2,
            model_name=MODEL,
            renderer_name=RENDERER,
            grader=self.grader,
            grader_concurrency=2,
        )

        async def run() -> tuple[float, float, dict, dict]:
            good_env, bad_env = await builder.make_envs()
            await good_env.initial_observation()
            tokenizer = good_env.renderer.tokenizer
            stop = good_env.stop_condition
            end = list(stop) if stop and isinstance(stop[0], int) else tokenizer.encode(stop[0])
            good = tokenizer.encode(format_artifact(self.canonical), add_special_tokens=False) + end
            bad = tokenizer.encode("I cannot write this project.", add_special_tokens=False) + end
            good_result, bad_result = await asyncio.gather(good_env.step(good), bad_env.step(bad))
            return good_result.reward, bad_result.reward, good_result.metrics, bad_result.metrics

        good_reward, bad_reward, good_metrics, bad_metrics = asyncio.run(run())
        self.assertEqual(good_reward, 1.0)
        self.assertEqual(good_metrics["format_error"], 0.0)
        self.assertEqual(bad_reward, 0.0)
        self.assertEqual(bad_metrics["format_error"], 1.0)


class GraderExceptionTests(unittest.TestCase):
    def _grade(self, fake_env: type) -> object:
        task = TaskRef(Path("/nonexistent"), "t", "f", "snowflake")
        with tempfile.TemporaryDirectory() as scratch:
            grader = GraderConfig(taskgen_root=TASKGEN, attempts_root=Path(scratch))
            with mock.patch("elt_environment.grading.DeclarativeEltEnv", fake_env):
                return grade_artifact(task, {"main.tf": ""}, grader)

    def test_an_exception_from_the_grader_discards_the_group_and_still_closes(self) -> None:
        closed = []

        class Raising:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def reset(self, task_id: str) -> None:
                raise OSError("disk full")

            def close(self) -> None:
                closed.append(True)

        with self.assertLogs("elt_environment.grading", "ERROR"):
            outcome = self._grade(Raising)
        self.assertEqual(outcome, DiscardGroup("grader_exception"))
        self.assertEqual(closed, [True])

    def test_a_constructor_failure_discards_the_group_and_closes_nothing(self) -> None:
        class Unbuildable:
            def __init__(self, *args, **kwargs) -> None:
                raise OSError("runtime image unavailable")

        with self.assertLogs("elt_environment.grading", "ERROR"):
            outcome = self._grade(Unbuildable)
        self.assertEqual(outcome, DiscardGroup("grader_exception"))

    def test_a_cleanup_failure_stops_the_run(self) -> None:
        class LeakyClose:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def reset(self, task_id: str) -> None:
                pass

            def step(self, files: dict) -> SimpleNamespace:
                signal = SimpleNamespace(
                    label_valid=True,
                    reward=0.5,
                    r_el=1.0,
                    r_t=0.0,
                    el_pass=True,
                    policy_violation=False,
                    first_failed_phase="mart",
                )
                return SimpleNamespace(signal=signal, harness_fault_code="")

            def close(self) -> None:
                raise OSError("busy")

        with self.assertRaises(GraderCleanupError):
            self._grade(LeakyClose)


if __name__ == "__main__":
    unittest.main()
