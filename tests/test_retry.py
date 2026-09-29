from __future__ import annotations

import asyncio
import threading
import time
import unittest
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

import tinker
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, StepResult

from elt_environment.env import EltDataset, GroupRetryExhausted, RetryWholeGroup, run_grader
from elt_environment.outcomes import DiscardGroupSignal, ValidReward
from elt_environment.tasks import TaskRef


class ScriptedPolicy(TokenCompleter):
    async def __call__(self, model_input, stop, *, max_tokens=None) -> TokensWithLogprobs:
        return TokensWithLogprobs(tokens=[7], maybe_logprobs=[0.0])


class ScriptedEnv(Env):
    def __init__(self, outcome: object, attempt: int, log: list | None = None) -> None:
        self.outcome = outcome
        self.attempt = attempt
        self.log = log

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        return tinker.ModelInput.from_ints([1]), []

    async def step(self, action, *, extra=None) -> StepResult:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        if self.log is not None:
            await asyncio.sleep(0.05)
            self.log.append(("step", self.attempt))
        return StepResult(
            reward=float(self.outcome),
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={"attempt": self.attempt},
        )


@dataclass
class ScriptedBuilder(EnvGroupBuilder):
    plan: list[list[object]]
    calls: int = field(default=0)
    log: list | None = None

    async def make_envs(self) -> Sequence[Env]:
        outcomes = self.plan[min(self.calls, len(self.plan) - 1)]
        self.calls += 1
        if self.log is not None:
            self.log.append(("make", self.calls))
        return [ScriptedEnv(outcome, self.calls, self.log) for outcome in outcomes]


def _run(strategy: RetryWholeGroup, builder: ScriptedBuilder):
    return asyncio.run(strategy.execute(builder, ScriptedPolicy()))


class RetryWholeGroupTests(unittest.TestCase):
    def test_labelled_group_is_returned_on_the_first_attempt(self) -> None:
        builder = ScriptedBuilder([[0.0, 0.5, 1.0, 0.0]])
        result = _run(RetryWholeGroup(max_attempts=3), builder)
        self.assertEqual(builder.calls, 1)
        self.assertEqual([t.transitions[-1].reward for t in result.trajectories], [0.0, 0.5, 1.0, 0.0])
        self.assertEqual(result.errors, [])

    def test_one_unlabelled_member_discards_the_group_and_resamples_it_whole(self) -> None:
        builder = ScriptedBuilder(
            [[1.0, DiscardGroupSignal("grader_child_died"), 0.5, 0.0], [0.0, 0.0, 0.5, 1.0]]
        )
        result = _run(RetryWholeGroup(max_attempts=3), builder)
        self.assertEqual(builder.calls, 2)
        self.assertEqual(len(result.trajectories), 4)
        self.assertEqual({t.transitions[-1].metrics["attempt"] for t in result.trajectories}, {2})
        self.assertEqual([t.transitions[-1].reward for t in result.trajectories], [0.0, 0.0, 0.5, 1.0])
        self.assertEqual([e.error_message for e in result.errors], ["grader_child_died"])

    def test_every_sibling_finishes_before_the_group_is_sampled_again(self) -> None:
        log: list[tuple[str, int]] = []
        builder = ScriptedBuilder(
            [[1.0, DiscardGroupSignal("tool_deadline"), 0.5, 0.0], [0.0, 0.0, 0.5, 1.0]], log=log
        )
        _run(RetryWholeGroup(max_attempts=3), builder)
        expected = [("make", 1)] + [("step", 1)] * 3 + [("make", 2)] + [("step", 2)] * 4
        self.assertEqual(log, expected)

    def test_numeric_zero_is_a_label_and_is_not_retried(self) -> None:
        builder = ScriptedBuilder([[0.0, 0.0, 0.0, 0.0]])
        _run(RetryWholeGroup(max_attempts=3), builder)
        self.assertEqual(builder.calls, 1)

    def test_exhaustion_raises_instead_of_returning_a_partial_group(self) -> None:
        builder = ScriptedBuilder([[1.0, DiscardGroupSignal("tool_deadline")]])
        with self.assertRaises(GroupRetryExhausted):
            _run(RetryWholeGroup(max_attempts=2), builder)
        self.assertEqual(builder.calls, 2)

    def test_other_errors_propagate_without_retry(self) -> None:
        builder = ScriptedBuilder([[1.0, ValueError("bug")]])
        with self.assertRaises(ValueError):
            _run(RetryWholeGroup(max_attempts=3), builder)
        self.assertEqual(builder.calls, 1)

    def test_exhaustion_is_not_caught_so_the_trainer_never_shrinks_a_batch(self) -> None:
        self.assertFalse(RetryWholeGroup().catches_group_errors)


class GraderConcurrencyTests(unittest.TestCase):
    def test_cancelled_rollouts_keep_their_grader_slot_until_the_grade_returns(self) -> None:
        workers = 3
        state = {"active": 0, "peak": 0, "graded": 0}
        lock = threading.Lock()

        def slow_grade(task, files, grader):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.3)
            with lock:
                state["active"] -= 1
                state["graded"] += 1
            return ValidReward(0.0)

        async def scenario() -> None:
            with mock.patch("elt_environment.env.grade_artifact", slow_grade):
                first = [asyncio.ensure_future(run_grader(None, {}, None, workers)) for _ in range(workers)]
                for _ in range(200):
                    if state["active"] == workers:
                        break
                    await asyncio.sleep(0.01)
                for rollout in first:
                    rollout.cancel()
                await asyncio.gather(*first, return_exceptions=True)
                second = [asyncio.ensure_future(run_grader(None, {}, None, workers)) for _ in range(workers)]
                await asyncio.gather(*second)

        asyncio.run(scenario())
        self.assertEqual(state["peak"], workers)
        self.assertEqual(state["graded"], 2 * workers)


class DatasetTests(unittest.TestCase):
    def _dataset(self, count: int, groups_per_batch: int, epochs: int = 1) -> EltDataset:
        tasks = [TaskRef(Path("/nonexistent"), f"t{index}", f"f{index}", "snowflake") for index in range(count)]
        return EltDataset(
            tasks,
            groups_per_batch=groups_per_batch,
            epochs=epochs,
            seed=0,
            shuffle=True,
            make_builder=lambda task: task,
        )

    def _batches(self, dataset: EltDataset) -> list[list[TaskRef]]:
        return [list(dataset.get_batch(index)) for index in range(len(dataset))]

    def test_every_task_is_trained_and_every_batch_is_full(self) -> None:
        for count, groups_per_batch, epochs in ((9, 8, 1), (5, 8, 1), (43, 8, 4), (16, 8, 1), (1, 4, 2)):
            with self.subTest(count=count, groups_per_batch=groups_per_batch, epochs=epochs):
                batches = self._batches(self._dataset(count, groups_per_batch, epochs))
                self.assertTrue(batches)
                self.assertTrue(all(len(batch) == groups_per_batch for batch in batches))
                seen = [task.task_id for batch in batches for task in batch]
                for index in range(count):
                    self.assertGreaterEqual(seen.count(f"t{index}"), epochs)

    def test_an_exact_multiple_adds_no_padding(self) -> None:
        self.assertEqual(len(self._dataset(16, 8)), 2)

    def test_an_empty_task_list_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self._dataset(0, 8)


if __name__ == "__main__":
    unittest.main()
