from __future__ import annotations

import asyncio
import unittest
from collections.abc import Sequence
from dataclasses import dataclass, field

import tinker
from tinker_cookbook.completers import StopCondition, TokenCompleter, TokensWithLogprobs
from tinker_cookbook.rl.types import Env, EnvGroupBuilder, StepResult

from elt_environment.env import GroupRetryExhausted, RetryWholeGroup
from elt_environment.outcomes import DiscardGroupSignal


class ScriptedPolicy(TokenCompleter):
    async def __call__(self, model_input, stop, *, max_tokens=None) -> TokensWithLogprobs:
        return TokensWithLogprobs(tokens=[7], maybe_logprobs=[0.0])


class ScriptedEnv(Env):
    def __init__(self, outcome: object, attempt: int) -> None:
        self.outcome = outcome
        self.attempt = attempt

    async def initial_observation(self) -> tuple[tinker.ModelInput, StopCondition]:
        return tinker.ModelInput.from_ints([1]), []

    async def step(self, action, *, extra=None) -> StepResult:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
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

    async def make_envs(self) -> Sequence[Env]:
        outcomes = self.plan[min(self.calls, len(self.plan) - 1)]
        self.calls += 1
        return [ScriptedEnv(outcome, self.calls) for outcome in outcomes]


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

    def test_strategy_catches_group_errors_so_the_trainer_skips_an_exhausted_group(self) -> None:
        self.assertTrue(RetryWholeGroup().catches_group_errors)


if __name__ == "__main__":
    unittest.main()
