from __future__ import annotations

import math
import pickle
import unittest

from elt_environment.outcomes import (
    DiscardGroup,
    DiscardGroupSignal,
    ValidReward,
    require_tinker_reward,
)


class ValidRewardTests(unittest.TestCase):
    def test_valid_reward_is_numeric_and_immutable(self) -> None:
        source_metrics = {"r_el": 1, "r_t": 0.5}
        outcome = ValidReward(0.75, source_metrics)
        source_metrics["r_el"] = 0

        self.assertEqual(require_tinker_reward(outcome), 0.75)
        self.assertEqual(dict(outcome.metrics), {"r_el": 1.0, "r_t": 0.5})
        with self.assertRaises(TypeError):
            outcome.metrics["r_el"] = 0.0  # type: ignore[index]

    def test_invalid_reward_is_rejected(self) -> None:
        for value in (-0.01, 1.01, math.inf, -math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ValidReward(value)

    def test_unlabelled_outcome_raises_before_tinker_result(self) -> None:
        with self.assertRaises(DiscardGroupSignal) as raised:
            require_tinker_reward(DiscardGroup("grader_timeout"))
        self.assertEqual(raised.exception.reason_code, "grader_timeout")

    def test_reason_codes_are_sanitized(self) -> None:
        for code in ("", "GraderTimeout", "grader-timeout", "private/path", "x" * 65):
            with self.subTest(code=code), self.assertRaises(ValueError):
                DiscardGroup(code)

    def test_metrics_are_finite_and_have_stable_names(self) -> None:
        with self.assertRaises(ValueError):
            ValidReward(1.0, {"hidden/population": 1.0})
        with self.assertRaises(ValueError):
            ValidReward(1.0, {"r_el": math.nan})

    def test_outcomes_and_control_signal_survive_spawn_pickling(self) -> None:
        reward = pickle.loads(pickle.dumps(ValidReward(0.75, {"r_el": 1.0})))
        self.assertEqual(reward, ValidReward(0.75, {"r_el": 1.0}))
        self.assertEqual(dict(reward.metrics), {"r_el": 1.0})

        signal = pickle.loads(pickle.dumps(DiscardGroupSignal("grader_timeout")))
        self.assertEqual(signal.reason_code, "grader_timeout")


if __name__ == "__main__":
    unittest.main()
