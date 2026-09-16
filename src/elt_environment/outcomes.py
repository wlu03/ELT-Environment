"""Coordinator control flow at the taskgen-to-Tinker reward boundary.

This module does not calculate an ELT reward. The trusted taskgen projection
does that. It only makes the distinction between a measured numeric label and
an unmeasurable rollout impossible to erase accidentally.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, TypeAlias

_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True, slots=True)
class ValidReward:
    """A policy-attributable ELT label ready for Tinker.

    Metrics are numeric and sanitized because Tinker logging must not receive
    private rows, paths, SQL, credentials, or raw grader exceptions.
    """

    value: float
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        value = float(self.value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("reward must be a finite number in [0, 1]")

        clean_metrics: dict[str, float] = {}
        for name, raw_value in self.metrics.items():
            if not _REASON_CODE.fullmatch(name):
                raise ValueError(f"invalid metric name: {name!r}")
            metric = float(raw_value)
            if not math.isfinite(metric):
                raise ValueError(f"metric {name!r} must be finite")
            clean_metrics[name] = metric

        object.__setattr__(self, "value", value)
        object.__setattr__(self, "metrics", MappingProxyType(clean_metrics))

    def __reduce__(self) -> tuple[object, tuple[float, dict[str, float]]]:
        """Keep the immutable view pickle-safe for spawned rollout workers."""
        return (type(self), (self.value, dict(self.metrics)))


@dataclass(frozen=True, slots=True)
class DiscardGroup:
    """A task, harness, or infrastructure failure that is not a label.

    Only a stable sanitized code crosses this boundary. Detailed diagnostics
    remain in restricted harness logs.
    """

    reason_code: str

    def __post_init__(self) -> None:
        if not _REASON_CODE.fullmatch(self.reason_code):
            raise ValueError("reason_code must be a lowercase stable code")


GraderOutcome: TypeAlias = ValidReward | DiscardGroup


class DiscardGroupSignal(RuntimeError):
    """Control-flow exception raised before a Tinker StepResult is created."""

    def __init__(self, reason_code: str) -> None:
        if not _REASON_CODE.fullmatch(reason_code):
            raise ValueError("reason_code must be a lowercase stable code")
        self.reason_code = reason_code
        super().__init__(f"discard_grpo_group:{reason_code}")

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (type(self), (self.reason_code,))


def require_tinker_reward(outcome: GraderOutcome) -> float:
    """Return a real reward or force the caller to retry the complete group.

    Tinker Cookbook reward fields are floats. Converting a taskgen no-label
    result to ``0.0``, ``NaN``, or a partial sibling replacement would corrupt
    GRPO. The Tinker Env calls this before constructing its terminal result.
    """

    if isinstance(outcome, DiscardGroup):
        raise DiscardGroupSignal(outcome.reason_code)
    if not isinstance(outcome, ValidReward):
        raise TypeError("outcome must be ValidReward or DiscardGroup")
    return outcome.value
