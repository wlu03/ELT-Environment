"""Tinker orchestration for verified ELT-taskgen releases.

The package is intentionally small while the architecture is being frozen.
Task generation, scoring, and reward projection stay in :mod:`elt_taskgen`.
"""

from elt_environment.outcomes import (
    DiscardGroup,
    DiscardGroupSignal,
    GraderOutcome,
    ValidReward,
    require_tinker_reward,
)

__all__ = [
    "DiscardGroup",
    "DiscardGroupSignal",
    "GraderOutcome",
    "ValidReward",
    "require_tinker_reward",
]

__version__ = "0.0.0"

