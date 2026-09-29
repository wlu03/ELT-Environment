"""Score an artifact with ELT-taskgen's declarative environment and return a reward or a discard."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from elt_taskgen.training.dbt_runner import DbtRuntimeConfig
from elt_taskgen.training.env import DeclarativeEltEnv

from elt_environment.outcomes import DiscardGroup, GraderOutcome, ValidReward
from elt_environment.tasks import TaskRef

_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PHASES = ("none", "terraform", "sync", "dbt", "mart", "immutability", "workspace")


@dataclass(frozen=True)
class GraderConfig:
    """Paths and limits for the local grader."""

    taskgen_root: Path
    attempts_root: Path
    grader_deadline_s: float | None = None
    w_t: float = 1.0

    def runtime_config(self) -> DbtRuntimeConfig:
        image = Path(self.taskgen_root) / "runtime-images" / "dbt-duckdb"
        return DbtRuntimeConfig(
            python=image / ".venv" / "bin" / "python",
            manifest=image / "runtime.json",
        )


def _reason_code(code: str) -> str:
    return code if _REASON_CODE.fullmatch(code) else "harness_fault"


def grade_artifact(task: TaskRef, files: Mapping[str, str], config: GraderConfig) -> GraderOutcome:
    """Seal ``files`` into a fresh attempt and score them on every hidden population.

    Returns ``DiscardGroup`` when the grader produced no label, which happens
    only for task, harness or infrastructure faults. Blocks for the length of
    one grade, so callers run it in a worker thread.
    """

    Path(config.attempts_root).mkdir(parents=True, exist_ok=True)
    env = DeclarativeEltEnv(
        task.release_dir,
        attempts_root=Path(config.attempts_root),
        runtime_config=config.runtime_config(),
        grader_deadline_s=config.grader_deadline_s,
        w_t=config.w_t,
        verify_release=False,
    )
    try:
        env.reset(task.task_id)
        step = env.step(dict(files))
    finally:
        env.close()
    signal = step.signal
    if not signal.label_valid or signal.reward is None:
        return DiscardGroup(_reason_code(step.harness_fault_code or "unlabelled"))
    metrics = {
        "r_el": signal.r_el,
        "r_t": signal.r_t,
        "el_pass": float(signal.el_pass),
        "policy_violation": float(signal.policy_violation),
    }
    metrics.update({f"phase_{phase}": float(signal.first_failed_phase == phase) for phase in _PHASES})
    return ValidReward(signal.reward, metrics)
