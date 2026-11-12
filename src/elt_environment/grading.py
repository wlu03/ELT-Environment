"""Score an artifact with ELT-taskgen's declarative environment and return a reward or a discard."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from elt_taskgen.training.dbt_runner import DbtRuntimeConfig
from elt_taskgen.training.env import DeclarativeEltEnv

from elt_environment.outcomes import DiscardGroup, GraderOutcome, ValidReward
from elt_environment.tasks import TaskRef

_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PHASES = ("none", "terraform", "sync", "dbt", "mart", "immutability", "workspace")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraderConfig:
    """Paths and limits for the local grader."""

    taskgen_root: Path
    attempts_root: Path
    grader_deadline_s: float | None = None
    w_t: float = 1.0
    verify_release: bool = True
    command: tuple[str, ...] = ()

    def runtime_config(self) -> DbtRuntimeConfig:
        image = Path(self.taskgen_root) / "runtime-images" / "dbt-duckdb"
        return DbtRuntimeConfig(
            python=image / ".venv" / "bin" / "python",
            manifest=image / "runtime.json",
        )


def _reason_code(code: str) -> str:
    return code if _REASON_CODE.fullmatch(code) else "harness_fault"


class GraderCleanupError(RuntimeError):
    """An attempt directory could not be removed; the run stops before more accumulate."""


def _close(env: DeclarativeEltEnv, task_id: str) -> None:
    try:
        env.close()
    except Exception as error:
        raise GraderCleanupError(
            f"the attempt directory for {task_id} could not be removed"
        ) from error


def grade_artifact(task: TaskRef, files: Mapping[str, str], config: GraderConfig) -> GraderOutcome:
    """Seal ``files`` into a fresh attempt and score them on every hidden population.

    Returns ``DiscardGroup`` when the grader produced no label, which happens
    only for task, harness or infrastructure faults, or when anything from
    creating the attempts directory to scoring raised. The exception is logged
    here and the group is sampled again; an error outside this function is a
    coordinator error and propagates. With ``config.verify_release`` the
    release is checked against its checksums before every grade, so a release
    changed after admission is never scored. An attempt directory that cannot
    be removed raises ``GraderCleanupError``, which stops the run. Blocks for
    the length of one grade, so callers run it in a worker thread.
    """

    if config.command:
        return grade_command(task, files, config)
    env: DeclarativeEltEnv | None = None
    try:
        Path(config.attempts_root).mkdir(parents=True, exist_ok=True)
        env = DeclarativeEltEnv(
            task.release_dir,
            attempts_root=Path(config.attempts_root),
            runtime_config=config.runtime_config(),
            grader_deadline_s=config.grader_deadline_s,
            w_t=config.w_t,
            verify_release=config.verify_release,
        )
        env.reset(task.task_id)
        step = env.step(dict(files))
    except Exception:
        logger.exception("grader raised on %s", task.task_id)
        return DiscardGroup("grader_exception")
    finally:
        if env is not None:
            _close(env, task.task_id)
    signal = step.signal
    if not signal.label_valid or signal.reward is None:
        return DiscardGroup(_reason_code(step.harness_fault_code or "unlabelled"))
    metrics = {
        "r_el": signal.r_el,
        "r_t": signal.r_t,
        "el_pass": float(signal.el_pass),
        "policy_violation": float(signal.policy_violation),
    }
    metrics.update(
        {f"phase_{phase}": float(signal.first_failed_phase == phase) for phase in _PHASES}
    )
    return ValidReward(signal.reward, metrics)


def grade_command(task: TaskRef, files: Mapping[str, str], config: GraderConfig) -> GraderOutcome:
    """Invoke an operator-configured hidden-replay grader outside the agent.

    The command gets one JSON request path and writes an outcome JSON object
    to stdout. It must replay the artifact on fresh hidden populations, rather
    than reward success on the agent-visible development warehouse.
    """
    try:
        from elt_taskgen.training import load_workspace_package

        load_workspace_package(task.release_dir, task.task_id, verify=config.verify_release)
        config.attempts_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(prefix="grade-", dir=config.attempts_root) as directory:
            root = Path(directory)
            request = root / "request.json"
            request.write_text(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "release_dir": str(task.release_dir.resolve()),
                        "destination": task.destination,
                        "files": dict(files),
                    }
                )
            )
            request.chmod(0o600)
            with (root / "stdout").open("w+b") as stdout, (root / "stderr").open("w+b") as stderr:
                process = subprocess.Popen(
                    [*config.command, str(request)],
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=config.grader_deadline_s or 7200)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    return DiscardGroup("grader_timeout")
                if process.returncode:
                    return DiscardGroup("grader_command_failed")
                stdout.seek(0)
                raw = stdout.read(65537)
            if len(raw) > 65536:
                return DiscardGroup("grader_response_invalid")
            value = json.loads(raw)
            if value.get("label_valid") is False:
                return DiscardGroup(_reason_code(value.get("reason_code", "unlabelled")))
            if value.get("label_valid") is not True:
                return DiscardGroup("grader_response_invalid")
            return ValidReward(value["reward"], value.get("metrics", {}))
    except Exception:
        logger.exception("trusted grader command failed for %s", task.task_id)
        return DiscardGroup("grader_exception")
