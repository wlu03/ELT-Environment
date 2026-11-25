"""Send artifacts to a trusted hidden-replay grader and return a reward or a discard."""

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

from elt_environment.outcomes import DiscardGroup, GraderOutcome, ValidReward
from elt_environment.tasks import TaskRef, read_task

_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraderConfig:
    """Command, private scratch space and timeout for hidden replay."""

    attempts_root: Path
    command: tuple[str, ...]
    grader_deadline_s: float | None = None

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("a trusted hidden-replay grader command is required")


def _reason_code(code: str) -> str:
    return code if _REASON_CODE.fullmatch(code) else "harness_fault"


def _save_duckdb_diagnostics(stderr, path: Path, task_id: str) -> None:
    try:
        stderr.seek(0)
        output = stderr.read(512 * 1024 + 1)
        if not output:
            return
        path.parent.mkdir(mode=0o700, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as log:
            log.write(output[:512 * 1024])
            if len(output) > 512 * 1024:
                log.write(b"\n[grader diagnostics truncated]\n")
        logger.info("DuckDB grading diagnostics for %s saved to %s", task_id, path)
    except OSError:
        logger.warning("Could not save DuckDB grading diagnostics for %s", task_id, exc_info=True)


def grade_artifact(
    task: TaskRef, files: Mapping[str, str], config: GraderConfig, runtime: dict | None = None
) -> GraderOutcome:
    """Invoke an operator-configured hidden-replay grader outside the agent.

    The command gets one JSON request path and writes an outcome JSON object
    to stdout. It must replay the artifact on fresh hidden populations, rather
    than reward success on the agent-visible development warehouse.
    """
    try:
        if read_task(task.release_dir, task.task_id) != task:
            raise ValueError("task metadata changed after discovery")
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
                        **({"runtime": runtime} if runtime is not None else {}),
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
                finally:
                    if task.destination == "duckdb":
                        _save_duckdb_diagnostics(
                            stderr, config.attempts_root / "diagnostics" / f"{root.name}.log", task.task_id,
                        )
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
