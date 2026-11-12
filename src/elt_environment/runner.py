"""Own an attempt's process, public workspace, provider bridge and cleanup."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from elt_environment.bridge import Bridge

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENTS = ("spider", "swe", "openhands", "codex", "claude")
IMAGES = {
    "spider": "elt_agent-image",
    "swe": "elt-swe",
    "codex": "elt-swe",
    "claude": "elt-swe-claude",
    "openhands": "elt-swe-openhands",
}
MODULES = {
    "spider": ("openai", "docker", "gymnasium", "pandas", "requests", "yaml"),
    "swe": ("sweagent", "swerex", "litellm"),
    "codex": ("agents", "openai"),
    "claude": ("claude_agent_sdk",),
    "openhands": ("openhands.sdk", "openhands.tools", "lmnr"),
}


@dataclass(frozen=True)
class AgentConfig:
    agent: str = "spider"
    inputs_root: str = ""
    python: str = sys.executable
    agents_root: str = str(PROJECT_ROOT / "agents")
    attempts_root: str = str(PROJECT_ROOT / ".state/agent-attempts")
    network: str = "elt-docker_elt_network"
    image: str | None = None
    max_turns: int = 100
    max_tokens: int = 10240
    context_length: int = 65536
    timeout_s: float = 7200
    # Container calls must reach the host bridge. Linux hosts may supply a
    # gateway IP; Docker Desktop resolves host.docker.internal automatically.
    bridge_bind_host: str = "127.0.0.1"
    bridge_container_host: str = "host.docker.internal"

    def validate(self):
        if self.agent not in AGENTS:
            raise ValueError(f"agent must be one of {', '.join(AGENTS)}")
        if not self.inputs_root or not Path(self.inputs_root).is_dir():
            raise ValueError("inputs_root must contain installed development tasks")
        if min(self.max_turns, self.max_tokens, self.context_length, self.timeout_s) <= 0:
            raise ValueError("turn, token, context and timeout limits must be positive")
        if self.max_tokens >= self.context_length:
            raise ValueError("max_tokens must leave room for the prompt")
        if self.agent in {"claude", "openhands"} and self.bridge_bind_host == "127.0.0.1":
            raise ValueError("container model calls require bridge_bind_host=0.0.0.0")


def worker_environment() -> dict[str, str]:
    # No inherited provider/cloud keys, user PYTHONPATH, or Tinker credentials.
    names = (
        "PATH",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "SYSTEMROOT",
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_CONFIG",
    )
    return {name: os.environ[name] for name in names if name in os.environ}


def preflight(config: AgentConfig, tasks=(), *, docker: bool = True):
    config.validate()
    code = "import importlib; " + "; ".join(
        f"importlib.import_module({name!r})" for name in MODULES[config.agent]
    )
    if config.agent == "codex":
        code += "; from agents import Agent, Runner, ShellTool, ApplyPatchTool"
    checked = subprocess.run(
        [config.python, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=worker_environment(),
        check=False,
    )
    if checked.returncode:
        raise ValueError(
            f"{config.agent} dependencies are missing from {config.python}; "
            "install that agent's requirements in its worker environment"
        )
    for task in tasks:
        validate_input(config, task)
    if docker:
        for args in (
            ["info"],
            ["image", "inspect", config.image or IMAGES[config.agent]],
            ["network", "inspect", config.network],
        ):
            result = subprocess.run(["docker", *args], capture_output=True, timeout=30, check=False)
            if result.returncode:
                raise ValueError(f"Docker preflight failed: docker {' '.join(args)}")


def validate_input(config: AgentConfig, task):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task.task_id):
        raise ValueError("task id must be a single safe path component")
    if config.agent == "spider" and task.destination != "snowflake":
        raise ValueError("the original Spider tools support Snowflake only")
    source = Path(config.inputs_root).resolve() / task.task_id
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"installed input missing: {task.task_id}")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("installed task inputs must not contain symlinks")
        if path.name in {"private", "answer_key", "gold", "oracle", "release_manifest.json"}:
            raise ValueError("inputs_root must contain only installed public development inputs")
    values = yaml.safe_load((source / "config.yaml").read_text())
    if task.destination not in values:
        raise ValueError(f"installed input destination mismatch: {task.task_id}")
    serialized = json.dumps(values)
    if "<" in serialized and ">" in serialized:
        raise ValueError(f"install runtime credentials before running {task.task_id}")
    # Schemas and the requested output must match the frozen grading task.
    for name in ("data_model.yaml", "schemas"):
        public = task.public_dir / name
        installed = source / name
        if public.is_dir():
            expected = {
                p.relative_to(public): p.read_bytes() for p in public.rglob("*") if p.is_file()
            }
            actual = {
                p.relative_to(installed): p.read_bytes()
                for p in installed.rglob("*")
                if p.is_file()
            }
            if expected != actual:
                raise ValueError(f"installed {name} differs from release: {task.task_id}")
        elif not installed.is_file() or public.read_bytes() != installed.read_bytes():
            raise ValueError(f"installed {name} differs from release: {task.task_id}")


class AgentFault(RuntimeError):
    pass


class SubmissionViolation(ValueError):
    pass


def public_snapshot(root: Path) -> dict[str, str]:
    snapshot = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0] in {"elt", "codex", "claude", "openhands", "spider", ".git"}:
            continue
        if path.name.endswith("_credential.json"):
            continue
        if path.is_symlink():
            raise SubmissionViolation("public symlink")
        if path.is_file():
            snapshot[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def collect_artifact(workspace: Path) -> dict[str, str]:
    """Collect declarative source, excluding runtime caches and warehouse profiles.

    The existing private taskgen grader owns the allowed artifact surface and
    validates every submitted path. Extra Terraform/dbt source is retained so
    an unsupported submission is rejected rather than silently graded partly.
    """
    if workspace.is_symlink() or (workspace / "elt").is_symlink():
        raise SubmissionViolation("artifact root symlink")
    files = {}
    total = 0
    for parent, directories, names in os.walk(workspace / "elt", followlinks=False):
        base = Path(parent)
        for name in list(directories):
            child = base / name
            if child.is_symlink():
                raise SubmissionViolation("artifact directory symlink")
            if name in {".terraform", "target", "logs", "dbt_packages", "__pycache__"}:
                directories.remove(name)
        for name in names:
            path = base / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise SubmissionViolation("artifact must contain regular files")
            if name == "profiles.yml" or path.suffix not in {".tf", ".sql", ".yml", ".yaml"}:
                continue
            total += path.stat().st_size
            if total > 4 * 1024 * 1024 or len(files) >= 512:
                raise SubmissionViolation("artifact size limit")
            try:
                files[str(path.relative_to(workspace))] = path.read_text()
            except UnicodeError as error:
                raise SubmissionViolation("artifact is not UTF-8") from error
    return files


class AgentSession:
    def __init__(self, task, renderer, config: AgentConfig):
        self.task, self.config = task, config
        self.bridge = Bridge(renderer, bind_host=config.bridge_bind_host)
        self.process = None
        self.done = None
        self.lock = None
        self.log = None
        self.root = None
        self.container_name = None
        self.started = 0.0
        self.stopped = False

    async def start(self):
        self.config.validate()
        validate_input(self.config, self.task)
        attempts = Path(self.config.attempts_root).resolve()
        attempts.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Cross-process lock: upstream runners reset shared warehouse schemas.
        # Hold it through grading and cleanup, not just during initialization.
        lock_path = Path(self.config.inputs_root).resolve() / ".tinker-runtime.lock"
        self.lock = lock_path.open("a")
        while True:
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.1)
        self.started = time.monotonic()
        self.root = Path(tempfile.mkdtemp(prefix="agent-", dir=attempts))
        self.workspace = self.root / "workspace"
        self.snapshot = public_snapshot(Path(self.config.inputs_root) / self.task.task_id)
        await self.bridge.start()
        attempt_id = "tinker-" + uuid.uuid4().hex[:16]
        self.container_name = attempt_id + "-" + self.task.task_id
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            openhands_port = sock.getsockname()[1]
        spec = {
            **self.config.__dict__,
            "task_id": self.task.task_id,
            "destination": self.task.destination,
            "attempt_id": attempt_id,
            "attempt_root": str(self.root),
            "container_name": self.container_name,
            "workspace": str(self.workspace),
            "result_file": str(self.root / "result.json"),
            "openhands_port": openhands_port,
            "host_url": f"http://127.0.0.1:{self.bridge.port}/v1",
            "container_url": f"http://{self.config.bridge_container_host}:{self.bridge.port}/v1",
            "token": self.bridge.token,
            "image": self.config.image or IMAGES[self.config.agent],
            "inputs_root": str(Path(self.config.inputs_root).resolve()),
            "agents_root": str(Path(self.config.agents_root).resolve()),
        }
        spec_file = self.root / "worker.json"
        spec_file.write_text(json.dumps(spec))
        spec_file.chmod(0o600)
        self.log = (self.root / "worker.log").open("wb")
        # Run the standalone file so the worker does not need Tinker installed.
        self.process = await asyncio.create_subprocess_exec(
            self.config.python,
            str(Path(__file__).with_name("agent_worker.py")),
            str(spec_file),
            cwd=self.root,
            env=worker_environment(),
            stdout=self.log,
            stderr=self.log,
            start_new_session=True,
        )
        self.done = asyncio.create_task(self.process.wait())

    async def next_request(self):
        remaining = self.config.timeout_s - (time.monotonic() - self.started)
        if remaining <= 0:
            raise AgentFault("agent_wall_timeout")
        pending = asyncio.create_task(self.bridge.pending.get())
        try:
            completed, _ = await asyncio.wait(
                [pending, self.done, self.bridge.failure],
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not completed:
                raise AgentFault("agent_wall_timeout")
            if self.bridge.failure in completed:
                raise AgentFault(self.bridge.failure.result())
            if self.done in completed:
                if self.done.result() != 0:
                    raise AgentFault("agent_worker_failed")
                return None
            return pending.result()
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    def artifact(self):
        if public_snapshot(self.workspace) != self.snapshot:
            raise SubmissionViolation("public input changed")
        return collect_artifact(self.workspace)

    async def stop_worker(self):
        if self.stopped:
            return
        if self.process is not None and self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                await asyncio.wait_for(asyncio.shield(self.done), timeout=5)
            except TimeoutError:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.done
            except ProcessLookupError:
                pass
        if self.container_name:
            # The upstream finally block may not run after cancellation.
            process = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "-f",
                self.container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(process.wait(), timeout=15)
            except TimeoutError:
                process.kill()
                await process.wait()
                raise AgentFault("container_cleanup_failed")
            if process.returncode != 0:
                # A container already removed by the runner is fine; a still
                # running one or an unreachable daemon must stop the run.
                check = await asyncio.create_subprocess_exec(
                    "docker",
                    "ps",
                    "-q",
                    "--filter",
                    f"name=^{self.container_name}$",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    output, _ = await asyncio.wait_for(check.communicate(), timeout=15)
                except TimeoutError:
                    check.kill()
                    await check.wait()
                    raise AgentFault("container_cleanup_failed") from None
                if check.returncode or output.strip():
                    raise AgentFault("container_cleanup_failed")
        self.stopped = True

    async def close(self):
        try:
            try:
                await self.stop_worker()
            finally:
                await self.bridge.close()
        finally:
            if self.log:
                self.log.close()
                self.log = None
            if self.root:
                (self.root / "worker.json").unlink(missing_ok=True)
            if self.lock:
                self.lock.close()
                self.lock = None
