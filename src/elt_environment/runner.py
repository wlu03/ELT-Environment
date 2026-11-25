"""Own an attempt's process, public workspace, provider bridge and cleanup."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import shutil
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
from elt_environment.duckdb_target import stage_duckdb_input, validate_duckdb_workspace

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
    duckdb_local_root: str | None = None
    duckdb_control_port: int | None = None
    max_turns: int = 100
    max_tokens: int = 10240
    context_length: int = 65536
    timeout_s: float = 7200
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
    images = set()
    for task in tasks:
        validate_input(config, task)
        images.add(agent_image(config, task.destination))
    if docker:
        for args in (
            ["info"],
            *(["image", "inspect", image] for image in sorted(images or {agent_image(config)})),
            ["network", "inspect", config.network],
        ):
            result = subprocess.run(["docker", *args], capture_output=True, timeout=30, check=False)
            if result.returncode:
                raise ValueError(f"Docker preflight failed: docker {' '.join(args)}")


def agent_image(config: AgentConfig, destination: str | None = None) -> str:
    return config.image or ("elt-swe-duckdb" if destination == "duckdb" else IMAGES[config.agent])


def duckdb_storage_root(config: AgentConfig) -> Path:
    if not config.duckdb_local_root:
        raise ValueError("set duckdb_local_root to the host directory mounted as /local by Airbyte")
    root = Path(config.duckdb_local_root).resolve(strict=True)
    if not root.is_dir() or ":" in str(root):
        raise ValueError("duckdb_local_root must be a Docker-mountable directory")
    return root


def validate_input(config: AgentConfig, task):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task.task_id):
        raise ValueError("task id must be a single safe path component")
    if config.agent == "spider" and task.destination != "snowflake":
        raise ValueError("the original Spider tools support Snowflake only")
    if task.destination == "duckdb":
        if config.agent != "swe":
            raise ValueError("DuckDB development is supported with agent=swe")
        duckdb_storage_root(config)
    source = Path(config.inputs_root).resolve() / task.task_id
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"installed input missing: {task.task_id}")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("installed task inputs must not contain symlinks")
        if path.name in {"private", "answer_key", "gold", "oracle", "task_ir.json"}:
            raise ValueError("inputs_root must contain only installed public development inputs")
    values = yaml.safe_load((source / "config.yaml").read_text())
    if task.destination not in values:
        raise ValueError(f"installed input destination mismatch: {task.task_id}")
    if task.destination == "duckdb":
        validate_duckdb_workspace(values)
    serialized = json.dumps(values)
    if "<" in serialized and ">" in serialized:
        raise ValueError(f"install runtime credentials before running {task.task_id}")
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


def collect_artifact(workspace: Path, dbt_project: Path | None = None) -> dict[str, str]:
    """Collect declarative source, excluding runtime caches and warehouse profiles.

    The existing private taskgen grader owns the allowed artifact surface and
    validates every submitted path. Extra Terraform/dbt source is retained so
    an unsupported submission is rejected rather than silently graded partly.
    """
    if workspace.is_symlink() or (workspace / "elt").is_symlink():
        raise SubmissionViolation("artifact root symlink")
    files = {}
    total = 0
    roots = [(workspace / "elt", Path("elt"))]
    if dbt_project is not None:
        roots.append((dbt_project, Path("elt") / dbt_project.name))
    for root, destination in roots:
        for parent, directories, names in os.walk(root, followlinks=False):
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
                suffixes = {".sql", ".yml", ".yaml"}
                if root != dbt_project:
                    suffixes.add(".tf")
                if name == "profiles.yml" or path.suffix not in suffixes:
                    continue
                relative = str(destination / path.relative_to(root))
                if relative in files:
                    raise SubmissionViolation("artifact path collision")
                total += path.stat().st_size
                if total > 4 * 1024 * 1024 or len(files) >= 512:
                    raise SubmissionViolation("artifact size limit")
                try:
                    files[relative] = path.read_text()
                except UnicodeError as error:
                    raise SubmissionViolation("artifact is not UTF-8") from error
    return files


def submission_artifact(workspace: Path, baseline: dict, *, swe: bool, duckdb: bool):
    snapshot = public_snapshot(workspace)
    added = snapshot.keys() - baseline.keys()
    allowed = set()
    project = None
    if swe:
        allowed.update(
            path for path in added
            if path in {"model.patch", ".user.yml"} or re.fullmatch(r"logs/dbt\.log(?:\.\d+)?", path)
        )
    if duckdb:
        protected_roots = {Path(path).parts[0] for path in baseline} | {
            "elt", "schemas", "documentation", "logs",
        }
        allowed.update(
            path for path in added
            if len(Path(path).parts) == 1 and Path(path).suffix == ".sql"
        )
        projects = [
            Path(path).parent for path in sorted(added)
            if len(Path(path).parts) == 2 and Path(path).name == "dbt_project.yml"
            and Path(path).parts[0] not in protected_roots
        ]
        if len(projects) == 1:
            candidate = projects[0]
            project = workspace / candidate
            allowed.update(path for path in added if Path(path).parts[0] == candidate.name)
    if {path: digest for path, digest in snapshot.items() if path not in allowed} != baseline:
        raise SubmissionViolation("public input changed")
    return collect_artifact(workspace, project)


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
        self.duckdb_storage = None
        self.started = 0.0
        self.stopped = False

    async def start(self):
        self.config.validate()
        validate_input(self.config, self.task)
        attempts = Path(self.config.attempts_root).resolve()
        attempts.mkdir(parents=True, exist_ok=True, mode=0o700)
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
        inputs = Path(self.config.inputs_root).resolve()
        if self.task.destination == "duckdb":
            self.duckdb_storage = Path(
                tempfile.mkdtemp(prefix="elt-duckdb-", dir=duckdb_storage_root(self.config))
            )
            self.duckdb_storage.chmod(0o733)
            staged = self.root / "inputs"
            stage_duckdb_input(
                inputs / self.task.task_id, staged / self.task.task_id, self.duckdb_storage
            )
            inputs = staged
        self.snapshot = public_snapshot(inputs / self.task.task_id)
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
            "image": agent_image(self.config, self.task.destination),
            "inputs_root": str(inputs),
            "duckdb_storage": str(self.duckdb_storage) if self.duckdb_storage else None,
            "agents_root": str(Path(self.config.agents_root).resolve()),
        }
        spec_file = self.root / "worker.json"
        spec_file.write_text(json.dumps(spec))
        spec_file.chmod(0o600)
        self.log = (self.root / "worker.log").open("wb")
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
        return submission_artifact(
            self.workspace, self.snapshot,
            swe=self.config.agent == "swe", duckdb=self.task.destination == "duckdb",
        )

    def duckdb_replay_receipt(self) -> dict:
        from elt_taskgen.runtime.process_compliance import (
            ProcessComplianceError,
            find_terraform_state,
            terraform_state_connections,
        )

        state = find_terraform_state(self.workspace / "elt")
        try:
            ids = (
                sorted(terraform_state_connections(state))
                if state and state.stat().st_size <= 4 * 1024 * 1024 else []
            )
        except ProcessComplianceError:
            ids = []
        return {"connection_ids": ids, "storage": str(self.duckdb_storage)}

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
            try:
                if self.log:
                    self.log.close()
                    self.log = None
                if self.root:
                    (self.root / "worker.json").unlink(missing_ok=True)
                if self.duckdb_storage and self.stopped:
                    shutil.rmtree(self.duckdb_storage)
                    self.duckdb_storage = None
            finally:
                if self.lock:
                    self.lock.close()
                    self.lock = None
