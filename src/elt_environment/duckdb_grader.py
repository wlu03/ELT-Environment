"""Replay registered Airbyte connections and dbt on private DuckDB populations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import time
from uuid import UUID, uuid4

import yaml
from elt_taskgen.models import task_from_json
from elt_taskgen.runtime import source_server
from elt_taskgen.runtime.airbyte import (
    AirbyteClient, AirbyteSyncFailure, access_token_provider,
)
from elt_taskgen.runtime.source_environment import SourceEnvironment, _compose_document, _safe_project
from elt_taskgen.verification.upstream_eval import compare_mart

from elt_environment.duckdb_target import duckdb_config, duckdb_profile
from elt_environment.tasks import read_task

HIDDEN = ("primary", "resampled", "counterfactual", "stress")
DBT_IMAGE = "elt-swe-duckdb"
TERMINAL_JOBS = {"succeeded", "failed", "cancelled", "canceled", "incomplete"}
READ_WAREHOUSE = """
import json
import sys
import duckdb

schema, counts, marts = json.loads(sys.argv[1])
con = duckdb.connect('/duckdb/warehouse.duckdb', read_only=True, config={
    'enable_external_access': 'false', 'threads': '1', 'memory_limit': '768MB',
})
def ident(value):
    return '"' + value.replace('"', '""') + '"'
relations = {row[0].casefold(): row[0] for row in con.execute(
    'SELECT table_name FROM information_schema.tables WHERE lower(table_schema) = lower(?)',
    [schema],
).fetchall()}
result = {}
for name in counts + marts:
    physical = '_airbyte_raw_' + name if counts else name
    if physical.casefold() not in relations:
        continue
    relation = ident(schema) + '.' + ident(relations[physical.casefold()])
    if counts:
        result[name] = con.execute('SELECT count(*) FROM ' + relation).fetchone()[0]
        continue
    try:
        query = con.execute('SELECT * FROM ' + relation)
        rows = query.fetchmany(100001)
        if len(rows) <= 100000:
            result[name] = [[item[0] for item in query.description], rows]
    except duckdb.Error:
        pass
print(json.dumps(result, default=str))
"""


def sandbox(storage: Path, *args: str, project: Path | None = None) -> subprocess.CompletedProcess:
    name = "elt-duckdb-grade-" + uuid4().hex
    mount = [] if project is None else ["-v", f"{project}:/submission:ro"]
    try:
        volume = f"{storage}:/duckdb" + (":ro" if project is None else "")
        result = subprocess.run(
            [
                "docker", "run", "--rm", "--pull=never", "--name", name, "--network", "none",
                "--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only",
                "--pids-limit=128", "--memory=2g", "--cpus=2", "--tmpfs", "/tmp:rw,size=256m",
                "-e", "HOME=/tmp", "-e", "DBT_SEND_ANONYMOUS_USAGE_STATS=false", *mount,
                "-v", volume, "--entrypoint", args[0], DBT_IMAGE, *args[1:],
            ],
            capture_output=True, text=True, check=False, timeout=240,
        )
        if result.returncode in (125, 126, 127):
            result.check_returncode()
        return result
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15, check=False)


def warehouse_rows(storage: Path, schema: str, counts: list[str], marts: list[str]) -> dict:
    result = sandbox(storage, "python", "-c", READ_WAREHOUSE, json.dumps([schema, counts, marts]))
    result.check_returncode()
    return json.loads(result.stdout)


def stage_submission(files: dict[str, str], root: Path) -> Path | None:
    if not isinstance(files, dict) or len(files) > 512:
        raise ValueError("invalid submission")
    size = 0
    for name, content in files.items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute() or len(relative.parts) < 2 or relative.parts[0] != "elt"
            or ".." in relative.parts or "\\" in name or relative.as_posix() != name
            or relative.suffix not in {".tf", ".sql", ".yml", ".yaml"}
            or relative.name == "profiles.yml" or not isinstance(content, str)
        ):
            raise ValueError("invalid submission path")
        size += len(content.encode("utf-8"))
        if size > 4 * 1024 * 1024:
            raise ValueError("invalid submission size")
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    projects = list((root / "elt").rglob("dbt_project.yml"))
    if len(projects) != 1:
        return None
    (root / "profiles.yml").write_text(yaml.safe_dump(duckdb_profile()))
    return projects[0].parent


def run_dbt(storage: Path, root: Path, project: Path, population: str) -> None:
    result = sandbox(
        storage, "dbt", "run", "--project-dir",
        f"/submission/{project.relative_to(root).as_posix()}",
        "--profiles-dir", "/submission", "--target-path", "/tmp/target",
        "--log-path", "/tmp/logs", project=root,
    )
    if result.returncode not in (0, 1, 2):
        result.check_returncode()


def settle_jobs(client: AirbyteClient, connections: list[str]) -> None:
    deadline = time.monotonic() + 480
    for connection in connections:
        jobs = client.request("GET", f"jobs?connectionId={connection}&limit=100")
        if not isinstance(jobs, Mapping) or not isinstance(jobs.get("data"), list):
            raise ValueError("Airbyte did not return a development job list")
        for entry in jobs["data"]:
            if entry.get("status") in TERMINAL_JOBS:
                continue
            job = int(entry["jobId"])
            while client.job_status(job) not in TERMINAL_JOBS:
                if time.monotonic() >= deadline:
                    raise TimeoutError("development Airbyte jobs did not stop")
                time.sleep(3)


def reward(el: bool, marts: float) -> dict:
    phase = "el" if not el else "t" if marts < 1.0 else "pass"
    return {
        "label_valid": True,
        "reward": (float(el) + marts) / 2,
        "metrics": {"el_pass": float(el), "r_el": float(el), "r_t": marts, f"phase_{phase}": 1.0},
    }


def replay(request: dict, runtime: dict, client: AirbyteClient, temp: Path) -> dict:
    release = Path(runtime["release_dir"]).resolve(strict=True)
    if release != Path(request["release_dir"]).resolve() or runtime["task_id"] != request["task_id"]:
        raise ValueError("runtime does not own the requested bundle and task")
    task = read_task(release, runtime["task_id"])
    if task.destination != "duckdb" or request["destination"] != "duckdb":
        raise ValueError("DuckDB replay requires a DuckDB task")
    source = SourceEnvironment(**{
        field: Path(value) if field in {
            "release_dir", "environment_dir", "rendered_root", "manifest_path", "compose_path",
        } else value for field, value in runtime["development"].items()
    })
    if source.task_id != task.task_id or source.population != "development":
        raise ValueError("runtime does not own the development source fixtures")
    config = duckdb_config(yaml.safe_load((task.public_dir / "config.yaml").read_text()))
    if config["definition_id"] != runtime["definition_id"]:
        raise ValueError("runtime connector differs from the released connector")
    schema = config["schema"]
    private = release / "private" / task.task_id
    metadata = private / "task_ir.json"
    if not metadata.exists():
        metadata = private / "semantic" / "task_ir.json"
    ir = task_from_json(metadata.read_text())
    counts, gold = {}, {}
    for population in HIDDEN:
        answer = private / "answer_key" / "gold" / population
        counts[population] = json.loads((answer / "stage1_counts.json").read_text())
        if set(counts[population]) != {table.name for table in ir.tables}:
            raise ValueError("stage-1 gold disagrees with TaskIR")
        gold[population] = {mart.name: (answer / f"{mart.name}.csv").read_text() for mart in ir.marts}
        if not (private / "populations" / population / "rendered").is_dir() or not gold[population]:
            raise ValueError("hidden replay assets are incomplete")
    submitted = request["runtime"]
    storage = Path(submitted["storage"])
    if (
        storage.is_symlink() or storage.parent != Path(runtime["local_root"]).resolve()
        or not storage.is_dir() or not storage.name.startswith("elt-duckdb-")
    ):
        raise ValueError("attempt storage is outside the private replay runtime")
    connection_ids = {str(UUID(value)) for value in submitted["connection_ids"]}
    workspace = str(UUID(runtime["workspace_id"]))
    registered = {entry["connectionId"]: entry for entry in client.list_connections(workspace)}
    connections = sorted(connection_ids & registered.keys())
    if not connections or not any(name.endswith(".tf") for name in request["files"]):
        return reward(False, 0.0)
    expected_path = f"/local/{storage.name}/warehouse.duckdb"
    for id_ in {registered[key]["destinationId"] for key in connections}:
        destination = client.request("GET", f"destinations/{UUID(id_)}")
        if (
            destination.get("definitionId") != runtime["definition_id"]
            or destination["configuration"].get("destination_path") != expected_path
            or destination["configuration"].get("schema", "main") != schema
            or destination["configuration"].get("motherduck_api_key")
        ):
            raise ValueError("registered destination is outside the attempt DuckDB path")

    project_root = temp / "submission"
    project = stage_submission(request["files"], project_root)
    settle_jobs(client, sorted(registered))
    source.stop()
    loaded, passed = [], []
    for population in HIDDEN:
        with tempfile.TemporaryDirectory(dir=storage.parent, prefix="replay-") as scratch:
            previous = Path(scratch) / "development"
            storage.rename(previous)
            storage.mkdir(mode=0o733)
            storage.chmod(0o733)
            env_dir = temp / population
            env_dir.mkdir()
            project_name = _safe_project(task.task_id, population, str(temp))
            rendered = private / "populations" / population / "rendered"
            hidden = SourceEnvironment(
                release, task.task_id, population, env_dir, rendered, source.manifest_path,
                env_dir / "compose.yaml", project_name, project_name + "-source-network",
            )
            hidden.compose_path.write_text(yaml.safe_dump(_compose_document(
                rendered_root=rendered, manifest_path=source.manifest_path,
                server_script=Path(source_server.__file__), network=hidden.network,
            )))
            try:
                hidden.start()
                hidden.seed()
                try:
                    for connection in connections:
                        client.trigger_and_wait_receipt([connection], poll_interval=5, timeout=480)
                except AirbyteSyncFailure as error:
                    print(f"Airbyte replay failed on {population}, job {error.job_id}", file=sys.stderr)
                    return {"label_valid": False, "reason_code": "airbyte_replay_incomplete"}
                el = (storage / "warehouse.duckdb").is_file() and (
                    warehouse_rows(storage, schema, sorted(counts[population]), []) == counts[population]
                )
                loaded.append(el)
                if el and project:
                    run_dbt(storage, project_root, project, population)
                    observed = warehouse_rows(storage, "elt_models", [], sorted(gold[population]))
                    matches = {
                        mart.name for mart in ir.marts
                        if mart.name in observed and compare_mart(
                            gold[population][mart.name],
                            [dict(zip(observed[mart.name][0], row)) for row in observed[mart.name][1]],
                            mart, actual_columns=observed[mart.name][0],
                        )
                    }
                    passed.append(matches)
                else:
                    passed.append(set())
            finally:
                hidden.stop()
    mart_count = len(ir.marts)
    el_pass = all(loaded)
    result = reward(el_pass, len(set.intersection(*passed)) / mart_count if el_pass else 0.0)
    for population, el, marts in zip(HIDDEN, loaded, passed):
        result["metrics"][f"el_{population}"] = float(el)
        result["metrics"][f"t_{population}"] = len(marts) / mart_count
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    try:
        runtime = json.loads(args.runtime.read_text())
        request = json.loads(args.request.read_text())
        airbyte = runtime["airbyte_url"]
        credentials = json.loads(Path(runtime["airbyte_credentials_file"]).read_text())
        client = AirbyteClient(
            airbyte,
            token_provider=access_token_provider(
                airbyte, credentials["client_id"], credentials["client_secret"],
            ),
            timeout=45,
        )
        with tempfile.TemporaryDirectory(prefix="duckdb-replay-", dir=args.request.parent) as temp:
            result = replay(request, runtime, client, Path(temp))
    except Exception as error:
        print(f"DuckDB replay failed: {type(error).__name__}", file=sys.stderr)
        result = {"label_valid": False, "reason_code": "duckdb_replay_failed"}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
