"""Replay registered Airbyte connections and dbt on private DuckDB populations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import time
import traceback
from urllib.parse import urlsplit
from uuid import UUID, uuid4

import yaml
from elt_taskgen.models import task_from_json
from elt_taskgen.runtime import source_server
from elt_taskgen.runtime.airbyte import (
    AIRBYTE_TERMINAL_JOB_STATUSES, AirbyteClient, AirbyteError, AirbyteSyncFailure,
    access_token_provider,
)
from elt_taskgen.runtime.process_compliance import workspace_connection_streams
from elt_taskgen.runtime.source_environment import SourceEnvironment, _compose_document, _safe_project
from elt_taskgen.verification.upstream_eval import compare_mart

from elt_environment.airbyte_runtime import ReplayAirbyteClient, airbyte_lease
from elt_environment.duckdb_target import duckdb_config, duckdb_destination_path, duckdb_profile
from elt_environment.tasks import read_task

HIDDEN = ("primary", "resampled", "counterfactual", "stress")
DBT_IMAGE = "elt-swe-duckdb"
DBT_RUNTIME_ERROR = re.compile(
    r"could not set lock on file|conflicting lock is held|no space left on device|"
    r"failed to create (?:a )?thread|can't start new thread|Out of Memory Error:|"
    r"(?:io error|permission denied).*?(?:/duckdb/warehouse\.duckdb|/tmp/(?:target|logs))",
    re.IGNORECASE,
)
DBT_PROJECT_ERROR = re.compile(
    r"^(?:Compilation|Parsing|Dependency) Error\b|^Runtime Error\s+"
    r"(?:Could not find profile named|Syntax error near line)\b|"
    r"^Error encountered in /submission/.+/dbt_project\.yml$",
    re.MULTILINE,
)
DBT_LOG_TAIL = 12000
S3_MISSING_BUCKET = re.compile(
    r"\bThe bucket (?P<quote>['\"]?)(?P<bucket>[\w.-]+)(?P=quote) does not exist\b"
)


class DbtReplayError(RuntimeError):
    """Hidden dbt execution could not produce a reliable grading result."""

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


def sandbox(
    storage: Path, *args: str, project: Path | None = None, output: Path | None = None,
    timeout: float = 240, memory_gb: int = 2,
) -> subprocess.CompletedProcess:
    name = "elt-duckdb-grade-" + uuid4().hex
    mount = [] if project is None else ["-v", f"{project}:/submission:ro"]
    if output is not None:
        mount.extend(["-v", f"{output}:/results"])
    try:
        volume = f"{storage}:/duckdb" + (":ro" if project is None else "")
        result = subprocess.run(
            [
                "docker", "run", "--rm", "--pull=never", "--name", name, "--network", "none",
                "--cap-drop=ALL", "--security-opt=no-new-privileges", "--read-only",
                "--pids-limit=128", f"--memory={memory_gb}g", "--cpus=2", "--tmpfs", "/tmp:rw,size=256m",
                "-e", "HOME=/tmp", "-e", "DBT_SEND_ANONYMOUS_USAGE_STATS=false", *mount,
                "-v", volume, "--entrypoint", args[0], DBT_IMAGE, *args[1:],
            ],
            capture_output=True, text=True, check=False, timeout=timeout,
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


def dbt_package_manifest(project: Path) -> tuple[Path, dict] | None:
    for name in ("packages.yml", "dependencies.yml"):
        path = project / name
        if not path.is_file():
            continue
        try:
            document = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            return None
        if isinstance(document, dict) and isinstance(document.get("packages"), list):
            return path, document
    return None


def local_dbt_package(root: Path, project: Path, location: object) -> Path | None:
    if not isinstance(location, str) or not location:
        return None
    relative = PurePosixPath(location)
    if relative.is_absolute():
        if not relative.is_relative_to("/workspace"):
            return None
        relative = relative.relative_to("/workspace")
        if not relative.parts or relative.parts[0] != "elt":
            relative = PurePosixPath("elt") / relative
        candidate = root / relative
    else:
        candidate = project / relative
    candidate = candidate.resolve()
    if (candidate.is_relative_to(root.resolve()) and candidate != project.resolve()
            and (candidate / "dbt_project.yml").is_file()):
        return candidate
    return None


def vendored_dbt_copies(project: Path, projects: set[Path], declaration: dict):


    reference = declaration.get("local") or declaration.get("package")
    if not isinstance(reference, str):
        return
    name = PurePosixPath(reference).name
    for candidate in projects:
        if candidate == project or candidate.name != name or not candidate.is_relative_to(project):
            continue
        try:
            document = yaml.safe_load((candidate / "dbt_project.yml").read_text())
        except yaml.YAMLError:
            continue
        if isinstance(document, dict) and document.get("name") == name:
            yield candidate


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
    projects = {path.parent.resolve() for path in (root / "elt").rglob("dbt_project.yml")}
    dependencies = set()
    for project in projects:
        declared = dbt_package_manifest(project)
        if not declared:
            continue
        for package in declared[1]["packages"]:
            if not isinstance(package, dict):
                continue
            if local := local_dbt_package(root, project, package.get("local")):
                dependencies.add(local)
            dependencies.update(vendored_dbt_copies(project, projects, package))
    projects.difference_update(dependencies)
    if len(projects) != 1:
        return None
    (root / "profiles.yml").write_text(yaml.safe_dump(duckdb_profile()))
    return root / projects.pop().relative_to(root.resolve())


def dbt_replay_status(result: subprocess.CompletedProcess) -> str:
    errors = []
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict) or not isinstance(info := event.get("info"), dict):
            continue
        if info.get("level") in ("error", "fatal"):
            data = event.get("data")
            errors.append((info.get("name"), data if isinstance(data, dict) else {}, info.get("msg", "")))
    if any(
        DBT_RUNTIME_ERROR.search(str(data.get("exc") or data.get("msg") or message))
        for _, data, message in errors
    ):
        return "grader_error"
    if result.returncode == 0:
        return "success"
    startup_errors = [data.get("exc", "") for name, data, _ in errors if name == "MainEncounteredError"]
    if startup_errors:
        if result.returncode in (1, 2) and all(DBT_PROJECT_ERROR.search(str(error)) for error in startup_errors):
            return "project_error"
        return "grader_error"
    if result.returncode == 1 and any(name == "RunResultError" for name, _, _ in errors):
        return "model_error"
    return "grader_error"


def log_dbt(population: str, status: str, **details) -> None:
    print(json.dumps({"phase": "hidden_dbt", "population": population, "status": status, **details}),
          file=sys.stderr)


def log_tail(output: str | bytes | None) -> str:
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    return (output or "")[-DBT_LOG_TAIL:]


DBT_WITH_LOCAL_PACKAGES = """
cp -R /submission /tmp/dbt-source
project="/tmp/dbt-source/$1"
dbt --no-use-colors --log-format json deps --project-dir "$project" \
    --profiles-dir /tmp/dbt-source --log-path /tmp/logs
exec dbt --no-use-colors --log-format json run --project-dir "$project" \
    --profiles-dir /tmp/dbt-source --target-path /tmp/target --log-path /tmp/logs
"""


def run_dbt(storage: Path, root: Path, project: Path, population: str, *,
            install_packages: bool = False, timeout: float = 240, memory_gb: int = 2) -> None:
    relative_project = project.relative_to(root).as_posix()
    arguments = (
        "dbt", "--no-use-colors", "--log-format", "json", "run",
        "--project-dir", f"/submission/{relative_project}", "--profiles-dir", "/submission",
        "--target-path", "/tmp/target", "--log-path", "/tmp/logs",
    )
    if install_packages:
        arguments = ("bash", "-ec", DBT_WITH_LOCAL_PACKAGES, "dbt-replay", relative_project)
    try:
        result = sandbox(storage, *arguments, project=root, timeout=timeout, memory_gb=memory_gb)
    except (OSError, subprocess.SubprocessError) as error:
        log_dbt(
            population, "grader_error", error=type(error).__name__,
            returncode=getattr(error, "returncode", None),
            stdout=log_tail(getattr(error, "stdout", None)),
            stderr=log_tail(getattr(error, "stderr", None)),
        )
        raise DbtReplayError from error
    status = dbt_replay_status(result)
    log_dbt(population, status, returncode=result.returncode,
            stdout=log_tail(result.stdout), stderr=log_tail(result.stderr))
    if status == "grader_error":
        raise DbtReplayError


def settle_jobs(client: AirbyteClient, connections: list[str], *, timeout: float = 480,
                cancel: bool = False) -> None:
    deadline = time.monotonic() + timeout
    for connection in connections:
        jobs = client.request("GET", f"jobs?connectionId={connection}&limit=100")
        if not isinstance(jobs, Mapping) or not isinstance(jobs.get("data"), list):
            raise ValueError("Airbyte did not return a development job list")
        for entry in jobs["data"]:
            if entry.get("status") in AIRBYTE_TERMINAL_JOB_STATUSES:
                continue
            job = int(entry["jobId"])
            if cancel:
                try:
                    client.request("DELETE", f"jobs/{job}")
                except AirbyteError:
                    if client.job_status(job) not in AIRBYTE_TERMINAL_JOB_STATUSES:
                        raise
            while client.job_status(job) not in AIRBYTE_TERMINAL_JOB_STATUSES:
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


def missing_required_streams(required: set[str], connections: list[dict]) -> set[str] | None:
    """Return missing raw tables when Airbyte exposes every connection's stream selection."""
    for connection in connections:
        config = connection.get("configurations")
        streams = config.get("streams") if isinstance(config, Mapping) else None
        if not isinstance(streams, list) or any(
            not isinstance(stream, Mapping) or not isinstance(stream.get("name"), str)
            or not stream["name"] for stream in streams
        ):
            return None
    selected = {
        stream.casefold()
        for names in workspace_connection_streams(connections).values()
        for stream in names
    }
    return {table for table in required if table.casefold() not in selected}


def invalid_s3_stream(
    client: AirbyteClient, error: AirbyteSyncFailure, hidden: SourceEnvironment,
) -> bool:
    """Prove the submitted S3 connection requests an absent bucket or literal object."""
    try:
        job = client.request("POST", "../../v1/jobs/get", {"id": error.job_id})
        failures = [
            failure for record in job.get("attempts", [])
            for failure in (record.get("attempt", record).get("failureSummary") or {}).get("failures", [])
        ]
        messages = [
            failure.get("internalMessage") or "" for failure in failures
            if failure.get("failureOrigin") == "source"
            and failure.get("failureType") == "config_error"
        ]
        missing_buckets = {
            match["bucket"] for message in messages for match in S3_MISSING_BUCKET.finditer(message)
        }
        missing_objects = any("No files were identified in the stream" in msg for msg in messages)
        if not missing_buckets and not missing_objects:
            return False
        connection = client.request("GET", f"connections/{error.connection_id}")
        source = client.request("GET", f"sources/{connection['sourceId']}")
        settings = source["configuration"]
        bucket = settings.get("bucket")
        if not isinstance(bucket, str) or not bucket:
            return False
        manifest = json.loads(hidden.manifest_path.read_text())["tables"]
        if bucket in missing_buckets:
            public = hidden.release_dir / "public" / hidden.task_id / "config.yaml"
            config = yaml.safe_load(public.read_text())
            urls = [urlsplit(row["path"]) for row in config["aws_s3"]["data"]]
            expected = {url.netloc for url in urls}
            seeded = {entry["bucket"] for entry in manifest.values() if entry.get("backend") == "s3"}
            if (
                urls and all(url.scheme == "s3" and url.netloc for url in urls)
                and expected == seeded and bucket not in seeded
            ):
                return True
        if not missing_objects:
            return False
        native = [(name, entry) for name, entry in manifest.items()
                  if entry.get("backend") == "s3" and entry.get("bucket") == bucket]
        if not native:
            return False
        objects = set()
        for name, entry in native:
            directory = hidden._artifact_dir(entry, "rendered_dir", name)
            pattern = entry.get("rendered_parts_glob") or entry.get("parts_glob") or "part-*.jsonl"
            parts = list(directory.glob(pattern))
            if not parts:
                return False
            if entry.get("object_key"):
                objects.add(entry["object_key"].strip("/"))
            else:
                prefix = (entry.get("key_prefix") or name).strip("/")
                objects.update(f"{prefix}/{part.name}" for part in parts)
        for name, entry in manifest.items():
            if entry.get("backend") == "files":
                objects.add(hidden._artifact_file(entry, "rendered_file", name).name)

        selected = (connection.get("configurations") or {}).get("streams")
        if not isinstance(selected, list):
            return False
        stream_names = {stream["name"] for stream in selected}
        for stream in settings.get("streams", []):
            patterns = stream.get("globs")
            if stream.get("name") not in stream_names or not isinstance(patterns, list):
                continue
            if patterns and all(
                isinstance(pattern, str) and pattern and not set(pattern) & set("*?[]{}!\\")
                and pattern == pattern.strip("/")
                and pattern not in objects for pattern in patterns
            ):
                return True
    except (AirbyteError, KeyError, TypeError, AttributeError, ValueError, OSError, yaml.YAMLError):

        pass
    return False


def registered_submission(request: dict, runtime: dict, client: AirbyteClient, schema: str):
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
    if not any(name.endswith(".tf") for name in request["files"]):
        connections = []
    expected_path = duckdb_destination_path(storage, runtime.get("airbyte_directory", "/local"))
    for id_ in {registered[key]["destinationId"] for key in connections}:
        destination = client.request("GET", f"destinations/{UUID(id_)}")
        if (
            destination.get("definitionId") != runtime["definition_id"]
            or destination["configuration"].get("destination_path") != expected_path
            or destination["configuration"].get("schema", "main") != schema
            or destination["configuration"].get("motherduck_api_key")
        ):
            raise ValueError("registered destination is outside the attempt DuckDB path")
    return storage, registered, connections


@contextmanager
def fresh_storage(storage: Path):

    with tempfile.TemporaryDirectory(dir=storage.parent, prefix="replay-") as scratch:
        storage.rename(Path(scratch) / "previous")
        storage.mkdir(mode=0o733)
        storage.chmod(0o733)
        yield


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
    storage, registered, connections = registered_submission(request, runtime, client, schema)
    if not connections:
        return reward(False, 0.0)

    required = {table.name for table in ir.tables}
    missing = missing_required_streams(required, [registered[key] for key in connections])
    if missing is None:
        missing = missing_required_streams(required, [
            client.request("GET", f"connections/{key}") for key in connections
        ])
    if missing:
        result = reward(False, 0.0)
        result["metrics"]["missing_stream_count"] = float(len(missing))
        return result

    project_root = temp / "submission"
    project = stage_submission(request["files"], project_root)
    settle_jobs(client, sorted(registered), timeout=180, cancel=True)
    source.stop()
    loaded, passed = [], []
    for population in HIDDEN:
        with fresh_storage(storage):
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
                    if invalid_s3_stream(client, error, hidden):
                        result = reward(False, 0.0)
                        result["metrics"]["s3_stream_configuration_error"] = 1.0
                        return result

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
                    log_dbt(population, "compared", readable_marts=sorted(observed),
                            matching_marts=sorted(matches))
                else:
                    passed.append(set())
                    if el:
                        log_dbt(population, "missing_unique_project")
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
        client = ReplayAirbyteClient(
            airbyte,
            token_provider=access_token_provider(
                airbyte, credentials["client_id"], credentials["client_secret"],
            ),
            timeout=45,
        )
        with airbyte_lease("elt-bench-grading.lock", Path(runtime["local_root"])):
            with tempfile.TemporaryDirectory(prefix="duckdb-replay-", dir=args.request.parent) as temp:
                result = replay(request, runtime, client, Path(temp))
    except DbtReplayError:
        result = {"label_valid": False, "reason_code": "duckdb_dbt_replay_failed"}
    except Exception as error:
        print(f"DuckDB replay failed: {type(error).__name__}", file=sys.stderr)
        for frame in traceback.extract_tb(error.__traceback__)[-8:]:
            print(f"  {Path(frame.filename).name}:{frame.lineno} in {frame.name}", file=sys.stderr)
        result = {"label_valid": False, "reason_code": "duckdb_replay_failed"}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
