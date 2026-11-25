"""Replay an upstream ELT-Bench submission in DuckDB and apply its evaluator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import pandas as pd
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from sqlglot import exp, parse_one
import yaml
from elt_taskgen.runtime.airbyte import (
    AirbyteClient, AirbyteError, AirbyteSyncConflict, AirbyteSyncFailure, access_token_provider,
)

from elt_environment.airbyte_runtime import ReplayAirbyteClient, airbyte_lease
from elt_environment.duckdb_grader import (
    DbtReplayError, dbt_package_manifest, fresh_storage, local_dbt_package, registered_submission,
    reward, run_dbt, sandbox, settle_jobs, stage_submission, warehouse_rows,
)
from elt_environment.duckdb_target import duckdb_config
from elt_environment.tasks import read_task


EXPORT_MARTS = """
import json
from pathlib import Path
import sys
import duckdb

with duckdb.connect('/duckdb/warehouse.duckdb', read_only=True, config={
    'enable_external_access': 'false', 'threads': '1', 'memory_limit': '768MB',
    'temp_directory': '/results/.spill',
}) as database:
    for index, sql in enumerate(json.loads(sys.argv[1])):
        try:
            results = database.execute(sql).fetchdf()
        except (duckdb.CatalogException, duckdb.BinderException, duckdb.ConversionException) as error:
            print(f'Mart {index} could not be read: {type(error).__name__}', file=sys.stderr)
            continue
        results.to_csv(Path('/results') / f'{index}.csv', index=False)
"""


def mart_queries(evaluation: Path, task: str, marts: list[str]) -> list[str]:
    """Port the upstream evaluator's logical task schema to dbt's output schema."""
    queries = []
    for mart in marts:
        path = evaluation / "sql" / task / f"{mart}.sql"
        default = exp.select("*").from_(exp.to_table(mart, quoted=True))
        query = parse_one(path.read_text(), read="snowflake") if path.is_file() else default
        if not isinstance(query, exp.Select):
            raise ValueError(f"{path}: expected evaluation SELECT")
        for table in query.find_all(exp.Table):
            if table.catalog or table.db.casefold() not in {"", task.casefold()}:
                raise ValueError(f"{path}: unexpected evaluation schema")
            table.set("db", exp.to_identifier("elt_models"))
        queries.append(query.sql(dialect="duckdb"))
    return queries


def compare_marts(storage: Path, evaluation: Path, gold: Path, task: str, output: Path, *,
                  timeout: float = 240):
    outputs = sorted(gold.glob("*.csv"))
    if not outputs:
        raise ValueError(f"{task}: no benchmark reference outputs")
    marts = [path.stem for path in outputs]
    output.mkdir()
    result = sandbox(storage, "python", "-c", EXPORT_MARTS,
                     json.dumps(mart_queries(evaluation, task, marts)), output=output, timeout=timeout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    result.check_returncode()

    spec = importlib.util.spec_from_file_location("elt_bench_stage2", evaluation / "eva_stage2.py")
    if spec is None or spec.loader is None:
        raise ImportError("ELT-Bench Stage 2 evaluator is unavailable")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    sort_keys = evaluator.load_sort_keys(str(evaluation / "sort_key.json")).get(task, {})
    tables = {}
    for index, path in enumerate(outputs):
        observed = output / f"{index}.csv"
        if not observed.is_file():
            tables[path.stem] = {"match": False, "status": "unreadable"}
            continue
        keys = sort_keys.get(path.stem, [])
        expected = evaluator.sort_by_keys(pd.read_csv(path, dtype=str), keys)
        actual = evaluator.sort_by_keys(pd.read_csv(observed, dtype=str), keys)
        tables[path.stem] = evaluator.check_corretness(expected, actual)
    return tables


DBT_UTILS_PATH = Path("vendor/dbt_utils")


def copy_dbt_utils(package: Path, target: Path) -> None:
    target.mkdir(parents=True)
    for name in ("dbt_project.yml", "LICENSE"):
        shutil.copyfile(package / name, target / name)
    shutil.copytree(package / "macros", target / "macros")


def uses_offline_dbt_utils(declaration: dict) -> bool:
    if declaration.get("local") in (
        "/workspace/vendor/dbt_utils", "../vendor/dbt_utils", "../../vendor/dbt_utils",
    ):
        return True
    if declaration.get("package") != "dbt-labs/dbt_utils":
        return False
    version = declaration.get("version")
    constraints = [version] if isinstance(version, str) else version
    if not isinstance(constraints, list) or not constraints or not all(
        isinstance(value, str) and value.strip() for value in constraints
    ):
        return False
    try:
        normalized = (value if value[0] in "<>=!~" else f"=={value}"
                      for value in (value.strip() for value in constraints))
        return "1.3.0" in SpecifierSet(",".join(normalized))
    except InvalidSpecifier:
        return False


def prepare_dbt_packages(root: Path, project: Path, dbt_utils: Path | None) -> bool:
    """Prepare declared offline dependencies for dbt deps inside the replay sandbox."""
    project = project.resolve()
    manifest = dbt_package_manifest(project)
    if not manifest or not manifest[1]["packages"]:
        return False
    path, document = manifest
    trusted = (root / DBT_UTILS_PATH).resolve()
    packages = []
    for declaration in document["packages"]:
        if not isinstance(declaration, dict):
            return False
        local = local_dbt_package(root, project, declaration.get("local"))
        if local is None and dbt_utils and uses_offline_dbt_utils(declaration):
            local = trusted
        if local is None:
            return False
        packages.append(local)
    if dbt_utils and trusted in packages:
        copy_dbt_utils(dbt_utils, trusted)
    document["packages"] = [{"local": os.path.relpath(local, project)} for local in packages]
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return True


def wait_for_airbyte_slot(client: AirbyteClient) -> None:


    deadline = time.monotonic() + 120
    while any(
        client.request("GET", f"jobs?jobType=sync&status={status}&limit=1")["data"]
        for status in ("incomplete", "pending", "running")
    ):
        if time.monotonic() >= deadline:
            print("Airbyte remained busy, starting the serialized benchmark replay", file=sys.stderr)
            return
        time.sleep(5)


def replay_connections(client: AirbyteClient, registered: list[str], connections: list[str],
                       store: Path, *, sync_timeout: float = 1500) -> list[int]:
    for attempt in range(2):

        settle_jobs(client, registered, timeout=180, cancel=True)
        wait_for_airbyte_slot(client)
        jobs = []
        with fresh_storage(store):
            try:
                for index, connection in enumerate(connections):
                    if index:
                        wait_for_airbyte_slot(client)
                    receipt = client.trigger_and_wait_receipt(
                        [connection], poll_interval=5, timeout=sync_timeout,
                    )
                    jobs.append(receipt.job_ids[connection])
                    print(json.dumps({"stage": "airbyte", "job_ids": receipt.job_ids}), file=sys.stderr)
            except AirbyteSyncConflict:
                if attempt:
                    raise
            else:
                return jobs

        print("Airbyte replay start conflicted, retrying from an empty DuckDB file", file=sys.stderr)
        time.sleep(5)


def failed_attempts(client: AirbyteClient, job_id: int) -> list[dict]:
    job = client.request("POST", "../../v1/jobs/get", {"id": job_id})
    return [item["attempt"] for item in job["attempts"] if item["attempt"]["status"] == "failed"]


def replay_rows(client: AirbyteClient, registered: list[str], connections: list[str], store: Path,
                schema: str, counts: dict, *, sync_timeout: float = 1500) -> dict:
    for attempt in range(2):
        try:
            jobs = replay_connections(client, registered, connections, store, sync_timeout=sync_timeout)
        except AirbyteSyncFailure as error:
            if attempt or not any(
                failure.get("failureType") == "system_error"
                for item in failed_attempts(client, error.job_id)
                for failure in (item.get("failureSummary") or {}).get("failures", [])
            ):
                raise
            print(f"Airbyte replay job {error.job_id} hit a system error, retrying replay", file=sys.stderr)
            continue
        actual = (warehouse_rows(store, schema, sorted(counts), [])
                  if (store / "warehouse.duckdb").is_file() else {})
        print(json.dumps({"stage": "el", "expected": counts, "actual": actual}), file=sys.stderr)
        if actual == counts or not any(failed_attempts(client, job) for job in jobs):
            return actual
        if attempt:
            raise AirbyteError("Replayed row counts differed after an Airbyte job retried")
        print("Replayed row counts differed after an Airbyte job retried, replaying from an empty file",
              file=sys.stderr)


def replay(request: dict, runtime: dict, client: AirbyteClient, temp: Path) -> dict:
    release = Path(runtime["release_dir"]).resolve(strict=True)
    task_id = runtime["task_id"]
    if release != Path(request["release_dir"]).resolve() or task_id != request["task_id"]:
        raise ValueError("runtime does not own the requested ELT-Bench task")
    task = read_task(release, task_id)
    if task.destination != "duckdb" or request["destination"] != "duckdb":
        raise ValueError("ELT-Bench replay requires DuckDB")
    config = duckdb_config(yaml.safe_load((task.public_dir / "config.yaml").read_text()))
    if config["definition_id"] != runtime["definition_id"]:
        raise ValueError("runtime connector differs from the released connector")
    evaluation = Path(runtime["evaluation"])
    counts = json.loads((evaluation / "table.json").read_text())[task_id]
    gold = Path(runtime["gold"])
    total = len(list(gold.glob("*.csv")))
    if not counts or not total:
        raise ValueError("benchmark reference assets are incomplete")
    store, registered, connections = registered_submission(request, runtime, client, config["schema"])
    if not connections:
        return reward(False, 0.0)

    submission = temp / "submission"
    project = stage_submission(request["files"], submission)

    with airbyte_lease("elt-bench-grading.lock", store.parent):
        try:
            actual = replay_rows(client, sorted(registered), connections, store, config["schema"], counts,
                                 sync_timeout=runtime.get("sync_timeout_s", 1500))
        except AirbyteSyncFailure as error:
            print(f"ELT-Bench Airbyte replay job {error.job_id} ended with {error.status}", file=sys.stderr)
            return {"label_valid": False, "reason_code": "airbyte_replay_incomplete"}
        el = actual == counts
        tables = {}
        if el and project:
            dbt_utils = Path(runtime["dbt_utils"]) if runtime.get("dbt_utils") else None
            install_packages = prepare_dbt_packages(submission, project, dbt_utils)
            timeout = runtime.get("dbt_timeout_s", 240)

            memory_gb = 5 if sum(counts.values()) >= 50_000_000 else 2
            run_dbt(store, submission, project, "elt_bench",
                    install_packages=install_packages, timeout=timeout, memory_gb=memory_gb)
            tables = compare_marts(store, evaluation, gold, task_id, temp / "marts", timeout=timeout)
            print(json.dumps({"stage": "t", "marts": tables}), file=sys.stderr)
    passed = sum(record["match"] for record in tables.values())
    result = reward(el, passed / total)
    result["metrics"].update(marts_passed=float(passed), marts_total=float(total))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    try:
        runtime = json.loads(args.runtime.read_text())
        request = json.loads(args.request.read_text())
        url = runtime["airbyte_url"]
        creds = json.loads(Path(runtime["airbyte_credentials_file"]).read_text())
        client = ReplayAirbyteClient(url, token_provider=access_token_provider(
            url, creds["client_id"], creds["client_secret"],
        ), timeout=45)
        with tempfile.TemporaryDirectory(prefix="elt-bench-replay-", dir=args.request.parent) as temp:
            result = replay(request, runtime, client, Path(temp))
    except DbtReplayError:
        result = {"label_valid": False, "reason_code": "duckdb_dbt_replay_failed"}
    except Exception as error:
        print(f"ELT-Bench DuckDB replay failed: {type(error).__name__}: {error}", file=sys.stderr)
        result = {"label_valid": False, "reason_code": "elt_bench_replay_failed"}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
