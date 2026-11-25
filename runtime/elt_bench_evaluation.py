"""Run Inkling 2 Small on the upstream ELT-Bench tasks with the DuckDB verifier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import fcntl
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import tempfile
from threading import Thread
from uuid import uuid4
from zipfile import ZipFile

import yaml

from duckdb_evaluation import (
    ENVIRONMENT, IMAGE, LOCAL_ROOT, MODELS, TASKGEN, DuckDBGateway, connect_egress,
    environment_variables, inkling_tokenizer_ready, read, register_duckdb, run_probe,
    runtime_smoke, safe_agent_network, start_desktop_relay, write,
)
from elt_bench_sources import check_task, load_task, stage_sources
from elt_environment.airbyte_runtime import ReplayAirbyteClient, airbyte_lease
from elt_environment.duckdb_grader import settle_jobs
from elt_environment.elt_bench_grader import DBT_UTILS_PATH, copy_dbt_utils, mart_queries
from elt_taskgen.destinations import SOURCE_CONNECTOR_CONTRACTS
from elt_taskgen.runtime import source_server
from elt_taskgen.runtime.airbyte import AirbyteError, access_token_provider, definition_versions
from elt_taskgen.runtime.airbyte_gateway import BasicCredential, make_handler
from elt_taskgen.runtime.process import SubprocessRunner
from elt_taskgen.runtime.source_environment import SourceEnvironment, _compose_document


BENCH = ENVIRONMENT.parent / "ELT-Bench"
STATE = ENVIRONMENT / ".state/elt-bench-100"
AIRBYTE = "airbyte-abctl-control-plane"
DUCKDB_IMAGE_TAG = "0.6.0-state-alias-bounded-v4"


@dataclass(frozen=True)
class BenchmarkRuntime:
    project_prefix: str = "elt-bench100"
    storage_name: str = "elt-bench-100"

    @property
    def storage(self) -> Path:
        return LOCAL_ROOT / self.storage_name

    @property
    def airbyte_directory(self) -> str:
        return f"/local/{self.storage_name}"

    @property
    def hosts(self) -> dict[str, str]:
        return {f"elt-{kind}": f"{self.project_prefix}-{kind}" for kind in (
            "postgres", "mongodb", "localstack", "api", "files",
        )}


@dataclass(frozen=True)
class BenchmarkSources(SourceEnvironment):
    """Use unique Docker service names on the shared Airbyte control plane."""

    hosts: dict[str, str]

    def _compose(self, *args):
        return super()._compose(*(self.hosts.get(arg, arg) for arg in args))

    def start(self, runner=None, *, airbyte_container=AIRBYTE):
        runner = runner or SubprocessRunner(timeout=180)
        networks = json.loads(runner.run((
            "docker", "inspect", "--format", "{{json .NetworkSettings.Networks}}", airbyte_container,
        )).stdout)
        prefix = self.project.rsplit("-", 1)[0]
        own_network = re.compile(rf"{re.escape(prefix)}-[0-9a-f]{{12}}-sources")
        if any(own_network.fullmatch(name) for name in networks):
            raise RuntimeError("This ELT-Bench namespace already has a source network attached")
        runner.run(self._compose("up", "--detach", "--wait"))
        runner.run(("docker", "network", "connect", self.network, airbyte_container))

    def seed(self, runner=None):
        runner = runner or SubprocessRunner(timeout=900)
        if (self.environment_dir / "seed" / "postgres.sh").is_file():
            runner.run(self._compose(
                "exec", "--no-TTY", "elt-postgres", "bash", "-e", "/original/postgres.sh", "/original",
            ))
        manifest = read(self.manifest_path)
        manifest["tables"] = {name: entry for name, entry in manifest["tables"].items()
                              if entry["backend"] in {"s3", "mongodb"}}
        seed_manifest = self.environment_dir / "seed.json"
        write(seed_manifest, manifest)
        SourceEnvironment.seed(replace(self, manifest_path=seed_manifest), runner)


def file_tls(root: Path, host: str) -> Path:
    tls = root / "tls"
    tls.mkdir(mode=0o700)
    authority = TASKGEN / "source_tls"
    shutil.copyfile(authority / "ca.crt", tls / "ca.crt")
    csr = root / "server.csr"
    extensions = root / "server.ext"
    extensions.write_text(f"subjectAltName=DNS:{host},DNS:localhost\nextendedKeyUsage=serverAuth\n")
    subprocess.run([
        "openssl", "req", "-new", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN={host}",
        "-keyout", str(tls / "server.key"), "-out", str(csr),
    ], check=True, capture_output=True, timeout=30)
    subprocess.run([
        "openssl", "x509", "-req", "-in", str(csr), "-CA", str(authority / "ca.crt"),
        "-CAkey", str(authority / "ca.key"), "-set_serial", "0x" + uuid4().hex,
        "-days", "7", "-extfile", str(extensions), "-out", str(tls / "server.crt"),
    ], check=True, capture_output=True, timeout=30)
    return tls


def prepare_sources(root: Path, release: Path, config: dict, manifest: dict, task: str,
                    runtime: BenchmarkRuntime):
    hosts = runtime.hosts
    project = f"{runtime.project_prefix}-{uuid4().hex[:12]}"
    network, solver = project + "-sources", project + "-solver"
    manifest_path, compose_path = root / "sources.json", root / "compose.yaml"
    rendered = root / "seed" / "rendered"
    write(manifest_path, manifest)
    compose = _compose_document(rendered_root=rendered, manifest_path=manifest_path,
                                server_script=Path(source_server.__file__), network=network)
    services = compose["services"]
    safe_agent_network(compose, solver)
    if "postgres" in config:
        services["elt-postgres"]["volumes"] = [f"{root / 'seed'}:/original:ro"]
    else:
        services.pop("elt-postgres")
    if "mongodb" in config:
        mongo = services["elt-mongodb"]

        mongo["command"].extend([
            "--wiredTigerCacheSizeGB", "0.75", "--setParameter", "tcmallocReleaseRate=33554432",
        ])
        test = mongo["healthcheck"]["test"]
        test[1] = test[1].replace("elt-mongodb:27017", hosts["elt-mongodb"] + ":27017")
    else:
        services.pop("elt-mongodb")
    if "aws_s3" not in config:
        services.pop("elt-localstack")
    if "flat_files" in config:
        tls = file_tls(root, hosts["elt-files"])
        volumes = services["elt-files"]["volumes"]
        services["elt-files"]["volumes"] = [
            f"{tls}:/tls:ro" if item.endswith(":/tls:ro") else item for item in volumes
        ]
    else:
        services.pop("elt-files")
    compose["services"] = {hosts[name]: value for name, value in services.items()}
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=False))
    return BenchmarkSources(release, task, "benchmark", root, rendered, manifest_path,
                            compose_path, project, network, hosts), solver


def register_sources(api, workspace, config, rest, *, large_load=False):
    airbyte = config["Airbyte"]["config"]
    definitions = set()
    available = definition_versions(api.list_source_definitions(workspace))
    for section, contract in SOURCE_CONNECTOR_CONTRACTS.items():
        if section not in config or section == "flat_files":
            continue
        definition = airbyte.setdefault(contract.definition_key, contract.definition_id)
        if available.get(definition) != contract.connector_version:
            raise ValueError(f"Airbyte {section} requires connector {contract.connector_version}")
        if section == "postgres" and large_load:
            response = api.request("POST", "../../v1/source_definitions/create_custom", {
                "workspaceId": workspace,
                "sourceDefinition": {
                    "name": "ELT-Bench PostgreSQL", "dockerRepository": "local/elt-source-postgres",
                    "dockerImageTag": f"{contract.connector_version}-bounded-fetch",
                    "documentationUrl": "https://docs.airbyte.com/integrations/sources/postgres",
                    "resourceRequirements": {
                        "default": {"memory_request": "1Gi", "memory_limit": "4Gi"},
                    },
                },
            })
            definition = response["sourceDefinitionId"]
            airbyte[contract.definition_key] = definition
        definitions.add(definition)
    if rest:
        definition = api.publish_declarative_source_definition(
            workspace, name="ELT-Bench REST", manifest=rest,
        )
        airbyte["custom_api_definition_id"] = definition
        definitions.add(definition)
    if "flat_files" in config:
        response = api.request("POST", "../../v1/source_definitions/create_custom", {
            "workspaceId": workspace,
            "sourceDefinition": {
                "name": "ELT-Bench File", "dockerRepository": "ghcr.io/wlu03/elt-taskgen-source-file",
                "dockerImageTag": "0.6.0-ca-temporal1",
                "documentationUrl": "https://docs.airbyte.com/integrations/sources/file",
            },
        })
        airbyte["files_definition_id"] = response["sourceDefinitionId"]
        definitions.add(response["sourceDefinitionId"])
    return definitions


def stage_task(task, release, inputs, config):
    public = release / "public" / task.name
    private = release / "private" / task.name
    public.mkdir(parents=True)
    private.mkdir(parents=True, mode=0o700)
    shutil.copyfile(task.original / "data_model.yaml", public / "data_model.yaml")
    shutil.copytree(task.bench / "elt-bench" / "schemas" / task.name, public / "schemas")
    shutil.copytree(task.bench / "documentation", public / "documentation",
                    ignore=shutil.ignore_patterns("destination_snowflake.md", "*_credential.json"))
    if task.dbt_utils:
        copy_dbt_utils(task.dbt_utils, public / DBT_UTILS_PATH)
        (public / "documentation" / "dbt_utils.md").write_text(
            "dbt_utils 1.3.0 is available offline at /workspace/vendor/dbt_utils.\n"
            "For a project in /workspace/elt, declare it in elt/packages.yml and run dbt deps there.\n"
            "\n```yaml\npackages:\n  - local: /workspace/vendor/dbt_utils\n```\n"
        )
    public_config = yaml.safe_load(yaml.safe_dump(config))
    airbyte = public_config["Airbyte"]["config"]
    for name in ("username", "password", "workspace_id"):
        airbyte.pop(name, None)
    (public / "config.yaml").write_text(yaml.safe_dump(public_config, sort_keys=False))
    write(private / "task_ir.json", {"task_id": task.name, "family_id": "elt-bench:" + task.name})
    installed = inputs / task.name
    shutil.copytree(public, installed)
    (installed / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return installed


def run_task(task, database, database_names, api_archive, client, credentials, url, output, smoke,
             bench_runtime, row_count):
    output.mkdir(parents=True, exist_ok=True, mode=0o700)

    sync_timeout = max(1500, 600 + row_count // 10_000)
    with tempfile.TemporaryDirectory(prefix="elt-bench100-") as temporary:
        root = Path(temporary)
        release, inputs = root / "bundle", root / "inputs"
        seed = root / "seed"
        seed.mkdir()
        config, manifest = stage_sources(task, database, database_names, api_archive, seed,
                                         bench_runtime.hosts)
        source, network = prepare_sources(root, release, config, manifest, task.name, bench_runtime)
        shared = read(TASKGEN / "secrets/airbyte-gateway.json")
        scoped = BasicCredential("airbyte", secrets.token_urlsafe(32))
        relay = source.project + "-relay"
        server = gateway = None
        workspace = client.create_workspace(f"ELT-Bench Inkling {task.name}")
        try:
            sources = register_sources(client, workspace, config, task.rest, large_load=row_count > 6_000_000)
            definition = register_duckdb(client, workspace, image_tag=DUCKDB_IMAGE_TAG)
            config["duckdb"] = {"config": {"definition_id": definition, "schema": "main",
                                           "airbyte_directory": bench_runtime.airbyte_directory}}
            config["Airbyte"]["config"].update(
                username=scoped.username, password=scoped.password, workspace_id=workspace,
                server_url="http://elt-egress:7801/api/public/v1/",
            )
            installed = stage_task(task, release, inputs, config)
            hosts = set(bench_runtime.hosts.values())
            hosts.update(entry["bucket"] for entry in manifest["tables"].values() if "bucket" in entry)
            gateway = DuckDBGateway(
                scoped, shared, workspace, {"sources": sources, "destinations": {definition}},
                hosts, "main", local_root=bench_runtime.storage,
                airbyte_directory=bench_runtime.airbyte_directory,
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(gateway))
            Thread(target=server.serve_forever, daemon=True).start()
            source.start()
            source.seed(SubprocessRunner(timeout=max(900, 300 + row_count // 100_000)))
            port = start_desktop_relay(relay, "elt-solver", server.server_port, "disabled.invalid",
                                       {"ELT_EVAL_SWE_IMAGE": IMAGE})
            connect_egress(relay, network)
            if smoke:
                runtime_smoke(network, port, installed)
                return None
            runtime = root / "grader.json"
            runtime_config = {
                "release_dir": str(release), "task_id": task.name, "workspace_id": workspace,
                "definition_id": definition, "airbyte_url": url,
                "airbyte_credentials_file": str(credentials), "local_root": str(bench_runtime.storage),
                "airbyte_directory": bench_runtime.airbyte_directory,
                "evaluation": str(task.bench / "evaluation"),
                "gold": str(task.gold),
                "sync_timeout_s": sync_timeout,
                "dbt_timeout_s": min(1800, 240 + row_count // 100_000),
            }
            if task.dbt_utils:
                runtime_config["dbt_utils"] = str(task.dbt_utils)
            write(runtime, runtime_config)
            return run_probe(release=release, inputs=inputs, output=output, network=network,
                             port=port, runtime=runtime, local_root=bench_runtime.storage,
                             grader="elt_environment.elt_bench_grader",
                             agent_timeout_s=max(7200, sync_timeout + 900),
                             grader_deadline_s=max(7200, 2 * sync_timeout + 1800))
        finally:
            try:
                if server:
                    server.shutdown()
                    server.server_close()
                if gateway:
                    connections = [item["connectionId"] for item in client.list_connections(workspace)]
                    settle_jobs(client, connections, timeout=180, cancel=True)
            except (AirbyteError, TimeoutError) as error:
                print(f"{task.name}: stopping workspace jobs during teardown: {error}", flush=True)
            finally:
                try:
                    subprocess.run(["docker", "rm", "-f", relay], capture_output=True, timeout=25)
                    source.stop()
                finally:
                    client.delete_workspace(workspace)


def report(output, roster, failures, status):
    scores = {}
    for task in roster:
        path = output / "inkling" / task / "summary.json"
        if path.is_file():
            result = read(path)
            if not result["groups_discarded"]:
                scores[task] = result["mean_reward"]
    write(output / "suite_summary.json", {
        "benchmark": "ELT-Bench++", "backend": "duckdb", "model": MODELS["inkling"][0],
        "status": status, "task_count": len(roster), "scored": len(scores),
        "full_passes": sum(value == 1.0 for value in scores.values()),
        "mean_reward": sum(scores.values()) / len(scores) if scores else None,
        "scores": scores, "unscored": failures,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=BENCH)
    parser.add_argument("--assets", type=Path, default=STATE / "assets")
    parser.add_argument("--output", type=Path, default=STATE / "run")
    parser.add_argument("--runtime-namespace", help="Use a separate Docker and DuckDB namespace.")
    parser.add_argument("--task-id", action="append", help="Evaluate only the named task(s).")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="Validate assets without Docker or inference.")
    action.add_argument("--smoke", action="store_true", help="Start the selected task runtime without inference.")
    args = parser.parse_args()
    bench_runtime = BenchmarkRuntime()
    if args.runtime_namespace is not None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,19}", args.runtime_namespace):
            parser.error("--runtime-namespace must be a short lowercase Docker name")
        name = f"elt-bench-rerun-{args.runtime_namespace}"
        bench_runtime = BenchmarkRuntime(name, name)
    bench, assets, output = (value.resolve() for value in (args.bench, args.assets, args.output))
    counts = read(bench / "evaluation" / "table.json")
    roster = sorted(path.parent.name for path in (bench / "elt-bench" / "snowflake").glob("*/config.yaml"))
    if len(roster) != 100 or set(roster) != counts.keys():
        parser.error("The ELT-Bench checkout must contain the 100 upstream tasks and count entries.")
    selected = args.task_id or (["amplitude"] if args.smoke else roster)
    if set(selected) - set(roster):
        parser.error("Unknown upstream ELT-Bench task ID.")
    tasks = sorted(set(selected), key=lambda name: (sum(counts[name].values()), name))
    rest = yaml.safe_load((bench / "setup" / "elt_snowflake.yaml").read_text())
    with ZipFile(assets / "data_db.zip") as database, ZipFile(assets / "data_api.zip") as api_archive:
        database_names = {name.casefold(): name for name in database.namelist()}
        tasks = [load_task(bench, assets, name, rest, bench_runtime.hosts) for name in tasks]
        for task in tasks:
            check_task(task, counts[task.name], database_names, api_archive)
            mart_queries(bench / "evaluation", task.name, [p.stem for p in task.gold.glob("*.csv")])
        print(f"Validated upstream assets for {len(tasks)} of {len(roster)} tasks", flush=True)
        if args.check:
            return
        if not args.smoke and not inkling_tokenizer_ready():
            parser.error("Install the Inkling Tinker runtime described in README.md.")
        url = environment_variables()["AIRBYTE_HARNESS_URL"]
        credentials = TASKGEN / "secrets/airbyte-base.json"
        creds = read(credentials)
        client = ReplayAirbyteClient(url, token_provider=access_token_provider(
            url, creds["client_id"], creds["client_secret"],
        ), timeout=100)
        bench_runtime.storage.mkdir(mode=0o711, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        failures = {}
        with (bench_runtime.storage / "runtime.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not args.smoke:
                report(output, roster, failures, "running")
            for task in tasks:
                dest = output / ("smoke" if args.smoke else "inkling") / task.name
                summary = dest / "summary.json"
                if summary.is_file() and not read(summary)["groups_discarded"]:
                    print(f"{task.name}: reusing saved score", flush=True)
                    continue
                print(f"Starting Inkling 2 Small on ELT-Bench task {task.name}", flush=True)
                for attempt in range(2 if not args.smoke else 1):
                    try:
                        with airbyte_lease("airbyte-attempt.lock"):
                            result = run_task(task, database, database_names, api_archive, client,
                                              credentials, url, dest, args.smoke, bench_runtime,
                                              sum(counts[task.name].values()))
                        if result is None or not result["groups_discarded"]:
                            failures.pop(task.name, None)
                            print(f"{task.name}: " + ("smoke passed" if result is None else
                                  f"reward={result['mean_reward']:.6f}"), flush=True)
                            break
                        failures[task.name] = result["per_task"][task.name]
                    except Exception as error:
                        failures[task.name] = f"{type(error).__name__}: {error}"
                    print(f"{task.name}: attempt {attempt + 1} unscored: {failures[task.name]}", flush=True)
                if not args.smoke:
                    report(output, roster, failures, "running")
            if not args.smoke:
                complete = not failures and len(tasks) == len(roster)
                report(output, roster, failures, "complete" if complete else "incomplete")
        if failures:
            raise SystemExit(f"{len(failures)} selected tasks remain unscored")


if __name__ == "__main__":
    main()
