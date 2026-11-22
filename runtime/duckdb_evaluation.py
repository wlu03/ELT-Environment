"""Evaluate the frozen 50-task DuckDB suite using isolated SWE-agent sessions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
from threading import Lock, Thread
from urllib.parse import urlsplit
from uuid import uuid4

import yaml

ENVIRONMENT = Path(__file__).resolve().parents[1]
TASKGEN = ENVIRONMENT.parent / "ELT-taskgen"
sys.path.insert(0, str(TASKGEN / "tools/elt_bench"))
sys.path.insert(0, str(ENVIRONMENT / "src"))

from attempt import (  # noqa: E402
    PREFIX, GatewayBridge, connect_egress, prepare_airbyte, settle_development_syncs,
    start_desktop_relay,
)
from common import environment_variables  # noqa: E402
from security import safe_agent_network  # noqa: E402
from elt_taskgen.runtime import source_server
from elt_taskgen.runtime.airbyte import AirbyteClient, access_token_provider
from elt_taskgen.runtime.airbyte_gateway import BasicCredential, ForwardResult, make_handler
from elt_taskgen.runtime.source_environment import SourceEnvironment, _compose_document, _safe_project
from elt_environment.agent_worker import start_private_swe

MODELS = {
    "inkling": ("thinkingmachines/Inkling-2-Small", "tml_v0"),
    "nemotron": ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
                 "nemotron3_ultra_disable_thinking"),
}
IMAGE = "elt-swe-duckdb"
LOCAL_ROOT = Path.home() / ".airbyte/abctl/data/elt-environment-duckdb"
WORKER = ENVIRONMENT.parent / "elt_agent_eval_run/agentvenv/bin/python"
COORDINATOR = ENVIRONMENT / ".venv/bin/python"
CONNECTOR = "local/elt-destination-duckdb"


class DuckDBGateway(GatewayBridge):
    """Limit DuckDB destinations to the storage opened by the current runner."""

    def __init__(self, scoped, shared, workspace, definitions, hosts, schema):
        super().__init__(scoped, shared, workspace, definitions, "", "", hosts)
        self.schema = schema
        self.previous_paths = set(LOCAL_ROOT.iterdir())
        self.sync_lock = Lock()

    def forward(self, method, path, headers, body):
        if method == "POST" and path == PREFIX + "destinations":
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            authorization = next(
                (value for name, value in headers.items() if name.lower() == "authorization"), None,
            )
            if (
                self.authorized(authorization) and isinstance(payload, dict)
                and self._in_destination_scope(payload)
                and isinstance(payload.get("configuration"), dict)
                and payload["configuration"].get("destination_path") == "/duckdb/warehouse.duckdb"
            ):
                message = (
                    "Airbyte runs in a different container. Set configuration.destination_path "
                    "to duckdb.config.destination_path from /workspace/config.yaml. "
                    "Use /duckdb/warehouse.duckdb for solver and dbt reads."
                )
                return ForwardResult(
                    403, "Forbidden", (("Content-Type", "application/json"),),
                    json.dumps({"message": message}).encode(),
                )
        if method == "POST" and urlsplit(path).path == PREFIX + "jobs":
            with self.sync_lock:
                return super().forward(method, path, headers, body)
        return super().forward(method, path, headers, body)

    def _in_destination_scope(self, payload):
        definition = payload.get("definitionId")
        return (
            payload.get("workspaceId") == self.workspace
            and isinstance(definition, str)
            and definition in self.definitions["destinations"]
        )

    def _allowed(self, method, parts, query, body):
        if method == "POST" and parts == ["destinations"]:
            config = body.get("configuration")
            active = [
                path for path in LOCAL_ROOT.iterdir()
                if path not in self.previous_paths and path.is_dir() and not path.is_symlink()
                and re.fullmatch(r"elt-duckdb-[a-z0-9_]+", path.name)
            ]
            return (
                not query and self._in_destination_scope(body)
                and isinstance(config, dict) and set(config) <= {"destination_path", "schema"}
                and config.get("schema", "main") == self.schema and len(active) == 1
                and config.get("destination_path") == f"/local/{active[0].name}/warehouse.duckdb"
            )
        if method == "POST" and parts == ["jobs"] and self.job_connections:
            headers = {"Authorization": self.shared}
            last = next(reversed(self.job_connections))
            response = self._send("GET", f"{PREFIX}jobs/{last}", headers, b"", None)
            if response.status != 200:
                return False
            job = json.loads(response.body)
            status = job.get("data", job.get("job", job)).get("status")
            if status not in {"succeeded", "failed", "cancelled", "canceled", "incomplete"}:
                return False
        return super()._allowed(method, parts, query, body)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def runtime_smoke(network, control_port, stage):
    """Check the isolated solver, provider mirror, and scoped gateway without inference."""
    container = "elt-duckdb-smoke-" + uuid4().hex[:12]
    try:
        start_private_swe(
            {"container_name": container, "image": IMAGE, "duckdb_control_port": control_port},
            ["--network", network, "-v", f"{stage}:/workspace"],
        )
        script = """
import base64, json, urllib.request, yaml
from urllib.error import HTTPError
config = yaml.safe_load(open('/workspace/config.yaml'))
c = config['Airbyte']['config']
pair = base64.b64encode((c['username'] + ':' + c['password']).encode()).decode()
def request(path, payload=None):
    req = urllib.request.Request(
        c['server_url'] + path, data=json.dumps(payload).encode() if payload else None,
        headers={'Authorization': 'Basic ' + pair, 'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as reply:
            return reply.status, json.load(reply)
    except HTTPError as exc:
        return exc.code, json.load(exc)
assert request('workspaces/' + c['workspace_id'])[0] == 200
assert request('workspaces')[0] == 403
body = {
    'workspaceId': c['workspace_id'],
    'definitionId': config['duckdb']['config']['definition_id'],
    'configuration': {'destination_path': '/duckdb/warehouse.duckdb'},
}
status, error = request('destinations', body)
assert status == 403 and 'duckdb.config.destination_path' in error['message'], (status, error)
status, error = request('destinations', {**body, 'workspaceId': 'outside-this-task'})
assert status == 403 and 'duckdb.config.destination_path' not in error['message'], (status, error)
"""
        for cmd in (["python", "-c", script],
                    ["terraform", "-chdir=/workspace/elt", "init", "-input=false"]):
            subprocess.run(["docker", "exec", container, *cmd],
                           check=True, timeout=45, stdout=subprocess.DEVNULL)
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=20)


def run_attempt(args, record, label, api, credentials_file, airbyte_url):
    task = record["task_id"]
    answer = Path(record["private"]) / "answer_key"
    public = Path(record["public"])
    original = args.suite / "bundle_duckdb"
    output = args.output / label / task
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix="duckdb-eval-") as directory:
        root = Path(directory)
        release, inputs = root / "bundle", root / "installed"
        frozen = original / "private" / task
        stage = inputs / task
        project = _safe_project(task, "development", directory)
        network, solver_network = project + "-source-network", project + "-solver"
        rendered = frozen / "populations" / "development" / "rendered"
        source = SourceEnvironment(
            release, task, "development", root, rendered, answer / "sources_serving.json",
            root / "compose.yaml", project, network,
        )
        compose = _compose_document(
            rendered_root=rendered, manifest_path=source.manifest_path,
            server_script=Path(source_server.__file__).resolve(), network=network,
        )
        safe_agent_network(compose, solver_network)
        source.compose_path.write_text(yaml.safe_dump(compose))
        shared = read(TASKGEN / "secrets/airbyte-gateway.json")
        scoped = BasicCredential("airbyte", secrets.token_urlsafe(32))
        relay = "elt-duckdb-relay-" + uuid4().hex[:12]
        workspace = api.create_workspace(f"ELT DuckDB {label} {task[:70]}")
        gateway = server = None
        try:
            config = yaml.safe_load((original / "public" / task / "config.yaml").read_text())
            _, definitions = prepare_airbyte(public, answer, task, api, workspace, config)
            definition = api.request("POST", "../../v1/destination_definitions/create_custom", {
                "workspaceId": workspace,
                "destinationDefinition": {
                    "name": "ELT DuckDB with protocol aliases", "dockerRepository": CONNECTOR,
                    "dockerImageTag": "0.6.0-state-alias",
                    "documentationUrl": "https://docs.airbyte.com/integrations/destinations/duckdb",
                },
            })["destinationDefinitionId"]
            definitions["destinations"] = {definition}
            config["duckdb"]["config"]["definition_id"] = definition
            airbyte = config["Airbyte"]["config"]
            airbyte.pop("snowflake_definition_id", None)
            airbyte.update(username=scoped.username, password=scoped.password, workspace_id=workspace,
                           server_url="http://elt-egress:7801/api/public/v1/")
            hosts = {"elt-postgres", "elt-mongodb", "elt-localstack", "elt-api", "elt-files"}
            hosts.update(item["bucket"] for item in read(source.manifest_path)["tables"].values()
                         if item.get("bucket"))
            gateway = DuckDBGateway(scoped, shared, workspace, definitions, hosts,
                                    config["duckdb"]["config"]["schema"])
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(gateway))
            Thread(target=server.serve_forever, daemon=True).start()
            source.start()
            source.seed()
            port = start_desktop_relay(relay, "elt-solver", server.server_port, "disabled.invalid",
                                       {"ELT_EVAL_SWE_IMAGE": IMAGE})
            connect_egress(relay, solver_network)
            shutil.copytree(original / "public" / task, release / "public" / task)
            released = release / "public" / task / "config.yaml"
            release_config = yaml.safe_load(released.read_text())
            release_config["duckdb"]["config"]["definition_id"] = definition
            released.write_text(yaml.safe_dump(release_config, sort_keys=False))
            shutil.copytree(frozen, release / "private" / task, copy_function=os.link)
            shutil.copytree(released.parent, stage)
            (stage / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            runtime = root / "grader-runtime.json"
            write(runtime, {
                "release_dir": str(release), "task_id": task, "workspace_id": workspace,
                "definition_id": definition, "airbyte_url": airbyte_url,
                "airbyte_credentials_file": str(credentials_file), "local_root": str(LOCAL_ROOT),
                "development": {key: str(value) for key, value in asdict(source).items()},
            })
            if args.smoke:
                runtime_smoke(solver_network, port, stage)
                return True
            model, renderer = MODELS[label]
            command = [
                str(COORDINATOR), "-m", "elt_environment.probe",
                f"releases_root={release}", f"inputs_root={inputs}", "agent=swe",
                f"agent_python={WORKER}", f"agent_network={solver_network}",
                f"duckdb_local_root={LOCAL_ROOT}", f"duckdb_control_port={port}",
                f"grader_command={shlex.join((str(COORDINATOR), '-m', 'elt_environment.duckdb_grader', '--runtime', str(runtime)))}",
                f"model_name={model}", f"renderer_name={renderer}", "eval_families=0",
                "samples=1", "max_group_attempts=1", "context_length=65536", f"out_dir={output}",
            ]
            with (output / "probe.log").open("w") as log:
                log_path = Path(log.name)
                log_path.chmod(0o600)
                subprocess.run(command, cwd=ENVIRONMENT, stdout=log, stderr=subprocess.STDOUT,
                               check=True)
            result = read(output / "summary.json")
            if result["groups_discarded"]:
                print(f"{label} {task} unscored: {result['per_task'][task]}", flush=True)
                return False
            print(f"{label} {task} reward={result['mean_reward']}", flush=True)
            return True
        finally:
            try:
                if server:
                    server.shutdown()
                    server.server_close()
                if gateway:
                    settle_development_syncs(api, gateway.job_connections)
            finally:
                try:
                    subprocess.run(["docker", "rm", "-f", relay], capture_output=True, timeout=25)
                    source.stop()
                finally:
                    api.delete_workspace(workspace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--output", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.limit <= 50:
        parser.error("--limit must be between 1 and 50")
    if not args.smoke:
        subprocess.run([
            str(COORDINATOR), "-c",
            "from tinker_cookbook.tokenizer_utils import get_tokenizer; "
            "tok = get_tokenizer('thinkingmachines/Inkling-2-Small'); "
            "assert type(tok.tml_tokenizer).__name__ == 'H200kBaseChatTokenizer', "
            "'Inkling 2 Small requires the H200k tokenizer (Cookbook 0.6.0 or newer)'",
        ], check=True, timeout=45)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    roster = args.suite.parent / "elt_bench_gpt4o"
    records = list(read(roster / "readiness.json")["records"].values())
    records.sort(key=lambda row: (not row["task_id"].startswith("dbt__snapchat_ads__"), row["task_id"]))
    records = records[:1 if args.smoke else args.limit]
    env = environment_variables()
    url = env["AIRBYTE_HARNESS_URL"]
    credentials_file = TASKGEN / "secrets/airbyte-base.json"
    credential = read(credentials_file)
    api = AirbyteClient(url, token_provider=access_token_provider(
        url, credential["client_id"], credential["client_secret"],
    ), timeout=100)
    def evaluate(record, label):
        with (roster / "spider-runtime.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(f"starting {label} {record['task_id']}", flush=True)
            return run_attempt(args, record, label, api, credentials_file, url)

    pending = []
    for record in records:
        for label in (["smoke"] if args.smoke else MODELS):
            summary = args.output / label / record["task_id"] / "summary.json"
            if summary.is_file() and not read(summary)["groups_discarded"]:
                continue
            if not evaluate(record, label):
                pending.append((record, label))
    unscored = sum(not evaluate(record, label) for record, label in pending)
    if unscored:
        raise SystemExit(f"Evaluation finished with {unscored} unscored attempts after retry")
    print("complete", flush=True)


if __name__ == "__main__":
    main()
