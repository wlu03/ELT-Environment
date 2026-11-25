"""Evaluate the frozen 50-task DuckDB suite using isolated SWE-agent sessions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from enum import Enum, auto
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
    PREFIX, GatewayBridge, connect_egress, prepare_airbyte,
    start_desktop_relay,
)
from common import environment_variables  # noqa: E402
from security import safe_agent_network  # noqa: E402
from elt_taskgen.runtime import source_server
from elt_taskgen.runtime.airbyte import (
    AIRBYTE_TERMINAL_JOB_STATUSES, AirbyteError, access_token_provider,
)
from elt_taskgen.runtime.airbyte_gateway import BasicCredential, ForwardResult, make_handler
from elt_taskgen.runtime.source_environment import SourceEnvironment, _compose_document, _safe_project
from elt_environment.agent_worker import SWE_STARTUP_FAILURES, start_private_swe
from elt_environment.airbyte_runtime import (
    LOCAL_AIRBYTE_ROOT, ReplayAirbyteClient, airbyte_lease, is_transient_airbyte_error,
)
from elt_environment.duckdb_grader import settle_jobs
from elt_environment.duckdb_target import duckdb_destination_path

MODELS = {
    "inkling": ("thinkingmachines/Inkling-2-Small", "tml_v0"),
    "nemotron": ("nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
                 "nemotron3_ultra_disable_thinking"),
}
IMAGE = "elt-swe-duckdb"
LOCAL_ROOT = LOCAL_AIRBYTE_ROOT
WORKER = ENVIRONMENT.parent / "elt_agent_eval_run/agentvenv/bin/python"
COORDINATOR = ENVIRONMENT / ".venv/bin/python"
CONNECTOR = "local/elt-destination-duckdb"


class AttemptResult(Enum):
    SCORED = auto()
    UNSCORED = auto()
    STARTUP_FAILURE = auto()


class DuckDBGateway(GatewayBridge):
    """Limit DuckDB destinations to the storage opened by the current runner."""

    def __init__(self, scoped, shared, workspace, definitions, hosts, schema, *,
                 local_root=LOCAL_ROOT, airbyte_directory="/local"):
        super().__init__(scoped, shared, workspace, definitions, "", "", hosts)
        self.schema = schema
        self.local_root = local_root
        self.airbyte_directory = airbyte_directory
        self.previous_paths = set(local_root.iterdir())
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
                path for path in self.local_root.iterdir()
                if path not in self.previous_paths and path.is_dir() and not path.is_symlink()
                and re.fullmatch(r"elt-duckdb-[a-z0-9_]+", path.name)
            ]
            return (
                not query and self._in_destination_scope(body)
                and isinstance(config, dict) and set(config) <= {"destination_path", "schema"}
                and config.get("schema", "main") == self.schema and len(active) == 1
                and config.get("destination_path") == duckdb_destination_path(
                    active[0], self.airbyte_directory,
                )
            )
        if method == "POST" and parts == ["jobs"] and self.job_connections:
            headers = {"Authorization": self.shared}
            last = next(reversed(self.job_connections))
            response = self._send("GET", f"{PREFIX}jobs/{last}", headers, b"", None)
            if response.status != 200:
                return False
            job = json.loads(response.body)
            status = job.get("data", job.get("job", job)).get("status")
            if status not in AIRBYTE_TERMINAL_JOB_STATUSES:
                return False
        return super()._allowed(method, parts, query, body)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def inkling_tokenizer_ready():
    check = subprocess.run([
        str(COORDINATOR), "-c",
        "import sys; "
        "from tinker_cookbook.tokenizer_utils import get_tokenizer; "
        "tok = get_tokenizer(sys.argv[1]); "
        "sys.exit(type(getattr(tok, 'tml_tokenizer', None)).__name__ != 'H200kBaseChatTokenizer')",
        MODELS["inkling"][0],
    ], timeout=45)
    return check.returncode == 0


def register_duckdb(api, workspace, *, image_tag="0.6.0-state-alias"):
    return api.request("POST", "../../v1/destination_definitions/create_custom", {
        "workspaceId": workspace,
        "destinationDefinition": {
            "name": "ELT DuckDB with protocol aliases", "dockerRepository": CONNECTOR,
            "dockerImageTag": image_tag,
            "documentationUrl": "https://docs.airbyte.com/integrations/destinations/duckdb",
        },
    })["destinationDefinitionId"]


def run_probe(*, release, inputs, output, network, port, runtime, label="inkling",
              local_root=LOCAL_ROOT, grader="elt_environment.duckdb_grader", agent_feedback=False,
              agent_timeout_s=None, grader_deadline_s=None):
    model, renderer = MODELS[label]
    grader_command = shlex.join((str(COORDINATOR), "-m", grader, "--runtime", str(runtime)))
    command = [
        str(COORDINATOR), "-m", "elt_environment.probe",
        f"releases_root={release}", f"inputs_root={inputs}", "agent=swe",
        f"agent_python={WORKER}", f"agent_network={network}",
        f"duckdb_local_root={local_root}", f"duckdb_control_port={port}",
        f"grader_command={grader_command}", f"model_name={model}", f"renderer_name={renderer}",
        "eval_families=0", "samples=1", "max_group_attempts=1", "context_length=65536",
        f"out_dir={output}",
    ]
    if agent_feedback:
        command.append("agent_feedback=True")
    if agent_timeout_s is not None:
        command.append(f"agent_timeout_s={agent_timeout_s}")
    if grader_deadline_s is not None:
        command.append(f"grader_deadline_s={grader_deadline_s}")
    with (output / "probe.log").open("w") as log:
        Path(log.name).chmod(0o600)
        subprocess.run(command, cwd=ENVIRONMENT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return read(output / "summary.json")


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
    local_root = LOCAL_ROOT / "agent-feedback" if args.agent_feedback else LOCAL_ROOT
    airbyte_directory = "/local/agent-feedback" if args.agent_feedback else "/local"
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
        workspace_name = f"ELT DuckDB {label} {task[:70]} {uuid4().hex[:12]}"
        workspace = gateway = server = None
        probe_started = False
        try:
            workspace = api.create_workspace(workspace_name)
            config = yaml.safe_load((original / "public" / task / "config.yaml").read_text())
            _, definitions = prepare_airbyte(public, answer, task, api, workspace, config)
            definition = register_duckdb(api, workspace)
            definitions["destinations"] = {definition}
            config["duckdb"]["config"]["definition_id"] = definition
            config["duckdb"]["config"]["airbyte_directory"] = airbyte_directory
            airbyte = config["Airbyte"]["config"]
            airbyte.pop("snowflake_definition_id", None)
            airbyte.update(username=scoped.username, password=scoped.password, workspace_id=workspace,
                           server_url="http://elt-egress:7801/api/public/v1/")
            hosts = {"elt-postgres", "elt-mongodb", "elt-localstack", "elt-api", "elt-files"}
            hosts.update(item["bucket"] for item in read(source.manifest_path)["tables"].values()
                         if item.get("bucket"))
            gateway = DuckDBGateway(scoped, shared, workspace, definitions, hosts,
                                    config["duckdb"]["config"]["schema"],
                                    local_root=local_root, airbyte_directory=airbyte_directory)
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
                "airbyte_credentials_file": str(credentials_file), "local_root": str(local_root),
                "airbyte_directory": airbyte_directory,
                "development": {key: str(value) for key, value in asdict(source).items()},
            })
            if args.smoke:
                runtime_smoke(solver_network, port, stage)
                return AttemptResult.SCORED
            probe_started = True
            result = run_probe(release=release, inputs=inputs, output=output, network=solver_network,
                               port=port, runtime=runtime, label=label, local_root=local_root,
                               agent_feedback=args.agent_feedback)
            if result["groups_discarded"]:
                print(f"{label} {task} unscored: {result['per_task'][task]}", flush=True)
                reasons = result.get("discard_reasons_by_task", {}).get(task, [])
                if reasons and SWE_STARTUP_FAILURES.issuperset(reasons):
                    return AttemptResult.STARTUP_FAILURE
                return AttemptResult.UNSCORED
            print(f"{label} {task} reward={result['mean_reward']}", flush=True)
            return AttemptResult.SCORED
        except AirbyteError as error:
            if probe_started or not is_transient_airbyte_error(error):
                raise
            reason = "airbyte_setup_unavailable"
            write(output / "summary.json", {
                "model": MODELS[label][0] if label in MODELS else label,
                "rollouts": 0, "groups_discarded": 1,
                "per_task": {task: reason}, "discard_reasons_by_task": {task: [reason]},
            })
            print(f"{label} {task} unscored: {reason}", flush=True)
            return AttemptResult.STARTUP_FAILURE
        finally:
            try:
                if server:
                    server.shutdown()
                    server.server_close()
                if gateway:
                    try:
                        settle_jobs(api, sorted(set(gateway.job_connections.values())), timeout=180, cancel=True)
                    except AirbyteError as error:
                        if not is_transient_airbyte_error(error):
                            raise
                        print(f"{label} {task} Airbyte unavailable while checking development jobs", flush=True)
                    except TimeoutError:
                        print(f"{label} {task} development jobs exceeded their cleanup deadline", flush=True)
            finally:
                try:
                    subprocess.run(["docker", "rm", "-f", relay], capture_output=True, timeout=25)
                    source.stop()
                finally:
                    cleanup_workspace(api, output, workspace, workspace_name)


def cleanup_workspace(api, output, workspace, workspace_name):
    receipt = output / f"cleanup-{workspace_name.split()[-1]}.json"
    try:
        if workspace is None:
            matches = [row["workspaceId"] for row in api.list_workspaces()
                       if row.get("name") == workspace_name]
            if len(matches) > 1:
                raise ValueError("multiple Airbyte workspaces matched a single attempt")
            if matches:
                workspace = matches[0]
        if workspace:
            api.delete_workspace(workspace)
        receipt.unlink(missing_ok=True)
    except AirbyteError as error:
        if not is_transient_airbyte_error(error):
            raise
        write(receipt, {"workspace_id": workspace, "workspace_name": workspace_name})
        print("Airbyte workspace cleanup deferred until the API is reachable", flush=True)


def run_with_startup_retry(args, record, label, api, credentials_file, url):
    result = run_attempt(args, record, label, api, credentials_file, url)
    if result is not AttemptResult.STARTUP_FAILURE:
        return result
    task = record["task_id"]
    output = args.output / label / task
    archive = Path(tempfile.mkdtemp(prefix="startup-failure-", dir=output))
    for filename in ("summary.json", "probe.log"):
        if (output / filename).is_file():
            (output / filename).replace(archive / filename)
    print(f"retrying {label} {task} with fresh runtime after pre-inference failure", flush=True)
    return run_attempt(args, record, label, api, credentials_file, url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--output", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--model", choices=MODELS, help="Run only this model.")
    parser.add_argument("--agent-feedback", action="store_true")
    parser.add_argument("--checkpoint-after", type=int, choices=range(1, 101), metavar="SCORES")
    parser.add_argument("--retry-unscored", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 50:
        parser.error("--limit must be between 1 and 50")
    if not args.smoke and not inkling_tokenizer_ready():
        parser.exit(
            1, "Inkling 2 Small requires the H200k tokenizer from Tinker Cookbook 0.6.0 or newer. "
            "See README.md for the Inkling evaluation runtime.\n",
        )
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.agent_feedback:
        (LOCAL_ROOT / "agent-feedback").mkdir(mode=0o711, exist_ok=True)
    roster = args.suite.parent / "elt_bench_gpt4o"
    records = list(read(roster / "readiness.json")["records"].values())
    records.sort(key=lambda row: (not row["task_id"].startswith("dbt__snapchat_ads__"), row["task_id"]))
    if args.task_id:
        records = [record for record in records if record["task_id"] in args.task_id]
        if len(records) != len(set(args.task_id)):
            parser.error("every --task-id must identify a task in the frozen roster")
    records = records[:1 if args.smoke else args.limit]
    env = environment_variables()
    url = env["AIRBYTE_HARNESS_URL"]
    credentials_file = TASKGEN / "secrets/airbyte-base.json"
    credential = read(credentials_file)
    api = ReplayAirbyteClient(url, token_provider=access_token_provider(
        url, credential["client_id"], credential["client_secret"],
    ), timeout=100)
    def evaluate(record, label):
        with (roster / "spider-runtime.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with airbyte_lease("airbyte-attempt.lock"):
                for receipt in args.output.glob("*/*/cleanup-*.json"):
                    saved = read(receipt)
                    cleanup_workspace(api, receipt.parent, saved["workspace_id"], saved["workspace_name"])
                print(f"starting {label} {record['task_id']}", flush=True)
                return run_with_startup_retry(args, record, label, api, credentials_file, url)

    def checkpoint_reached():
        if args.checkpoint_after is None:
            return False
        scores = sorted(
            str(path.relative_to(args.output)) for path in args.output.glob("*/*/summary.json")
            if not read(path)["groups_discarded"]
        )
        if len(scores) < args.checkpoint_after:
            return False
        write(args.output / f"checkpoint-{args.checkpoint_after:02d}.json", {
            "status": "awaiting_review", "scored_attempts": len(scores), "results": scores,
        })
        print(f"Paused at {len(scores)} scored attempts for environment review", flush=True)
        return True

    pending = []
    startup_failures = 0
    labels = (args.model,) if args.model else tuple(MODELS)
    if args.smoke:
        labels = ("smoke",)
    for record in records:
        for label in labels:
            if checkpoint_reached():
                return
            summary = args.output / label / record["task_id"] / "summary.json"
            if summary.is_file():
                if not read(summary)["groups_discarded"]:
                    continue
                if args.checkpoint_after is not None and not args.retry_unscored:
                    pending.append((record, label))
                    continue
            elif args.retry_unscored:
                continue
            result = evaluate(record, label)
            if result is AttemptResult.STARTUP_FAILURE:
                startup_failures += 1
            elif result is AttemptResult.UNSCORED:
                pending.append((record, label))
    if checkpoint_reached():
        return
    unscored = startup_failures + len(pending)
    if not args.retry_unscored:
        unscored = startup_failures
        for record, label in pending:
            summary = args.output / label / record["task_id"] / "summary.json"
            if summary.is_file() and not read(summary)["groups_discarded"]:
                continue
            unscored += evaluate(record, label) is not AttemptResult.SCORED
            if checkpoint_reached():
                return
    if unscored:
        raise SystemExit(f"Evaluation finished with {unscored} unscored attempts after retry")
    print("complete", flush=True)


if __name__ == "__main__":
    main()
