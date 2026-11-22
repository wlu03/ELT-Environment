"""Launch a vendored agent in its own interpreter, changing model transport only.

Run by runner.py; not imported into the training process. Provider keys here
are per-attempt gateway tokens, never Tinker or commercial provider keys.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path

FOLDERS = {
    "spider": "spider-agent",
    "swe": "SWE-agent",
    "codex": "codex_agent",
    "claude": "claude_agent",
    "openhands": "openhands",
}


def load(path: Path):
    spec = importlib.util.spec_from_file_location("elt_vendored_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_spider(spec: dict, root: Path):
    module = types.ModuleType("spider_agent.agent.models")

    def call_llm(body):
        request = urllib.request.Request(
            spec["host_url"] + "/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json", "Authorization": "Bearer " + spec["token"]},
        )
        with urllib.request.urlopen(request, timeout=spec["timeout_s"]) as response:
            reply = json.load(response)
        return True, reply["choices"][0]["message"]["content"], 0.0

    module.call_llm = call_llm
    sys.modules[module.__name__] = module
    sys.path.insert(0, str(root / "spider-agent"))
    from common import prepare_destination
    from spider_agent.agent.agents import PromptAgent
    from spider_agent.envs.spider_agent import Spider_Agent_Env

    prepare_destination(spec["destination"], Path(spec["inputs_root"]) / spec["task_id"])
    public_input = Path(spec["attempt_root"]) / "spider-input"
    shutil.copytree(
        Path(spec["inputs_root"]) / spec["task_id"],
        public_input,
        ignore=shutil.ignore_patterns("*_credential.json"),
    )
    instruction = (
        "You are required to build an ELT pipeline by extracting data from multiple sources, "
        "such as custom APIs, Postgres, MongoDB, flat files and cloud service S3. The extracted "
        "data will be loaded into a target system, Snowflake, followed by writing transformation "
        "queries to construct final tables for downstream use."
    )
    agent = PromptAgent(model="tinker", max_memory_length=25, max_steps=spec["max_turns"])
    env = Spider_Agent_Env(
        env_config={
            "init_args": {
                "name": spec["container_name"],
                "work_dir": "/workspace",
                "network": spec["network"],
            },
            "image_name": spec["image"],
        },
        task_config={
            "instance_id": spec["task_id"],
            "instruction": instruction,
            "config": [{"type": "copy_all_subfiles", "parameters": {"dirs": [str(public_input)]}}],
        },
        cache_dir=str(Path(spec["attempt_root"]) / "cache"),
        mnt_dir=spec["workspace"],
    )
    agent.set_env_and_task(env)
    try:
        done, result = agent.run()
        trajectory = agent.get_trajectory()
        env.post_process()
        Path(spec["result_file"]).write_text(
            json.dumps({"finished": done, "result": result, **trajectory}, default=str)
        )
    finally:
        subprocess.run(
            ["docker", "rm", "-f", spec["container_name"]], capture_output=True, check=False
        )


def run_swe(spec: dict, root: Path):
    import yaml

    sys.path.insert(0, str(root / "SWE-agent"))
    upstream = load(root / "SWE-agent/run_elt.py")
    from common import prepare_destination
    from sweagent.run.run_single import RunSingle, RunSingleConfig

    inputs = Path(spec["inputs_root"]) / spec["task_id"]
    duckdb = spec["destination"] == "duckdb"
    if not duckdb:
        prepare_destination(spec["destination"], inputs)
    upstream.copy_task_input(inputs, Path(spec["workspace"]))
    config_path = Path(spec["attempt_root"]) / "swe.yaml"
    upstream.write_destination_config(spec["destination"], config_path)
    config = yaml.safe_load(config_path.read_text())
    docker_args = ["--network", spec["network"], "-v", spec["workspace"] + ":/workspace"]
    if duckdb:
        docker_args.extend(["-v", spec["duckdb_storage"] + ":/duckdb"])
        config["agent"]["templates"]["instance_template"] += """

For this DuckDB task, use duckdb.config from /workspace/config.yaml for the Airbyte
destination definition UUID, destination_path, and schema. Airbyte writes to
the /local/... path from config.yaml. Use /duckdb/warehouse.duckdb for solver
queries and dbt. With Airbyte provider
0.6.5, use airbyte_destination_custom and JSON-encode the destination_path and
schema in its configuration. Set the connection
namespace to the destination, use the original stream names, and run one sync
at a time. Wait for each sync to finish before starting the next or opening dbt.
The connector writes <schema>._airbyte_raw_<stream> with a JSON _airbyte_data
column. Extract and cast source fields from that JSON in your models or staging
queries. The loader and dbt share the same file at /duckdb/warehouse.duckdb.
Keep the dbt project in /workspace/elt or in a new directory directly under
/workspace. Set profile: elt_environment in dbt_project.yml and pass
--profiles-dir /workspace to dbt. The installed profile writes to elt_models.
"""
    config.setdefault("agent", {})["model"] = {
        "name": "openai/gpt-4.1",
        "api_base": spec["host_url"],
        "api_key": spec["token"],
        "per_instance_cost_limit": 0,
        "total_cost_limit": 0,
        "completion_kwargs": {"max_tokens": spec["max_tokens"]},
    }
    deployment = {"type": "docker", "image": spec["image"], "docker_args": docker_args}
    if duckdb and spec.get("duckdb_control_port"):
        deployment = start_private_swe(spec, docker_args)
    config["env"] = {
        "repo": {"type": "preexisting", "repo_name": "workspace"},
        "deployment": deployment,
    }
    config["problem_statement"] = {"type": "empty", "id": spec["task_id"]}
    config["output_dir"] = str(Path(spec["attempt_root"]) / "swe-results")
    empty_env = Path(spec["attempt_root"]) / "empty.env"
    empty_env.touch()
    config["env_var_path"] = str(empty_env)
    os.chdir(root / "SWE-agent")
    run = RunSingle.from_config(RunSingleConfig.model_validate(config))
    if deployment["type"] == "docker":
        run.env.deployment._get_container_name = lambda: spec["container_name"]
    try:
        run.run()
    finally:
        run.env.close()


def start_private_swe(spec: dict, docker_args: list[str]) -> dict:
    token = secrets.token_urlsafe(32)
    name = spec["container_name"]
    command = [
        "docker", "run", "--rm", "--detach", "--pull=never",
        "--name", name, "--network-alias", "elt-solver",
        "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=256",
        "--memory=4g", "--cpus=2", "--read-only",
        "--tmpfs", "/tmp:rw,exec,nosuid,size=512m",
        "--tmpfs", "/root:rw,exec,nosuid,size=64m",
        *docker_args, "--entrypoint", "swerex-remote", spec["image"], "--auth-token", token,
    ]
    host, port = "http://127.0.0.1", spec["duckdb_control_port"]
    request = urllib.request.Request(f"{host}:{port}/is_alive", headers={"X-API-Key": token})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    result = subprocess.run(command, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f"could not start private SWE-ReX (Docker exit {result.returncode})")
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            with opener.open(request, timeout=2) as response:
                if response.status == 200:
                    return {"type": "remote", "host": host, "port": port, "auth_token": token,
                            "timeout": 30}
        except OSError:
            pass
        time.sleep(0.5)
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=20)
    raise RuntimeError("private SWE-ReX did not become ready")


async def run_sdk(spec: dict, root: Path):
    agent = spec["agent"]
    upstream = load(root / FOLDERS[agent] / "run.py")
    upstream.IMAGE_NAME = spec["image"]
    upstream.NETWORK_NAME = spec["network"]
    model = "tinker"
    if agent == "codex":
        from openai import AsyncOpenAI

        from agents import set_default_openai_client, set_tracing_disabled

        set_tracing_disabled(True)
        set_default_openai_client(
            AsyncOpenAI(
                base_url=spec["host_url"],
                api_key=spec["token"],
                max_retries=0,
                timeout=spec["timeout_s"],
            ),
            use_for_tracing=False,
        )
    elif agent == "claude":
        os.environ["ANTHROPIC_BASE_URL"] = spec["container_url"].removesuffix("/v1")
        os.environ["ANTHROPIC_API_KEY"] = spec["token"]
        os.environ["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
        model = "claude-sonnet-4-5"
    else:
        original_llm = upstream.LLM

        def llm(**kwargs):
            return original_llm(**kwargs, base_url=spec["container_url"])

        upstream.LLM = llm
        upstream.AGENT_SERVER_PORT = spec["openhands_port"]
        model = "openai/gpt-4.1"
    out_dir = Path(spec["workspace"])
    out_dir.mkdir(parents=True, exist_ok=True)
    result = out_dir / (agent if agent != "codex" else "codex") / "result.json"
    await upstream._run_task_body(
        db=spec["task_id"],
        destination=spec["destination"],
        credential=None,
        instance_id=spec["attempt_id"] + "/" + spec["task_id"],
        experiment_id=spec["attempt_id"],
        out_dir=out_dir,
        result_path=result,
        inputs_root=Path(spec["inputs_root"]),
        model=model,
        max_turns=spec["max_turns"] + 1,
    )
    result_data = json.loads(result.read_text())
    Path(spec["result_file"]).write_text(json.dumps(result_data))
    if not result_data.get("finished", False):
        raise RuntimeError("vendored SDK reported an unsuccessful session")


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    root = Path(spec["agents_root"])
    sys.path.insert(0, str(root))
    os.environ.update(
        {
            "OPENAI_API_KEY": spec["token"],
            "LLM_API_KEY": spec["token"],
            "OPENAI_BASE_URL": spec["host_url"],
            "OPENAI_API_BASE": spec["host_url"],
        }
    )
    if spec["agent"] == "spider":
        run_spider(spec, root)
    elif spec["agent"] == "swe":
        run_swe(spec, root)
    else:
        asyncio.run(run_sdk(spec, root))


if __name__ == "__main__":
    main()
