"""Launch a vendored agent in its own interpreter, changing model transport only.

Run by runner.py; not imported into the training process. Provider keys here
are per-attempt gateway tokens, never Tinker or commercial provider keys.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import subprocess
import sys
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
    # Replace only call_llm; importing upstream models.py would load every
    # commercial provider even though no such provider is used here.
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
    # The instruction and agent/environment classes are the upstream runner's.
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

    prepare_destination(spec["destination"], Path(spec["inputs_root"]) / spec["task_id"])
    upstream.copy_task_input(Path(spec["inputs_root"]) / spec["task_id"], Path(spec["workspace"]))
    config_path = Path(spec["attempt_root"]) / "swe.yaml"
    upstream.write_destination_config(spec["destination"], config_path)
    config = yaml.safe_load(config_path.read_text())
    config.setdefault("agent", {})["model"] = {
        "name": "openai/gpt-4.1",
        "api_base": spec["host_url"],
        "api_key": spec["token"],
        "per_instance_cost_limit": 0,
        "total_cost_limit": 0,
        "completion_kwargs": {"max_tokens": spec["max_tokens"]},
    }
    config["env"] = {
        "repo": {"type": "preexisting", "repo_name": "workspace"},
        "deployment": {
            "type": "docker",
            "image": spec["image"],
            "docker_args": ["--network", spec["network"], "-v", spec["workspace"] + ":/workspace"],
        },
    }
    config["problem_statement"] = {"type": "empty", "id": spec["task_id"]}
    config["output_dir"] = str(Path(spec["attempt_root"]) / "swe-results")
    empty_env = Path(spec["attempt_root"]) / "empty.env"
    empty_env.touch()
    config["env_var_path"] = str(empty_env)
    os.chdir(root / "SWE-agent")
    # The framework owns history, tools, retries and termination. Its shell
    # runs in the same /workspace bind mount used by the other agents.
    run = RunSingle.from_config(RunSingleConfig.model_validate(config))
    # Resource naming only: let the coordinator reap the exact deployment if
    # this worker is cancelled before the upstream cleanup hook can run.
    run.env.deployment._get_container_name = lambda: spec["container_name"]
    try:
        run.run()
    finally:
        run.env.close()


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
    # Call the original task body, with the same prompts, tools and SDK loop.
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
        # Leave call N+1 reachable so the coordinator's budget wins before
        # an SDK turn-limit exception can obscure the final tool results.
        max_turns=spec["max_turns"] + 1,
    )
    # Keep coordinator status outside the writable agent mount.
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
