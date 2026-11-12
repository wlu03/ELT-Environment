"""Sample and grade multi-turn agent trajectories without updating the model.

Run this before GRPO: a task whose group rewards are all equal gives no
gradient, so a model that scores 0 on every task cannot improve.

Example::

    See README.md for runtime and grader configuration.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import shlex
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import chz
import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout

from elt_environment.admission import admit_tasks
from elt_environment.env import (
    EltGroupBuilder,
    GroupRetryExhausted,
    RetryWholeGroup,
    cached_renderer,
)
from elt_environment.grading import GraderConfig
from elt_environment.runner import AgentConfig
from elt_environment.tasks import TaskRef, discover_tasks, split_by_family
from elt_environment.train import (
    DEFAULT_MODEL,
    DEFAULT_RELEASES,
    PROJECT_ROOT,
    STATE_ROOT,
    TASKGEN_ROOT,
    agent_config,
    check_run,
    default_renderer_name,
    load_env_file,
)

logger = logging.getLogger(__name__)


@chz.chz
class ProbeConfig:
    """Which model, which tasks, and how many samples per task."""

    releases_root: str = str(DEFAULT_RELEASES)
    agent: str = "spider"
    inputs_root: str = ""
    agent_python: str = AgentConfig().python
    max_turns: int = 100
    agent_timeout_s: float = 7200
    agent_image: str | None = None
    agent_network: str = "elt-docker_elt_network"
    bridge_bind_host: str = "127.0.0.1"
    bridge_container_host: str = "host.docker.internal"
    grader_command: str = ""
    allow_declarative_grader: bool = False
    taskgen_root: str = str(TASKGEN_ROOT)
    model_name: str = DEFAULT_MODEL
    model_path: str | None = None
    renderer_name: str | None = None
    context_length: int = 65536
    max_tokens: int = 10240
    temperature: float = 1.0
    tasks: int = 6
    task_ids: str | None = None
    samples: int = 4
    eval_families: int = 5
    seed: int = 0
    grader_concurrency: int = 8
    verify_releases: bool = True
    out_dir: str | None = None


def choose_tasks(train: Sequence[TaskRef], count: int, task_ids: str | None) -> list[TaskRef]:
    """Return the tasks named in ``task_ids``, or ``count`` tasks spread evenly over ``train``.

    A name that is not an admitted training task, a count below one, or an
    empty choice raises ``ValueError`` before any sampling starts.
    """

    if task_ids:
        wanted = set(task_ids.split(","))
        chosen = [task for task in train if task.task_id in wanted]
        missing = wanted - {task.task_id for task in chosen}
        if missing:
            raise ValueError(f"not admitted training tasks: {', '.join(sorted(missing))}")
    else:
        if count < 1:
            raise ValueError("tasks must be at least 1")
        chosen = train[:: max(1, len(train) // count)][:count]
    if not chosen:
        raise ValueError("no training task to probe")
    return chosen


async def probe(config: ProbeConfig) -> dict:
    """Grade ``samples`` agent attempts per chosen task and summarize rewards."""

    if config.samples < 1:
        raise ValueError("samples must be at least 1")
    check_run(config)
    renderer_name = config.renderer_name or default_renderer_name(config.model_name)
    renderer = cached_renderer(config.model_name, renderer_name)
    admitted, _ = admit_tasks(
        discover_tasks(Path(config.releases_root)),
        verify=config.verify_releases,
        agent=agent_config(config),
    )
    train, _ = split_by_family([a.task for a in admitted], config.eval_families, config.seed)
    chosen = choose_tasks(train, config.tasks, config.task_ids)
    out_dir = Path(
        config.out_dir or STATE_ROOT / "probes" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    grader = GraderConfig(
        taskgen_root=Path(config.taskgen_root),
        attempts_root=STATE_ROOT / "attempts",
        verify_release=config.verify_releases,
        command=tuple(shlex.split(config.grader_command)),
    )
    service = tinker.ServiceClient()
    sampling = await service.create_sampling_client_async(
        model_path=config.model_path, base_model=None if config.model_path else config.model_name
    )
    policy = TinkerTokenCompleter(
        sampling, max_tokens=config.max_tokens, temperature=config.temperature
    )
    strategy = RetryWholeGroup(max_attempts=3)

    async def one(task) -> dict:
        builder = EltGroupBuilder(
            task=task,
            group_size=config.samples,
            model_name=config.model_name,
            renderer_name=renderer_name,
            grader=grader,
            grader_concurrency=config.grader_concurrency,
            agent=agent_config(config),
        )
        try:
            group = await do_group_rollout(builder, policy, strategy=strategy)
        except GroupRetryExhausted as error:
            return {"task_id": task.task_id, "discarded": str(error)}
        task_dir = out_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        rewards, metrics = [], []
        for index, trajectory in enumerate(group.trajectories_G):
            last = trajectory.transitions[-1]
            turns = []
            for transition in trajectory.transitions:
                message, _ = renderer.parse_response(transition.ac.tokens)
                turns.append(
                    {
                        "message": renderer.to_openai_message(message),
                        "tokens": transition.ac.tokens,
                        "logprobs": transition.ac.logprobs,
                        "reward": transition.reward,
                    }
                )
            (task_dir / f"trajectory_{index}.json").write_text(json.dumps(turns, indent=2))
            rewards.append(last.reward)
            metrics.append(dict(last.metrics))
        return {"task_id": task.task_id, "rewards": rewards, "metrics": metrics}

    results = await asyncio.gather(*(one(task) for task in chosen))
    scored = [r for r in results if "rewards" in r]
    all_metrics = [m for r in scored for m in r["metrics"]]
    rollouts = len(all_metrics)
    phases = collections.Counter(
        next(
            (k[len("phase_") :] for k, v in m.items() if k.startswith("phase_") and v == 1.0),
            "format",
        )
        for m in all_metrics
    )
    summary = {
        "model": config.model_path or config.model_name,
        "renderer": renderer_name,
        "agent": config.agent,
        "tasks": len(chosen),
        "rollouts": rollouts,
        "mean_reward": sum(sum(r["rewards"]) for r in scored) / max(1, rollouts),
        "el_pass_rate": sum(m.get("el_pass", 0.0) for m in all_metrics) / max(1, rollouts),
        "full_reward_rate": sum(1 for r in scored for x in r["rewards"] if x == 1.0)
        / max(1, rollouts),
        "format_error_rate": sum(m.get("format_error", 0.0) for m in all_metrics)
        / max(1, rollouts),
        "first_failed_phase": dict(phases),
        "groups_with_reward_variance": sum(1 for r in scored if len(set(r["rewards"])) > 1),
        "groups_discarded": len(results) - len(scored),
        "per_task": {r["task_id"]: r.get("rewards", r.get("discarded")) for r in results},
        "replies": str(out_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_env_file(PROJECT_ROOT / ".env")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set; put it in .env or the environment")
    print(json.dumps(asyncio.run(probe(chz.entrypoint(ProbeConfig))), indent=2))
