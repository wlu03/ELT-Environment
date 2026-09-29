"""Sample and grade replies from a model without training, to measure its pass rates.

Run this before GRPO: a task whose group rewards are all equal gives no
gradient, so a model that scores 0 on every task cannot improve.

Example::

    .venv/bin/python -m elt_environment.probe tasks=6 samples=4
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.rollouts import do_group_rollout

from elt_environment.admission import admit_tasks
from elt_environment.env import EltGroupBuilder, GroupRetryExhausted, RetryWholeGroup, cached_renderer
from elt_environment.grading import GraderConfig
from elt_environment.tasks import discover_tasks, split_by_family
from elt_environment.train import (
    DEFAULT_MODEL,
    DEFAULT_RELEASES,
    PROJECT_ROOT,
    STATE_ROOT,
    TASKGEN_ROOT,
    default_renderer_name,
    load_env_file,
)

logger = logging.getLogger(__name__)


@chz.chz
class ProbeConfig:
    """Which model, which tasks, and how many samples per task."""

    releases_root: str = str(DEFAULT_RELEASES)
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


async def probe(config: ProbeConfig) -> dict:
    """Grade ``samples`` replies for each chosen training task and summarize the rewards."""

    renderer_name = config.renderer_name or default_renderer_name(config.model_name)
    renderer = cached_renderer(config.model_name, renderer_name)
    admitted, _ = admit_tasks(
        discover_tasks(Path(config.releases_root)),
        renderer,
        context_length=config.context_length,
        max_tokens=config.max_tokens,
        verify=config.verify_releases,
    )
    train, _ = split_by_family([a.task for a in admitted], config.eval_families, config.seed)
    if config.task_ids:
        wanted = set(config.task_ids.split(","))
        chosen = [task for task in train if task.task_id in wanted]
    else:
        chosen = train[:: max(1, len(train) // config.tasks)][: config.tasks]
    out_dir = Path(config.out_dir or STATE_ROOT / "probes" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    grader = GraderConfig(
        taskgen_root=Path(config.taskgen_root),
        attempts_root=STATE_ROOT / "attempts",
        verify_release=config.verify_releases,
    )
    service = tinker.ServiceClient()
    sampling = await service.create_sampling_client_async(
        model_path=config.model_path, base_model=None if config.model_path else config.model_name
    )
    policy = TinkerTokenCompleter(sampling, max_tokens=config.max_tokens, temperature=config.temperature)
    strategy = RetryWholeGroup(max_attempts=3)

    async def one(task) -> dict:
        builder = EltGroupBuilder(
            task=task,
            group_size=config.samples,
            model_name=config.model_name,
            renderer_name=renderer_name,
            grader=grader,
            grader_concurrency=config.grader_concurrency,
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
            message, _ = renderer.parse_response(last.ac.tokens)
            (task_dir / f"reply_{index}.txt").write_text(renderers.get_text_content(message))
            rewards.append(last.reward)
            metrics.append(dict(last.metrics))
        return {"task_id": task.task_id, "rewards": rewards, "metrics": metrics}

    results = await asyncio.gather(*(one(task) for task in chosen))
    scored = [r for r in results if "rewards" in r]
    all_metrics = [m for r in scored for m in r["metrics"]]
    rollouts = len(all_metrics)
    phases = collections.Counter(
        next((k[len("phase_") :] for k, v in m.items() if k.startswith("phase_") and v == 1.0), "format")
        for m in all_metrics
    )
    summary = {
        "model": config.model_path or config.model_name,
        "renderer": renderer_name,
        "tasks": len(chosen),
        "rollouts": rollouts,
        "mean_reward": sum(sum(r["rewards"]) for r in scored) / max(1, rollouts),
        "el_pass_rate": sum(m.get("el_pass", 0.0) for m in all_metrics) / max(1, rollouts),
        "full_reward_rate": sum(1 for r in scored for x in r["rewards"] if x == 1.0) / max(1, rollouts),
        "format_error_rate": sum(m.get("format_error", 0.0) for m in all_metrics) / max(1, rollouts),
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
