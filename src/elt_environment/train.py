"""Run GRPO on Tinker over frozen ELT-taskgen releases.

Example::

    .venv/bin/python -m elt_environment.train max_steps=20
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.rl.train import Config, main

from elt_environment.env import EltDatasetBuilder, RetryWholeGroup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASKGEN_ROOT = PROJECT_ROOT.parent / "ELT-taskgen"
DEFAULT_RELEASES = TASKGEN_ROOT / "runs" / "batch50c_20260923" / "workspace" / "releases"
STATE_ROOT = PROJECT_ROOT / ".state"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line settings for one training run."""

    releases_root: str = str(DEFAULT_RELEASES)
    taskgen_root: str = str(TASKGEN_ROOT)
    model_name: str = DEFAULT_MODEL
    renderer_name: str | None = None
    context_length: int = 65536
    lora_rank: int = 32
    learning_rate: float = 1e-5
    group_size: int = 8
    groups_per_batch: int = 8
    epochs: int = 4
    max_steps: int | None = None
    max_tokens: int = 10240
    temperature: float = 1.0
    eval_families: int = 5
    eval_every: int = 5
    save_every: int = 5
    seed: int = 0
    grader_concurrency: int = 8
    max_group_attempts: int = 3
    verify_releases: bool = True
    log_path: str | None = None
    wandb_project: str | None = None
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def load_env_file(path: Path) -> None:
    """Copy ``KEY=value`` lines from ``path`` into the environment when the key is unset."""

    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def default_renderer_name(model_name: str) -> str:
    """Return the recommended renderer with thinking turned off when the model offers one.

    The prompt is up to about 55,000 tokens, so a 64K context leaves no room
    for long reasoning before the reply.
    """

    names = model_info.get_recommended_renderer_names(model_name)
    without_thinking = [name for name in names if name.endswith("disable_thinking")]
    return without_thinking[0] if without_thinking else names[0]


def build_config(cli: CLIConfig) -> Config:
    """Translate the command-line settings into the Cookbook training config."""

    renderer_name = cli.renderer_name or default_renderer_name(cli.model_name)
    run_name = (
        f"elt-{cli.model_name.replace('/', '-')}-g{cli.group_size}x{cli.groups_per_batch}"
        f"-lr{cli.learning_rate}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    )
    log_path = cli.log_path or str(STATE_ROOT / "runs" / run_name)
    dataset_builder = EltDatasetBuilder(
        releases_root=cli.releases_root,
        taskgen_root=cli.taskgen_root,
        attempts_root=str(STATE_ROOT / "attempts"),
        model_name=cli.model_name,
        renderer_name=renderer_name,
        group_size=cli.group_size,
        groups_per_batch=cli.groups_per_batch,
        epochs=cli.epochs,
        eval_families=cli.eval_families,
        seed=cli.seed,
        max_tokens=cli.max_tokens,
        context_length=cli.context_length,
        grader_concurrency=cli.grader_concurrency,
        verify_releases=cli.verify_releases,
    )
    return Config(
        learning_rate=cli.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli.model_name,
        recipe_name="elt_environment",
        renderer_name=renderer_name,
        lora_rank=cli.lora_rank,
        max_tokens=cli.max_tokens,
        temperature=cli.temperature,
        log_path=log_path,
        eval_every=cli.eval_every,
        save_every=cli.save_every,
        max_steps=cli.max_steps,
        rollout_error_tolerance=RetryWholeGroup(max_attempts=cli.max_group_attempts),
        remove_constant_reward_groups=False,
        wandb_project=cli.wandb_project,
        wandb_name=run_name if cli.wandb_project else None,
    )


async def cli_main(cli: CLIConfig) -> None:
    load_env_file(PROJECT_ROOT / ".env")
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY is not set; put it in .env or the environment")
    config = build_config(cli)
    cli_utils.check_log_dir(config.log_path, behavior_if_exists=cli.behavior_if_log_dir_exists)
    logger.info("log path: %s", config.log_path)
    await main(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(cli_main(chz.entrypoint(CLIConfig)))
