"""Tinker Cookbook environment, group builder, rollout strategy and dataset for ELT tasks."""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition, TokenCompleter
from tinker_cookbook.rl.rollout_strategy import RolloutResult, RolloutStrategy
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    RolloutError,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

from elt_environment.admission import admit_tasks
from elt_environment.artifact import ArtifactFormatError, parse_artifact
from elt_environment.grading import GraderConfig, grade_artifact
from elt_environment.outcomes import DiscardGroupSignal, require_tinker_reward
from elt_environment.prompt import build_messages
from elt_environment.tasks import TaskRef, discover_tasks, split_by_family

logger = logging.getLogger(__name__)

_GRADER_SLOTS: dict[int, asyncio.Semaphore] = {}


@functools.lru_cache(maxsize=8)
def cached_renderer(model_name: str, renderer_name: str) -> renderers.Renderer:
    """Return one renderer per model and renderer name for the whole process."""

    return renderers.get_renderer(renderer_name, tokenizer=get_tokenizer(model_name))


def _grader_slots(limit: int) -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    slots = _GRADER_SLOTS.get(loop_id)
    if slots is None:
        slots = _GRADER_SLOTS[loop_id] = asyncio.Semaphore(limit)
    return slots


class EltEnv(Env):
    """One attempt at one task: the policy writes the whole artifact in a single reply."""

    def __init__(
        self,
        task: TaskRef,
        messages: list[renderers.Message],
        renderer: renderers.Renderer,
        grader: GraderConfig,
        grader_concurrency: int,
    ) -> None:
        self.task = task
        self.messages = messages
        self.renderer = renderer
        self.grader = grader
        self.grader_concurrency = grader_concurrency

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return self.renderer.build_generation_prompt(self.messages), self.stop_condition

    def _final(self, reward: float, metrics: Metrics) -> StepResult:
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, _ = self.renderer.parse_response(action)
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        try:
            files = parse_artifact(renderers.get_text_content(message))
        except ArtifactFormatError:
            return self._final(0.0, {"format_error": 1.0, "r_el": 0.0, "r_t": 0.0, "el_pass": 0.0})
        async with _grader_slots(self.grader_concurrency):
            outcome = await asyncio.to_thread(grade_artifact, self.task, files, self.grader)
        reward = require_tinker_reward(outcome)
        return self._final(reward, {"format_error": 0.0, **outcome.metrics})


@dataclass(frozen=True)
class EltGroupBuilder(EnvGroupBuilder):
    """``group_size`` independent attempts at the same task."""

    task: TaskRef
    group_size: int
    model_name: str
    renderer_name: str
    grader: GraderConfig
    grader_concurrency: int

    async def make_envs(self) -> Sequence[Env]:
        renderer = cached_renderer(self.model_name, self.renderer_name)
        messages = build_messages(self.task)
        return [
            EltEnv(self.task, messages, renderer, self.grader, self.grader_concurrency)
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["elt", self.task.origin]


class GroupRetryExhausted(RuntimeError):
    """Every attempt at a group had at least one rollout without a label."""


@dataclass(frozen=True)
class RetryWholeGroup(RolloutStrategy):
    """Discard the whole group when any rollout has no label, then sample a fresh group.

    No trajectory from a discarded attempt is kept, so a group is never a mix
    of attempts. After ``max_attempts`` discards the group is skipped and the
    trainer counts it in ``rollout_errors/groups_skipped``.
    """

    max_attempts: int = 3

    @property
    def catches_group_errors(self) -> bool:
        return True

    async def execute(
        self,
        env_group_builder: EnvGroupBuilder,
        policy: TokenCompleter,
    ) -> RolloutResult:
        from tinker_cookbook.rl.rollouts import do_single_rollout

        errors: list[RolloutError] = []
        for _ in range(self.max_attempts):
            envs = await env_group_builder.make_envs()
            rollouts = [asyncio.ensure_future(do_single_rollout(policy, env)) for env in envs]
            try:
                trajectories = await asyncio.gather(*rollouts)
            except DiscardGroupSignal as signal:
                await _cancel_all(rollouts)
                errors.append(RolloutError(error_type="discard_group", error_message=signal.reason_code))
                continue
            except BaseException:
                await _cancel_all(rollouts)
                raise
            return RolloutResult(trajectories=list(trajectories), envs=envs, errors=errors)
        raise GroupRetryExhausted(
            f"{self.max_attempts} attempts each had a rollout without a label: "
            + ", ".join(error.error_message for error in errors)
        )


async def _cancel_all(rollouts: Sequence[asyncio.Future]) -> None:
    for rollout in rollouts:
        rollout.cancel()
    await asyncio.gather(*rollouts, return_exceptions=True)


class EltDataset(RLDataset):
    """Batches of task groups in a seeded order; each epoch is shuffled again."""

    def __init__(
        self,
        tasks: Sequence[TaskRef],
        *,
        groups_per_batch: int,
        epochs: int,
        seed: int,
        shuffle: bool,
        make_builder: Callable[[TaskRef], EltGroupBuilder],
    ) -> None:
        if groups_per_batch < 1:
            raise ValueError("groups_per_batch must be at least 1")
        rng = random.Random(seed)
        order: list[TaskRef] = []
        for _ in range(max(1, epochs)):
            epoch = list(tasks)
            if shuffle:
                rng.shuffle(epoch)
            order.extend(epoch)
        self.order = order
        self.groups_per_batch = groups_per_batch
        self.make_builder = make_builder

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.groups_per_batch
        return [self.make_builder(task) for task in self.order[start : start + self.groups_per_batch]]

    def __len__(self) -> int:
        return len(self.order) // self.groups_per_batch


@chz.chz
class EltDatasetBuilder(RLDatasetBuilder):
    """Discover releases, admit tasks, split by family, and build train and eval datasets."""

    releases_root: str
    taskgen_root: str
    attempts_root: str
    model_name: str
    renderer_name: str
    group_size: int = 8
    groups_per_batch: int = 8
    epochs: int = 4
    eval_families: int = 5
    seed: int = 0
    max_tokens: int = 10240
    context_length: int = 65536
    grader_concurrency: int = 8
    grader_deadline_s: float | None = None
    verify_releases: bool = True

    async def __call__(self) -> tuple[EltDataset, EltDataset | None]:
        renderer = cached_renderer(self.model_name, self.renderer_name)
        tasks = discover_tasks(Path(self.releases_root))
        admitted, excluded = await asyncio.to_thread(
            admit_tasks,
            tasks,
            renderer,
            context_length=self.context_length,
            max_tokens=self.max_tokens,
            verify=self.verify_releases,
        )
        for exclusion in excluded:
            logger.warning("excluded %s: %s", exclusion.task_id, exclusion.reason)
        train, evaluation = split_by_family(
            [admission.task for admission in admitted], self.eval_families, self.seed
        )
        logger.info(
            "tasks: %d found, %d admitted, %d train, %d eval (%s)",
            len(tasks),
            len(admitted),
            len(train),
            len(evaluation),
            ", ".join(task.task_id for task in evaluation),
        )
        grader = GraderConfig(
            taskgen_root=Path(self.taskgen_root),
            attempts_root=Path(self.attempts_root),
            grader_deadline_s=self.grader_deadline_s,
        )

        def builder(group_size: int) -> Callable[[TaskRef], EltGroupBuilder]:
            return lambda task: EltGroupBuilder(
                task=task,
                group_size=group_size,
                model_name=self.model_name,
                renderer_name=self.renderer_name,
                grader=grader,
                grader_concurrency=self.grader_concurrency,
            )

        train_dataset = EltDataset(
            train,
            groups_per_batch=self.groups_per_batch,
            epochs=self.epochs,
            seed=self.seed,
            shuffle=True,
            make_builder=builder(self.group_size),
        )
        eval_dataset = (
            EltDataset(
                evaluation,
                groups_per_batch=len(evaluation),
                epochs=1,
                seed=self.seed,
                shuffle=False,
                make_builder=builder(1),
            )
            if evaluation
            else None
        )
        return train_dataset, eval_dataset
