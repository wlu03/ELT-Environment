"""Tinker Cookbook environment, group builder, rollout strategy and dataset for ELT tasks."""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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
from elt_environment.outcomes import DiscardGroupSignal, GraderOutcome, require_tinker_reward
from elt_environment.prompt import build_messages
from elt_environment.tasks import TaskRef, discover_tasks, split_by_family

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def cached_renderer(model_name: str, renderer_name: str) -> renderers.Renderer:
    """Return one renderer per model and renderer name for the whole process."""

    return renderers.get_renderer(renderer_name, tokenizer=get_tokenizer(model_name))


@functools.lru_cache(maxsize=None)
def _grader_pool(workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="elt-grader")


async def run_grader(
    task: TaskRef, files: Mapping[str, str], grader: GraderConfig, workers: int
) -> GraderOutcome:
    """Grade on a shared pool of ``workers`` threads and return the outcome.

    A pool thread stays occupied until its grade returns, even when the
    awaiting rollout is cancelled, so at most ``workers`` grades run at once.
    A cancelled grade that has not started is removed from the queue.
    """

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_grader_pool(workers), grade_artifact, task, files, grader)


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
        outcome = await run_grader(self.task, files, self.grader, self.grader_concurrency)
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

    Every rollout of an attempt runs to completion before the next attempt
    starts, so no grader from a discarded attempt is still running when the
    fresh group is graded; a grade cannot outlast taskgen's grader deadline.
    No trajectory from a discarded attempt is kept, so a group is never a mix
    of attempts. After ``max_attempts`` discards, ``GroupRetryExhausted``
    propagates and stops the run, so the trainer never trains on a batch
    with fewer groups than configured.
    """

    max_attempts: int = 3

    @property
    def catches_group_errors(self) -> bool:
        return False

    async def execute(
        self,
        env_group_builder: EnvGroupBuilder,
        policy: TokenCompleter,
    ) -> RolloutResult:
        from tinker_cookbook.rl.rollouts import do_single_rollout

        errors: list[RolloutError] = []
        for _ in range(self.max_attempts):
            envs = await env_group_builder.make_envs()
            results = await asyncio.gather(
                *(do_single_rollout(policy, env) for env in envs), return_exceptions=True
            )
            failures = [result for result in results if isinstance(result, BaseException)]
            for failure in failures:
                if not isinstance(failure, DiscardGroupSignal):
                    raise failure
            if failures:
                errors.extend(
                    RolloutError(error_type="discard_group", error_message=failure.reason_code)
                    for failure in failures
                )
                continue
            return RolloutResult(trajectories=list(results), envs=envs, errors=errors)
        raise GroupRetryExhausted(
            f"{self.max_attempts} attempts each had a rollout without a label: "
            + ", ".join(error.error_message for error in errors)
        )


class EltDataset(RLDataset):
    """Complete batches of task groups in a seeded order.

    Each epoch is shuffled again. When the task occurrences do not fill the
    last batch, it is filled from a further pass over the tasks, so every task
    is trained at least ``epochs`` times and every batch has
    ``groups_per_batch`` groups.
    """

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
        if not tasks:
            raise ValueError("a dataset needs at least one task")
        rng = random.Random(seed)

        def one_pass() -> list[TaskRef]:
            items = list(tasks)
            if shuffle:
                rng.shuffle(items)
            return items

        order: list[TaskRef] = []
        for _ in range(max(1, epochs)):
            order.extend(one_pass())
        while len(order) % groups_per_batch:
            order.extend(one_pass()[: groups_per_batch - len(order) % groups_per_batch])
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
        if not train:
            raise ValueError(
                f"no training tasks: {len(tasks)} found, {len(admitted)} admitted, "
                f"{len(evaluation)} held out for evaluation"
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
            verify_release=self.verify_releases,
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
