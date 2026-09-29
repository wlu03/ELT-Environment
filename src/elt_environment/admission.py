"""Keep the tasks whose release verifies and whose prompt and known-correct reply fit the model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from elt_taskgen.training import load_workspace_package
from elt_taskgen.training.canonical import render_canonical_project
from tinker_cookbook.renderers import Renderer

from elt_environment.artifact import format_artifact
from elt_environment.prompt import EXAMPLE_TASK_ID, BundleRedactionError, build_messages
from elt_environment.tasks import TaskRef


@dataclass(frozen=True)
class Admission:
    """Token counts for one admitted task."""

    task: TaskRef
    prompt_tokens: int
    reply_tokens: int


@dataclass(frozen=True)
class Exclusion:
    """A task left out of training, with the reason."""

    task_id: str
    reason: str


def admit_tasks(
    tasks: Sequence[TaskRef],
    renderer: Renderer,
    *,
    context_length: int,
    max_tokens: int,
    verify: bool = True,
) -> tuple[list[Admission], list[Exclusion]]:
    """Split ``tasks`` into admitted tasks and exclusions.

    A task is excluded when its release fails to load or verify, when a
    credential in its bundle cannot be redacted, when the known-correct reply
    is longer than ``max_tokens``, or when the prompt plus ``max_tokens``
    exceeds ``context_length``. An excluded task is not
    scored; it is a task or configuration problem, not a policy failure.
    """

    admitted: list[Admission] = []
    excluded: list[Exclusion] = []
    for task in tasks:
        if task.task_id == EXAMPLE_TASK_ID:
            excluded.append(Exclusion(task.task_id, "example_task"))
            continue
        try:
            package = load_workspace_package(task.release_dir, task.task_id, verify=verify)
            reply = format_artifact(render_canonical_project(package))
        except Exception as error:
            excluded.append(Exclusion(task.task_id, f"release_invalid:{type(error).__name__}"))
            continue
        try:
            messages = build_messages(task)
        except BundleRedactionError:
            excluded.append(Exclusion(task.task_id, "bundle_redaction_failed"))
            continue
        prompt_tokens = renderer.build_generation_prompt(messages).length
        reply_tokens = len(renderer.tokenizer.encode(reply, add_special_tokens=False))
        if reply_tokens > max_tokens:
            excluded.append(Exclusion(task.task_id, "reply_over_max_tokens"))
        elif prompt_tokens + max_tokens > context_length:
            excluded.append(Exclusion(task.task_id, "prompt_over_context"))
        else:
            admitted.append(Admission(task, prompt_tokens, reply_tokens))
    return admitted, excluded
