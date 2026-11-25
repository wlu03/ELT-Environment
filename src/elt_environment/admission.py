"""Verify frozen tasks and installed development inputs before sampling."""

from dataclasses import dataclass

from elt_environment.runner import AgentConfig, validate_input
from elt_environment.tasks import TaskRef, read_task


@dataclass(frozen=True)
class Admission:
    task: TaskRef


@dataclass(frozen=True)
class Exclusion:
    task_id: str
    reason: str


def admit_tasks(tasks, *, agent: AgentConfig):
    """Validate task identity and installed inputs before sampling."""
    admitted, excluded = [], []
    for task in tasks:
        try:
            if read_task(task.release_dir, task.task_id) != task:
                raise ValueError("task metadata changed after discovery")
            validate_input(agent, task)
        except Exception as error:  # noqa: BLE001 -- package failures are admission exclusions
            excluded.append(Exclusion(task.task_id, f"input_invalid:{type(error).__name__}"))
        else:
            admitted.append(Admission(task))
    return admitted, excluded
