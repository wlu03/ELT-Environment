"""Verify frozen tasks and installed development inputs before sampling."""

from dataclasses import dataclass

from elt_taskgen.training import load_workspace_package

from elt_environment.runner import AgentConfig, validate_input
from elt_environment.tasks import TaskRef


@dataclass(frozen=True)
class Admission:
    task: TaskRef


@dataclass(frozen=True)
class Exclusion:
    task_id: str
    reason: str


def admit_tasks(tasks, *, verify=True, agent: AgentConfig):
    """Admit packages; measure context on each actual request in EltEnv."""
    admitted, excluded = [], []
    for task in tasks:
        try:
            load_workspace_package(task.release_dir, task.task_id, verify=verify)
            validate_input(agent, task)
        except Exception as error:  # noqa: BLE001 -- package failures are admission exclusions
            excluded.append(Exclusion(task.task_id, f"input_invalid:{type(error).__name__}"))
        else:
            admitted.append(Admission(task))
    return admitted, excluded
