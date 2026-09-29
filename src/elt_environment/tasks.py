"""Find frozen ELT-taskgen releases and split their tasks by family."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MANIFEST_NAME = "release_manifest.json"


@dataclass(frozen=True)
class TaskRef:
    """One task inside a frozen release."""

    release_dir: Path
    task_id: str
    family: str
    destination: str

    @property
    def public_dir(self) -> Path:
        return self.release_dir / "public" / self.task_id

    @property
    def origin(self) -> str:
        return self.task_id.split("__", 1)[0]


def discover_tasks(releases_root: Path) -> list[TaskRef]:
    """Return every task under ``releases_root``, sorted by task id.

    ``releases_root`` is one release directory or a directory of single-task
    releases. Every task must have a family and a destination entry in its
    manifest; a task with no family would otherwise become its own family and
    could be split from related tasks across training and evaluation.
    """

    root = Path(releases_root)
    if (root / MANIFEST_NAME).is_file():
        manifests = [root / MANIFEST_NAME]
    else:
        manifests = sorted(root.glob(f"*/{MANIFEST_NAME}"))
    if not manifests:
        raise FileNotFoundError(f"no {MANIFEST_NAME} under {root}")
    tasks: list[TaskRef] = []
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        families = manifest.get("families") or {}
        destinations = manifest.get("destinations") or {}
        for task_id in manifest["tasks"]:
            if task_id not in families or task_id not in destinations:
                raise ValueError(f"{path}: task {task_id} has no family or destination entry")
            tasks.append(
                TaskRef(
                    release_dir=path.parent,
                    task_id=task_id,
                    family=families[task_id],
                    destination=destinations[task_id],
                )
            )
    tasks.sort(key=lambda task: task.task_id)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("a task id appears in more than one release")
    return tasks


def split_by_family(
    tasks: Sequence[TaskRef], eval_families: int, seed: int
) -> tuple[list[TaskRef], list[TaskRef]]:
    """Hold out ``eval_families`` whole families for evaluation.

    Families are ranked by a seeded SHA-256 hash and both lists are sorted by
    task id, so the result does not depend on input order and never places
    one family on both sides.
    """

    families = sorted(
        {task.family for task in tasks},
        key=lambda family: hashlib.sha256(f"{seed}:{family}".encode()).hexdigest(),
    )
    held_out = set(families[: max(0, eval_families)])
    ordered = sorted(tasks, key=lambda task: task.task_id)
    train = [task for task in ordered if task.family not in held_out]
    evaluation = [task for task in ordered if task.family in held_out]
    return train, evaluation
