"""Discover public/private task bundles and split their tasks by family."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from elt_taskgen.destinations import destination_from_config
from elt_taskgen.models import validate_task_id_segment


@dataclass(frozen=True)
class TaskRef:
    """One parent task inside a public/private bundle."""

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


def read_task(release_dir: Path, task_id: str) -> TaskRef:
    """Read family from private TaskIR and destination from public config."""
    validate_task_id_segment(task_id)
    public = release_dir / "public" / task_id
    private = release_dir / "private" / task_id
    metadata = private / "task_ir.json"
    if not metadata.exists() and not metadata.is_symlink():
        metadata = private / "semantic" / "task_ir.json"
    config = public / "config.yaml"
    for path in (metadata, config):
        for parent in (path, *path.parents):
            if parent.is_symlink():
                raise ValueError(f"task inputs must not traverse symlinks: {path}")
            if parent == release_dir:
                break
    if any(path.is_symlink() for path in public.rglob("*")):
        raise ValueError(f"public task inputs must not contain symlinks: {public}")
    values = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or values.get("task_id") != task_id:
        raise ValueError(f"{metadata}: TaskIR task_id does not match {task_id}")
    family = values.get("family_id")
    if not isinstance(family, str) or not family.strip():
        raise ValueError(f"{metadata}: task has no family_id")
    destination_config = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(destination_config, dict):
        raise TypeError(f"{config}: expected a destination configuration object")
    destination = destination_from_config(destination_config).value
    return TaskRef(release_dir, task_id, family, destination)


def discover_tasks(releases_root: Path) -> list[TaskRef]:
    """Discover parent tasks in one bundle or a directory of bundles.

    Each parent has matching public/private directories and a private TaskIR.
    Public EL/T variants without their own private task are not extra examples.
    A missing family is an error: guessing it could leak related tasks across
    training and evaluation splits.
    """

    root = Path(releases_root).absolute()
    bundles = (
        [root]
        if (root / "public").is_dir()
        else [path.parent for path in sorted(root.glob("*/public"))]
    )
    tasks: list[TaskRef] = []
    for bundle in bundles:
        for private in sorted((bundle / "private").iterdir()):
            if private.is_dir() and (bundle / "public" / private.name).is_dir():
                tasks.append(read_task(bundle, private.name))
    if not tasks:
        raise FileNotFoundError(f"no public/private task bundles under {root}")
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
