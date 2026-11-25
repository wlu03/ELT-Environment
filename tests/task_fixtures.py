"""Small public/private bundles for coordinator tests."""

import json
from pathlib import Path


def write_bundle(root: Path, task_id="t", family="f", destination="snowflake", *, semantic=False):
    public = root / "public" / task_id
    private = root / "private" / task_id
    if semantic:
        private /= "semantic"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    (private / "task_ir.json").write_text(json.dumps({"task_id": task_id, "family_id": family}))
    (public / "config.yaml").write_text(f"{destination}: {{config: {{database: dev}}}}\n")
    (public / "data_model.yaml").write_text("models: []\n")
    (public / "schemas").mkdir(exist_ok=True)
    return public
