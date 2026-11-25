"""Check the public source roster against applied Terraform connections."""

import json
from pathlib import Path

import yaml


def missing_streams(config: dict, state: dict) -> list[str]:
    expected = {
        name
        for source in ("postgres", "mongodb", "custom_api")
        for name in config.get(source, {}).get("config", {}).get("tables", [])
    }
    expected.update(row["table"] for row in config.get("flat_files", []))
    expected.update(row["table"] for row in config.get("aws_s3", {}).get("data", []))
    applied = {
        stream["name"].casefold()
        for resource in state.get("resources", [])
        if resource["type"] == "airbyte_connection"
        for instance in resource["instances"]
        for stream in (instance["attributes"].get("configurations") or {}).get("streams") or []
    }
    return sorted(name for name in expected if name.casefold() not in applied)


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parent
    state = workspace / "elt" / "terraform.tfstate"
    if not state.is_file():
        raise SystemExit("Run Terraform apply in /workspace/elt before checking streams.")
    config = yaml.safe_load((workspace / "config.yaml").read_text())
    missing = missing_streams(config, json.loads(state.read_text()))
    if missing:
        raise SystemExit("Missing streams in applied connections: " + ", ".join(missing))
    print("All configured streams are present in applied connections.")
