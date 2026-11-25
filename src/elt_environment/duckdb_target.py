"""Prepare the filesystem-backed development target for a DuckDB task."""

import re
import shutil
from pathlib import Path
from uuid import UUID

import yaml


DBT_PROFILE = "elt_environment"


def duckdb_config(document: dict) -> dict:
    section = document.get("duckdb")
    config = section.get("config") if isinstance(section, dict) else None
    if not isinstance(config, dict):
        raise ValueError("duckdb.config must be a YAML mapping")
    try:
        definition = str(UUID(str(config["definition_id"])))
    except (KeyError, ValueError) as exc:
        raise ValueError("duckdb.config.definition_id must be the Airbyte connector UUID") from exc
    schema = config.get("schema", "main")
    if not isinstance(schema, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
        raise ValueError("duckdb.config.schema must be a simple identifier")
    if schema.casefold() == "elt_models":
        raise ValueError("duckdb.config.schema must differ from the dbt schema elt_models")
    return {"definition_id": definition, "schema": schema}


def validate_duckdb_workspace(document: dict) -> None:
    duckdb_config(document)
    try:
        config = document["Airbyte"]["config"]
        workspace = UUID(str(config["workspace_id"]))
        ready = workspace.int != 0 and all(
            isinstance(config[field], str) and config[field].strip()
            for field in ("server_url", "username", "password")
        )
    except (KeyError, TypeError, ValueError):
        ready = False
    if not ready:
        raise ValueError("install a scoped Airbyte development workspace for this DuckDB task")


def duckdb_destination_path(storage: Path, directory: str = "/local") -> str:
    if not isinstance(directory, str) or not re.fullmatch(r"/local(?:/[a-zA-Z0-9_-]+)*", directory):
        raise ValueError("DuckDB Airbyte storage must be in /local")
    return f"{directory}/{storage.name}/warehouse.duckdb"


def stage_duckdb_input(source: Path, target: Path, storage: Path, *, agent_feedback=False) -> None:
    """Give Airbyte and dbt two container paths to the same attempt file."""
    document = yaml.safe_load((source / "config.yaml").read_text())
    connector = duckdb_config(document)
    connector["destination_path"] = duckdb_destination_path(
        storage, document["duckdb"]["config"].get("airbyte_directory", "/local"),
    )
    document["duckdb"]["config"] = connector
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("*_credential.json"))
    (target / "config.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    (target / "profiles.yml").write_text(yaml.safe_dump(duckdb_profile()))
    if agent_feedback:
        shutil.copyfile(Path(__file__).with_name("development_preflight.py"), target / "check_streams.py")


def duckdb_profile() -> dict:
    return {DBT_PROFILE: {"target": "development", "outputs": {"development": {
        "type": "duckdb", "path": "/duckdb/warehouse.duckdb", "schema": "elt_models", "threads": 1,
    }}}}
