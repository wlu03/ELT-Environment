"""Render a task's public bundle into the chat messages the policy receives."""

from __future__ import annotations

import csv
import io
import json
import os
import re
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from elt_taskgen.export.eltbench import public_column_description
from tinker_cookbook.renderers import Message

from elt_environment.tasks import TaskRef

RENDERER_VERSION = "bundle-v4"
EXAMPLE_TASK_ID = "gate__five_backend_probe"
EXAMPLE_REPLY = files("elt_environment").joinpath("example_reply.txt").read_text(encoding="utf-8")
CREDENTIAL_PLACEHOLDER = "<supplied as a Terraform variable>"

_CREDENTIAL_KEY = re.compile(
    r"password|passwd|secret|access_key|api_key|token|private_key|credential", re.IGNORECASE
)
_SCALAR_LINE = re.compile(r"^(?P<indent>[ \t]*)(?P<key>[A-Za-z0-9_-]+):[ \t]+(?P<value>\S.*?)[ \t]*$")
_EMPTY_VALUES = frozenset({"''", '""', "~", "null"})

_README = "documentation/README.md"
_EXCLUDED_NAMES = frozenset({"check_job_status.py"})
_LEADING_FILES = (_README, "config.yaml", "data_model.yaml")
_BUNDLE_FILES = frozenset({_README, "config.yaml", "data_model.yaml", "elt/main.tf"})
_MIN_SCANNED_CREDENTIAL_LENGTH = 5
_SCHEMA_HEADER = ["column_name", "column_description"]
_SOURCE_TABLE_HEADING = re.compile(r"### (?P<table>\S+)  \(source backend: [^)]+\)")
_COLUMN_LINE = re.compile(
    r"- `(?P<name>[^`]+)`: \S+ (?P<null>NULL|NOT NULL) —"
    r"(?: (?P<description>.*?))?(?: one of: (?P<values>.*)\.)?"
)

SYSTEM_PROMPT = """You write the files for one ELT task. The user message contains the task bundle: the specification in documentation/README.md, whose Source tables section lists the columns of every source table, the source and destination settings in config.yaml, the data models in data_model.yaml, and the provided elt/main.tf.

Write a complete project:
- elt/main.tf: Terraform for the Airbyte provider airbytehq/airbyte version 0.6.5. Keep the provided required_providers block. Create one Airbyte source for each source section in config.yaml, the destination from config.yaml, and one airbyte_connection for each source. Each connection uses namespace_definition = "destination" and syncs exactly the tables listed for its source, each with sync_mode "full_refresh_append".
- elt/dbt_project.yml: a dbt project with config-version 2 and profile elt_taskgen.
- elt/models/sources.yml: dbt sources whose database and schema are the destination database and schema in config.yaml, listing every source table.
- elt/models/<model name>.sql: one model for each model in data_model.yaml, with the same name and exactly its columns.

The grader checks these rules. A reply that breaks one scores 0.
- main.tf contains only terraform, provider, resource and variable blocks. Do not use output, locals, data, module, count, for_each or dynamic.
- These values are supplied when the project is applied, so each one is a variable: every password, secret, access key and client credential, which config.yaml shows as '<supplied as a Terraform variable>'; the Airbyte workspace id; the destination host, role, warehouse, username and password; and every value that config.yaml leaves empty. Declare each one as an empty variable block with no default, for example variable "postgres_password" {}, and reference it as var.postgres_password. Every source and the destination use the same workspace id variable.
- Every other value comes from config.yaml and is written literally: hosts, ports, database and schema names, buckets, endpoints, regions, URLs, connection strings and definition ids. The destination database and schema are literals.
- A Snowflake destination configuration has exactly these eight keys and no others. The host variable takes the place of the account field in config.yaml, and there is no top-level password:
  configuration = {
    host             = var.destination_host
    role             = var.destination_role
    warehouse        = var.destination_warehouse
    database         = "<database from the snowflake section of config.yaml>"
    schema           = "<schema from the snowflake section of config.yaml>"
    username         = var.destination_username
    number_data_type = "NUMBER(38,9)"
    credentials      = { username_and_password = { password = var.destination_password } }
  }
- Every variable referenced in main.tf is declared in main.tf.
- The airbyte provider block sets only server_url, client_id, client_secret, username and password.
- The dbt project consists only of elt/dbt_project.yml, elt/models/sources.yml and elt/models/*.sql. Do not write profiles.yml, packages.yml, macros, seeds, snapshots, analyses, Python models or hooks. The grader supplies the dbt profile.
- Models are materialized as tables in the default schema. Do not set a custom schema.
- Each model is one SELECT query, which may use WITH clauses, and reads the raw tables through {{ source('<source name>', '<table name>') }}.
- Model SQL must run unchanged on DuckDB, Snowflake, Databricks and Redshift, so it uses only these constructs: WITH, SELECT, SELECT DISTINCT, joins, WHERE, GROUP BY, HAVING, QUALIFY, ORDER BY, LIMIT, OFFSET, UNION, INTERSECT, EXCEPT, subqueries in FROM and IN; arithmetic including %, comparisons, AND, OR, NOT, IN, LIKE, ILIKE, BETWEEN, IS NULL and ||; COUNT, COUNT(DISTINCT), SUM, AVG, MIN, MAX and aggregate FILTER (WHERE ...); window functions ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD and LAST_VALUE; CASE, CAST, TRY_CAST, COALESCE, NULLIF, ROUND, CEIL, FLOOR, POWER, CONCAT, LENGTH, LOWER, TRIM, SUBSTRING, REGEXP_LIKE, REGEXP_REPLACE, DATE_TRUNC, DATEADD, DATEDIFF, EXTRACT, TO_CHAR, and MD5 over CAST(... AS VARCHAR). Anything else is refused, including UPPER, ABS, GREATEST, LEAST, EXISTS, BOOL_AND, TO_NUMBER, CONCAT_WS and SPLIT_PART.

Reply with every file in its own block and nothing else inside a block:
<file path="elt/main.tf">
contents of the file
</file>

Below is a correct reply for a different task, gate__five_backend_probe, which has one table on each of the five source backends. It shows the accepted form of every block. It shows one of its two models. Take every value for your task from your task's config.yaml, not from this example.

"""


class BundleError(ValueError):
    """The public bundle cannot be shown to the model."""


class BundleRedactionError(BundleError):
    """A credential value in the bundle could not be replaced safely."""


class BundleLayoutError(BundleError):
    """The public tree holds a file, directory or symlink outside the documented layout."""


def _masked(node: Any) -> Any:
    if isinstance(node, Mapping):
        return {
            key: CREDENTIAL_PLACEHOLDER
            if _CREDENTIAL_KEY.search(str(key))
            and value not in (None, "")
            and not isinstance(value, (Mapping, list))
            else _masked(value)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_masked(value) for value in node]
    return node


def redact_credentials(text: str) -> str:
    """Replace the value of every credential-named YAML key with ``CREDENTIAL_PLACEHOLDER``.

    The result must parse to the original document with exactly those values
    replaced. Otherwise ``BundleRedactionError`` is raised and nothing is sent.
    """

    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        match = _SCALAR_LINE.match(body)
        if match and _CREDENTIAL_KEY.search(match["key"]) and match["value"] not in _EMPTY_VALUES:
            line = f"{match['indent']}{match['key']}: '{CREDENTIAL_PLACEHOLDER}'{line[len(body):]}"
        lines.append(line)
    redacted = "".join(lines)
    try:
        faithful = yaml.safe_load(redacted) == _masked(yaml.safe_load(text))
    except yaml.YAMLError as error:
        raise BundleRedactionError("config is not valid YAML") from error
    if not faithful:
        raise BundleRedactionError("redaction changed values other than credentials")
    return redacted


def _role(relative: str) -> str | None:
    """Return "show", "schema" or "skip" for a documented public file, and None for any other."""

    parts = relative.split("/")
    name = parts[-1]
    if relative in _BUNDLE_FILES:
        return "show"
    if len(parts) == 2 and parts[0] == "schemas" and name.endswith(".csv"):
        return "schema"
    if len(parts) == 2 and parts[0] == "documentation" and name.endswith(".md"):
        return "skip"
    if len(parts) == 3 and parts[0] == "destinations":
        if name in {"README.md", "config.yaml", f"{parts[1]}_credential.json"}:
            return "skip"
    if len(parts) == 1 and (name in _EXCLUDED_NAMES or name.endswith("_credential.json")):
        return "skip"
    return None


def _public_files(public: Path) -> list[str]:
    """Return every file under ``public`` relative to it; symlinks and special files are refused."""

    if public.is_symlink() or public.parent.is_symlink():
        raise BundleLayoutError(f"{public} is reached through a symlink")
    if not public.is_dir():
        raise BundleLayoutError(f"{public} is not a directory")
    found: list[str] = []
    for directory, subdirectories, names in os.walk(public, followlinks=False):
        for name in subdirectories:
            if (Path(directory) / name).is_symlink():
                raise BundleLayoutError(f"symlink in the public tree: {name}")
        for name in names:
            path = Path(directory) / name
            relative = path.relative_to(public).as_posix()
            if path.is_symlink() or not path.is_file():
                raise BundleLayoutError(f"not a regular file: {relative}")
            found.append(relative)
    return sorted(found)


def _secret_strings(node: Any) -> set[str]:
    values: set[str] = set()
    if isinstance(node, Mapping):
        for key, value in node.items():
            if _CREDENTIAL_KEY.search(str(key)) and isinstance(value, str):
                values.add(value)
            values |= _secret_strings(value)
    elif isinstance(node, list):
        for value in node:
            values |= _secret_strings(value)
    return values


def _credential_values(public: Path, relatives: Sequence[str]) -> set[str]:
    """Return the credential values in every public ``config.yaml`` and ``*_credential.json``.

    Values shorter than ``_MIN_SCANNED_CREDENTIAL_LENGTH`` characters are left
    out: they cannot be told apart from ordinary text, and a secret that short
    protects nothing.
    """

    values: set[str] = set()
    for relative in relatives:
        name = relative.rsplit("/", 1)[-1]
        if name != "config.yaml" and not name.endswith("_credential.json"):
            continue
        text = (public / relative).read_text(encoding="utf-8")
        try:
            document = yaml.safe_load(text) if name == "config.yaml" else json.loads(text)
        except (yaml.YAMLError, ValueError) as error:
            raise BundleRedactionError(f"{relative} could not be parsed") from error
        values |= _secret_strings(document)
    return {value for value in values if len(value) >= _MIN_SCANNED_CREDENTIAL_LENGTH}


def _column_spec(match: re.Match[str]) -> SimpleNamespace:
    return SimpleNamespace(
        description=match["description"] or "",
        enum_values=match["values"].split(", ") if match["values"] else [],
        nullable=match["null"] == "NULL",
    )


def _readme_rows(readme: str) -> dict[str, list[list[str]]]:
    """Return the schema CSV rows taskgen writes for each table in the README's Source tables."""

    tables: dict[str, list[list[str]]] = {}
    rows: list[list[str]] | None = None
    for line in readme.splitlines():
        heading = _SOURCE_TABLE_HEADING.fullmatch(line)
        if heading:
            rows = tables.setdefault(heading["table"], [])
        elif line.startswith("#"):
            rows = None
        elif rows is not None and (column := _COLUMN_LINE.fullmatch(line)):
            rows.append([column["name"], public_column_description(_column_spec(column))])
    return tables


def _listed_in_readme(
    relative: str, text: str, readme_rows: Mapping[str, list[list[str]]]
) -> bool:
    """Return whether ``relative`` is a schema CSV whose rows equal the README's rows for it."""

    folder, _, name = relative.partition("/")
    if folder != "schemas" or "/" in name or not name.endswith(".csv"):
        return False
    rows = [row for row in csv.reader(io.StringIO(text)) if row]
    if not rows or rows[0] != _SCHEMA_HEADER:
        return False
    return rows[1:] == readme_rows.get(name.removesuffix(".csv"))


def bundle_files(task: TaskRef) -> list[tuple[str, str]]:
    """Return the public files shown to the policy as ``(path, text)`` in display order.

    The public tree must hold only the documented layout, with no symlinks;
    anything else raises ``BundleLayoutError``. Shown are
    ``documentation/README.md``, ``config.yaml``, ``data_model.yaml``,
    ``elt/main.tf`` and any ``schemas/<table>.csv`` whose rows differ from the
    rows taskgen writes from the README's Source tables entry for that table.
    The other documentation files are the same for every task, and the example
    reply in the system prompt shows the accepted form of every block. Other
    destinations, credential templates and the job-status script do not affect
    the artifact. Credential values in every YAML file are replaced by
    ``CREDENTIAL_PLACEHOLDER``, and no credential value from any public config
    or credential file may remain in a shown file; otherwise
    ``BundleRedactionError`` is raised and nothing is sent.
    """

    public = task.public_dir
    relatives = _public_files(public)
    roles = {relative: _role(relative) for relative in relatives}
    unknown = [relative for relative, role in roles.items() if role is None]
    if unknown:
        raise BundleLayoutError(f"undocumented public files: {', '.join(unknown)}")
    selected: list[tuple[str, str]] = []
    for relative, role in roles.items():
        if role == "skip":
            continue
        text = (public / relative).read_text(encoding="utf-8")
        if relative.endswith(".yaml"):
            text = redact_credentials(text)
        selected.append((relative, text))
    readme_rows = _readme_rows(dict(selected).get(_README, ""))
    selected = [item for item in selected if not _listed_in_readme(*item, readme_rows)]
    secrets = _credential_values(public, relatives)
    for relative, text in selected:
        if any(value in text for value in secrets):
            raise BundleRedactionError(f"{relative} contains a credential value")
    rank = {name: index for index, name in enumerate(_LEADING_FILES)}
    selected.sort(key=lambda item: (rank.get(item[0], len(rank)), item[0]))
    return selected


def build_messages(task: TaskRef) -> list[Message]:
    """Return the system instructions and the user message holding the bundle."""

    body = "\n\n".join(
        f"----- BEGIN FILE {relative} -----\n{text}\n----- END FILE {relative} -----"
        for relative, text in bundle_files(task)
    )
    user = (
        f"Task bundle for {task.task_id}. The destination is {task.destination}.\n\n"
        f"{body}\n\n"
        'Write the files now, each in a <file path="elt/..."> block.'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT + EXAMPLE_REPLY},
        {"role": "user", "content": user},
    ]
