"""Render a task's public bundle into the chat messages the policy receives."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

import yaml
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
_SCHEMA_HEADER = ["column_name", "column_description"]
_SOURCE_TABLE_HEADING = re.compile(r"### (?P<table>\S+)  \(source backend: [^)]+\)")
_COLUMN_LINE = re.compile(r"- `(?P<column>[^`]+)`: ")

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


class BundleRedactionError(ValueError):
    """A credential value in the bundle could not be replaced safely."""


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


def _excluded(relative: str) -> bool:
    parts = relative.split("/")
    name = parts[-1]
    if parts[0] == "destinations" or name in _EXCLUDED_NAMES or name.endswith("_credential.json"):
        return True
    return parts[0] == "documentation" and relative != _README


def _readme_columns(readme: str) -> dict[str, list[str]]:
    """Return the column names that the README's Source tables section lists for each table."""

    tables: dict[str, list[str]] = {}
    columns: list[str] | None = None
    for line in readme.splitlines():
        heading = _SOURCE_TABLE_HEADING.fullmatch(line)
        if heading:
            columns = tables.setdefault(heading["table"], [])
        elif line.startswith("#"):
            columns = None
        elif columns is not None and (column := _COLUMN_LINE.match(line)):
            columns.append(column["column"])
    return tables


def _listed_in_readme(relative: str, text: str, readme_columns: Mapping[str, list[str]]) -> bool:
    """Return whether ``relative`` is a schema CSV whose columns the README lists in order."""

    folder, _, name = relative.partition("/")
    if folder != "schemas" or "/" in name or not name.endswith(".csv"):
        return False
    rows = [row for row in csv.reader(io.StringIO(text)) if row]
    if not rows or rows[0] != _SCHEMA_HEADER:
        return False
    return [row[0] for row in rows[1:]] == readme_columns.get(name.removesuffix(".csv"))


def bundle_files(task: TaskRef) -> list[tuple[str, str]]:
    """Return the public files shown to the policy as ``(path, text)`` in display order.

    Only regular files under ``public/<task_id>/`` are read. From
    ``documentation/`` only the README is shown: the Airbyte and Terraform
    reference documents are the same for every task, and the example reply in
    the system prompt shows the accepted form of every block. A
    ``schemas/<table>.csv`` file is left out when the README's Source tables
    section lists the same columns, because that section gives the same
    descriptions with each column's type and nullability. Other destinations,
    credential templates and the job-status script are left out because they do
    not affect the artifact. Credential values in ``config.yaml`` are replaced
    by ``CREDENTIAL_PLACEHOLDER`` before the text leaves this process.
    """

    public = task.public_dir
    selected: list[tuple[str, str]] = []
    for path in sorted(public.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(public).as_posix()
        if _excluded(relative):
            continue
        text = path.read_text(encoding="utf-8")
        if path.name == "config.yaml":
            text = redact_credentials(text)
        selected.append((relative, text))
    readme_columns = _readme_columns(dict(selected).get(_README, ""))
    selected = [item for item in selected if not _listed_in_readme(*item, readme_columns)]
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
