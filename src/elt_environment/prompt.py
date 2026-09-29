"""Render a task's public bundle into the chat messages the policy receives."""

from __future__ import annotations

from importlib.resources import files

from tinker_cookbook.renderers import Message

from elt_environment.tasks import TaskRef

RENDERER_VERSION = "bundle-v2"
EXAMPLE_TASK_ID = "gate__five_backend_probe"
EXAMPLE_REPLY = files("elt_environment").joinpath("example_reply.txt").read_text(encoding="utf-8")

_DESTINATION_DOCUMENTS = {
    "snowflake": "destination_snowflake.md",
    "databricks": "destination_databricks.md",
    "redshift": "destination_redshift.md",
}
_DESTINATION_ONLY_DOCUMENTS = {"databricks_authentication.md": "databricks"}
_EXCLUDED_DOCUMENTS = frozenset({"trigger_job.md"})
_EXCLUDED_NAMES = frozenset({"check_job_status.py"})
_LEADING_FILES = ("documentation/README.md", "config.yaml", "data_model.yaml")

SYSTEM_PROMPT = """You write the files for one ELT task. The user message contains the task bundle: the specification in documentation/README.md, the source and destination settings in config.yaml, the data models in data_model.yaml, the source table schemas, the provided elt/main.tf, and Airbyte and Terraform reference documents.

Write a complete project:
- elt/main.tf: Terraform for the Airbyte provider airbytehq/airbyte version 0.6.5. Keep the provided required_providers block. Create one Airbyte source for each source section in config.yaml, the destination from config.yaml, and one airbyte_connection for each source. Each connection uses namespace_definition = "destination" and syncs exactly the tables listed for its source, each with sync_mode "full_refresh_append".
- elt/dbt_project.yml: a dbt project with config-version 2 and profile elt_taskgen.
- elt/models/sources.yml: dbt sources whose database and schema are the destination database and schema in config.yaml, listing every source table.
- elt/models/<model name>.sql: one model for each model in data_model.yaml, with the same name and exactly its columns.

The grader checks these rules. A reply that breaks one scores 0.
- main.tf contains only terraform, provider, resource and variable blocks. Do not use output, locals, data, module, count, for_each or dynamic.
- These values are supplied when the project is applied, so each one is a variable: every password, secret, access key and client credential, even when config.yaml shows its value, such as the Postgres password or the S3 access keys; the Airbyte workspace id; the destination host, role, warehouse, username and password; and every value that config.yaml leaves empty. Declare each one as an empty variable block with no default, for example variable "postgres_password" {}, and reference it as var.postgres_password. Every source and the destination use the same workspace id variable.
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


def _excluded(relative: str, destination: str) -> bool:
    parts = relative.split("/")
    name = parts[-1]
    if parts[0] == "destinations" or name in _EXCLUDED_NAMES or name.endswith("_credential.json"):
        return True
    if parts[0] != "documentation":
        return False
    if name in _EXCLUDED_DOCUMENTS:
        return True
    if name in _DESTINATION_DOCUMENTS.values() and name != _DESTINATION_DOCUMENTS.get(destination):
        return True
    required = _DESTINATION_ONLY_DOCUMENTS.get(name)
    return required is not None and required != destination


def bundle_files(task: TaskRef) -> list[tuple[str, str]]:
    """Return the public files shown to the policy as ``(path, text)`` in display order.

    Only regular files under ``public/<task_id>/`` are read. Other
    destinations, credential templates, the job-status script and the
    sync-trigger guide are left out because they do not affect the artifact.
    """

    public = task.public_dir
    selected: list[tuple[str, str]] = []
    for path in sorted(public.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(public).as_posix()
        if _excluded(relative, task.destination):
            continue
        selected.append((relative, path.read_text(encoding="utf-8")))
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
