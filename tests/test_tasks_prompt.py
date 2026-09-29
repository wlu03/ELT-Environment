from __future__ import annotations

import csv
import io
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import hcl2
import yaml
from elt_taskgen.export.eltbench import public_column_description

from elt_environment.artifact import format_artifact, parse_artifact
from elt_environment.prompt import (
    CREDENTIAL_PLACEHOLDER,
    EXAMPLE_REPLY,
    EXAMPLE_TASK_ID,
    BundleRedactionError,
    build_messages,
    bundle_files,
    redact_credentials,
)
from elt_environment.tasks import TaskRef, discover_tasks, split_by_family

CONFIG = """Airbyte:
  config:
    password: ''
    server_url: http://airbyte:80/api/public/v1/
aws_s3:
  AWS_ACCESS_KEY_ID: key-id-value
  AWS_SECRET_ACCESS_KEY: "secret value"
  AWS_ENDPOINT_URL: http://localstack:4566
postgres:
  config:
    host: elt-postgres
    password: pg-password-value  # set by the harness
    user: postgres
"""


class RedactionTests(unittest.TestCase):
    def test_credential_values_are_replaced_and_everything_else_is_kept(self) -> None:
        redacted = redact_credentials(CONFIG)
        for secret in ("key-id-value", "secret value", "pg-password-value"):
            self.assertNotIn(secret, redacted)
        document = yaml.safe_load(redacted)
        self.assertEqual(document["aws_s3"]["AWS_ACCESS_KEY_ID"], CREDENTIAL_PLACEHOLDER)
        self.assertEqual(document["aws_s3"]["AWS_SECRET_ACCESS_KEY"], CREDENTIAL_PLACEHOLDER)
        self.assertEqual(document["postgres"]["config"]["password"], CREDENTIAL_PLACEHOLDER)
        self.assertEqual(document["Airbyte"]["config"]["password"], "")
        self.assertEqual(document["postgres"]["config"]["user"], "postgres")
        self.assertEqual(document["aws_s3"]["AWS_ENDPOINT_URL"], "http://localstack:4566")

    def test_a_credential_the_line_rule_cannot_replace_refuses_the_bundle(self) -> None:
        for text in (
            "postgres:\n  password: |\n    multi\n    line\n",
            "sources:\n  - password: listed-value\n",
            "postgres: {password: inline-value}\n",
        ):
            with self.subTest(text=text), self.assertRaises(BundleRedactionError):
                redact_credentials(text)


README_TABLES = """## Source tables

### orders  (source backend: postgres)
Source table orders.

- `id`: bigint NOT NULL — Order id.
- `note`: text NULL — Free text.
- primary key: id

### items  (source backend: files)
- `id`: bigint NOT NULL — Item id.

### Relationships

- `sku`: text NULL — Listed under another heading, so not a column of items.
"""
SCHEMA_HEADER = "column_name,column_description\n"
README_COLUMN = re.compile(
    r"- `(?P<name>[^`]+)`: \S+ (?P<null>NULL|NOT NULL) —"
    r"(?: (?P<description>.*?))?(?: one of: (?P<values>.*)\.)?"
)


class BundleSelectionTests(unittest.TestCase):
    def test_readme_is_the_only_document_and_repeated_schemas_are_left_out(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            task = TaskRef(Path(root), "t", "f", "snowflake")
            files = {
                "documentation/README.md": README_TABLES,
                "documentation/source_postgres.md": "reference",
                "documentation/destination_snowflake.md": "reference",
                "config.yaml": "snowflake:\n  config:\n    database: d\n",
                "schemas/orders.csv": (
                    SCHEMA_HEADER + "id,Order id.\nnote,Free text. may be NULL.\n"
                ),
                "schemas/items.csv": SCHEMA_HEADER + "id,Item id.\nsku,Stock unit.\n",
                "schemas/customers.csv": SCHEMA_HEADER + "id,Customer id.\n",
            }
            for relative, content in files.items():
                (task.public_dir / relative).parent.mkdir(parents=True, exist_ok=True)
                (task.public_dir / relative).write_text(content, encoding="utf-8")
            paths = [path for path, _ in bundle_files(task)]
        self.assertEqual(
            paths,
            [
                "documentation/README.md",
                "config.yaml",
                "schemas/customers.csv",
                "schemas/items.csv",
            ],
        )


def _schema_rows_from_readme(readme: str, table: str) -> list[list[str]] | None:
    """Rebuild the rows of ``schemas/<table>.csv`` from the README the way taskgen writes them."""

    section = re.search(
        rf"^### {re.escape(table)}  \(source backend: [^)]+\)\n(.*?)(?=^#|\Z)",
        readme,
        re.MULTILINE | re.DOTALL,
    )
    if section is None:
        return None
    rows = [SCHEMA_HEADER.strip().split(",")]
    for line in section[1].splitlines():
        match = README_COLUMN.fullmatch(line)
        if match:
            column = SimpleNamespace(
                description=match["description"] or "",
                enum_values=match["values"].split(", ") if match["values"] else [],
                nullable=match["null"] == "NULL",
            )
            rows.append([match["name"], public_column_description(column)])
    return rows


def _attribute_paths(main_tf: str) -> set[tuple[str, ...]]:
    """Return every provider and resource attribute path in ``main_tf``, without values."""

    paths: set[tuple[str, ...]] = set()

    def walk(node: object, prefix: tuple[str, ...]) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                paths.add((*prefix, key))
                walk(value, (*prefix, key))
        elif isinstance(node, list):
            for value in node:
                walk(value, prefix)

    document = hcl2.loads(main_tf)
    for block in document.get("resource", []):
        for kind, named in block.items():
            for body in named.values():
                walk(body, ("resource", kind))
    for block in document.get("provider", []):
        for kind, body in block.items():
            walk(body, ("provider", kind))
    return paths


TASKGEN = Path(__file__).resolve().parents[2] / "ELT-taskgen"
RELEASES = TASKGEN / "runs" / "batch50c_20260923" / "workspace" / "releases"
FIXTURE = TASKGEN / "tests" / "fixtures" / "semantic_gate" / "release"


@unittest.skipUnless(FIXTURE.is_dir(), "taskgen fixture release is not on disk")
class ExampleReplyTests(unittest.TestCase):
    def test_packaged_example_matches_the_canonical_render_of_the_fixture(self) -> None:
        from elt_taskgen.training import load_workspace_package
        from elt_taskgen.training.canonical import render_canonical_project

        canonical = render_canonical_project(
            load_workspace_package(FIXTURE, EXAMPLE_TASK_ID, verify=True)
        )
        example = parse_artifact(EXAMPLE_REPLY)
        self.assertEqual(example, {path: canonical[path] for path in example})
        self.assertEqual(
            set(example), {"main.tf", "dbt_project.yml", "models/sources.yml", "models/event_wide.sql"}
        )
        self.assertEqual(EXAMPLE_REPLY, format_artifact(example) + "\n")


def _fake(task_id: str, family: str) -> TaskRef:
    return TaskRef(Path("/nonexistent"), task_id, family, "snowflake")


class SplitTests(unittest.TestCase):
    def test_split_is_family_disjoint_and_seeded(self) -> None:
        tasks = [_fake(f"t{index}", f"f{index % 7}") for index in range(30)]
        train, evaluation = split_by_family(tasks, eval_families=2, seed=3)
        self.assertEqual(len(train) + len(evaluation), len(tasks))
        self.assertFalse({t.family for t in train} & {t.family for t in evaluation})
        self.assertEqual(len({t.family for t in evaluation}), 2)
        self.assertEqual(split_by_family(list(reversed(tasks)), 2, 3)[1], evaluation)
        self.assertNotEqual(split_by_family(tasks, 2, 4)[1], evaluation)


@unittest.skipUnless(RELEASES.is_dir(), "batch50c releases are not on disk")
class ReleaseAndPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = discover_tasks(RELEASES)
        cls.task = next(t for t in cls.tasks if t.task_id == "dlt__workable")

    def test_discovers_every_release(self) -> None:
        self.assertEqual(len(self.tasks), 50)
        self.assertEqual(self.task.family, "dlt__workable")
        self.assertEqual(self.task.destination, "snowflake")

    def test_every_bundle_shows_the_readme_config_data_model_and_starter_main_tf(self) -> None:
        for task in self.tasks:
            with self.subTest(task=task.task_id):
                self.assertEqual(
                    [path for path, _ in bundle_files(task)],
                    ["documentation/README.md", "config.yaml", "data_model.yaml", "elt/main.tf"],
                )

    def test_every_schema_file_left_out_is_rebuilt_exactly_from_the_readme(self) -> None:
        checked = 0
        for task in self.tasks:
            readme = (task.public_dir / "documentation" / "README.md").read_text(encoding="utf-8")
            for schema in sorted((task.public_dir / "schemas").glob("*.csv")):
                with self.subTest(task=task.task_id, table=schema.stem):
                    shown = list(csv.reader(io.StringIO(schema.read_text(encoding="utf-8"))))
                    self.assertEqual(_schema_rows_from_readme(readme, schema.stem), shown)
                    checked += len(shown) - 1
        self.assertEqual(checked, 8314)

    def test_example_shows_every_terraform_attribute_a_canonical_reply_uses(self) -> None:
        from elt_taskgen.training import load_workspace_package
        from elt_taskgen.training.canonical import render_canonical_project

        example = _attribute_paths(parse_artifact(EXAMPLE_REPLY)["main.tf"])
        for task in self.tasks:
            package = load_workspace_package(task.release_dir, task.task_id, verify=False)
            canonical = render_canonical_project(package)["main.tf"]
            with self.subTest(task=task.task_id):
                self.assertLessEqual(_attribute_paths(canonical), example)

    def test_prompt_is_deterministic_and_names_no_private_material(self) -> None:
        first, second = build_messages(self.task), build_messages(self.task)
        self.assertEqual(first, second)
        text = first[0]["content"] + first[1]["content"]
        for marker in ("answer_key", "private/", "/gold/", ".duckdb", "task_ir", str(RELEASES)):
            self.assertNotIn(marker, text)

    def test_no_task_sends_a_credential_value_to_the_model(self) -> None:
        checked = 0
        for task in self.tasks:
            original = yaml.safe_load((task.public_dir / "config.yaml").read_text(encoding="utf-8"))
            shown = yaml.safe_load(dict(bundle_files(task))["config.yaml"])
            fields = [("aws_s3", key) for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")]
            with self.subTest(task=task.task_id):
                for section, key in fields:
                    if (original.get(section) or {}).get(key):
                        self.assertEqual(shown[section][key], CREDENTIAL_PLACEHOLDER)
                        checked += 1
                if ((original.get("postgres") or {}).get("config") or {}).get("password"):
                    self.assertEqual(shown["postgres"]["config"]["password"], CREDENTIAL_PLACEHOLDER)
                    checked += 1
                self.assertEqual(
                    shown["snowflake"]["config"]["database"], original["snowflake"]["config"]["database"]
                )
        self.assertEqual(checked, 45 + 45 + 38)

    def test_system_prompt_carries_the_example_and_the_user_message_carries_the_task(self) -> None:
        system, user = build_messages(self.task)
        self.assertTrue(system["content"].endswith(EXAMPLE_REPLY))
        self.assertIn(EXAMPLE_TASK_ID, system["content"])
        self.assertNotIn(EXAMPLE_TASK_ID, user["content"])
        self.assertIn("dlt__workable", user["content"])
        self.assertFalse(any(task.task_id == EXAMPLE_TASK_ID for task in self.tasks))


if __name__ == "__main__":
    unittest.main()
