from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import hcl2
import yaml

from elt_environment.artifact import format_artifact, parse_artifact
from elt_environment.prompt import (
    CREDENTIAL_PLACEHOLDER,
    EXAMPLE_REPLY,
    EXAMPLE_TASK_ID,
    BundleLayoutError,
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

### products  (source backend: s3)
- `id`: bigint NOT NULL — Product id.
- `kind`: text NOT NULL — Kind. one of: a, b.

### Relationships

- `sku`: text NULL — Listed under another heading, so not a column of items.
"""
SCHEMA_HEADER = "column_name,column_description\n"
BUNDLE = {
    "documentation/README.md": README_TABLES,
    "documentation/source_postgres.md": "reference",
    "documentation/destination_snowflake.md": "reference",
    "config.yaml": (
        "postgres:\n  config:\n    password: pg-secret-value\n"
        "snowflake:\n  config:\n    database: d\n"
    ),
    "snowflake_credential.json": '{"account": "", "password": "wh-secret-value", "user": ""}',
    "check_job_status.py": "print()\n",
    "destinations/snowflake/README.md": "snowflake\n",
    "destinations/snowflake/config.yaml": (
        "snowflake:\n  config:\n    password: dest-secret-value\n"
    ),
    "destinations/snowflake/snowflake_credential.json": "{}\n",
    "data_model.yaml": "models: []\n",
    "elt/main.tf": "terraform {}\n",
    "schemas/orders.csv": SCHEMA_HEADER + "id,Order id.\nnote,Free text. may be NULL.\n",
    "schemas/items.csv": SCHEMA_HEADER + "id,Item id.\nsku,Stock unit.\n",
    "schemas/customers.csv": SCHEMA_HEADER + "id,Customer id.\n",
    "schemas/products.csv": (
        SCHEMA_HEADER + "id,Product identifier.\nkind,Kind. always exactly one of 'a', 'b'.\n"
    ),
}


class BundleSelectionTests(unittest.TestCase):
    def _task(self, root: str, files: dict[str, str]) -> TaskRef:
        task = TaskRef(Path(root), "t", "f", "snowflake")
        for relative, content in files.items():
            (task.public_dir / relative).parent.mkdir(parents=True, exist_ok=True)
            (task.public_dir / relative).write_text(content, encoding="utf-8")
        return task

    def test_documented_files_are_shown_or_skipped_and_repeated_schemas_are_left_out(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            shown = dict(bundle_files(self._task(root, BUNDLE)))
        self.assertEqual(
            list(shown),
            [
                "documentation/README.md",
                "config.yaml",
                "data_model.yaml",
                "elt/main.tf",
                "schemas/customers.csv",
                "schemas/items.csv",
                "schemas/products.csv",
            ],
        )
        self.assertNotIn("pg-secret-value", shown["config.yaml"])

    def test_an_undocumented_file_or_a_symlink_refuses_the_bundle(self) -> None:
        cases = {
            "root file": lambda public: (public / "notes.txt").write_text("x"),
            "nested file": lambda public: (public / "elt" / "secrets.yaml").write_text("x"),
            "linked directory": lambda public: os.symlink(public.parent, public / "linked"),
            "linked file": lambda public: os.symlink(
                public / "config.yaml", public / "documentation" / "link.md"
            ),
        }
        for name, tamper in cases.items():
            with tempfile.TemporaryDirectory() as root, self.subTest(case=name):
                task = self._task(root, BUNDLE)
                tamper(task.public_dir)
                with self.assertRaises(BundleLayoutError):
                    bundle_files(task)

    def test_a_credential_value_in_any_shown_file_refuses_the_bundle(self) -> None:
        for relative, content in (
            ("documentation/README.md", README_TABLES + "\nThe password is pg-secret-value.\n"),
            ("data_model.yaml", "models: []\nnote: wh-secret-value\n"),
            ("elt/main.tf", 'terraform {}\nlocals { x = "dest-secret-value" }\n'),
        ):
            files = dict(BUNDLE, **{relative: content})
            with tempfile.TemporaryDirectory() as root, self.subTest(file=relative):
                with self.assertRaises(BundleRedactionError):
                    bundle_files(self._task(root, files))

    def test_a_credential_value_too_short_to_search_for_does_not_refuse_the_bundle(self) -> None:
        files = dict(
            BUNDLE,
            **{
                "config.yaml": "aws_s3:\n  AWS_ACCESS_KEY_ID: test\n",
                "documentation/README.md": README_TABLES + "\nA test of the bundle.\n",
            },
        )
        with tempfile.TemporaryDirectory() as root:
            shown = dict(bundle_files(self._task(root, files)))
        self.assertIn("A test of the bundle.", shown["documentation/README.md"])
        self.assertNotIn("test", shown["config.yaml"])

    def test_credential_keys_in_every_yaml_file_are_redacted(self) -> None:
        data_model = "models:\n- name: m\n  api_token: model-secret\n"
        files = dict(BUNDLE, **{"data_model.yaml": data_model})
        with tempfile.TemporaryDirectory() as root:
            shown = dict(bundle_files(self._task(root, files)))
        model = yaml.safe_load(shown["data_model.yaml"])["models"][0]
        self.assertEqual(model["api_token"], CREDENTIAL_PLACEHOLDER)


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


class DiscoverTests(unittest.TestCase):
    def _release(self, root: Path, manifest: dict) -> None:
        (root / "release_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_a_task_without_a_family_or_destination_entry_is_refused(self) -> None:
        for manifest in (
            {"tasks": {"t": "0"}, "destinations": {"t": "snowflake"}},
            {"tasks": {"t": "0"}, "families": {"t": "f"}},
        ):
            with tempfile.TemporaryDirectory() as root, self.subTest(manifest=manifest):
                self._release(Path(root), manifest)
                with self.assertRaises(ValueError):
                    discover_tasks(Path(root))

    def test_family_and_destination_come_from_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self._release(
                Path(root),
                {"tasks": {"t": "0"}, "families": {"t": "f"}, "destinations": {"t": "databricks"}},
            )
            (task,) = discover_tasks(Path(root))
        self.assertEqual((task.task_id, task.family, task.destination), ("t", "f", "databricks"))


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
