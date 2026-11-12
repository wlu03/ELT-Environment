"""Snowflake reset, the committed configs, and the evaluator must agree.

The reset helper emits unquoted upper-cased identifiers because Airbyte and
evaluation/db_connectors.py address the configured names without quotes.
These tests prove all three renderings resolve to the same object for the
real committed corpus, without any cloud credentials, docker, or network.

Run from the ELT-Bench repository root:

    python3 -m unittest discover -s agents/tests -v

Requires PyYAML. evaluation/db_connectors.py imports pandas at module level;
a machine without pandas gets an empty placeholder module for the import only
(the identifier helpers under test never touch it).
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

AGENTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENTS_DIR.parent
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

import yaml  # noqa: E402

from common import snowflake_identifier, warehouse  # noqa: E402


def load_module(path: Path, name: str):
    """Load a non-package module (e.g. evaluation/db_connectors.py) by path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_db_connectors():
    inserted_pandas_stub = False
    if "pandas" not in sys.modules:
        try:
            import pandas  # noqa: F401
        except ImportError:
            sys.modules["pandas"] = types.ModuleType("pandas")
            inserted_pandas_stub = True
    try:
        return load_module(
            REPO_ROOT / "evaluation" / "db_connectors.py",
            "elt_bench_evaluation_db_connectors",
        )
    finally:
        if inserted_pandas_stub:
            del sys.modules["pandas"]


class SnowflakeIdentifierNormalizationTest(unittest.TestCase):
    def test_helper_matches_evaluator_for_all_committed_configs(self):
        db_connectors = load_db_connectors()
        # No credentials needed: the connection is lazy (BaseConnector.conn).
        connector = db_connectors.SnowflakeConnector({})
        config_paths = sorted(
            (REPO_ROOT / "elt-bench" / "snowflake").glob("*/config.yaml")
        )
        self.assertTrue(config_paths, "no committed snowflake configs found")
        for config_path in config_paths:
            with self.subTest(task=config_path.parent.name):
                # A parse failure here fails the subTest: this doubles as a
                # strict-parse gate for every committed config.yaml.
                with config_path.open(encoding="utf-8") as handle:
                    config = yaml.safe_load(handle)
                destination_config = config["snowflake"]["config"]
                for key in ("database", "schema"):
                    name = destination_config[key]
                    self.assertEqual(
                        connector._norm(connector._quote(name)),
                        snowflake_identifier(name),
                        f"{key} {name!r}: the unquoted name the evaluator "
                        "emits does not fold to the object the reset creates",
                    )

    def test_config_namespace_matches_task_identity_on_every_destination(self):
        """The destination namespace each config declares must be the task's
        own identity, because reset targets the config value while stage-1 and
        stage-2 evaluation address the evaluation/table.json task name. A
        mismatch means reset and evaluation touch different objects (the
        works_cycles defect: config said work_cycles, evaluation reads
        works_cycles)."""
        import json

        table_keys = set(
            json.loads(
                (REPO_ROOT / "evaluation" / "table.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        namespace_field = {"snowflake": "database", "databricks": "schema",
                          "redshift": "schema"}
        for destination, field in namespace_field.items():
            config_paths = sorted(
                (REPO_ROOT / "elt-bench" / destination).glob("*/config.yaml")
            )
            self.assertTrue(
                config_paths, f"no committed {destination} configs found"
            )
            for config_path in config_paths:
                task = config_path.parent.name
                with self.subTest(destination=destination, task=task):
                    self.assertIn(
                        task, table_keys,
                        "task directory has no evaluation/table.json entry",
                    )
                    with config_path.open(encoding="utf-8") as handle:
                        config = yaml.safe_load(handle)
                    declared = config[destination]["config"][field]
                    if destination == "snowflake":
                        self.assertEqual(
                            snowflake_identifier(declared),
                            snowflake_identifier(task),
                            f"{field} {declared!r} does not resolve to the "
                            f"task identity {task!r} the evaluator addresses",
                        )
                        self.assertEqual(
                            config[destination]["config"]["schema"],
                            "AIRBYTE_SCHEMA",
                        )
                    else:
                        self.assertEqual(
                            str(declared).lower(), task.lower(),
                            f"{field} {declared!r} does not match the task "
                            f"identity {task!r} the evaluator addresses",
                        )

    def test_reset_sql_is_unquoted_uppercase(self):
        statements: list[str] = []

        class RecordingCursor:
            def execute(self, sql):
                statements.append(sql)

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        class RecordingConnection:
            def cursor(self):
                return RecordingCursor()

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        fake_connector = types.ModuleType("snowflake.connector")
        fake_connector.connect = lambda **kwargs: RecordingConnection()
        fake_package = types.ModuleType("snowflake")
        fake_package.connector = fake_connector
        with mock.patch.dict(
            sys.modules,
            {"snowflake": fake_package, "snowflake.connector": fake_connector},
        ):
            namespace = warehouse._prepare_snowflake(
                {
                    "account": "a",
                    "password": "p",
                    "username": "u",
                    "database": "asana",
                    "schema": "AIRBYTE_SCHEMA",
                    "role": "AIRBYTE_ROLE",
                }
            )
        self.assertEqual(
            statements,
            [
                "CREATE DATABASE IF NOT EXISTS ASANA",
                "DROP SCHEMA IF EXISTS ASANA.AIRBYTE_SCHEMA CASCADE",
                "CREATE SCHEMA ASANA.AIRBYTE_SCHEMA",
                "GRANT OWNERSHIP ON SCHEMA ASANA.AIRBYTE_SCHEMA "
                "TO ROLE AIRBYTE_ROLE",
            ],
        )
        self.assertEqual(namespace, "ASANA.AIRBYTE_SCHEMA")

    def test_unrepresentable_identifier_rejected(self):
        for bad in ("my-db", "1db", 'we"ird', "", "pay$roll"):
            with self.subTest(identifier=bad):
                with self.assertRaises(ValueError):
                    snowflake_identifier(bad)


if __name__ == "__main__":
    unittest.main()
