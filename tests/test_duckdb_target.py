import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from task_fixtures import write_bundle

from elt_environment.duckdb_grader import (
    invalid_s3_stream, missing_required_streams, stage_submission,
)
from elt_environment.duckdb_target import stage_duckdb_input
from elt_environment.runner import (
    AgentConfig, AgentSession, SubmissionViolation, agent_image, public_snapshot,
    submission_artifact, validate_input,
)
from elt_environment.tasks import discover_tasks
from elt_taskgen.runtime.airbyte import AirbyteSyncFailure
from elt_taskgen.runtime.source_environment import SourceEnvironment


CONNECTOR = "457da65e-d465-4e75-8afd-2ad38ed7ed0e"
WORKSPACE = "25493451-6d20-4747-965b-d1409299bdea"


def duckdb_task(bundle: Path) -> Path:
    public = write_bundle(bundle)
    (public / "config.yaml").write_text(yaml.safe_dump({
        "duckdb": {"config": {"definition_id": CONNECTOR, "schema": "raw"}},
        "Airbyte": {"config": {
            "workspace_id": WORKSPACE, "server_url": "http://gateway/api/public/v1/",
            "username": "scoped", "password": "example-dev-password",
        }},
    }))
    return public


class DuckDBTargetTests(unittest.TestCase):
    def test_discover_and_validate_swe_task(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            duckdb_task(root)
            task, = discover_tasks(root)
            self.assertEqual(task.destination, "duckdb")
            config = AgentConfig(agent="swe", inputs_root=str(root / "public"), duckdb_local_root=str(root))
            validate_input(config, task)
            self.assertEqual(agent_image(config, task.destination), "elt-swe-duckdb")
            with self.assertRaisesRegex(ValueError, "duckdb_local_root"):
                validate_input(AgentConfig(agent="swe", inputs_root=str(root / "public")), task)
            with self.assertRaisesRegex(ValueError, "agent=swe"):
                validate_input(AgentConfig(agent="codex", inputs_root=str(root / "public")), task)
            settings = task.public_dir / "config.yaml"
            settings.write_text(settings.read_text().replace(WORKSPACE, "development-template"))
            with self.assertRaisesRegex(ValueError, "scoped Airbyte development workspace"):
                validate_input(config, task)

    def test_snapshot_preparation_gives_each_attempt_a_distinct_airbyte_path(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            source = duckdb_task(root)
            original = (source / "config.yaml").read_bytes()
            (source / "snowflake_credential.json").write_text("{}")
            for name in ("attempt-a", "attempt-b"):
                target = root / "staged" / name
                storage = root / "local" / name
                stage_duckdb_input(source, target, storage)
                config = yaml.safe_load((target / "config.yaml").read_text())
                self.assertEqual(config["duckdb"]["config"], {
                    "definition_id": CONNECTOR,
                    "destination_path": f"/local/{name}/warehouse.duckdb",
                    "schema": "raw",
                })
                self.assertEqual(config["Airbyte"]["config"]["workspace_id"], WORKSPACE)
                profile = yaml.safe_load((target / "profiles.yml").read_text())
                self.assertEqual(profile["elt_environment"]["outputs"]["development"], {
                    "type": "duckdb", "path": "/duckdb/warehouse.duckdb", "schema": "elt_models", "threads": 1,
                })
                self.assertFalse((target / "snowflake_credential.json").exists())
            self.assertEqual((source / "config.yaml").read_bytes(), original)

    def test_destination_does_not_mix_warehouse_configs(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            config = duckdb_task(root) / "config.yaml"
            with config.open("a") as output:
                output.write("snowflake: {config: {database: dev}}\n")
            with self.assertRaisesRegex(ValueError, "single destination"):
                discover_tasks(root)

    def test_hidden_project_uses_trusted_profile_and_rejects_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            project = stage_submission({"elt/dbt_project.yml": "profile: elt_environment"}, root)
            self.assertEqual(project, root / "elt")
            self.assertTrue((root / "profiles.yml").is_file())
            for path in ("elt/../outside.sql", "elt/profiles.yml", "other/dbt_project.yml"):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "submission path"):
                    stage_submission({path: ""}, root)

    def test_real_swe_submit_does_not_invalidate_duckdb_pipeline(self):
        submit = Path(__file__).resolve().parents[1] / "agents/SWE-agent/tools/submit/bin/submit"
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "config.yaml").write_text("public: true\n")
            (root / "elt").mkdir()
            baseline = public_snapshot(root)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "elt/main.tf").write_text("terraform {}\n")
            (root / ".user.yml").write_text("id: example\n")
            (root / "logs").mkdir()
            (root / "logs/dbt.log").write_text("dbt run\n")
            subprocess.run(["bash", str(submit)], cwd=root, check=True, capture_output=True,
                           env={**os.environ, "ROOT": str(root)})
            self.assertTrue((root / "model.patch").is_file())
            self.assertEqual(submission_artifact(root, baseline, swe=True, duckdb=True), {
                "elt/main.tf": "terraform {}\n",
            })
            unexpected = root / "logs/unexpected.sql"
            unexpected.write_text("select 1\n")
            with self.assertRaisesRegex(SubmissionViolation, "public input changed"):
                submission_artifact(root, baseline, swe=True, duckdb=True)
            unexpected.unlink()
            (root / "config.yaml").write_text("public: false\n")
            with self.assertRaisesRegex(SubmissionViolation, "public input changed"):
                submission_artifact(root, baseline, swe=True, duckdb=True)

    def test_sibling_dbt_project_is_staged_under_elt(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch) / "workspace"
            root.mkdir()
            (root / "profiles.yml").write_text("provided\n")
            baseline = public_snapshot(root)
            project = root / "my_models"
            project.mkdir()
            (project / "dbt_project.yml").write_text("profile: elt_environment\n")
            (project / "model.sql").write_text("select 1\n")
            (project / "do_not_collect.tf").write_text("terraform {}\n")
            (root / "stg_customer.sql").write_text("select 42\n")
            artifact = submission_artifact(root, baseline, swe=True, duckdb=True)
            self.assertEqual(set(artifact), {"elt/my_models/dbt_project.yml", "elt/my_models/model.sql"})
            staging = root.parent / "hidden"
            self.assertEqual(stage_submission(artifact, staging), staging / "elt/my_models")
            (project / "private.sql").symlink_to(root / "profiles.yml")
            with self.assertRaisesRegex(SubmissionViolation, "public symlink"):
                submission_artifact(root, baseline, swe=True, duckdb=True)

    def test_provided_root_sql_stays_protected(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            public_sql = root / "source.sql"
            public_sql.write_text("select 1\n")
            baseline = public_snapshot(root)
            public_sql.write_text("select 2\n")
            with self.assertRaisesRegex(SubmissionViolation, "public input changed"):
                submission_artifact(root, baseline, swe=True, duckdb=True)

    def test_missing_streams_require_complete_airbyte_metadata(self):
        required = {"public_users", "public_events"}
        configured = {
            "connectionId": "one", "prefix": "public_",
            "configurations": {"streams": [{"name": "Users"}]},
        }
        empty = {"connectionId": "two", "configurations": {"streams": []}}
        unknown = {"connectionId": "two"}
        self.assertEqual(missing_required_streams(required, [configured, empty]), {"public_events"})
        self.assertIsNone(missing_required_streams(required, [configured, unknown]))
        self.assertEqual(missing_required_streams({"public_users"}, [configured, empty]), set())

    def test_s3_replay_error_requires_a_selected_missing_literal(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "parts").mkdir()
            (root / "parts/part-00000.jsonl").write_text("{}\n")
            (root / "mirror.parquet").touch()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"tables": {
                "users": {"backend": "s3", "bucket": "fixture", "rendered_dir": "parts",
                          "object_key": "users.jsonl"},
                "mirror": {"backend": "files", "rendered_file": "mirror.parquet"},
            }}))
            hidden = SourceEnvironment(
                root, "task", "primary", root, root, manifest, root / "compose.yaml", "task", "net",
            )
            error = AirbyteSyncFailure("connection", 100, "failed")
            failure = {
                "failureOrigin": "source", "failureType": "config_error",
                "internalMessage": "No files were identified in the stream",
            }
            connection = {"sourceId": "source", "configurations": {"streams": [{"name": "users"}]}}
            for glob, expected in (
                ("wrong.jsonl", True), ("users.jsonl", False), ("mirror.parquet", False),
                ("**/wrong.jsonl", False),
            ):
                with self.subTest(glob=glob):
                    client = Mock()
                    client.request.side_effect = [
                        {"attempts": [{"attempt": {"failureSummary": {"failures": [failure]}}}]},
                        connection,
                        {"configuration": {"bucket": "fixture",
                                           "streams": [{"name": "users", "globs": [glob]}]}},
                    ]
                    self.assertEqual(invalid_s3_stream(client, error, hidden), expected)


class DuckDBSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_duckdb_session_passes_attempt_storage_to_worker_and_cleans_it(self):
        class FinishedProcess:
            returncode = 0

            async def wait(self):
                return self.returncode

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            duckdb_task(root)
            task, = discover_tasks(root)
            storage = root / "local"
            storage.mkdir()
            config = AgentConfig(
                agent="swe", inputs_root=str(root / "public"), duckdb_local_root=str(storage),
                attempts_root=str(root / "attempts"),
            )
            session = AgentSession(task, None, config)
            with patch("elt_environment.runner.asyncio.create_subprocess_exec", return_value=FinishedProcess()):
                await session.start()
                try:
                    spec = json.loads((session.root / "worker.json").read_text())
                    shared = Path(spec["duckdb_storage"])
                    self.assertEqual(shared.parent, storage.resolve())
                    self.assertTrue(shared.is_dir())
                    self.assertEqual(stat.S_IMODE(shared.stat().st_mode), 0o733)
                    self.assertEqual(spec["image"], "elt-swe-duckdb")
                    staged = Path(spec["inputs_root"]) / task.task_id
                    config = yaml.safe_load((staged / "config.yaml").read_text())
                    self.assertEqual(config["duckdb"]["config"]["destination_path"],
                                     f"/local/{shared.name}/warehouse.duckdb")
                    self.assertIn("profiles.yml", session.snapshot)
                    state = session.workspace / "elt" / "terraform.tfstate"
                    state.parent.mkdir(parents=True)
                    connection = "95977bc0-1d4d-4505-ad88-3ec3823c425d"
                    state.write_text(json.dumps({"resources": [{
                        "type": "airbyte_connection", "instances": [{
                            "attributes": {"connection_id": connection},
                        }],
                    }]}))
                    self.assertEqual(session.duckdb_replay_receipt(), {
                        "storage": str(shared), "connection_ids": [connection],
                    })
                finally:
                    await session.close()
                await session.close()
            self.assertEqual(list(storage.iterdir()), [])
