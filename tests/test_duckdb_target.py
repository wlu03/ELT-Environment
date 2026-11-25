import fcntl
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import yaml
from task_fixtures import write_bundle

from elt_environment.airbyte_runtime import ReplayAirbyteClient, airbyte_lease
from elt_environment.duckdb_grader import (
    DbtReplayError, invalid_s3_stream, missing_required_streams, run_dbt, settle_jobs,
    stage_submission,
)
from elt_environment.duckdb_target import stage_duckdb_input
from elt_environment.elt_bench_grader import (
    prepare_dbt_packages, replay_connections, replay_rows, wait_for_airbyte_slot,
)
from elt_environment.grading import GraderConfig, grade_artifact
from elt_environment.outcomes import DiscardGroup
from elt_environment.runner import (
    AgentConfig, AgentSession, SubmissionViolation, agent_image, public_snapshot,
    submission_artifact, validate_input,
)
from elt_environment.tasks import discover_tasks
from elt_taskgen.runtime.airbyte import (
    AirbyteError, AirbyteSyncConflict, AirbyteSyncFailure, AirbyteSyncReceipt,
)
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


class AirbyteRetryTests(unittest.TestCase):
    def test_generated_and_reference_replays_share_the_same_lock(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            reference = root / "elt-bench-100"
            reference.mkdir()
            name = "elt-bench-grading.lock"
            with patch("elt_environment.airbyte_runtime.LOCAL_AIRBYTE_ROOT", root):
                with airbyte_lease(name, reference), (root / name).open("a") as contender:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with airbyte_lease(name, root):
                    self.assertFalse((reference / name).exists())

    def test_replays_read_after_airbyte_timeout(self):
        opener = Mock(side_effect=[TimeoutError(), BytesIO(b'{"status":"succeeded"}')])
        client = ReplayAirbyteClient("http://airbyte/api/public/v1", bearer_token="test", opener=opener)
        with patch("elt_environment.airbyte_runtime.time.sleep") as sleep:
            self.assertEqual(client.job_status(100), "succeeded")
        self.assertEqual(opener.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_post_and_authorization_errors_are_not_retried(self):
        for method, error in (
            ("POST", TimeoutError()),
            ("GET", HTTPError("http://airbyte", 403, "Forbidden", {}, None)),
        ):
            with self.subTest(method=method):
                opener = Mock(side_effect=error)
                client = ReplayAirbyteClient("http://airbyte", bearer_token="test", opener=opener)
                with self.assertRaises(AirbyteError):
                    client.request(method, "workspaces")
                opener.assert_called_once()

    def test_delete_accepts_workspace_already_removed_after_timeout(self):
        opener = Mock(side_effect=[
            TimeoutError(), HTTPError("http://airbyte", 404, "Missing", {}, None),
        ])
        client = ReplayAirbyteClient("http://airbyte", bearer_token="test", opener=opener)
        with patch("elt_environment.airbyte_runtime.time.sleep"):
            client.delete_workspace(WORKSPACE)
        self.assertEqual(opener.call_count, 2)


class DuckDBTargetTests(unittest.TestCase):
    def test_benchmark_replay_waits_for_another_sync_between_retries(self):
        client = Mock()
        client.request.side_effect = [{"data": [{"jobId": 12}]}] + [{"data": []}] * 3
        with patch("elt_environment.elt_bench_grader.time.sleep") as sleep:
            wait_for_airbyte_slot(client)
        sleep.assert_called_once_with(5)
        first_query = client.request.call_args_list[0].args[1]
        self.assertIn("status=incomplete", first_query)
        self.assertEqual(client.request.call_count, 4)

    def test_busy_airbyte_worker_does_not_discard_the_submission(self):
        client = Mock()
        client.request.return_value = {"data": [{"jobId": 12}]}
        with (
            patch("elt_environment.elt_bench_grader.time.monotonic", side_effect=[0, 120]),
            patch("elt_environment.elt_bench_grader.sys.stderr"),
        ):
            wait_for_airbyte_slot(client)
        client.request.assert_called_once()

    def test_development_job_settles_after_airbyte_retry(self):
        client = Mock()
        client.request.return_value = {"data": [{"jobId": 42, "status": "incomplete"}]}
        client.job_status.side_effect = ["incomplete", "running", "succeeded"]
        with patch("elt_environment.duckdb_grader.time.sleep"):
            settle_jobs(client, ["connection"])
        self.assertEqual([call.args for call in client.job_status.call_args_list], [(42,)] * 3)

    def test_cancel_stops_retrying_workspace_jobs_before_replay(self):
        client = Mock()
        client.request.return_value = {"data": [
            {"jobId": 41, "status": "succeeded"}, {"jobId": 42, "status": "incomplete"},
        ]}
        client.job_status.side_effect = ["incomplete", "cancelled"]
        with patch("elt_environment.duckdb_grader.time.sleep") as sleep:
            settle_jobs(client, ["connection"], cancel=True)
        client.request.assert_called_with("DELETE", "jobs/42")
        self.assertEqual(client.request.call_count, 2)
        sleep.assert_called_once_with(3)

    def test_conflicted_replay_restarts_all_connections_with_empty_storage(self):
        client = Mock()
        client.request.return_value = {"data": []}
        attempted = []
        with tempfile.TemporaryDirectory() as scratch:
            store = Path(scratch) / "attempt"
            store.mkdir()
            warehouse = store / "warehouse.duckdb"
            warehouse.write_text("development")

            def trigger(connections, **kwargs):
                connection, = connections
                attempted.append(connection)
                if connection == "first":
                    self.assertFalse(warehouse.exists())
                    warehouse.write_text("first replay")
                elif len(attempted) == 2:
                    warehouse.write_text("conflicting job")
                    raise AirbyteSyncConflict("sync already running")
                else:
                    self.assertEqual(warehouse.read_text(), "first replay")
                return AirbyteSyncReceipt({connection: len(attempted)}, {connection: "succeeded"})

            client.trigger_and_wait_receipt.side_effect = trigger
            with (
                patch("elt_environment.elt_bench_grader.settle_jobs") as settle,
                patch("elt_environment.elt_bench_grader.time.sleep"),
                patch("elt_environment.elt_bench_grader.sys.stderr"),
            ):
                replay_connections(client, ["first", "second"], ["first", "second"], store)
            self.assertEqual(settle.call_count, 2)
            self.assertEqual(attempted, ["first", "second", "first", "second"])

    def test_row_count_mismatch_after_airbyte_retry_gets_a_clean_replay(self):
        client = Mock()
        client.request.return_value = {"attempts": [{"attempt": {"status": "failed"}}]}
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            (store / "warehouse.duckdb").touch()
            with (
                patch("elt_environment.elt_bench_grader.replay_connections",
                      side_effect=[[21], [22]]) as replay,
                patch("elt_environment.elt_bench_grader.warehouse_rows",
                      side_effect=[{"stream": 150}, {"stream": 100}]),
                patch("elt_environment.elt_bench_grader.sys.stderr"),
            ):
                actual = replay_rows(client, ["connection"], ["connection"], store,
                                     "main", {"stream": 100})
        self.assertEqual(actual, {"stream": 100})
        self.assertEqual(replay.call_count, 2)
        client.request.assert_called_once_with("POST", "../../v1/jobs/get", {"id": 21})

    def test_hidden_dbt_preserves_model_failures_and_flags_runtime_failures(self):
        for returncode, event, data, status in (
            (0, None, {}, "success"),
            (1, "RunResultError", {"msg": "Catalog Error: table is missing"}, "model_error"),
            (2, "MainEncounteredError", {"exc": "Compilation Error\n  Invalid ref"}, "project_error"),
            (2, "MainEncounteredError", {"exc": "Runtime Error\n  Could not find profile named 'bad'"},
             "project_error"),
            (1, "RunResultError", {"msg": "IO Error: Could not set lock on file"}, "grader_error"),
            (1, "RunResultError", {"msg": "Out of Memory Error: failed to pin block of size 256.0 KiB"},
             "grader_error"),
            (2, "MainEncounteredError", {"exc": 'IO Error: could not read "/duckdb/warehouse.duckdb"'},
             "grader_error"),
            (137, None, {}, "grader_error"),
        ):
            with self.subTest(returncode=returncode, status=status):
                stdout = json.dumps({"info": {"level": "error", "name": event}, "data": data}) if event else ""
                result = subprocess.CompletedProcess([], returncode, stdout, "")
                with patch("elt_environment.duckdb_grader.sandbox", return_value=result), patch(
                    "elt_environment.duckdb_grader.sys.stderr"
                ) as stderr:
                    root = Path("/tmp/submission")
                    if status == "grader_error":
                        with self.assertRaises(DbtReplayError):
                            run_dbt(Path("/tmp/db"), root, root / "elt", "primary")
                    else:
                        run_dbt(Path("/tmp/db"), root, root / "elt", "primary")
                log = json.loads(stderr.write.call_args_list[0].args[0])
                self.assertEqual(log["status"], status)
                self.assertEqual(log["population"], "primary")
                self.assertEqual(log["stdout"], stdout)

    def test_hidden_dbt_timeout_makes_replay_unscorable_and_keeps_output(self):
        timeout = subprocess.TimeoutExpired(["dbt"], 240, output=b"dbt startup")
        with patch("elt_environment.duckdb_grader.sandbox", side_effect=timeout), patch(
            "elt_environment.duckdb_grader.sys.stderr"
        ) as stderr:
            root = Path("/tmp/submission")
            with self.assertRaises(DbtReplayError):
                run_dbt(Path("/tmp/db"), root, root / "elt", "stress")
        log = json.loads(stderr.write.call_args_list[0].args[0])
        self.assertEqual(log["status"], "grader_error")
        self.assertEqual(log["stdout"], "dbt startup")

    def test_hidden_dbt_diagnostics_survive_grading_without_leaving_private_requests(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            duckdb_task(root / "bundle")
            task, = discover_tasks(root / "bundle")
            response = json.dumps({"label_valid": False, "reason_code": "duckdb_dbt_replay_failed"})
            config = GraderConfig(root / "attempts", (
                sys.executable, "-c",
                "import sys; print('private dbt error', file=sys.stderr); "
                f"print({response!r})",
            ))
            self.assertEqual(grade_artifact(task, {}, config), DiscardGroup("duckdb_dbt_replay_failed"))
            log, = (config.attempts_root / "diagnostics").iterdir()
            self.assertEqual(log.read_text(), "private dbt error\n")
            self.assertEqual(stat.S_IMODE(log.stat().st_mode), 0o600)
            self.assertEqual(list(config.attempts_root.iterdir()), [log.parent])

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

    def test_stage_keeps_the_airbyte_storage_namespace(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            source = duckdb_task(root)
            config = yaml.safe_load((source / "config.yaml").read_text())
            config["duckdb"]["config"]["airbyte_directory"] = "/local/elt-bench-100"
            (source / "config.yaml").write_text(yaml.safe_dump(config))
            stage_duckdb_input(source, root / "staged", root / "attempt")
            staged = yaml.safe_load((root / "staged" / "config.yaml").read_text())
            self.assertEqual(staged["duckdb"]["config"]["destination_path"],
                             "/local/elt-bench-100/attempt/warehouse.duckdb")

    def test_feedback_checks_only_configured_and_applied_streams(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            source = duckdb_task(root)
            config_path = source / "config.yaml"
            config = yaml.safe_load(config_path.read_text())
            config.update(
                postgres={"config": {"tables": ["customers"]}},
                mongodb={"config": {"tables": ["settings"]}},
                custom_api={"config": {"tables": ["invoices"]}},
                flat_files=[{"table": "payments"}],
                aws_s3={"data": [{"table": "items"}]},
            )
            config_path.write_text(yaml.safe_dump(config))
            target = root / "staged"
            stage_duckdb_input(source, root / "baseline", root / "attempt")
            self.assertFalse((root / "baseline" / "check_streams.py").exists())
            stage_duckdb_input(source, target, root / "attempt", agent_feedback=True)
            self.assertIn("check_streams.py", public_snapshot(target))
            state = target / "elt" / "terraform.tfstate"
            state.parent.mkdir(exist_ok=True)
            state.write_text(json.dumps({"resources": [{
                "type": "airbyte_connection",
                "instances": [{"attributes": {"configurations": {"streams": [
                    {"name": "Customers"}, {"name": "invoices"}, {"name": "items"},
                ]}}}],
            }, {
                "type": "airbyte_source",
                "instances": [{"attributes": {"configurations": {"streams": [{"name": "payments"}]}}}],
            }]}))
            result = subprocess.run([sys.executable, str(target / "check_streams.py")],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stderr.strip(), "Missing streams in applied connections: payments, settings")

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

    def test_declared_local_package_does_not_count_as_a_second_project(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            files = {
                "elt/dbt_project.yml": "name: pipeline\nprofile: elt_environment\n",
                "elt/packages.yml": "packages:\n  - local: local_packages/dbt_utils\n",
                "elt/local_packages/dbt_utils/dbt_project.yml": "name: dbt_utils\n",
            }
            project = stage_submission(files, root)
            self.assertEqual(project, root / "elt")
            self.assertTrue(prepare_dbt_packages(root, project, None))
            self.assertEqual(yaml.safe_load((project / "packages.yml").read_text()), {
                "packages": [{"local": "local_packages/dbt_utils"}],
            })
            files["elt/independent/dbt_project.yml"] = "name: independent\n"
            self.assertIsNone(stage_submission(files, root / "ambiguous"))

    def test_offline_dbt_utils_is_restored_only_when_declared(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            package = root / "package"
            (package / "macros").mkdir(parents=True)
            (package / "dbt_project.yml").write_text("name: dbt_utils\nversion: '1.3.0'\n")
            (package / "LICENSE").write_text("test license\n")
            staged = root / "submission"
            project = stage_submission({"elt/dbt_project.yml": "name: pipeline\n"}, staged)
            self.assertFalse(prepare_dbt_packages(staged, project, package))
            self.assertFalse((staged / "vendor").exists())
            (project / "dependencies.yml").write_text(
                "packages:\n  - local: /workspace/vendor/dbt_utils\n",
            )
            self.assertTrue(prepare_dbt_packages(staged, project, package))
            self.assertTrue((staged / "vendor/dbt_utils/dbt_project.yml").is_file())
            self.assertEqual(yaml.safe_load((project / "dependencies.yml").read_text()), {
                "packages": [{"local": "../vendor/dbt_utils"}],
            })

    def test_copy_of_declared_offline_package_does_not_hide_the_pipeline(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            project = stage_submission({
                "elt/dbt_project.yml": "name: pipeline\n",
                "elt/packages.yml": "packages:\n  - local: /workspace/vendor/dbt_utils\n",
                "elt/packages/dbt_utils/dbt_project.yml": "name: dbt_utils\n",
            }, root)
            self.assertEqual(project, root / "elt")

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

    def test_s3_wrong_bucket_requires_agreement_between_public_and_seeded_buckets(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            config = root / "public" / "task" / "config.yaml"
            config.parent.mkdir(parents=True)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"tables": {
                "users": {"backend": "s3", "bucket": "provided-bucket"},
            }}))
            hidden = SourceEnvironment(
                root, "task", "primary", root, root, manifest, root / "compose.yaml", "task", "net",
            )
            error = AirbyteSyncFailure("connection", 100, "failed")
            for public, applied, reported, origin, expected in (
                ("provided-bucket", "wrong-bucket", "wrong-bucket", "source", True),
                ("provided-bucket", "wrong_bucket", "wrong_bucket", "source", True),
                ("provided-bucket", "provided-bucket", "provided-bucket", "source", False),
                ("different-bucket", "wrong-bucket", "wrong-bucket", "source", False),
                ("provided-bucket", "wrong-bucket", "different-bucket", "source", False),
                ("provided-bucket", "wrong-bucket", "wrong-bucket", "destination", False),
            ):
                with self.subTest(public=public, applied=applied, reported=reported, origin=origin):
                    config.write_text(yaml.safe_dump({
                        "aws_s3": {"data": [{"path": f"s3://{public}/users.jsonl"}]},
                    }))
                    failure = {
                        "failureOrigin": origin, "failureType": "config_error",
                        "internalMessage": f"3 streams with errors: The bucket {reported} does not exist.",
                    }
                    client = Mock()
                    client.request.side_effect = [
                        {"attempts": [{"attempt": {"failureSummary": {"failures": [failure]}}}]},
                        {"sourceId": "source"},
                        {"configuration": {"bucket": applied}},
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
