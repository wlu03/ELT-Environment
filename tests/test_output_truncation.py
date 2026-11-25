"""A capped response preserves the score for work completed in earlier turns."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from elt_environment.agent_worker import (
    SWE_SHELL_RECOVERY_FAILED, SWE_TOOL_STATE_FAILED, SWEStartupError, read_swe_tool_state,
    recover_swe_shell, start_private_swe, write_swe_status,
)
from elt_environment.env import EltEnv
from elt_environment.outcomes import ValidReward
from elt_environment.runner import AgentConfig, AgentFault, AgentSession


class OutputTruncationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.session = Mock()
        self.session.stop_worker = AsyncMock()
        self.session.artifact.return_value = {"elt/main.tf": "terraform {}"}
        self.session.duckdb_replay_receipt.return_value = {"connection_ids": ["test-id"]}
        self.renderer = Mock()
        self.renderer.get_stop_sequences.return_value = [1]
        self.renderer.parse_response.return_value = (
            {"role": "assistant", "content": ""},
            SimpleNamespace(is_clean=False),
        )
        self.task = SimpleNamespace(destination="duckdb")
        self.env = EltEnv(
            self.task, self.renderer, Mock(), 1, AgentConfig(max_tokens=4),
            session_factory=lambda *_: self.session,
        )

    async def assert_replayed(self, action, extra):
        self.env.pending = SimpleNamespace(reply=asyncio.get_running_loop().create_future())
        with patch("elt_environment.env.run_grader", new_callable=AsyncMock) as grader:
            grader.return_value = ValidReward(0.75, {"el_pass": 1.0})
            result = await self.env.step(action, extra=extra)
        self.assertEqual(result.reward, 0.75)
        self.assertTrue(result.episode_done)
        self.assertEqual(result.metrics["output_truncated"], 1.0)
        self.assertEqual(result.metrics["format_error"], 0.0)
        self.assertFalse(self.env.pending.reply.done())
        self.session.stop_worker.assert_awaited_once()
        grader.assert_awaited_once()
        self.assertEqual(grader.call_args.args[-1], {"connection_ids": ["test-id"]})

    async def test_provider_length_stop_prevents_execution_and_grades_saved_artifact(self):
        await self.assert_replayed([5, 6], {"stop_reason": "length"})
        self.renderer.parse_response.assert_not_called()

    async def test_full_length_missing_stop_reason_replays_legacy_truncation(self):
        await self.assert_replayed([5, 6, 7, 8], None)

    async def test_genuine_format_failure_keeps_its_own_metric(self):
        with patch("elt_environment.env.run_grader", new_callable=AsyncMock) as grader:
            result = await self.env.step([5, 6, 7, 8], extra={"stop_reason": "stop"})
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.metrics["format_error"], 1.0)
        self.assertFalse(result.metrics.get("output_truncated"))
        grader.assert_not_awaited()

    async def test_reported_swe_exit_preserves_replay_score(self):
        self.env.agent = AgentConfig(agent="swe")
        self.session.swe_exit_status.return_value = "submitted (exit_format)"
        with patch("elt_environment.env.run_grader", new_callable=AsyncMock) as grader:
            grader.return_value = ValidReward(0.75)
            result = await self.env._grade_workspace(budget=False)
        self.assertEqual(result.reward, 0.75)
        self.assertEqual(result.metrics["solver_exit_status/submitted (exit_format)"], 1.0)
        self.assertEqual(result.metrics["format_error"], 0.0)

    async def test_worker_uses_structured_startup_failure(self):
        session = AgentSession(SimpleNamespace(task_id="test"), self.renderer, AgentConfig(agent="swe"))
        loop = asyncio.get_running_loop()
        session.started = loop.time()
        session.done = loop.create_future()
        session.done.set_result(1)
        session.bridge.failure = loop.create_future()
        with tempfile.TemporaryDirectory() as directory:
            session.root = Path(directory)
            spec = {"result_file": str(session.root / "result.json")}
            write_swe_status(spec, startup_failure="swe_rex_relay_unavailable")
            with self.assertLogs("elt_environment.runner", level="WARNING"):
                with self.assertRaisesRegex(AgentFault, "^swe_rex_relay_unavailable$"):
                    await session.next_request()
            write_swe_status(spec, startup_failure="unrecognized", exit_status="submitted")
            self.assertEqual(session.swe_exit_status(), "submitted")
            with self.assertLogs("elt_environment.runner", level="WARNING"):
                with self.assertRaisesRegex(AgentFault, "^agent_worker_failed$"):
                    await session.next_request()

    async def test_runtime_failure_discards_even_when_worker_autosubmits(self):
        session = AgentSession(SimpleNamespace(task_id="test"), self.renderer, AgentConfig(agent="swe"))
        loop = asyncio.get_running_loop()
        session.started = loop.time()
        session.done = loop.create_future()
        session.done.set_result(0)
        session.bridge.failure = loop.create_future()
        with tempfile.TemporaryDirectory() as directory:
            session.root = Path(directory)
            for reason in (SWE_SHELL_RECOVERY_FAILED, SWE_TOOL_STATE_FAILED):
                with self.subTest(reason=reason):
                    write_swe_status(
                        {"result_file": str(session.root / "result.json")},
                        exit_status="submitted (exit_error)", runtime_failure=reason,
                    )
                    with self.assertRaisesRegex(AgentFault, f"^{reason}$"):
                        await session.next_request()


class ShellRecoveryTests(unittest.TestCase):
    def test_timeout_drains_poll_output_before_reading_state(self):
        marker = "123456789abcdef0" * 2
        shell = Mock()
        shell.communicate.side_effect = [
            '{"jobId":5391,"status":"running"}\nKeyboardInterrupt\n' + marker + "\n",
            '{"open_file":"n/a","working_dir":"/workspace/elt"}',
        ]
        with patch("elt_environment.agent_worker.secrets.token_hex", return_value=marker):
            recover_swe_shell(shell.interrupt, shell.communicate)
        shell.interrupt.assert_called_once()
        command = shell.communicate.call_args.args[0]
        self.assertNotIn(marker, command)
        self.assertEqual(shell.communicate("state"), '{"open_file":"n/a","working_dir":"/workspace/elt"}')

    def test_stale_output_cannot_acknowledge_a_recovery_command(self):
        first, second = "a" * 32, "b" * 32
        communicate = Mock(side_effect=["stale output", first + "\n"])
        with (
            patch("elt_environment.agent_worker.secrets.token_hex", side_effect=[first, second]),
            self.assertRaisesRegex(RuntimeError, "^swe_shell_recovery_failed$"),
        ):
            recover_swe_shell(Mock(), communicate)
        self.assertEqual(communicate.call_count, 2)


class ToolStateTests(unittest.TestCase):
    def test_background_job_json_and_completion_notice_do_not_corrupt_state(self):
        marker = "a" * 32
        payload = '{"open_file": "/workspace/elt/main.tf", "working_dir": "/workspace/elt"}'
        communicate = Mock(return_value=(
            '{"jobId":7232,"status":"running"}'
            f"SWE_STATE_BEGIN{marker}{payload}SWE_STATE_END{marker}\n"
            "[1]+ Done curl"
        ))
        with patch("elt_environment.agent_worker.secrets.token_hex", return_value=marker):
            state = read_swe_tool_state("_state", communicate)
        self.assertEqual(state, {"open_file": "/workspace/elt/main.tf", "working_dir": "/workspace/elt"})
        command = communicate.call_args.args[0]
        self.assertNotIn(f"SWE_STATE_BEGIN{marker}", command)
        self.assertIn("$(_state)", command)

    def test_retries_corrupted_frame_with_fresh_marker(self):
        first, second = "a" * 32, "b" * 32
        communicate = Mock(side_effect=[
            f"SWE_STATE_BEGIN{first}{{brokenSWE_STATE_END{first}",
            f'SWE_STATE_BEGIN{second}{{"working_dir":"/workspace"}}SWE_STATE_END{second}',
        ])
        with patch("elt_environment.agent_worker.secrets.token_hex", side_effect=[first, second]):
            self.assertEqual(read_swe_tool_state("_state", communicate), {"working_dir": "/workspace"})
        self.assertEqual(communicate.call_count, 2)

    def test_unmarked_or_stale_output_is_never_accepted_as_state(self):
        first, second = "a" * 32, "b" * 32
        communicate = Mock(side_effect=[
            '{"working_dir":"/workspace"}',
            f'SWE_STATE_BEGIN{first}{{"working_dir":"/workspace"}}SWE_STATE_END{first}',
        ])
        with (
            patch("elt_environment.agent_worker.secrets.token_hex", side_effect=[first, second]),
            self.assertRaisesRegex(RuntimeError, "^swe_tool_state_failed$"),
        ):
            read_swe_tool_state("_state", communicate)


class SolverStartupTests(unittest.TestCase):
    def test_timed_out_solver_is_removed_before_retry(self):
        opener = Mock()
        response = MagicMock()
        response.__enter__.return_value.status = 200
        opener.open.side_effect = [ConnectionError(), response]
        with (
            patch("elt_environment.agent_worker.urllib.request.build_opener", return_value=opener),
            patch("elt_environment.agent_worker.time.monotonic", side_effect=[0, 0, 46, 46, 46]),
            patch("elt_environment.agent_worker.time.sleep"),
            patch("elt_environment.agent_worker.local_swe_is_ready", return_value=False),
            patch("elt_environment.agent_worker.subprocess.run") as run,
        ):
            run.return_value.returncode = 0
            result = start_private_swe({
                "container_name": "test-solver", "image": "test-image",
                "duckdb_control_port": 3333,
            }, [])

        self.assertEqual(result["port"], 3333)
        self.assertEqual(
            [call.args[0][:4] for call in run.call_args_list],
            [["docker", "run", "--rm", "--detach"], ["docker", "rm", "-f", "test-solver"],
             ["docker", "run", "--rm", "--detach"]],
        )

    def test_container_and_relay_failures_have_different_codes(self):
        spec = {"container_name": "test-solver", "image": "test-image", "duckdb_control_port": 3333}
        for local_ready, reason in (
            (True, "swe_rex_relay_unavailable"),
            (False, "swe_rex_container_unavailable"),
        ):
            with (
                self.subTest(reason=reason),
                patch("elt_environment.agent_worker.time.monotonic", side_effect=[0, 46, 46, 92]),
                patch("elt_environment.agent_worker.subprocess.run", return_value=Mock(returncode=0)),
                patch("elt_environment.agent_worker.local_swe_is_ready", return_value=local_ready),
                self.assertRaisesRegex(SWEStartupError, f"^{reason}$"),
            ):
                start_private_swe(spec, [])

    def test_docker_start_failure_is_typed(self):
        spec = {"container_name": "test-solver", "image": "test-image", "duckdb_control_port": 3333}
        with (
            patch("elt_environment.agent_worker.subprocess.run", return_value=Mock(returncode=1)),
            self.assertRaisesRegex(SWEStartupError, "^swe_rex_docker_start_failed$"),
        ):
            start_private_swe(spec, [])
