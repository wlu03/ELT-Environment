"""A capped response preserves the score for work completed in earlier turns."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from elt_environment.env import EltEnv
from elt_environment.outcomes import ValidReward
from elt_environment.runner import AgentConfig


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
