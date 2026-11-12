"""A failed destination reset must abort the attempt, not limp through.

prepare_destination wraps every preparation failure in
DestinationPreparationError, and each runner propagates it before copying
seed files, starting a container, or opening an agent session.

Run from the ELT-Bench repository root:

    python3 -m unittest discover -s agents/tests -v

The per-runner tests skip when the runner's SDK is not installed. No
network, docker, or credentials are touched.
"""

import asyncio
import importlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

AGENTS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENTS_DIR.parent
if str(AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(AGENTS_DIR))

from common import (  # noqa: E402
    DestinationPreparationError,
    prepare_destination,
    warehouse,
)


def load_module(path: Path, name: str):
    """Load a non-package module (a runner's run.py) by file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PrepareDestinationWrapsFailuresTest(unittest.TestCase):
    def test_prepare_destination_wraps_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_input_dir = Path(tmp) / "asana"
            task_input_dir.mkdir()
            (task_input_dir / "config.yaml").write_text(
                "snowflake:\n"
                "  config:\n"
                "    database: asana\n"
                "    schema: AIRBYTE_SCHEMA\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                warehouse, "_prepare_snowflake", side_effect=RuntimeError("boom")
            ):
                with self.assertRaises(DestinationPreparationError) as ctx:
                    prepare_destination("snowflake", task_input_dir)
            self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
            self.assertIn("boom", str(ctx.exception))

            # Config-load failures take the same typed path.
            missing_config_dir = Path(tmp) / "empty_task"
            missing_config_dir.mkdir()
            with self.assertRaises(DestinationPreparationError) as ctx:
                prepare_destination("snowflake", missing_config_dir)
            self.assertIsNotNone(ctx.exception.__cause__)


class RunnerAbortsBeforeSessionTest(unittest.TestCase):
    def _import_or_skip(self, name: str) -> None:
        try:
            module = importlib.import_module(name)
        except ImportError:
            self.skipTest(f"{name} is not installed")
        if getattr(module, "__file__", None) is None:
            # A bare namespace package (e.g. this repo's agents/ directory
            # picked up from the working directory), not an installed SDK.
            self.skipTest(f"{name} resolves to a namespace package, not an SDK")

    def _assert_runner_fails_closed(self, runner_path: Path, module_name: str):
        runner = load_module(runner_path, module_name)
        calls: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_dir = tmp_path / "out"
            out_dir.mkdir()
            result_path = out_dir / "x" / "result.json"
            with mock.patch.object(
                runner,
                "prepare_destination",
                side_effect=DestinationPreparationError("injected"),
            ), mock.patch.object(
                runner,
                "copy_initial_files",
                side_effect=lambda *a, **k: calls.append("copy_initial_files"),
            ), mock.patch.object(
                runner,
                "ensure_container",
                side_effect=lambda *a, **k: calls.append("ensure_container"),
            ):
                with self.assertRaises(DestinationPreparationError):
                    asyncio.run(
                        runner._run_task_body(
                            db="asana",
                            destination="snowflake",
                            credential=None,
                            instance_id="t/asana",
                            experiment_id="t",
                            out_dir=out_dir,
                            result_path=result_path,
                            inputs_root=tmp_path,
                            model="m",
                            max_turns=1,
                        )
                    )
            # No seed copy and no container means no agent session started.
            self.assertEqual(calls, [])
            # Nothing for evaluation to consume.
            self.assertFalse(result_path.exists())

    def test_codex_runner_aborts_before_session(self):
        self._import_or_skip("agents")
        self._assert_runner_fails_closed(
            AGENTS_DIR / "codex_agent" / "run.py", "elt_bench_codex_runner"
        )

    def test_claude_runner_aborts_before_session(self):
        self._import_or_skip("claude_agent_sdk")
        self._assert_runner_fails_closed(
            AGENTS_DIR / "claude_agent" / "run.py", "elt_bench_claude_runner"
        )

    def test_openhands_runner_aborts_before_session(self):
        self._import_or_skip("openhands.sdk")
        self._assert_runner_fails_closed(
            AGENTS_DIR / "openhands" / "run.py", "elt_bench_openhands_runner"
        )


if __name__ == "__main__":
    unittest.main()
