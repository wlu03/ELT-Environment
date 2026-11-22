from __future__ import annotations

import asyncio
import io
import unittest
from unittest import mock

from tinker_cookbook.rl import train
from tinker_cookbook.utils.ml_log import Logger, MultiplexLogger
from tqdm import tqdm

from elt_environment.progress import training_progress


class RecordingLogger(Logger):
    def __init__(self):
        self.records = []

    def log_hparams(self, config):
        pass

    def log_metrics(self, metrics, step=None):
        self.records.append((metrics, step))


class TrainingProgressTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.bars = []
        self.sink = RecordingLogger()
        self.logger = MultiplexLogger([self.sink])

        def make_bar(**kwargs):
            bar = tqdm(**{**kwargs, "file": self.output, "mininterval": 0, "miniters": 1})
            self.bars.append(bar)
            return bar

        patcher = mock.patch("elt_environment.progress.tqdm", side_effect=make_bar)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_resume_and_completed_batches_drive_progress_and_preserve_metrics(self):
        metrics = {"progress/batch": 2, "reward": 0.75}

        async def training(**kwargs):
            self.assertIs(kwargs["ml_logger"], self.logger)
            self.assertEqual(self.bars[0].total, 4)
            self.assertEqual(self.bars[0].n, 2)
            self.logger.log_metrics({"eval/reward": 1.0}, step=2)
            self.assertEqual(self.bars[0].n, 2)
            self.logger.log_metrics(metrics, step=2)
            self.assertEqual(self.bars[0].n, 3)
            self.logger.log_metrics(metrics, step=2)
            self.assertEqual(self.bars[0].n, 3)
            self.logger.log_metrics({"progress/batch": 3}, step=3)
            return "trained"

        with mock.patch.object(train, "do_sync_training", training):
            with training_progress():
                result = await train.do_sync_training(
                    start_batch=2, end_batch=4, ml_logger=self.logger
                )
            self.assertIs(train.do_sync_training, training)
        self.assertEqual(result, "trained")
        self.assertEqual(self.bars[0].n, 4)
        self.assertIn("4/4", self.output.getvalue())
        self.assertIs(self.sink.records[1][0], metrics)
        self.assertEqual(self.sink.records[1][1], 2)
        self.assertEqual(self.logger.loggers, [self.sink])

    async def test_failure_and_cancellation_close_partial_bar_and_restore_trainer(self):
        for error in (RuntimeError("training failed"), asyncio.CancelledError()):

            async def training(error=error, **kwargs):
                self.logger.log_metrics({"progress/batch": 0}, step=0)
                raise error

            with (
                self.subTest(error=type(error)),
                mock.patch.object(train, "do_sync_training", training),
            ):
                with self.assertRaises(type(error)), training_progress():
                    await train.do_sync_training(start_batch=0, end_batch=5, ml_logger=self.logger)
                self.assertIs(train.do_sync_training, training)
            self.assertEqual(self.bars[-1].n, 1)
            self.assertTrue(self.bars[-1].disable)
            self.assertEqual(self.logger.loggers, [self.sink])

    async def test_already_completed_resume_stays_within_step_limit(self):
        training = mock.AsyncMock()
        with mock.patch.object(train, "do_sync_training", training), training_progress():
            await train.do_sync_training(start_batch=8, end_batch=5, ml_logger=self.logger)
        self.assertEqual((self.bars[0].n, self.bars[0].total), (5, 5))

    async def test_disabled_progress_leaves_training_untouched(self):
        training = mock.AsyncMock()
        with mock.patch.object(train, "do_sync_training", training), training_progress(False):
            self.assertIs(train.do_sync_training, training)
            await train.do_sync_training(start_batch=0, end_batch=2, ml_logger=self.logger)
        self.assertEqual(self.bars, [])
        training.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
