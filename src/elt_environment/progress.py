"""Display completed training batches alongside Cookbook's sampling bars."""

from contextlib import contextmanager

from tinker_cookbook.rl import train
from tinker_cookbook.utils.ml_log import Logger
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class _TrainingProgress(Logger):
    def __init__(self, bar):
        self.bar = bar

    def log_hparams(self, config):
        pass

    def log_metrics(self, metrics, step=None):
        batch = metrics.get("progress/batch")
        if batch is not None:
            completed = min(batch + 1, self.bar.total)
            self.bar.update(max(0, completed - self.bar.n))


@contextmanager
def training_progress(enabled=True):
    """Add an overall bar for one synchronous CLI training run.

    Cookbook 0.5.7 has no progress callback. Wrap its sync entry point only
    during this run so the bar uses the trainer's actual resume/end indices
    and completed-batch metrics, without duplicating its training loop.
    """
    if not enabled:
        yield
        return

    original = train.do_sync_training

    async def with_progress(**kwargs):
        with (
            tqdm(
                total=kwargs["end_batch"],
                initial=min(kwargs["start_batch"], kwargs["end_batch"]),
                desc="Training",
                unit="step",
                dynamic_ncols=True,
            ) as bar,
            logging_redirect_tqdm(),
        ):
            progress = _TrainingProgress(bar)
            loggers = kwargs["ml_logger"].loggers
            loggers.append(progress)
            try:
                return await original(**kwargs)
            finally:
                loggers.remove(progress)

    train.do_sync_training = with_progress
    try:
        yield
    finally:
        train.do_sync_training = original
