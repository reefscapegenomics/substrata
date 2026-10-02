"""Logging configuration and helpers for the substrata package."""

import logging
from typing import Optional
from contextlib import contextmanager
import joblib

logger: logging.Logger = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO, format_string: Optional[str] = None
) -> None:
    """Set up logging for the substrata package only.

    Attaches a StreamHandler directly to the substrata logger and blocks
    propagation to the root logger, so third-party library output (kaleido,
    choreographer, etc.) is unaffected.

    Args:
        level: Logging level to use. Defaults to logging.INFO.
        format_string: Custom format string for log messages.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(levelname)s - %(message)s"

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to integrate joblib.Parallel with tqdm progress bars.

    Usage:
        with tqdm_joblib(tqdm(total=N)):
            Parallel(n_jobs=-1)(delayed(func)(x) for x in items)

    Args:
        tqdm_object: An instance of tqdm configured with the desired total.
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()
