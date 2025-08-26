"""Logging configuration and helpers for the substrata package."""

import logging
from typing import Optional
from contextlib import contextmanager
import joblib

logger: logging.Logger = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO, format_string: Optional[str] = None
) -> None:
    """Set up logging configuration for the substrata package.

    Args:
        level: Logging level to use. Defaults to logging.INFO.
        format_string: Custom format string for log messages.
                      If None, uses a default format.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(level=level, format=format_string, datefmt="%Y-%m-%d %H:%M:%S")


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
