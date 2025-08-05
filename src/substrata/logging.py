"""Logging configuration for the substrata package."""

import logging
from typing import Optional

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
