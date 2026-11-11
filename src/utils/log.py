"""Logging setup shared by every entry point.

Replaces the old `print(...)` calls with proper levels:

    INFO     normal operation (models loaded, features enabled, stats)
    WARNING  a feature fell back or was disabled
    ERROR    something failed but the process can continue
"""

from __future__ import annotations

import logging
import sys

DEFAULT_FORMAT = "%(asctime)s  %(levelname)-7s  %(name)-24s  %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"


def setup_logging(level: int = logging.INFO, quiet: bool = False) -> None:
    """Configure the root logger once. Safe to call from any entry point."""
    root = logging.getLogger()
    if root.handlers:                     # already configured (imported twice)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(DEFAULT_FORMAT, DEFAULT_DATEFMT))
    root.addHandler(handler)
    root.setLevel(logging.ERROR if quiet else level)

    # third-party libraries are chatty at INFO
    for noisy in ("ultralytics", "openvino", "PIL", "matplotlib", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Shorthand used by every module."""
    return logging.getLogger(name)
