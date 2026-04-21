"""Logging bootstrap helpers for local CLI demos."""

from __future__ import annotations

import logging
import sys


class _LevelFormatter(logging.Formatter):
    """Simple formatter that adds a short per-level visual prefix."""

    LEVEL_PREFIX = {
        logging.DEBUG: "DBG",
        logging.INFO: "INF",
        logging.WARNING: "WRN",
        logging.ERROR: "ERR",
        logging.CRITICAL: "CRT",
    }

    def format(self, record: logging.LogRecord) -> str:
        record.level_tag = self.LEVEL_PREFIX.get(record.levelno, record.levelname)
        return super().format(record)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once for the CLI demo.

    The output is intentionally simple but useful for tracing node execution,
    tool calls, warnings, and unexpected errors during local iteration.
    """

    root_logger = logging.getLogger()
    resolved_level = getattr(logging, level.upper(), logging.INFO)

    if root_logger.handlers:
        root_logger.setLevel(resolved_level)
        for handler in root_logger.handlers:
            handler.setLevel(resolved_level)
        return

    formatter = _LevelFormatter(
        fmt="%(asctime)s | %(level_tag)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(resolved_level)
    stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)

    root_logger.setLevel(resolved_level)
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)
