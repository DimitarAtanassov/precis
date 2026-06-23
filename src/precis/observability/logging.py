"""Structured logging setup.

Library and service code emits structured events through structlog instead of
``print``, so the same code path is debuggable in a terminal (console renderer)
and ingestible by log aggregators in production (JSON renderer). Logs go to
stderr to keep stdout reserved for actual program/CLI output.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog
from structlog.typing import FilteringBoundLogger


def configure_logging(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure process-wide structured logging. Idempotent and cheap.

    Args:
        level: Minimum level name (e.g. ``"INFO"``, ``"DEBUG"``).
        json_logs: Emit JSON lines (production) instead of the console renderer.
    """
    level_number = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level_number),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a bound structured logger, optionally named after a module."""
    return cast(FilteringBoundLogger, structlog.get_logger(name))
