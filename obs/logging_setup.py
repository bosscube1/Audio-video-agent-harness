"""Logging setup for the Gemini Live Agent.

Provides a project root logger that writes structured JSON Lines to a rotating
file under the user's platform-specific log directory and plain text to stderr.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import platformdirs

orjson: Any | None
try:
    import orjson
except ImportError:
    orjson = None

_APP_NAME = "GeminiLiveAgent"
_ROOT_LOGGER_NAME = "gemini_live_agent"
_LOG_FILENAME = "agent.log"
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 4

_RUN_ID: str = str(uuid.uuid4())

# Standard attributes on a logging.LogRecord; anything else is treated as `extra`.
_STD_RECORD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "run_id",
        "taskName",
    }
)

__all__ = [
    "get_logger",
    "get_run_id",
    "setup_logging",
]


class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=UTC
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": self.run_id,
        }

        # Include any extra fields attached to the record.
        for key, value in record.__dict__.items():
            if key not in _STD_RECORD_ATTRS:
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        if orjson is not None:
            encoded: bytes = orjson.dumps(payload, default=str)
            return encoded.decode("utf-8")
        return json.dumps(payload, default=str, ensure_ascii=False)


def _log_directory() -> Path:
    """Return the platform-specific log directory for this application."""
    return Path(platformdirs.user_log_dir(appname=_APP_NAME, appauthor=False))


def get_run_id() -> str:
    """Return the current run's UUID4 identifier (hyphenated, lowercase)."""
    return _RUN_ID


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the project root logger."""
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")


def setup_logging(run_id: str | None = None, debug: bool = False) -> logging.Logger:
    """Configure and return the project root logger.

    Args:
        run_id: Optional run identifier. If omitted, the module-level run_id is used.
        debug: When True, set the root logger level to DEBUG; otherwise INFO.

    Returns:
        The configured ``gemini_live_agent`` logger.
    """
    global _RUN_ID
    if run_id is not None:
        _RUN_ID = run_id

    log_dir = _log_directory()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / _LOG_FILENAME

    root_logger = logging.getLogger(_ROOT_LOGGER_NAME)
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.handlers.clear()
    root_logger.propagate = False

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(_JSONFormatter(run_id=_RUN_ID))
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root_logger.addHandler(console_handler)

    return root_logger
