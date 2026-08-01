"""JSON Lines audit journal.

Every record is flushed immediately to disk so crashes still leave a trail.
Serialization prefers ``orjson`` but falls back to the stdlib ``json`` module.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import platformdirs

try:
    import orjson
except ImportError:  # pragma: no cover - orjson is a required dependency
    orjson = None  # type: ignore[assignment]


class _Serializer:
    """Thin adapter so ``orjson`` and ``json`` expose the same interface."""

    def __init__(self, module: ModuleType | None) -> None:
        self._module = module

    def dumps(self, obj: Any) -> str:
        if self._module is not None:
            payload: bytes = self._module.dumps(obj)
            return payload.decode("utf-8")
        return json.dumps(obj, default=str, separators=(",", ":"))


class Journal:
    """Append-only JSON Lines journal for a single run.

    Parameters
    ----------
    run_id:
        Identifier for the run; included in every record.
    log_dir:
        Directory that will contain ``{run_id}.jsonl``. Defaults to the
        user-specific log directory for ``GeminiLiveAgent``.
    """

    _serializer = _Serializer(orjson)

    def __init__(self, run_id: str, log_dir: Path | None = None) -> None:
        self.run_id = run_id
        self.log_dir = (
            log_dir
            if log_dir is not None
            else Path(platformdirs.user_log_dir("GeminiLiveAgent", appauthor=False)) / "journal"
        )
        # Atomic / idempotent directory creation.
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.log_dir / f"{run_id}.jsonl"
        self._file: Any | None = None

    def _ensure_open(self) -> Any:
        if self._file is None or self._file.closed:
            self._file = self._path.open("a", encoding="utf-8")
        return self._file

    def close(self) -> None:
        """Close the journal file handle, flushing any buffered writes."""
        if self._file is not None and not self._file.closed:
            self._file.close()

    # Backward-compatible alias; prefer ``close()``.
    _close = close

    def _now_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def write(self, record: dict[str, Any]) -> None:
        """Write a single JSON Line record, flushing immediately to disk."""
        enriched = {
            "run_id": self.run_id,
            "timestamp": self._now_iso(),
            **record,
        }
        line = self._serializer.dumps(enriched)
        f = self._ensure_open()
        f.write(line + "\n")
        f.flush()

    def tool_attempt(
        self,
        call_id: str,
        name: str,
        args: dict[str, Any],
        epoch: int,
    ) -> None:
        """Log a tool call just before any side effect is executed."""
        self.write(
            {
                "type": "tool_attempt",
                "call_id": call_id,
                "name": name,
                "args": args,
                "epoch": epoch,
            }
        )

    def tool_result(
        self,
        call_id: str,
        name: str,
        ok: bool,
        result: Any,
        *,
        orphaned: bool = False,
        epoch: int,
    ) -> None:
        """Log the outcome of a tool call after its side effect has run."""
        self.write(
            {
                "type": "tool_result",
                "call_id": call_id,
                "name": name,
                "ok": ok,
                "result": result,
                "orphaned": orphaned,
                "epoch": epoch,
            }
        )

    def gating_mode_change(self, mode: str, duration_minutes: int | None = None) -> None:
        """Log a change in safety/gating mode (e.g. autopilot -> confirmation)."""
        record: dict[str, Any] = {
            "type": "gating_mode_change",
            "mode": mode,
        }
        if duration_minutes is not None:
            record["duration_minutes"] = duration_minutes
        self.write(record)

    def connection_event(
        self, event: str, detail: str, *, epoch: int = 0
    ) -> None:
        """Log a connection lifecycle event (connect, disconnect, error, etc.)."""
        self.write(
            {
                "type": "connection_event",
                "event": event,
                "detail": detail,
                "epoch": epoch,
            }
        )

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, *exc: object) -> None:
        self._close()
