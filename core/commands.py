"""Frozen dataclasses that form the entire GUI -> core command vocabulary.

Commands are placed on a single ``asyncio.Queue`` and consumed by the
supervisor/session loop. No PySide6 here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(UTC)


class GatingMode(StrEnum):
    GUARDED = "guarded"
    YOLO = "yolo"


@dataclass(frozen=True, slots=True)
class Connect:
    """Start (or restart) a conversation with the given settings."""

    settings: dict[str, Any]
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class Disconnect:
    """Gracefully end the current session."""

    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class SendText:
    text: str
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class SetMicGate:
    """Open or close the microphone gate. False = muted."""

    open: bool
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class SetShareScreen:
    enabled: bool
    monitor: int | None = None
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class ApproveTool:
    call_id: str
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class DenyTool:
    call_id: str
    reason: str = "user_denied"
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class CancelTool:
    call_ids: list[str]
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class SetGatingMode:
    mode: GatingMode
    duration_minutes: int | None = None
    timestamp: datetime = field(default_factory=_utc_now)


# Convenience union.
AgentCommand = (
    Connect
    | Disconnect
    | SendText
    | SetMicGate
    | SetShareScreen
    | ApproveTool
    | DenyTool
    | CancelTool
    | SetGatingMode
)
