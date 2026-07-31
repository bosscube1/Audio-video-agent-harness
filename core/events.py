"""Frozen dataclasses that form the entire core -> GUI event vocabulary.

No PySide6, no asyncio primitives, no mutable state. These objects cross the
thread boundary in ``app/bridge.py`` and are consumed by the Qt list model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ConnectionState(StrEnum):
    IDLE = "idle"
    CONNECTING = "connecting"
    LIVE = "live"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class TranscriptSource(StrEnum):
    USER = "user"
    MODEL = "model"


@dataclass(frozen=True, slots=True)
class ConnectionStateChanged:
    state: ConnectionState
    detail: str = ""
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class PartialTranscript:
    text: str
    source: TranscriptSource
    turn_id: int = 0
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class TurnComplete:
    source: TranscriptSource
    text: str
    turn_id: int = 0
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class ToolCallReceived:
    call_id: str
    name: str
    args: dict[str, Any]
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class ToolApprovalRequested:
    call_id: str
    name: str
    args: dict[str, Any]
    allow_text: str = ""  # one-line policy summary for headless/CLI
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class ToolResultSent:
    call_id: str
    name: str
    ok: bool
    result: str
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class ToolCallCancelled:
    call_ids: list[str]
    reason: str
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class AudioLevel:
    source: str  # "mic" or "speaker"
    level: float  # 0.0 - 1.0 RMS-ish envelope
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class SessionError:
    message: str
    fatal: bool = False
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class ContextReset:
    """Emitted when a reconnect could not resume and history was rebuilt."""

    reason: str
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True, slots=True)
class DisconnectReason:
    """Reason a ``LiveSession`` ended.

    Use the module-level singletons ``USER_REQUEST``, ``API_CLOSE``, or build
    ``ERROR`` with a detail string.
    """

    code: str
    detail: str = ""
    timestamp: datetime = field(default_factory=_utc_now)


USER_REQUEST = DisconnectReason("user_request")
API_CLOSE = DisconnectReason("api_close")


@dataclass(frozen=True, slots=True)
class UsageUpdate:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    audio_tokens: int = 0
    video_tokens: int = 0
    total_tokens: int = 0
    estimated_usd: float = 0.0
    timestamp: datetime = field(default_factory=_utc_now)


# Convenience union used by type hints; not instantiated.
AgentEvent = (
    ConnectionStateChanged
    | PartialTranscript
    | TurnComplete
    | ToolCallReceived
    | ToolApprovalRequested
    | ToolResultSent
    | ToolCallCancelled
    | AudioLevel
    | SessionError
    | ContextReset
    | UsageUpdate
)
