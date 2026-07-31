"""Assemble partial transcripts into turns and enforce turn boundaries.

``TurnState`` consumes ``types.ServerContent`` from the Live API and emits the
``PartialTranscript`` / ``TurnComplete`` events that the rest of the core
publishes to the GUI.
"""

from __future__ import annotations

import asyncio
from typing import Any

from google.genai import types

from core.events import (
    AgentEvent,
    PartialTranscript,
    SessionError,
    TranscriptSource,
    TurnComplete,
)


class TurnState:
    """Accumulates user and model transcript text and emits coalesced events.

    Parameters
    ----------
    coalesce_ms:
        Minimum interval between successive ``PartialTranscript`` events for the
        same source. Updates inside the window are accumulated and emitted once
        when the window expires.
    """

    def __init__(self, coalesce_ms: float = 80.0) -> None:
        self._coalesce_ms = max(0.0, coalesce_ms)
        self._coalesce_s = self._coalesce_ms / 1000.0

        # Completed transcript for the active turn, used to build TurnComplete.
        self._user_text = ""
        self._model_text = ""
        self._user_turn_id = 0
        self._model_turn_id = 0

        # Text that has arrived since the last PartialTranscript emission.
        self._pending_user = ""
        self._pending_model = ""

        self._timer_user: asyncio.TimerHandle | None = None
        self._timer_model: asyncio.TimerHandle | None = None

        # Events produced by timer callbacks are buffered here because
        # ``consume_server_content`` is synchronous. Public methods drain it.
        self._outbox: list[AgentEvent] = []

    def consume_server_content(self, sc: types.LiveServerContent) -> list[AgentEvent]:
        """Process a server content chunk and return any events to emit."""
        events = self._drain_outbox()

        # Input transcription = what the model heard from the user.
        if sc.input_transcription and sc.input_transcription.text:
            self._pending_user += sc.input_transcription.text
            self._schedule_emit(TranscriptSource.USER)

        # Output transcription = what the model is saying.
        if sc.output_transcription and sc.output_transcription.text:
            self._pending_model += sc.output_transcription.text
            self._schedule_emit(TranscriptSource.MODEL)

        # Model turn parts may contain text in addition to audio inline data.
        model_turn = sc.model_turn
        if model_turn and model_turn.parts:
            for part in model_turn.parts:
                text = getattr(part, "text", None)
                if text:
                    self._pending_model += text
                    self._schedule_emit(TranscriptSource.MODEL)

        # Interrupted: model was cut off. Surface as a non-fatal session error
        # and reset the model transcript so the next turn starts fresh.
        if sc.interrupted:
            self._cancel_timer(TranscriptSource.MODEL)
            if self._pending_model:
                events.append(
                    PartialTranscript(
                        text=self._pending_model,
                        source=TranscriptSource.MODEL,
                        turn_id=self._model_turn_id,
                    )
                )
                self._model_text += self._pending_model
                self._pending_model = ""
            events.append(SessionError(message="interrupted", fatal=False))
            self._model_text = ""

        # Turn complete: flush and emit the final transcript for the source.
        if sc.turn_complete:
            events.extend(self._complete_turn(TranscriptSource.MODEL))

        events.extend(self._drain_outbox())
        return events

    def flush(self) -> list[AgentEvent]:
        """Emit any pending partial transcript text immediately.

        Called during shutdown so the GUI does not leave a half-rendered line.
        """
        events = self._drain_outbox()
        for source in (TranscriptSource.USER, TranscriptSource.MODEL):
            self._cancel_timer(source)
            pending = self._pop_pending(source)
            if pending:
                if source == TranscriptSource.USER:
                    self._user_text += pending
                else:
                    self._model_text += pending
                events.append(
                    PartialTranscript(
                        text=pending,
                        source=source,
                        turn_id=self._turn_id_for(source),
                    )
                )
        return events

    def _schedule_emit(self, source: TranscriptSource) -> None:
        """Schedule a coalesced emission for ``source`` if one is not pending."""
        if self._coalesce_s <= 0.0:
            self._emit_now(source)
            return

        timer = self._timer_for(source)
        if timer is not None and not timer.cancelled():
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Synchronous fallback: emit immediately if no loop is running.
            self._emit_now(source)
            return

        handle = loop.call_later(self._coalesce_s, self._emit_now, source)
        self._set_timer(source, handle)

    def _emit_now(self, source: TranscriptSource) -> None:
        """Emit accumulated text for ``source`` if it has changed."""
        self._set_timer(source, None)
        pending = self._pop_pending(source)
        if not pending:
            return

        turn_id = self._turn_id_for(source)
        if source == TranscriptSource.USER:
            self._user_text += pending
        else:
            self._model_text += pending

        self._outbox.append(
            PartialTranscript(text=pending, source=source, turn_id=turn_id)
        )

    def _complete_turn(self, source: TranscriptSource) -> list[AgentEvent]:
        """Flush pending text and emit ``TurnComplete`` for ``source``."""
        self._cancel_timer(source)
        events: list[AgentEvent] = []

        pending = self._pop_pending(source)
        if pending:
            if source == TranscriptSource.USER:
                self._user_text += pending
            else:
                self._model_text += pending
            events.append(
                PartialTranscript(
                    text=pending,
                    source=source,
                    turn_id=self._turn_id_for(source),
                )
            )

        full_text = self._clear_transcript(source)
        if full_text:
            events.append(
                TurnComplete(
                    source=source,
                    text=full_text,
                    turn_id=self._turn_id_for(source),
                )
            )

        self._increment_turn_id(source)
        return events

    def _pop_pending(self, source: TranscriptSource) -> str:
        if source == TranscriptSource.USER:
            text = self._pending_user
            self._pending_user = ""
            return text
        text = self._pending_model
        self._pending_model = ""
        return text

    def _clear_transcript(self, source: TranscriptSource) -> str:
        if source == TranscriptSource.USER:
            text = self._user_text
            self._user_text = ""
            return text
        text = self._model_text
        self._model_text = ""
        return text

    def _turn_id_for(self, source: TranscriptSource) -> int:
        return self._user_turn_id if source == TranscriptSource.USER else self._model_turn_id

    def _increment_turn_id(self, source: TranscriptSource) -> None:
        if source == TranscriptSource.USER:
            self._user_turn_id += 1
        else:
            self._model_turn_id += 1

    def _timer_for(self, source: TranscriptSource) -> asyncio.TimerHandle | None:
        return self._timer_user if source == TranscriptSource.USER else self._timer_model

    def _set_timer(
        self, source: TranscriptSource, handle: asyncio.TimerHandle | None
    ) -> None:
        if source == TranscriptSource.USER:
            self._timer_user = handle
        else:
            self._timer_model = handle

    def _cancel_timer(self, source: TranscriptSource) -> None:
        timer = self._timer_for(source)
        if timer is not None and not timer.cancelled():
            timer.cancel()
        self._set_timer(source, None)

    def _drain_outbox(self) -> list[AgentEvent]:
        """Return and clear the internal event buffer."""
        out = list(self._outbox)
        self._outbox.clear()
        return out

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - safety net
        raise AttributeError(name)
