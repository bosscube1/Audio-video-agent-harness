"""Unit tests for ``core.turn_state``."""

from __future__ import annotations

import asyncio

import pytest
from google.genai import types

from core.events import PartialTranscript, SessionError, TranscriptSource, TurnComplete
from core.turn_state import TurnState


def _server_content(
    *,
    input_text: str | None = None,
    output_text: str | None = None,
    turn_complete: bool = False,
    interrupted: bool = False,
    model_turn_text: str | None = None,
) -> types.LiveServerContent:
    parts: list[types.Part] = []
    if model_turn_text:
        parts.append(types.Part(text=model_turn_text))
    return types.LiveServerContent(
        input_transcription=types.Transcription(text=input_text)
        if input_text is not None
        else None,
        output_transcription=types.Transcription(text=output_text)
        if output_text is not None
        else None,
        turn_complete=turn_complete,
        interrupted=interrupted,
        model_turn=types.Content(parts=parts) if parts else None,
    )


def test_tracks_input_and_output_transcripts() -> None:
    ts = TurnState(coalesce_ms=0.0)
    events = ts.consume_server_content(
        _server_content(input_text="hello ", output_text="hi ")
    )
    assert len(events) == 2
    assert events[0] == PartialTranscript(
        text="hello ", source=TranscriptSource.USER, turn_id=0
    )
    assert events[1] == PartialTranscript(
        text="hi ", source=TranscriptSource.MODEL, turn_id=0
    )


def test_emits_turn_complete_and_clears_transcript() -> None:
    ts = TurnState(coalesce_ms=0.0)
    events = ts.consume_server_content(
        _server_content(output_text="hello world", turn_complete=True)
    )

    partials = [e for e in events if isinstance(e, PartialTranscript)]
    completes = [e for e in events if isinstance(e, TurnComplete)]
    assert partials == [
        PartialTranscript(text="hello world", source=TranscriptSource.MODEL, turn_id=0)
    ]
    assert completes == [
        TurnComplete(source=TranscriptSource.MODEL, text="hello world", turn_id=0)
    ]

    # A new turn starts with a fresh turn id and empty transcript.
    events = ts.consume_server_content(_server_content(output_text="next"))
    assert events == [
        PartialTranscript(text="next", source=TranscriptSource.MODEL, turn_id=1)
    ]


@pytest.mark.asyncio
async def test_interrupted_emits_error_and_clears_model() -> None:
    # Use a non-zero coalesce window so the partial text is still pending when
    # the interrupt arrives.
    ts = TurnState(coalesce_ms=1000.0)
    events = ts.consume_server_content(
        _server_content(output_text="partial ", interrupted=True)
    )

    assert events[0] == PartialTranscript(
        text="partial ", source=TranscriptSource.MODEL, turn_id=0
    )
    assert events[1] == SessionError(message="interrupted", fatal=False)

    # After an interrupt the model transcript is cleared.
    events = ts.consume_server_content(
        _server_content(output_text="new turn", turn_complete=True)
    )
    complete = [e for e in events if isinstance(e, TurnComplete)][0]
    assert complete.text == "new turn"


@pytest.mark.asyncio
async def test_flush_emits_pending_text() -> None:
    ts = TurnState(coalesce_ms=1000.0)
    ts.consume_server_content(_server_content(output_text="pending"))
    events = ts.flush()

    assert events == [
        PartialTranscript(text="pending", source=TranscriptSource.MODEL, turn_id=0)
    ]


@pytest.mark.asyncio
async def test_coalesces_multiple_updates() -> None:
    ts = TurnState(coalesce_ms=50.0)
    ts.consume_server_content(_server_content(output_text="a"))
    ts.consume_server_content(_server_content(output_text="b"))
    ts.consume_server_content(_server_content(output_text="c"))

    # The text is pending; flush returns the accumulated value once.
    events = ts.flush()
    assert len(events) == 1
    assert events[0] == PartialTranscript(
        text="abc", source=TranscriptSource.MODEL, turn_id=0
    )


@pytest.mark.asyncio
async def test_no_emit_when_text_unchanged() -> None:
    ts = TurnState(coalesce_ms=20.0)
    ts.consume_server_content(_server_content(output_text="x"))
    await asyncio.sleep(0.05)

    # Drain the coalesced event.
    ts.flush()

    # No new text has arrived, so another flush should be empty.
    await asyncio.sleep(0.05)
    assert ts.flush() == []
