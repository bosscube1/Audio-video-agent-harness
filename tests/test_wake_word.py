"""Wake-word gating tests.

Three layers are covered:
- ``core.wake_word.contains_wake_word`` — the transcript matcher.
- ``AppSettings`` — the strict prompt block is appended only when enabled.
- ``LiveSession`` — voice turns without the wake word produce no audio, no
  model transcript events, no tool dispatch, and no persisted turns, while
  addressed voice turns and typed text passes through normally.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest
from google.genai import types

from app.main_window import MainWindow
from audit.journal import Journal
from core.commands import Disconnect, SendText
from core.events import (
    ConnectionState,
    ConnectionStateChanged,
    PartialTranscript,
    SessionError,
    TranscriptSource,
    TurnComplete,
)
from core.session import LiveSession
from core.wake_word import contains_wake_word
from persist.store import Store
from settings.settings import AppSettings
from tests.fakes.fake_live import FakeLive, make_fake_client_factory

# ----------------------------------------------------------------------
# Matcher
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "wake_word"),
    [
        ("Bongo, what's the time?", "Bongo"),  # exact, leading
        ("hey bongo", "Bongo"),  # case-insensitive
        ("can you check that, Bongo?", "Bongo"),  # trailing + punctuation
        ("thanks BONGO!", "bongo"),  # shouting + punctuation
        ("hey bonggo", "Bongo"),  # ASR near-miss (1 edit)
        ("hey bingo what's up", "Bongo"),  # tolerated phonetic near-match
        ("hey computer, open the file", "hey computer"),  # multi-word phrase
        ("Al, ping the server", "Al"),  # short wake word, exact
    ],
)
def test_matcher_accepts_addressed_utterances(text: str, wake_word: str) -> None:
    assert contains_wake_word(text, wake_word)


@pytest.mark.parametrize(
    ("text", "wake_word"),
    [
        ("did you see the game last night", "Bongo"),  # background chatter
        ("", "Bongo"),  # empty transcript
        ("bongo", ""),  # empty wake word never matches
        ("xl", "Al"),  # short wake word: fuzzy matching disabled
        ("hey computron", "hey computer"),  # multi-word: fuzzy disabled
    ],
)
def test_matcher_rejects_unaddressed_utterances(text: str, wake_word: str) -> None:
    assert not contains_wake_word(text, wake_word)


# ----------------------------------------------------------------------
# Settings / prompt
# ----------------------------------------------------------------------


def _make_settings(tmp_path: Path, **overrides: Any) -> AppSettings:
    base: dict[str, Any] = {
        "model": "gemini-3.1-flash-live-preview",
        "mode": "text",
        "working_dir": tmp_path,
        "workspace_root": tmp_path,
        "share_screen": False,
        "minimize_to_tray": False,
    }
    base.update(overrides)
    return AppSettings(**base)


def _instruction_text(settings: AppSettings) -> str:
    config = settings.as_connect_config(tools=[])
    instruction = config.system_instruction
    assert isinstance(instruction, types.Content)
    assert instruction.parts is not None
    return "".join(
        part.text or "" for part in instruction.parts if isinstance(part, types.Part)
    )


def test_wake_word_prompt_block_absent_by_default(tmp_path: Path) -> None:
    text = _instruction_text(_make_settings(tmp_path))
    assert "Wake-Word Gating" not in text


def test_wake_word_prompt_block_appended_when_enabled(tmp_path: Path) -> None:
    text = _instruction_text(
        _make_settings(tmp_path, wake_word_enabled=True, wake_word="Zephyr")
    )
    assert "Wake-Word Gating (STRICT)" in text
    assert '"Zephyr"' in text
    assert "background" in text
    # Built-in tool knowledge stays intact.
    assert "Your Tools" in text


def test_enabled_wake_word_must_not_be_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="wake_word"):
        _make_settings(tmp_path, wake_word_enabled=True, wake_word="   ")


def test_wake_word_settings_round_trip(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "settings.settings.user_config_dir",
        lambda appname, appauthor=None, **kwargs: str(tmp_path),
    )
    settings = _make_settings(
        tmp_path, wake_word_enabled=True, wake_word="Zephyr"
    )
    settings.save()
    loaded = AppSettings.load(working_dir=tmp_path, workspace_root=tmp_path)
    assert loaded.wake_word_enabled is True
    assert loaded.wake_word == "Zephyr"


# ----------------------------------------------------------------------
# Session gating
# ----------------------------------------------------------------------


class RecordingSpeaker:
    """Speaker double that records writes and flushes."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.flushes = 0
        self.flushed = asyncio.Event()

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def write(self, pcm_bytes: bytes) -> None:
        self.writes.append(pcm_bytes)

    def flush(self) -> None:
        self.flushes += 1
        self.flushed.set()

    @property
    def is_playing(self) -> bool:
        return True

    @property
    def level(self) -> float:
        return 0.0


def _audio_message(data: bytes) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            model_turn=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=data, mime_type="audio/pcm")
                    )
                ]
            )
        )
    )


async def _next_event(queue: asyncio.Queue[Any]) -> Any:
    async with asyncio.timeout(5):
        return await queue.get()


async def _wait_for_live(queue: asyncio.Queue[Any]) -> list[Any]:
    """Drain events until the session reports LIVE; return what was seen."""
    seen: list[Any] = []
    while True:
        event = await _next_event(queue)
        seen.append(event)
        if (
            isinstance(event, ConnectionStateChanged)
            and event.state == ConnectionState.LIVE
        ):
            return seen


async def _run_session(
    monkeypatch: Any,
    tmp_path: Path,
    settings: AppSettings,
) -> tuple[FakeLive, LiveSession, asyncio.Queue[Any], asyncio.Queue[Any], Journal, Store]:
    live = FakeLive()
    monkeypatch.setattr("google.genai.Client", make_fake_client_factory(live))
    monkeypatch.setattr("settings.secrets.get_api_key", lambda: "fake-api-key")
    monkeypatch.setattr("media.devices.list_input_devices", lambda: [])
    monkeypatch.setattr("media.devices.list_output_devices", lambda: [])

    command_queue: asyncio.Queue[Any] = asyncio.Queue()
    event_queue: asyncio.Queue[Any] = asyncio.Queue()
    journal = Journal(run_id="wake-test", log_dir=tmp_path)
    store = Store(db_path=tmp_path / "state.db")
    session = LiveSession(
        settings=settings,
        command_queue=command_queue,
        event_queue=event_queue,
        journal=journal,
        store=store,
        run_id="wake-test",
    )
    return live, session, command_queue, event_queue, journal, store


@pytest.mark.asyncio
async def test_unaddressed_voice_turn_is_fully_suppressed(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Background speech: no audio, no model transcript, no persisted turn."""
    settings = _make_settings(tmp_path, wake_word_enabled=True)
    live, session, command_queue, event_queue, journal, store = await _run_session(
        monkeypatch, tmp_path, settings
    )
    speaker = RecordingSpeaker()
    session._speaker = speaker  # type: ignore[assignment]
    session._owns_speaker = False

    task = asyncio.create_task(session.run())
    try:
        await _wait_for_live(event_queue)
        fake_session = live.latest_session
        assert fake_session is not None

        # The server heard background chatter and (wrongly) started answering.
        fake_session.emit(_audio_message(b"\x01\x00" * 48))
        fake_session.say(
            "Oh I think they were talking about the game",
            input_text="did you see the game last night",
            turn_complete=True,
        )

        await asyncio.wait_for(speaker.flushed.wait(), timeout=5)
        # Let the remaining scripted messages drain through the receive loop.
        await asyncio.sleep(0.3)
        command_queue.put_nowait(Disconnect())
        await asyncio.wait_for(task, timeout=5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        journal._close()

    events: list[Any] = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())

    model_events = [
        e
        for e in events
        if isinstance(e, (PartialTranscript, TurnComplete))
        and e.source == TranscriptSource.MODEL
    ]
    assert model_events == []
    assert speaker.writes == []
    assert speaker.flushes >= 1
    assert any(
        isinstance(e, SessionError) and "wake-word gate" in e.message
        for e in events
    )
    # Ignored utterances are not persisted as turns.
    assert store.get_turns("wake-test") == []


@pytest.mark.asyncio
async def test_addressed_voice_turn_passes_through(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """An utterance containing the wake word gets a normal response."""
    settings = _make_settings(tmp_path, wake_word_enabled=True)
    live, session, command_queue, event_queue, journal, store = await _run_session(
        monkeypatch, tmp_path, settings
    )

    task = asyncio.create_task(session.run())
    events: list[Any] = []
    try:
        events.extend(await _wait_for_live(event_queue))
        fake_session = live.latest_session
        assert fake_session is not None

        fake_session.say(
            "Sure, happy to help.",
            input_text="hey Bongo, can you help me",
            turn_complete=True,
        )

        while not any(isinstance(e, TurnComplete) for e in events):
            events.append(await _next_event(event_queue))

        command_queue.put_nowait(Disconnect())
        await asyncio.wait_for(task, timeout=5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        journal._close()

    model_completions = [
        e
        for e in events
        if isinstance(e, TurnComplete) and e.source == TranscriptSource.MODEL
    ]
    assert len(model_completions) == 1
    assert "happy to help" in model_completions[0].text
    assert store.get_turns("wake-test") != []


@pytest.mark.asyncio
async def test_typed_text_is_exempt_from_gating(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Typed input needs no wake word even when gating is enabled."""
    settings = _make_settings(tmp_path, wake_word_enabled=True)
    live, session, command_queue, event_queue, journal, store = await _run_session(
        monkeypatch, tmp_path, settings
    )

    task = asyncio.create_task(session.run())
    events: list[Any] = []
    try:
        events.extend(await _wait_for_live(event_queue))
        fake_session = live.latest_session
        assert fake_session is not None

        command_queue.put_nowait(SendText(text="what is 2+2"))
        # Wait until the typed turn was actually forwarded to the API — that
        # is also the point where the text-turn exemption is armed.
        for _ in range(100):
            if fake_session.pop_outbox():
                break
            await asyncio.sleep(0.05)
        else:  # pragma: no cover - test harness failure
            pytest.fail("typed turn was never forwarded to the API")

        # The model replies without any voice transcript at all.
        fake_session.say("Four.", turn_complete=True)

        while not any(isinstance(e, TurnComplete) for e in events):
            events.append(await _next_event(event_queue))

        command_queue.put_nowait(Disconnect())
        await asyncio.wait_for(task, timeout=5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        journal._close()

    model_completions = [
        e
        for e in events
        if isinstance(e, TurnComplete) and e.source == TranscriptSource.MODEL
    ]
    assert len(model_completions) == 1
    assert "Four" in model_completions[0].text


@pytest.mark.asyncio
async def test_gating_disabled_leaves_background_turns_alone(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """With wake_word_enabled=False (default), behavior is unchanged."""
    settings = _make_settings(tmp_path)  # gating off
    live, session, command_queue, event_queue, journal, store = await _run_session(
        monkeypatch, tmp_path, settings
    )

    task = asyncio.create_task(session.run())
    events: list[Any] = []
    try:
        events.extend(await _wait_for_live(event_queue))
        fake_session = live.latest_session
        assert fake_session is not None

        fake_session.say(
            "Answering unprompted.",
            input_text="just some background chatter",
            turn_complete=True,
        )

        while not any(isinstance(e, TurnComplete) for e in events):
            events.append(await _next_event(event_queue))

        command_queue.put_nowait(Disconnect())
        await asyncio.wait_for(task, timeout=5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        journal._close()

    assert any(
        isinstance(e, TurnComplete) and e.source == TranscriptSource.MODEL
        for e in events
    )


# ----------------------------------------------------------------------
# GUI controls
# ----------------------------------------------------------------------


def test_gui_loads_wake_word_controls(qtbot: Any, tmp_path: Path) -> None:
    settings = _make_settings(tmp_path, wake_word_enabled=True, wake_word="Zephyr")
    window = MainWindow(settings)
    qtbot.addWidget(window)

    assert window._wake_word_check.isChecked()
    assert window._wake_word_edit.text() == "Zephyr"


def test_gui_wake_word_round_trips_into_built_settings(
    qtbot: Any, tmp_path: Path
) -> None:
    settings = _make_settings(tmp_path)
    window = MainWindow(settings)
    qtbot.addWidget(window)

    assert not window._wake_word_check.isChecked()
    assert window._wake_word_edit.text() == "Bongo"

    window._wake_word_check.setChecked(True)
    window._wake_word_edit.setText("  Zephyr  ")
    built = window._build_settings()
    assert built.wake_word_enabled is True
    assert built.wake_word == "Zephyr"
