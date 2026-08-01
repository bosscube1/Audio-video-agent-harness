"""Barge-in (interruption) tests.

When the user starts speaking mid-response, the server aborts the model turn
and sends ``server_content.interrupted``. The client must drop the audio it has
already buffered, otherwise the model keeps talking to the end of a response it
generated faster than real time, and the user's speech is merely queued behind
it.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest
from google.genai import types

from audit.journal import Journal
from core.commands import Disconnect
from core.session import LiveSession
from media.speaker import Speaker
from persist.store import Store
from settings.settings import AppSettings
from tests.fakes.fake_live import FakeLive, make_fake_client_factory


class _StubStream:
    """Minimal stand-in for ``sd.RawOutputStream``."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.aborts = 0
        self.starts = 0
        self.closes = 0

    def write(self, chunk: bytes) -> None:
        self.written.append(chunk)

    def abort(self) -> None:
        self.aborts += 1

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closes += 1


class RecordingSpeaker:
    """Speaker double that records writes and flushes for the session tests."""

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


# ----------------------------------------------------------------------
# Speaker.flush
# ----------------------------------------------------------------------


def test_flush_drops_queued_audio_and_resets_the_stream() -> None:
    """Queued chunks are discarded and the device buffer is aborted."""
    stream = _StubStream()
    created = [stream]

    def factory(**_kwargs: Any) -> _StubStream:
        new = _StubStream()
        created.append(new)
        return new

    speaker = Speaker(stream_factory=factory)
    speaker._stream = stream
    speaker._active = True

    speaker.write(b"\x01\x00" * 100)
    speaker.write(b"\x02\x00" * 100)
    assert speaker._queue.qsize() == 2

    speaker.flush()

    assert speaker._queue.qsize() == 0
    # The old stream is aborted and closed, and a fresh stream is opened for
    # the next turn (recreating avoids Windows MME errors restarting the same
    # aborted stream).
    assert stream.aborts == 1
    assert stream.closes == 1
    assert len(created) == 2
    assert created[1].starts == 1
    assert speaker._stream is created[1]
    assert speaker.level == 0.0


def test_writer_discards_chunks_queued_before_a_flush() -> None:
    """A chunk that survives the drain is still dropped by its generation tag."""
    stream = _StubStream()

    def factory(**_kwargs: Any) -> _StubStream:
        return _StubStream()

    speaker = Speaker(stream_factory=factory)
    speaker._stream = stream
    speaker._active = True

    stale = (speaker._generation, b"\x01\x00" * 100)
    speaker.flush()
    # Re-queue a chunk tagged with the pre-flush generation, as an in-flight
    # write() racing the flush would do.
    speaker._queue.put(stale)
    speaker._queue.put(None)

    speaker._writer()

    assert stream.written == []


def test_flush_is_a_noop_when_the_speaker_is_not_running() -> None:
    speaker = Speaker()
    speaker.flush()  # must not raise
    assert speaker._queue.qsize() == 0


def test_flush_survives_a_stream_that_fails_to_abort() -> None:
    """A device error during flush must not take the speaker down."""

    class _AngryStream(_StubStream):
        def abort(self) -> None:
            raise RuntimeError("device gone")

    def factory(**_kwargs: Any) -> _StubStream:
        return _StubStream()

    speaker = Speaker(stream_factory=factory)
    speaker._stream = _AngryStream()
    speaker._active = True
    speaker.write(b"\x01\x00" * 100)

    speaker.flush()

    assert speaker._queue.qsize() == 0
    assert not speaker._flushing.is_set()


# ----------------------------------------------------------------------
# Session wiring
# ----------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        model="gemini-3.1-flash-live-preview",
        mode="both",
        working_dir=tmp_path,
        workspace_root=tmp_path,
        share_screen=False,
    )


@pytest.mark.asyncio
async def test_interrupted_server_content_flushes_the_speaker(
    monkeypatch: Any,
    settings: AppSettings,
    tmp_path: Path,
) -> None:
    """A barge-in must flush playback, not just reset the transcript."""
    live = FakeLive()
    monkeypatch.setattr("google.genai.Client", make_fake_client_factory(live))
    monkeypatch.setattr("settings.secrets.get_api_key", lambda: "fake-api-key")
    monkeypatch.setattr("media.devices.list_input_devices", lambda: [])
    monkeypatch.setattr("media.devices.list_output_devices", lambda: [])

    speaker = RecordingSpeaker()
    command_queue: asyncio.Queue[Any] = asyncio.Queue()
    journal = Journal(run_id="barge-in", log_dir=tmp_path)
    session = LiveSession(
        settings=settings,
        command_queue=command_queue,
        event_queue=asyncio.Queue(),
        journal=journal,
        store=Store(db_path=tmp_path / "state.db"),
        run_id="barge-in",
        speaker=speaker,  # type: ignore[arg-type]
    )

    async def drive(fake_session: Any) -> None:
        fake_session.say("I was in the middle of a sentence")
        fake_session.interrupt()

    live.on_connect(drive)

    task = asyncio.create_task(session.run())
    try:
        await asyncio.wait_for(speaker.flushed.wait(), timeout=5)
        await command_queue.put(Disconnect())
        await asyncio.wait_for(task, timeout=5)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        journal._close()

    assert speaker.flushes == 1
    # The pre-interrupt audio was still played; only what was queued behind the
    # barge-in gets dropped (by Speaker.flush, covered above).
    assert speaker.writes == []


# ----------------------------------------------------------------------
# Connect config
# ----------------------------------------------------------------------


def test_connect_config_enables_interrupting_vad(settings: AppSettings) -> None:
    """Server-side VAD must be configured to interrupt on start of speech."""
    config = settings.as_connect_config(tools=[])

    realtime = config.realtime_input_config
    assert realtime is not None
    assert realtime.activity_handling == types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS

    detection = realtime.automatic_activity_detection
    assert detection is not None
    assert detection.disabled is False
    assert detection.prefix_padding_ms == settings.vad_prefix_padding_ms
    assert detection.silence_duration_ms == settings.vad_silence_duration_ms


def test_vad_settings_reach_the_connect_config(tmp_path: Path) -> None:
    """The vad_* settings are not dead config."""
    settings = AppSettings(
        working_dir=tmp_path,
        workspace_root=tmp_path,
        vad_prefix_padding_ms=120,
        vad_silence_duration_ms=450,
    )

    detection = settings.as_connect_config(
        tools=[]
    ).realtime_input_config.automatic_activity_detection  # type: ignore[union-attr]

    assert detection is not None
    assert detection.prefix_padding_ms == 120
    assert detection.silence_duration_ms == 450
