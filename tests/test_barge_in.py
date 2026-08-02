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
    """Minimal stand-in for a callback ``sd.RawOutputStream``."""

    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self.closes = 0

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1

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
# Speaker.flush / callback
# ----------------------------------------------------------------------


def _drain_callback(speaker: Speaker, frames: int) -> bytearray:
    """Invoke the PortAudio callback once and return the buffer it filled."""
    out = bytearray(b"\xff" * frames * 2)  # sentinel: must be fully overwritten
    speaker._audio_callback(out, frames, None, None)
    return out


def test_callback_plays_queued_audio_and_pads_silence() -> None:
    """Queued chunks fill the callback buffer; underrun pads with zeros."""
    speaker = Speaker(stream_factory=lambda **_kw: _StubStream())
    speaker._active = True

    speaker.write(b"\x01\x00" * 40)  # 80 bytes = 40 frames
    out = _drain_callback(speaker, frames=64)

    assert bytes(out[:80]) == b"\x01\x00" * 40
    assert bytes(out[80:]) == b"\x00" * 48  # silence for the remaining 24 frames


def test_callback_splits_a_large_chunk_across_periods() -> None:
    """A chunk bigger than one callback period is consumed piece by piece."""
    speaker = Speaker(stream_factory=lambda **_kw: _StubStream())
    speaker._active = True

    speaker.write(b"\x02\x00" * 100)  # 100 frames
    first = _drain_callback(speaker, frames=64)
    second = _drain_callback(speaker, frames=64)

    assert bytes(first) == b"\x02\x00" * 64
    assert bytes(second[:72]) == b"\x02\x00" * 36  # remaining 36 frames
    assert bytes(second[72:]) == b"\x00" * 56  # then silence


def test_flush_drops_queued_audio_and_pending_remainder() -> None:
    """After flush, the callback emits silence — without touching the stream."""
    stream = _StubStream()
    speaker = Speaker(stream_factory=lambda **_kw: stream)
    speaker._stream = stream
    speaker._active = True

    speaker.write(b"\x01\x00" * 100)
    _drain_callback(speaker, frames=16)  # leave a pending remainder
    speaker.write(b"\x02\x00" * 100)

    speaker.flush()

    assert speaker._queue.qsize() == 0
    assert speaker._pending == b""
    out = _drain_callback(speaker, frames=64)
    assert bytes(out) == b"\x00" * 128
    # The stream is never aborted/closed/recreated on flush.
    assert stream.closes == 0
    assert stream.stops == 0
    assert speaker.level == 0.0


def test_callback_discards_chunks_queued_before_a_flush() -> None:
    """A chunk queued after the drain but tagged pre-flush is still dropped."""
    speaker = Speaker(stream_factory=lambda **_kw: _StubStream())
    speaker._active = True

    speaker.flush()
    # Re-queue a chunk tagged with the pre-flush generation, as a racing
    # write() would produce.
    speaker._queue.put((speaker._generation - 1, b"\x01\x00" * 100))

    out = _drain_callback(speaker, frames=64)
    assert bytes(out) == b"\x00" * 128


def test_flush_is_a_noop_when_the_speaker_is_not_running() -> None:
    speaker = Speaker()
    speaker.flush()  # must not raise
    assert speaker._queue.qsize() == 0


def test_start_passes_a_low_latency_callback_stream() -> None:
    """The stream factory receives the callback and low-latency settings."""
    captured: dict[str, Any] = {}

    def factory(**kwargs: Any) -> _StubStream:
        captured.update(kwargs)
        return _StubStream()

    speaker = Speaker(stream_factory=factory)
    asyncio.run(speaker.start())

    assert captured["samplerate"] == 24_000
    assert captured["latency"] == "low"
    assert captured["callback"] == speaker._audio_callback


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
