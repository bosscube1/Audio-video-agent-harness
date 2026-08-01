"""Microphone capture for the Gemini Live Agent."""

from __future__ import annotations

import array
import asyncio
import contextlib
import math
import threading
from collections.abc import AsyncIterator
from typing import Any

import sounddevice as sd

from media import devices
from obs.logging_setup import get_logger

logger = get_logger(__name__)

INPUT_SAMPLE_RATE = 16_000  # Hz
CHANNELS = 1
DTYPE = "int16"
CHUNK_DURATION_MS = 100
INPUT_CHUNK_SAMPLES = int(INPUT_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
QUEUE_MAXSIZE = 4


class Microphone:
    """Captures audio from a microphone as 16 kHz 16-bit mono PCM chunks.

    Captured chunks are bridged from the PortAudio callback thread to the
    async event loop via a bounded asyncio queue. When the gate is closed,
    chunks are discarded instead of queued.
    """

    def __init__(self, device: str | int | None = None) -> None:
        if isinstance(device, str):
            self._device_index = devices.validate_device(device, "input")
        else:
            self._device_index = device

        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self._stream: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._gate_open = True

        self._last_level = 0.0
        self._level_lock = threading.Lock()

    # -- Public API --

    async def start(self) -> None:
        """Start capturing audio from the selected microphone."""
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._drain_queue()

        self._stream = sd.RawInputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=INPUT_CHUNK_SAMPLES,
            device=self._device_index,
            callback=self._audio_callback,
        )
        self._stream.start()
        logger.info("[MIC] Microphone active")

    async def stop(self) -> None:
        """Stop capturing and release the microphone."""
        self._running = False

        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Mic stop failed: %s", exc)
            try:
                self._stream.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Mic close failed: %s", exc)
            self._stream = None

        # Release any waiter in chunks().
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._enqueue_sentinel)
        logger.info("[MIC] Microphone stopped")

    async def chunks(self) -> AsyncIterator[bytes]:
        """Yield captured PCM chunks until the microphone is stopped."""
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

    def set_gate(self, open: bool) -> None:  # noqa: A002 - matches domain language
        """Open or close the audio gate.

        When closed, captured chunks are discarded instead of being queued.
        """
        self._gate_open = open

    @property
    def gate_open(self) -> bool:
        """Return whether the audio gate is currently open."""
        return self._gate_open

    @property
    def level(self) -> float:
        """Return the most recent input level, normalised to ``[0.0, 1.0]``."""
        with self._level_lock:
            return self._last_level

    # -- PortAudio callback (runs on the audio thread) --

    def _audio_callback(
        self, indata: Any, frames: int, time_info: Any, status: Any
    ) -> None:
        if status:
            logger.warning("Mic callback warning: %s", status)

        chunk = bytes(indata)

        if self._loop is None or self._loop.is_closed():
            return

        try:
            self._loop.call_soon_threadsafe(self._enqueue_chunk, chunk)
        except RuntimeError:  # pragma: no cover - event loop closed during shutdown
            logger.debug("Failed to enqueue mic chunk: event loop closed")

    # -- Queue bridging --

    def _enqueue_chunk(self, chunk: bytes) -> None:
        self._update_level(chunk)
        if not self._gate_open:
            return
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()  # drop oldest chunk
        self._queue.put_nowait(chunk)

    def _enqueue_sentinel(self) -> None:
        """Put a sentinel value to unblock chunk consumers."""
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        self._queue.put_nowait(None)

    def _drain_queue(self) -> None:
        """Drop any stale chunks left over from a previous session."""
        while not self._queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()

    def _update_level(self, chunk: bytes) -> None:
        if not chunk:
            return
        try:
            samples = array.array("h", chunk)
        except ValueError:
            return
        n = len(samples)
        if n == 0:
            return
        mean_square = sum(sample * sample for sample in samples) / n
        rms = math.sqrt(mean_square)
        with self._level_lock:
            self._last_level = min(rms / 32768.0, 1.0)
