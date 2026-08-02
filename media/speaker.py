"""Speaker playback for the Gemini Live Agent."""

from __future__ import annotations

import array
import math
import queue
import threading
from typing import Any

import sounddevice as sd

from media import devices
from obs.logging_setup import get_logger

logger = get_logger(__name__)

OUTPUT_SAMPLE_RATE = 24_000  # Hz
CHANNELS = 1
DTYPE = "int16"
_BYTES_PER_FRAME = CHANNELS * 2  # int16


class Speaker:
    """Plays 24 kHz 16-bit mono PCM audio through the selected speaker.

    The stream is a *callback* (pull) stream: PortAudio calls
    ``_audio_callback`` on its own real-time thread whenever it needs more
    frames, and the callback drains a queue of PCM chunks. There is no writer
    thread and no blocking ``stream.write()`` call anywhere.

    That design is what makes barge-in both instant and safe: ``flush()`` only
    bumps a generation counter and drops queued audio, so playback goes silent
    within one callback period. The stream itself is never aborted, closed, or
    recreated mid-conversation, which avoids the Windows MME errors ("wave
    header not prepared", "media data is still playing") and the native
    crash that tearing a stream down mid-write could cause.

    Thread-safety: the callback only reads ``self._generation`` and swaps
    ``self._pending`` via atomic (GIL-protected) assignments; ``flush()`` may
    run on the asyncio thread. Worst case, one stale chunk plays out after a
    flush — a few tens of milliseconds.
    """

    def __init__(
        self,
        device: str | int | None = None,
        *,
        stream_factory: Any = sd.RawOutputStream,
    ) -> None:
        if isinstance(device, str):
            self._device_index = devices.validate_device(device, "output")
        else:
            self._device_index = device

        self._stream_factory = stream_factory
        self._stream: Any | None = None
        self._active = False
        self._queue: queue.Queue[tuple[int, bytes]] = queue.Queue()
        self._lock = threading.Lock()
        self._generation = 0
        self._pending = b""  # unconsumed remainder of the current chunk

        self._last_level = 0.0
        self._level_lock = threading.Lock()

    # -- Public API --

    async def start(self) -> None:
        """Open the speaker output stream; PortAudio pulls audio via callback."""
        self._stream = self._stream_factory(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=self._device_index,
            latency="low",
            callback=self._audio_callback,
        )
        self._stream.start()
        self._active = True
        logger.info("[SPK] Speaker active")

    async def stop(self) -> None:
        """Stop and close the speaker output stream."""
        if self._stream is None:
            return

        self._active = False
        try:
            self._stream.stop()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Speaker stop failed: %s", exc)
        try:
            self._stream.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Speaker close failed: %s", exc)
        self._stream = None
        logger.info("[SPK] Speaker stopped")

    def write(self, pcm_bytes: bytes) -> None:
        """Queue PCM audio for playback. Never blocks the async loop."""
        if not self._active:
            return

        self._update_level(pcm_bytes)
        with self._lock:
            generation = self._generation
        self._queue.put((generation, pcm_bytes))

    def flush(self) -> None:
        """Drop all pending audio immediately, e.g. on barge-in.

        The generation bump makes the callback discard every queued chunk and
        the partially consumed remainder, so the stream falls silent within
        one callback period — no abort/close/recreate of the native stream.
        """
        with self._lock:
            self._generation += 1
        self._pending = b""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._level_lock:
            self._last_level = 0.0
        logger.info("[SPK] Playback flushed (barge-in)")

    @property
    def is_playing(self) -> bool:
        """Return whether the speaker output stream is active."""
        return self._active

    @property
    def level(self) -> float:
        """Return the most recent output level, normalised to ``[0.0, 1.0]``."""
        with self._level_lock:
            return self._last_level

    # -- PortAudio callback (real-time thread; never blocks) --

    def _audio_callback(self, outdata: Any, frames: int, _time: Any, status: Any) -> None:
        """Fill ``outdata`` with queued PCM, or silence when nothing is queued."""
        if status:
            logger.warning("Speaker callback status: %s", status)

        need = frames * _BYTES_PER_FRAME
        pos = 0
        while pos < need:
            if not self._pending:
                try:
                    generation, chunk = self._queue.get_nowait()
                except queue.Empty:
                    break  # underrun: the rest of the buffer stays silent
                if generation != self._generation or not self._active:
                    continue
                self._pending = chunk
            take = min(need - pos, len(self._pending))
            outdata[pos : pos + take] = self._pending[:take]
            self._pending = self._pending[take:]
            pos += take

        if pos < need:
            outdata[pos:need] = b"\x00" * (need - pos)

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
