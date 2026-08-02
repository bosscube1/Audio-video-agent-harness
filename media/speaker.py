"""Speaker playback for the Gemini Live Agent."""

from __future__ import annotations

import array
import contextlib
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


class Speaker:
    """Plays 24 kHz 16-bit mono PCM audio through the selected speaker.

    Playback runs on a dedicated writer thread so that ``write()`` never blocks
    the async event loop. A monotonic generation counter protects ``flush()``
    from races with queued ``write()`` calls: after ``flush()`` increments the
    generation, queued chunks tagged with the old generation are discarded by
    the writer.

    Thread-safety: ``self._lock`` serializes *all* stream access. The writer
    holds it for the whole check+write, and ``flush()``/``stop()`` hold it for
    the whole abort/close/recreate, so the native PortAudio stream is never
    torn down while another thread is inside ``stream.write()`` — closing an
    MME stream mid-write is a use-after-free that can kill the process without
    a Python traceback.
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
        self._queue: queue.Queue[tuple[int, bytes] | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._generation = 0

        self._last_level = 0.0
        self._level_lock = threading.Lock()

    # -- Public API --

    async def start(self) -> None:
        """Open the speaker output stream and start the writer thread."""
        self._stream = self._stream_factory(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            device=self._device_index,
        )
        self._stream.start()
        self._active = True
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
        logger.info("[SPK] Speaker active")

    async def stop(self) -> None:
        """Stop the writer thread and close the speaker output stream."""
        if self._stream is None:
            return

        self._active = False
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

        # Take the lock so a write that outlived the join finishes before the
        # stream is stopped and closed underneath it.
        with self._lock:
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

        Draining the queue alone is not enough to stop the model mid-sentence:
        PortAudio has already buffered whatever was handed to ``write()``. We
        abort the current stream, close it, and open a fresh one for the next
        turn. On Windows MME, restarting an aborted stream often fails with
        "media data is still playing", so we always recreate the stream
        instead.

        The whole teardown runs under ``self._lock``, which the writer thread
        also holds for the duration of its ``stream.write()`` call. An
        in-flight write therefore finishes before the stream is aborted and
        closed — tearing the stream down mid-write corrupts the native MME
        state and can crash the process outright.
        """
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        with self._lock:
            self._generation += 1
            stream = self._stream
            if stream is None or not self._active:
                return

            with contextlib.suppress(Exception):
                stream.abort()
            with contextlib.suppress(Exception):
                stream.close()

            try:
                new_stream = self._stream_factory(
                    samplerate=OUTPUT_SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    device=self._device_index,
                )
                new_stream.start()
                self._stream = new_stream
            except Exception as exc:  # pragma: no cover - device dependent
                logger.warning("Speaker stream recreate after flush failed: %s", exc)
                self._stream = None

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

    # -- Writer thread --

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:  # sentinel
                return

            generation, chunk = item
            # Hold the lock for the whole check+write so flush()/stop() cannot
            # tear the stream down mid-write.
            with self._lock:
                if (
                    not self._active
                    or generation != self._generation
                    or self._stream is None
                ):
                    continue
                try:
                    self._stream.write(chunk)
                except Exception as exc:  # pragma: no cover - real-time playback
                    logger.warning("Speaker write failed: %s", exc)

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
