"""Audio I/O for the Gemini Live Agent — microphone capture and speaker playback.

Input:  16 kHz · 16-bit · mono PCM  (what Gemini Live expects)
Output: 24 kHz · 16-bit · mono PCM  (what Gemini Live produces)
"""

import asyncio
import queue
import threading
from typing import Optional

from rich.console import Console

console = Console()

# ── Audio specifications for the Gemini Live API ──
INPUT_SAMPLE_RATE = 16_000  # 16 kHz for microphone → model
OUTPUT_SAMPLE_RATE = 24_000  # 24 kHz for model → speaker
CHANNELS = 1  # Mono
DTYPE = "int16"  # 16-bit signed PCM
CHUNK_DURATION_MS = 100  # Send audio in 100 ms chunks
INPUT_CHUNK_SAMPLES = int(INPUT_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


def check_audio_available() -> bool:
    """Return True if the system has both an input and output audio device."""
    try:
        import sounddevice as sd

        devices = list(sd.query_devices())
        has_input = any(d.get("max_input_channels", 0) > 0 for d in devices)
        has_output = any(d.get("max_output_channels", 0) > 0 for d in devices)
        return has_input and has_output
    except Exception:
        return False


# ─────────────────────────────────────────────
# Microphone Input
# ─────────────────────────────────────────────


class MicrophoneStream:
    """Captures audio from the default microphone as 16 kHz 16-bit mono PCM chunks.

    Uses an asyncio.Queue to bridge the sounddevice callback thread with
    the async event loop.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stream: Optional[object] = None  # sd.RawInputStream
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # -- sounddevice callback (runs on audio thread) --

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            console.print(f"[dim red]Mic warning: {status}[/dim red]", highlight=False)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, bytes(indata))

    # -- Public API --

    async def start(self) -> None:
        """Start capturing audio from the default microphone."""
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()
        self._stream = sd.RawInputStream(
            samplerate=INPUT_SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=INPUT_CHUNK_SAMPLES,
            callback=self._audio_callback,
        )
        self._stream.start()
        console.print("[dim green][MIC] Microphone active[/dim green]")

    async def read_chunk(self) -> bytes:
        """Read the next PCM audio chunk. Blocks until data is available."""
        return await self._queue.get()

    async def stop(self) -> None:
        """Stop capturing and release the microphone."""
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            console.print("[dim][MIC] Microphone stopped[/dim]")


# ─────────────────────────────────────────────
# Speaker Output
# ─────────────────────────────────────────────


class SpeakerStream:
    """Plays 24 kHz 16-bit mono PCM audio through the default speaker.

    Playback runs on a dedicated writer thread. `RawOutputStream.write` blocks
    until the device has buffer space, so calling it from the event loop would
    stall the receive loop (and therefore tool dispatch) while audio plays.
    """

    def __init__(self) -> None:
        self._stream: Optional[object] = None  # sd.RawOutputStream
        self._active = False
        self._queue: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    async def start(self) -> None:
        """Open the speaker output stream and start the writer thread."""
        import sounddevice as sd

        self._stream = sd.RawOutputStream(
            samplerate=OUTPUT_SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
        )
        self._stream.start()
        self._active = True
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
        console.print("[dim green][SPK] Speaker active[/dim green]")

    def _writer(self) -> None:
        """Drain the playback queue onto the device (runs on its own thread)."""
        while True:
            chunk = self._queue.get()
            if chunk is None:  # sentinel from stop()
                return
            if not self._active or self._stream is None:
                continue
            try:
                self._stream.write(chunk)
            except Exception:
                pass  # Swallow underrun / overrun during real-time playback

    def write(self, pcm_bytes: bytes) -> None:
        """Queue PCM audio for playback. Never blocks the async loop."""
        if self._active:
            self._queue.put(pcm_bytes)

    def flush(self) -> None:
        """Drop queued audio — called on barge-in to cut playback immediately."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.start()
            except Exception:
                pass

    async def stop(self) -> None:
        """Stop the writer thread and close the speaker output stream."""
        if self._stream is not None:
            self._active = False
            self._queue.put(None)
            if self._thread is not None:
                self._thread.join(timeout=2)
                self._thread = None
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            console.print("[dim][SPK] Speaker stopped[/dim]")
