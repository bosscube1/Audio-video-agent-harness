"""Screen capture for the Gemini Live Agent — streams the desktop as video.

The Live API takes realtime video as individual JPEG frames. Screen content
changes slowly compared to speech, so this captures at a low frame rate (1 fps
by default) and downscales before encoding — full-resolution frames burn
bandwidth and context for no gain in what the model can read.

Capture runs on its own thread: ``mss`` is blocking and its instances are bound
to the thread that created them, so it cannot share the default executor.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import threading
import time
from typing import Any

from obs.logging_setup import get_logger

logger = get_logger(__name__)


def _mss_factory(mss: Any) -> Any:
    """Return the screenshotter class. mss 10 renamed ``mss.mss`` to ``mss.MSS``."""
    return getattr(mss, "MSS", None) or mss.mss


DEFAULT_FPS = 1.0
DEFAULT_MAX_WIDTH = 1024  # downscale wider screens to this before encoding
DEFAULT_QUALITY = 70  # JPEG quality


class ScreenCapture:
    """Captures the desktop as JPEG frames at a fixed frame rate.

    Only the most recent frame is kept. If the sender falls behind, older
    frames are dropped rather than queued — stale screenshots are worse than
    no screenshot.
    """

    def __init__(
        self,
        fps: float = DEFAULT_FPS,
        monitor: int = 1,
        max_width: int = DEFAULT_MAX_WIDTH,
        quality: int = DEFAULT_QUALITY,
    ) -> None:
        self.fps = fps
        self.monitor = monitor
        self.max_width = max_width
        self.quality = quality
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=1)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    # -- Public API --

    async def start(self) -> None:
        """Begin capturing the screen."""
        import mss  # imported here so --help works without the dependency

        # Validate the monitor index up front rather than failing per-frame
        with _mss_factory(mss)() as sct:
            count = len(sct.monitors) - 1  # index 0 is the virtual "all monitors"
            if self.monitor < 0 or self.monitor > count:
                raise ValueError(
                    f"Monitor {self.monitor} does not exist "
                    f"(available: 0 for all, 1-{count} for individual screens)"
                )
            region = sct.monitors[self.monitor]

        self._loop = asyncio.get_running_loop()
        self._running = True
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()
        logger.info(
            "[SCREEN] Sharing monitor %d (%dx%d) at %g fps",
            self.monitor,
            region["width"],
            region["height"],
            self.fps,
        )

    async def read_frame(self) -> bytes:
        """Return the next JPEG frame. Blocks until one is available."""
        frame = await self._queue.get()
        if frame is None:
            raise asyncio.CancelledError("Screen capture stopped")
        return frame

    async def stop(self) -> None:
        """Stop capturing."""
        if self._running:
            self._running = False
            if self._loop is not None and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._enqueue_sentinel)
            if self._thread is not None:
                self._thread.join(timeout=2)
                self._thread = None
            logger.info("[SCREEN] Screen sharing stopped")

    # -- Capture thread --

    def _capture(self) -> None:
        import mss
        from PIL import Image

        interval = 1.0 / self.fps if self.fps > 0 else 1.0

        with _mss_factory(mss)() as sct:
            region = sct.monitors[self.monitor]
            while self._running:
                started = time.monotonic()
                try:
                    shot = sct.grab(region)
                    image = Image.frombytes(
                        "RGB", shot.size, shot.bgra, "raw", "BGRX"
                    )

                    if image.width > self.max_width:
                        height = round(image.height * self.max_width / image.width)
                        image = image.resize(
                            (self.max_width, height), Image.LANCZOS  # type: ignore[attr-defined]
                        )

                    buffer = io.BytesIO()
                    image.save(buffer, format="JPEG", quality=self.quality)
                    self._publish(buffer.getvalue())
                except Exception as exc:
                    logger.warning("Screen capture error: %s", exc)

                # Pace to the target frame rate, accounting for capture time
                elapsed = time.monotonic() - started
                if self._running and elapsed < interval:
                    time.sleep(interval - elapsed)

    def _publish(self, frame: bytes) -> None:
        """Hand a frame to the event loop, replacing any frame not yet sent."""
        if self._loop is None:
            return

        def put() -> None:
            if self._queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    self._queue.get_nowait()  # drop the stale frame
            self._queue.put_nowait(frame)

        try:
            self._loop.call_soon_threadsafe(put)
        except RuntimeError:
            logger.debug("Failed to publish screen frame: event loop closed")

    def _enqueue_sentinel(self) -> None:
        """Put a sentinel to unblock read_frame during shutdown."""
        if self._queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
        self._queue.put_nowait(None)
