"""Qt thread bridge that runs the asyncio Supervisor in a QThread.

Events flow from the asyncio core to the main thread via Qt signals.
Commands flow from the GUI to the asyncio core via ``loop.call_soon_threadsafe``
into the command queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any

from PySide6.QtCore import QThread, Signal, Slot

from audit.journal import Journal
from core.commands import AgentCommand, Disconnect
from core.events import AgentEvent, AudioLevel
from core.supervisor import Supervisor
from obs.logging_setup import get_run_id
from persist.store import Store
from settings.settings import AppSettings


class Bridge(QThread):
    """QThread that owns the asyncio event loop and the reconnect-capable Supervisor.

    Parameters
    ----------
    settings:
        Runtime configuration used to build the Supervisor and connect config.
    run_id:
        Optional run identifier. A fresh UUID is generated if omitted.
    """

    # Forward core events to the main thread. The payload is always a frozen
    # AgentEvent dataclass from ``core.events``.
    event_emitted = Signal(object)

    # Emitted when the Supervisor loop ends. Carries the disconnect code string.
    finished_with_reason = Signal(str)

    def __init__(self, settings: AppSettings, run_id: str | None = None) -> None:
        super().__init__()
        self._settings = settings
        self._run_id = run_id or get_run_id()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._command_queue: asyncio.Queue[AgentCommand] | None = None
        self._event_queue: asyncio.Queue[Any] | None = None
        self._supervisor_task: asyncio.Task[Any] | None = None
        self._bridge_task: asyncio.Task[Any] | None = None
        self._journal: Journal | None = None
        self._store: Store | None = None

        # Commands sent before the thread's loop exists are buffered here and
        # drained into the queue during run(). Without this, commands fired
        # immediately after start() — like the GUI's initial SetGatingMode —
        # are silently dropped.
        self._pending_lock = threading.Lock()
        self._pending_commands: list[AgentCommand] = []
        self._ready = False

        self._mic_level = 0.0
        self._speaker_level = 0.0
        self._level_lock = threading.Lock()
        self._stopped = False

    @property
    def mic_level(self) -> float:
        """Latest microphone level, safe to read from the GUI thread."""
        with self._level_lock:
            return self._mic_level

    @property
    def speaker_level(self) -> float:
        """Latest speaker level, safe to read from the GUI thread."""
        with self._level_lock:
            return self._speaker_level

    def run(self) -> None:
        """Create the asyncio loop, start the Supervisor, and bridge events."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._command_queue = asyncio.Queue()
        self._event_queue = asyncio.Queue()
        self._journal = Journal(run_id=self._run_id)
        self._store = Store()

        # Flush commands that arrived before the loop existed, preserving order.
        with self._pending_lock:
            pending = self._pending_commands
            self._pending_commands = []
            self._ready = True
        for command in pending:
            self._command_queue.put_nowait(command)

        supervisor = Supervisor(
            settings=self._settings,
            command_queue=self._command_queue,
            event_queue=self._event_queue,
            journal=self._journal,
            store=self._store,
            run_id=self._run_id,
        )

        self._supervisor_task = self._loop.create_task(supervisor.run())
        self._bridge_task = self._loop.create_task(self._bridge_events())

        reason_code = "unknown"
        try:
            reason = self._loop.run_until_complete(self._supervisor_task)
            reason_code = reason.code
        except Exception as exc:  # pragma: no cover - defensive
            reason_code = f"error: {exc}"
        finally:
            self._cleanup(reason_code)

    async def _bridge_events(self) -> None:
        """Read from the event queue and emit or cache events."""
        queue = self._event_queue
        if queue is None:
            return
        while True:
            try:
                event: AgentEvent = await queue.get()
            except asyncio.CancelledError:
                break

            if isinstance(event, AudioLevel):
                self._update_level(event)
                continue

            self.event_emitted.emit(event)

    def _update_level(self, event: AudioLevel) -> None:
        with self._level_lock:
            if event.source == "mic":
                self._mic_level = event.level
            elif event.source == "speaker":
                self._speaker_level = event.level

    @Slot(object)
    def send_command(self, command: AgentCommand) -> None:
        """Forward a command from the GUI thread to the asyncio core."""
        if self._stopped and not isinstance(command, Disconnect):
            return
        with self._pending_lock:
            loop = self._loop
            queue = self._command_queue
            if not self._ready or loop is None or queue is None or loop.is_closed():
                self._pending_commands.append(command)
                return
        loop.call_soon_threadsafe(queue.put_nowait, command)

    def stop(self) -> None:
        """Request the Supervisor to disconnect.

        This only queues a ``Disconnect`` command; callers that need to block
        until the thread has exited must call ``wait()`` afterwards (or listen
        for ``finished_with_reason``).
        """
        self._stopped = True
        self.send_command(Disconnect())

    def _cleanup(self, reason_code: str) -> None:
        """Cancel the bridge task, drain remaining events, and close the loop."""
        loop = self._loop
        if loop is None or loop.is_closed():
            self.finished_with_reason.emit(reason_code)
            return

        if self._bridge_task is not None and not self._bridge_task.done():
            self._bridge_task.cancel()
            try:
                with contextlib.suppress(asyncio.CancelledError):
                    loop.run_until_complete(self._bridge_task)
            except Exception:  # pragma: no cover - loop may already be closed
                pass

        # Emit any events that were still in the queue when the supervisor ended.
        if self._event_queue is not None:
            while not self._event_queue.empty():
                event = self._event_queue.get_nowait()
                if isinstance(event, AudioLevel):
                    self._update_level(event)
                else:
                    self.event_emitted.emit(event)

        if self._journal is not None:
            self._journal.close()
        if self._store is not None:
            self._store.close()

        if loop is not None and not loop.is_closed():
            loop.close()

        self.finished_with_reason.emit(reason_code)
