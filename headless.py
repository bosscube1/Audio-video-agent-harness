"""Headless terminal driver for the Gemini Live Agent.

Consumes the core event stream and drives ``Supervisor`` from the terminal.
No GUI dependencies; plain ``print`` output only.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
from typing import Any

from audit.journal import Journal
from core.commands import (
    AgentCommand,
    ApproveTool,
    DenyTool,
    Disconnect,
    GatingMode,
    SendText,
    SetGatingMode,
)
from core.events import (
    AgentEvent,
    AudioLevel,
    ConnectionStateChanged,
    ContextReset,
    PartialTranscript,
    SessionError,
    ToolApprovalRequested,
    ToolCallCancelled,
    ToolCallReceived,
    ToolResultSent,
    TranscriptSource,
    TurnComplete,
    UsageUpdate,
)
from core.supervisor import Supervisor
from obs.logging_setup import get_run_id, setup_logging
from persist.store import Store
from settings.settings import AppSettings


class _TerminalState:
    """Mutable terminal rendering state used while printing events."""

    def __init__(self) -> None:
        self.current_source: TranscriptSource | None = None
        self.line_started: bool = False
        self.pending_call_id: str | None = None


def _start_stdin_thread(
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[str],
    stop_event: threading.Event,
) -> threading.Thread:
    """Start a daemon thread that pushes stdin lines to ``queue`` thread-safely."""

    def _reader() -> None:
        while not stop_event.is_set():
            try:
                line = sys.stdin.readline()
            except (OSError, ValueError):
                break
            if not line:
                break
            loop.call_soon_threadsafe(queue.put_nowait, line.rstrip("\n").rstrip("\r"))

    thread = threading.Thread(target=_reader, daemon=True, name="stdin-reader")
    thread.start()
    return thread


def _finish_line(state: _TerminalState) -> None:
    if state.line_started:
        print()
        state.line_started = False
        state.current_source = None


def _print_event(event: AgentEvent, state: _TerminalState) -> None:
    if isinstance(event, ConnectionStateChanged):
        _finish_line(state)
        print(f"[STATE] {event.state.value}")
    elif isinstance(event, PartialTranscript):
        prefix = "[USER] " if event.source == TranscriptSource.USER else "[MODEL] "
        if state.line_started and state.current_source == event.source:
            print(event.text, end="", flush=True)
        else:
            _finish_line(state)
            print(f"{prefix}{event.text}", end="", flush=True)
            state.line_started = True
            state.current_source = event.source
    elif isinstance(event, TurnComplete):
        _finish_line(state)
    elif isinstance(event, ToolCallReceived):
        _finish_line(state)
        print(f"[TOOL] {event.name}({event.args})")
    elif isinstance(event, ToolApprovalRequested):
        state.pending_call_id = event.call_id
        _finish_line(state)
        print(f"[TOOL] {event.name}({event.args})")
        print("Allow? [y/N]: ", end="", flush=True)
    elif isinstance(event, ToolResultSent):
        _finish_line(state)
        marker = "[OK]" if event.ok else "[X]"
        print(f"{marker} {event.result}")
    elif isinstance(event, ToolCallCancelled):
        _finish_line(state)
        print("[CANCELLED]")
    elif isinstance(event, SessionError):
        _finish_line(state)
        marker = "[FATAL ERROR]" if event.fatal else "[ERROR]"
        print(f"{marker} {event.message}")
    elif isinstance(event, ContextReset):
        _finish_line(state)
        print(f"[RESET] {event.reason}")
    elif isinstance(event, AudioLevel):
        # Briefly ignored in headless mode.
        return
    elif isinstance(event, UsageUpdate):
        # Not surfaced in the terminal UI.
        return


async def _drain_events(event_queue: asyncio.Queue[AgentEvent], state: _TerminalState) -> None:
    while not event_queue.empty():
        event = event_queue.get_nowait()
        _print_event(event, state)


async def _interactive_loop(
    settings: AppSettings,
    command_queue: asyncio.Queue[AgentCommand],
    event_queue: asyncio.Queue[AgentEvent],
    stdin_queue: asyncio.Queue[str],
    session_task: asyncio.Task[Any],
) -> int:
    """Run the terminal event loop until the session ends."""
    state = _TerminalState()

    while not session_task.done():
        event_get = asyncio.create_task(event_queue.get())
        stdin_get = asyncio.create_task(stdin_queue.get())

        done, pending = await asyncio.wait(
            {session_task, event_get, stdin_get},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Never cancel the session task; only cancel the local wait tasks.
        for task in pending:
            if task is not session_task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        if session_task in done:
            break

        if event_get in done:
            event = event_get.result()
            _print_event(event, state)
            continue

        if stdin_get in done:
            line = stdin_get.result().strip()
            if line == "":
                continue

            lowered = line.lower()
            if lowered in {"exit", "quit", "/exit", "/quit"}:
                await command_queue.put(Disconnect())
                with contextlib.suppress(asyncio.CancelledError):
                    await session_task
                break

            if state.pending_call_id is not None:
                if lowered in {"y", "yes"}:
                    await command_queue.put(ApproveTool(call_id=state.pending_call_id))
                else:
                    await command_queue.put(
                        DenyTool(call_id=state.pending_call_id, reason="user_denied")
                    )
                state.pending_call_id = None
                continue

            if settings.mode != "voice":
                await command_queue.put(SendText(text=line))

    await _drain_events(event_queue, state)

    if session_task.cancelled():
        return 0

    exc = session_task.exception()
    if exc is not None:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1

    reason = session_task.result()
    return 0 if reason.code == "user_request" else 1


async def run_headless(settings: AppSettings) -> int:
    """Run the agent in headless terminal mode.

    Returns 0 for a clean exit and 1 for a fatal session error.
    """
    run_id = get_run_id()
    setup_logging(run_id=run_id, debug=settings.debug)

    journal = Journal(run_id=run_id)
    store = Store()

    command_queue: asyncio.Queue[AgentCommand] = asyncio.Queue()
    event_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

    loop = asyncio.get_running_loop()
    stdin_queue: asyncio.Queue[str] = asyncio.Queue()
    stop_stdin = threading.Event()
    _start_stdin_thread(loop, stdin_queue, stop_stdin)

    if settings.yolo:
        await command_queue.put(SetGatingMode(mode=GatingMode.YOLO))

    session_task = asyncio.create_task(
        Supervisor(
            settings=settings,
            command_queue=command_queue,
            event_queue=event_queue,
            journal=journal,
            store=store,
            run_id=run_id,
        ).run()
    )

    try:
        return await _interactive_loop(
            settings=settings,
            command_queue=command_queue,
            event_queue=event_queue,
            stdin_queue=stdin_queue,
            session_task=session_task,
        )
    except KeyboardInterrupt:
        await command_queue.put(Disconnect())
        with contextlib.suppress(asyncio.CancelledError):
            await session_task
        await _drain_events(event_queue, _TerminalState())
        return 0
    finally:
        stop_stdin.set()
