"""Conversation-level orchestrator that survives WebSocket reconnects.

A ``Supervisor`` owns the long-lived resources that must persist across
resumptions: the ``ToolDispatcher``, ``TurnState``, and media objects. It builds
a new ``LiveSession`` for each WebSocket epoch and handles resumption handles,
GoAway, failure budgets, and in-flight tool results.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from datetime import UTC, datetime, timedelta
from typing import Any

from google.genai import types

from audit.journal import Journal
from core.commands import AgentCommand
from core.dispatcher import ToolDispatcher
from core.events import (
    AudioLevel,
    ConnectionState,
    ConnectionStateChanged,
    ContextReset,
    DisconnectReason,
)
from core.session import LiveSession
from core.turn_state import TurnState
from media import devices
from media.microphone import Microphone
from media.screen import ScreenCapture
from media.speaker import Speaker
from obs.logging_setup import get_logger
from persist.store import Store
from policy import PolicyEngine
from settings.settings import AppSettings
from tools.registry import ToolResult

logger = get_logger(__name__)

_FAILURE_WINDOW = timedelta(minutes=5)
_HANDLE_TTL_MINUTES = 110
_MAX_BACKOFF_TRANSIENT = 30.0
_MAX_BACKOFF_QUOTA = 120.0
_METER_INTERVAL_SECONDS = 0.033
_BUDGETS: dict[str, int] = {
    "transient": 6,
    "rejected_handle": 6,
    "quota": 3,
}


class Supervisor:
    """Reconnect-capable conversation manager for the Gemini Live API.

    Parameters
    ----------
    settings:
        Runtime configuration.
    command_queue:
        Queue of commands from the GUI/driver.
    event_queue:
        Queue of events published to the GUI/driver.
    journal:
        Audit journal.
    store:
        Persistent SQLite store for turns, tool calls, and resumption handles.
    run_id:
        Identifier for this run.
    """

    def __init__(
        self,
        settings: AppSettings,
        command_queue: asyncio.Queue[AgentCommand],
        event_queue: asyncio.Queue[Any],
        journal: Journal,
        store: Store,
        run_id: str,
    ) -> None:
        self._settings = settings
        self._command_queue = command_queue
        self._event_queue = event_queue
        self._journal = journal
        self._store = store
        self._run_id = run_id

        self._epoch = 0
        self._resumption_handle: str | None = None
        self._pending_tool_results: list[tuple[str, str, ToolResult]] = []
        self._failure_log: dict[str, list[datetime]] = {}
        self._live_session: LiveSession | None = None

        self._mic: Microphone | None = None
        self._speaker: Speaker | None = None
        self._screen: ScreenCapture | None = None
        self._media_started = False

        self._turn_state = TurnState()
        self._dispatcher = ToolDispatcher(
            policy=PolicyEngine(
                allow_roots=[self._settings.workspace_root],
                deny_globs=[r"*\.git*", "__pycache__"],
                never_allow_regex=[
                    r"AIza[0-9A-Za-z\-_]{35}",
                    r"sk-[A-Za-z0-9]{20,}",
                    r"ghp_[A-Za-z0-9]{36}",
                ],
            ),
            journal=self._journal,
            event_queue=self._event_queue,
            run_id=self._run_id,
            epoch=self._epoch,
            store=self._store,
        )

    async def run(self) -> DisconnectReason:
        """Run the main reconnect loop until the user disconnects or we fail."""
        self._resumption_handle = self._store.get_active_resumption_handle(self._run_id)
        self._dispatcher.set_respond_callback(self._respond_to_model)

        meter_task: asyncio.Task[Any] | None = None
        try:
            await self._start_media()
            if self._mic is not None or self._speaker is not None:
                meter_task = asyncio.create_task(self._meter_loop())

            while True:
                self._dispatcher.set_epoch(self._epoch)
                live_session = LiveSession(
                    settings=self._settings,
                    command_queue=self._command_queue,
                    event_queue=self._event_queue,
                    journal=self._journal,
                    store=self._store,
                    run_id=self._run_id,
                    epoch=self._epoch,
                    resumption_handle=self._resumption_handle,
                    mic=self._mic,
                    speaker=self._speaker,
                    screen=self._screen,
                    dispatcher=self._dispatcher,
                    turn_state=self._turn_state,
                    on_resumption_handle=self._on_resumption_handle,
                    on_connected=self._on_connected,
                )
                self._live_session = live_session
                reason = await live_session.run()
                self._live_session = None

                if reason.code == "user_request":
                    return reason

                if reason.code in ("auth_error", "not_found"):
                    detail = reason.detail or f"{reason.code}"
                    await self._emit(
                        ConnectionStateChanged(ConnectionState.FAILED, detail=detail)
                    )
                    return reason

                category = self._classify(reason)

                if category == "go_away":
                    await self._emit(
                        ConnectionStateChanged(
                            ConnectionState.DRAINING, detail=reason.detail
                        )
                    )
                    self._dispatcher.deny_all_pending(reason="connection_lost")
                elif category in ("transient", "rejected_handle", "api_close", "quota"):
                    self._dispatcher.deny_all_pending(reason="connection_lost")

                if category == "rejected_handle":
                    if self._resumption_handle is not None:
                        self._store.delete_resumption_handle(self._resumption_handle)
                    self._resumption_handle = None
                    await self._emit(
                        ContextReset(reason="resumption handle rejected")
                    )

                if not self._check_failure_budget(category):
                    msg = f"failure budget exceeded for {category}"
                    await self._emit(
                        ConnectionStateChanged(ConnectionState.FAILED, detail=msg)
                    )
                    return DisconnectReason("failed", msg)

                backoff = self._backoff(category)
                await self._emit(
                    ConnectionStateChanged(
                        ConnectionState.RECONNECTING, detail=reason.detail
                    )
                )
                if backoff > 0:
                    await asyncio.sleep(backoff)

                self._epoch += 1
        finally:
            self._live_session = None
            await self._emit(ConnectionStateChanged(ConnectionState.DISCONNECTED))
            if meter_task is not None:
                meter_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await meter_task
            await self._stop_media()

    async def _on_resumption_handle(self, handle: str | None, resumable: bool) -> None:
        """Persist or clear a server-provided resumption handle."""
        if handle is None:
            self._resumption_handle = None
            return
        if resumable:
            created_at = datetime.now(UTC).isoformat()
            expires_at = (
                datetime.now(UTC) + timedelta(minutes=_HANDLE_TTL_MINUTES)
            ).isoformat()
            # Delete first so re-saving an existing handle does not violate the
            # primary-key constraint if the server reuses a handle.
            self._store.delete_resumption_handle(handle)
            self._store.save_resumption_handle(
                handle, self._run_id, created_at, expires_at
            )
            self._resumption_handle = handle
        else:
            self._store.delete_resumption_handle(handle)
            self._resumption_handle = None

    async def _on_connected(self, live_session: LiveSession) -> None:
        """Seed context on a fresh connect and drain queued tool results."""
        if self._resumption_handle is None:
            prior_turns = self._store.get_turns(self._run_id)
            if prior_turns:
                await self._emit(
                    ContextReset(reason="no resumption handle; context seeded from store")
                )
                await live_session.send_client_content(
                    turns=[
                        types.Content(
                            role=role,
                            parts=[types.Part(text=text)],
                        )
                        for role, text, _epoch in prior_turns
                    ],
                    turn_complete=True,
                )

        pending = self._pending_tool_results
        self._pending_tool_results = []
        for call_id, name, result in pending:
            await self._deliver_tool_result(live_session, call_id, name, result)

    async def _respond_to_model(
        self, call_id: str, name: str, result: ToolResult
    ) -> bool:
        """Deliver a completed tool result to the current live session."""
        session = self._live_session
        if session is None:
            self._pending_tool_results.append((call_id, name, result))
            return False

        try:
            await session.send_tool_response(call_id, name, result)
            return True
        except Exception as exc:
            logger.debug("send_tool_response failed for %s: %s", call_id, exc)
            try:
                await self._send_tool_result_as_note(session, call_id, name, result)
                return True
            except Exception as exc2:  # pragma: no cover - defensive
                logger.debug("tool result fallback note failed for %s: %s", call_id, exc2)
                self._pending_tool_results.append((call_id, name, result))
                return False

    async def _deliver_tool_result(
        self,
        live_session: LiveSession,
        call_id: str,
        name: str,
        result: ToolResult,
    ) -> None:
        """Deliver a queued tool result to a newly connected session."""
        try:
            await live_session.send_tool_response(call_id, name, result)
        except Exception as exc:
            logger.debug("queued tool result failed for %s: %s", call_id, exc)
            await self._send_tool_result_as_note(live_session, call_id, name, result)

    async def _send_tool_result_as_note(
        self,
        live_session: LiveSession,
        call_id: str,
        name: str,
        result: ToolResult,
    ) -> None:
        """Send a tool result as a client text turn when the fc_id is rejected."""
        text = f"System note: tool result for {name} ({call_id}): {result.message}"
        await live_session.send_client_content(
            turns=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=text)],
                )
            ],
            turn_complete=True,
        )

    async def _start_media(self) -> None:
        """Create and start the mic, speaker, and screen if the config requires."""
        if self._media_started:
            return

        if (
            self._settings.mode != "text"
            and self._audio_input_available()
            and self._mic is None
        ):
            self._mic = Microphone(device=self._settings.input_device)
            await self._mic.start()

        if self._audio_output_available() and self._speaker is None:
            self._speaker = Speaker(device=self._settings.output_device)
            await self._speaker.start()

        if self._settings.share_screen and self._screen is None:
            self._screen = ScreenCapture(
                fps=self._settings.screen_fps,
                monitor=self._settings.screen_monitor,
            )
            await self._screen.start()

        self._media_started = True

    async def _stop_media(self) -> None:
        """Stop the media objects owned by the Supervisor."""
        if self._screen is not None:
            try:
                await self._screen.stop()
            except Exception as exc:
                logger.warning("Screen stop failed: %s", exc)
            self._screen = None

        if self._mic is not None:
            try:
                await self._mic.stop()
            except Exception as exc:
                logger.warning("Mic stop failed: %s", exc)
            self._mic = None

        if self._speaker is not None:
            try:
                await self._speaker.stop()
            except Exception as exc:
                logger.warning("Speaker stop failed: %s", exc)
            self._speaker = None

        self._media_started = False

    async def _meter_loop(self) -> None:
        """Poll media levels and emit ``AudioLevel`` events."""
        try:
            while True:
                if self._mic is not None:
                    await self._emit(
                        AudioLevel(source="mic", level=self._mic.level)
                    )
                if self._speaker is not None:
                    await self._emit(
                        AudioLevel(source="speaker", level=self._speaker.level)
                    )
                await asyncio.sleep(_METER_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            pass

    def _classify(self, reason: DisconnectReason) -> str:
        """Map a session disconnect reason to the supervisor category."""
        if reason.code in (
            "transient",
            "rejected_handle",
            "quota",
            "go_away",
            "api_close",
        ):
            return reason.code
        return "transient"

    def _record_failure(self, category: str) -> None:
        """Append a timestamped failure for ``category``."""
        now = datetime.now(UTC)
        window = [
            t
            for t in self._failure_log.get(category, [])
            if now - t < _FAILURE_WINDOW
        ]
        window.append(now)
        self._failure_log[category] = window

    def _check_failure_budget(self, category: str) -> bool:
        """Return ``True`` if the category is still within its failure budget."""
        self._record_failure(category)
        budget = _BUDGETS.get(category)
        if budget is None:
            return True
        return len(self._failure_log[category]) <= budget

    def _backoff(self, category: str) -> float:
        """Return the randomized backoff for the next reconnect attempt."""
        attempt = max(0, len(self._failure_log.get(category, [])) - 1)
        if category in ("transient", "go_away", "api_close", "rejected_handle"):
            return random.uniform(0, min(_MAX_BACKOFF_TRANSIENT, 0.5 * (2 ** attempt)))
        if category == "quota":
            return random.uniform(0, min(_MAX_BACKOFF_QUOTA, 5.0 * (2 ** attempt)))
        return 0.0

    def _audio_input_available(self) -> bool:
        try:
            return len(devices.list_input_devices()) > 0
        except Exception:
            return False

    def _audio_output_available(self) -> bool:
        try:
            return len(devices.list_output_devices()) > 0
        except Exception:
            return False

    async def _emit(self, event: Any) -> None:
        await self._event_queue.put(event)
