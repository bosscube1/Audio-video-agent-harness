"""Single-epoch WebSocket driver for the Gemini Live API.

A ``LiveSession`` owns one Google Live API connection, the media send/receive
loops, and (optionally) a command dispatcher for that single epoch. It is
intentionally a single-socket object; resumption and reconnect logic live in
``core.supervisor.Supervisor``.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import google.genai as genai
import google.genai.errors
from google.genai import types

from audit.journal import Journal
from core.commands import (
    AgentCommand,
    ApproveTool,
    CancelTool,
    DenyTool,
    Disconnect,
    SendText,
    SetGatingMode,
    SetMicGate,
    SetShareScreen,
)
from core.dispatcher import ToolDispatcher
from core.events import (
    API_CLOSE,
    USER_REQUEST,
    AudioLevel,
    ConnectionState,
    ConnectionStateChanged,
    DisconnectReason,
    SessionError,
    SessionExpiring,
    TurnComplete,
    UsageUpdate,
)
from core.turn_state import TurnState
from media import devices
from media.microphone import INPUT_SAMPLE_RATE, Microphone
from media.screen import ScreenCapture
from media.speaker import Speaker
from obs.logging_setup import get_logger
from persist.store import Store
from policy import PolicyEngine
from settings.secrets import get_api_key
from settings.settings import AppSettings
from tools import build_function_declarations
from tools._state import set_policy_engine, set_workspace_root
from tools.registry import ToolResult

logger = get_logger(__name__)

_MAX_VIDEO_SEND_ERRORS = 5
_METER_INTERVAL_SECONDS = 0.033


class LiveSession:
    """Single-session manager for a Gemini Live connection.

    Parameters
    ----------
    settings:
        Runtime settings used to build the connect config and media objects.
    command_queue:
        Queue of commands from the GUI/driver.
    event_queue:
        Queue of events published to the GUI/driver.
    journal:
        Audit journal for tool attempts and results.
    store:
        Persistent store for run state (used by later phases).
    run_id:
        Identifier for this run.
    epoch:
        Resumption epoch (always ``0`` in Phase 2).
    resumption_handle:
        Optional handle to request session resumption from the API.
    mic:
        Optional injected microphone. If provided, the session will not start or
        stop it.
    speaker:
        Optional injected speaker. If provided, the session will not start or stop
        it.
    screen:
        Optional injected screen capture. If provided, the session will not start or
        stop it.
    dispatcher:
        Optional injected tool dispatcher. If provided, the session will not create
        one and will not set its respond callback.
    turn_state:
        Optional injected turn state. If provided, the session will not create one.
    on_resumption_handle:
        Optional callback invoked when the server updates resumption availability.
    on_connected:
        Optional callback invoked after the session connects successfully.
    respond_callback:
        Optional callback used to send tool responses when a dispatcher is created
        by the session.
    """

    def __init__(
        self,
        settings: AppSettings,
        command_queue: asyncio.Queue[AgentCommand],
        event_queue: asyncio.Queue[Any],
        journal: Journal,
        store: Store,
        run_id: str,
        epoch: int = 0,
        *,
        resumption_handle: str | None = None,
        mic: Microphone | None = None,
        speaker: Speaker | None = None,
        screen: ScreenCapture | None = None,
        dispatcher: ToolDispatcher | None = None,
        turn_state: TurnState | None = None,
        on_resumption_handle: Callable[[str, bool], Awaitable[None]] | None = None,
        on_connected: Callable[[LiveSession], Awaitable[None]] | None = None,
        respond_callback: Callable[[str, str, ToolResult], Awaitable[bool]] | None = None,
    ) -> None:
        self._settings = settings
        self._command_queue = command_queue
        self._event_queue = event_queue
        self._journal = journal
        self._store = store
        self._run_id = run_id
        self._epoch = epoch
        self._resumption_handle = resumption_handle

        self._turn_state = turn_state if turn_state is not None else TurnState()
        self._dispatcher = dispatcher

        self._mic = mic
        self._speaker = speaker
        self._screen = screen

        self._on_resumption_handle = on_resumption_handle
        self._on_connected = on_connected
        self._respond_callback = respond_callback

        self._owns_mic = mic is None
        self._owns_speaker = speaker is None
        self._owns_screen = screen is None
        self._owns_dispatcher = dispatcher is None
        self._owns_turn_state = turn_state is None

        self._session: Any | None = None
        self._latest_resumption_handle: str | None = None
        self._closed = False

        self._video_send_task: asyncio.Task[Any] | None = None
        self._shutdown_event = asyncio.Event()
        self._mic_gate_open = True

    async def run(self) -> DisconnectReason:
        """Connect to the API, spawn loops, and return when the session ends."""
        policy = PolicyEngine(
            allow_roots=[self._settings.workspace_root],
            deny_globs=[r"*\.git\*", "__pycache__"],
            never_allow_regex=[
                r"AIza[0-9A-Za-z\-_]{35}",
                r"sk-[A-Za-z0-9]{20,}",
                r"ghp_[A-Za-z0-9]{36}",
            ],
        )
        set_policy_engine(policy)
        set_workspace_root(self._settings.workspace_root)

        api_key = get_api_key()
        if api_key is None:
            raise ValueError("No API key configured")

        client = genai.Client(api_key=api_key)
        connect_config = self._settings.as_connect_config(
            tools=list(build_function_declarations()),
            resumption_handle=self._resumption_handle,
        )

        await self._emit(ConnectionStateChanged(ConnectionState.CONNECTING))

        try:
            async with client.aio.live.connect(
                model=self._settings.model,
                config=connect_config,
            ) as session:
                self._session = session
                await self._emit(ConnectionStateChanged(ConnectionState.LIVE))
                self._journal.connection_event(
                    "connected", self._settings.model, epoch=self._epoch
                )
                if self._on_connected is not None:
                    await self._on_connected(self)

                reason = await self._run_session(session, policy)
                return reason

        except Exception as exc:
            logger.exception("Session failed")
            reason = self._classify_disconnect(exc)
            await self._emit(
                ConnectionStateChanged(
                    ConnectionState.ERROR,
                    detail=reason.detail,
                )
            )
            return reason
        finally:
            await self._emit(ConnectionStateChanged(ConnectionState.DISCONNECTED))
            self._journal.connection_event("disconnected", "", epoch=self._epoch)
            self._session = None
            self._closed = True

    async def _run_session(
        self, session: Any, policy: PolicyEngine
    ) -> DisconnectReason:
        """Start media and the concurrent send/receive loops."""
        if self._dispatcher is None:
            self._dispatcher = ToolDispatcher(
                policy=policy,
                journal=self._journal,
                event_queue=self._event_queue,
                run_id=self._run_id,
                epoch=self._epoch,
                store=self._store,
            )
            respond_callback = self._respond_callback
            if respond_callback is None:
                respond_callback = self._make_respond_callback(session)
            self._dispatcher.set_respond_callback(respond_callback)

        try:
            if self._audio_output_available() and self._speaker is None:
                self._speaker = Speaker(device=self._settings.output_device)
                await self._speaker.start()

            if (
                self._settings.mode != "text"
                and self._audio_input_available()
                and self._mic is None
            ):
                self._mic = Microphone(device=self._settings.input_device)
                await self._mic.start()

            if self._settings.share_screen and self._screen is None:
                self._screen = ScreenCapture(
                    fps=self._settings.screen_fps,
                    monitor=self._settings.screen_monitor,
                )
                await self._screen.start()

            tasks: set[asyncio.Task[Any]] = set()
            tasks.add(
                asyncio.create_task(
                    self._receive_loop(session),
                    name="receive",
                )
            )
            tasks.add(
                asyncio.create_task(
                    self._send_loop(session),
                    name="send",
                )
            )

            if self._mic is not None:
                tasks.add(
                    asyncio.create_task(
                        self._audio_send_loop(session, self._mic),
                        name="audio_send",
                    )
                )
            if self._screen is not None:
                self._video_send_task = asyncio.create_task(
                    self._video_send_loop(session, self._screen),
                    name="video_send",
                )
                tasks.add(self._video_send_task)
            if (
                self._mic is not None or self._speaker is not None
            ) and (self._owns_mic or self._owns_speaker):
                tasks.add(
                    asyncio.create_task(
                        self._meter_loop(),
                        name="meter",
                    )
                )

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            reason: DisconnectReason | None = None
            for task in done:
                exc = task.exception()
                if exc is not None:
                    reason = self._classify_disconnect(exc)
                    break
                result = task.result()
                if isinstance(result, DisconnectReason):
                    reason = result

            if reason is None:
                reason = API_CLOSE

            # Signal the remaining loops to exit, then cancel them.
            self._shutdown_event.set()
            await self._stop_media()

            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            # Flush any partial transcript that arrived just before close.
            for event in self._turn_state.flush():
                await self._emit(event)

            return reason

        finally:
            await self._stop_media()

    async def _receive_loop(self, session: Any) -> DisconnectReason | None:
        """Consume server responses and dispatch side effects.

        ``session.receive()`` returns at the end of each model turn, so the
        outer ``while`` re-enters it for the next turn. This matches the
        Gemini Live SDK behavior and the original ``agent.py`` loop.
        """
        try:
            while not self._shutdown_event.is_set():
                async for response in session.receive():
                    if self._shutdown_event.is_set():
                        break

                    resumption_update = response.session_resumption_update
                    if resumption_update is not None:
                        if resumption_update.resumable:
                            self._latest_resumption_handle = resumption_update.new_handle
                            if self._on_resumption_handle is not None:
                                await self._on_resumption_handle(
                                    resumption_update.new_handle, True
                                )
                        else:
                            if self._on_resumption_handle is not None:
                                await self._on_resumption_handle(
                                    resumption_update.new_handle, False
                                )

                    go_away = response.go_away
                    if go_away is not None:
                        time_left = go_away.time_left
                        seconds = self._parse_time_left(time_left)
                        await self._emit(SessionExpiring(seconds=seconds))
                        return DisconnectReason("go_away", time_left)

                    server_content = response.server_content
                    if server_content:
                        # Barge-in: the server detected user speech and aborted
                        # the model turn. Kill queued playback first, before any
                        # other handling, so the model stops mid-sentence rather
                        # than finishing the response it already generated.
                        if server_content.interrupted and self._speaker is not None:
                            self._speaker.flush()

                        events = self._turn_state.consume_server_content(server_content)
                        for event in events:
                            await self._emit(event)
                            if isinstance(event, TurnComplete):
                                completed_at = datetime.now(UTC).isoformat()
                                unique_id = (
                                    f"{self._run_id}-{self._epoch}-{event.turn_id}"
                                )
                                self._store.append_turn(
                                    unique_id,
                                    self._run_id,
                                    event.source.value,
                                    event.text,
                                    completed_at,
                                    self._epoch,
                                )

                        # Play audio contained in the model turn.
                        model_turn = server_content.model_turn
                        if model_turn and model_turn.parts and self._speaker is not None:
                            for part in model_turn.parts:
                                inline_data = getattr(part, "inline_data", None)
                                if inline_data and inline_data.data:
                                    self._speaker.write(inline_data.data)

                    # Tool calls are handled off the receive path so approvals and
                    # blocking operations cannot stall message handling.
                    tool_call = response.tool_call
                    if tool_call and tool_call.function_calls:
                        for fc in tool_call.function_calls:
                            if self._dispatcher is not None:
                                asyncio.create_task(
                                    self._dispatcher.submit(
                                        fc.id,
                                        fc.name,
                                        fc.args or {},
                                    )
                                )

                    cancellation = response.tool_call_cancellation
                    if cancellation and self._dispatcher is not None:
                        self._dispatcher.cancel(cancellation.ids or [])

                    usage = response.usage_metadata
                    if usage:
                        # genai >= 2.x: output tokens are ``response_token_count``
                        # and per-modality counts live in the *_tokens_details
                        # lists. Older field names (completion/audio/video)
                        # silently miss, so read the real ones.
                        def _modality_tokens(details: Any, name: str) -> int:
                            return sum(
                                getattr(d, "token_count", 0) or 0
                                for d in (details or [])
                                if str(getattr(d, "modality", "")).upper() == name
                            )

                        prompt_details = getattr(usage, "prompt_tokens_details", None)
                        response_details = getattr(
                            usage, "response_tokens_details", None
                        )
                        audio_tokens = _modality_tokens(
                            prompt_details, "AUDIO"
                        ) + _modality_tokens(response_details, "AUDIO")
                        video_tokens = _modality_tokens(
                            prompt_details, "VIDEO"
                        ) + _modality_tokens(response_details, "VIDEO")
                        usage_event = UsageUpdate(
                            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                            cached_tokens=getattr(
                                usage, "cached_content_token_count", 0
                            )
                            or 0,
                            completion_tokens=getattr(
                                usage, "response_token_count", 0
                            )
                            or 0,
                            audio_tokens=audio_tokens,
                            video_tokens=video_tokens,
                            total_tokens=getattr(usage, "total_token_count", 0) or 0,
                        )
                        await self._emit(usage_event)
                        self._record_usage(usage_event)

        except asyncio.CancelledError:
            return None
        except Exception as exc:
            logger.exception("Receive loop failed")
            await self._emit(SessionError(message=str(exc), fatal=True))
            return self._classify_disconnect(exc)

        return API_CLOSE

    async def _send_loop(self, session: Any) -> DisconnectReason | None:
        """Consume commands from the GUI/driver and forward them to the API."""
        try:
            while True:
                cmd = await self._command_queue.get()

                if isinstance(cmd, Disconnect):
                    return USER_REQUEST

                if isinstance(cmd, SendText):
                    # TODO(Gemini 3.1): the docs suggest live text input may need
                    # to go through ``send_realtime_input`` rather than
                    # ``send_client_content``. Verify against a live socket.
                    await session.send_client_content(
                        turns=[
                            types.Content(
                                role="user",
                                parts=[types.Part(text=cmd.text)],
                            )
                        ],
                        turn_complete=True,
                    )

                elif isinstance(cmd, SetMicGate):
                    self._mic_gate_open = cmd.open
                    if self._mic is not None:
                        self._mic.set_gate(cmd.open)

                elif isinstance(cmd, SetShareScreen):
                    if cmd.enabled:
                        if self._screen is None:
                            self._screen = ScreenCapture(
                                fps=self._settings.screen_fps,
                                monitor=cmd.monitor
                                if cmd.monitor is not None
                                else self._settings.screen_monitor,
                            )
                            self._owns_screen = True
                            await self._screen.start()
                            self._video_send_task = asyncio.create_task(
                                self._video_send_loop(session, self._screen),
                                name="video_send_dynamic",
                            )
                        elif self._video_send_task is None or self._video_send_task.done():
                            self._video_send_task = asyncio.create_task(
                                self._video_send_loop(session, self._screen),
                                name="video_send_dynamic",
                            )
                    else:
                        if self._video_send_task is not None and not self._video_send_task.done():
                            self._video_send_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await self._video_send_task
                            self._video_send_task = None
                        if self._screen is not None and self._owns_screen:
                            await self._screen.stop()
                            self._screen = None

                elif isinstance(cmd, ApproveTool):
                    if self._dispatcher is not None:
                        self._dispatcher.approve(cmd.call_id)

                elif isinstance(cmd, DenyTool):
                    if self._dispatcher is not None:
                        self._dispatcher.deny(cmd.call_id, reason=cmd.reason)

                elif isinstance(cmd, CancelTool):
                    if self._dispatcher is not None:
                        self._dispatcher.cancel(cmd.call_ids)

                elif isinstance(cmd, SetGatingMode):
                    if self._dispatcher is not None:
                        self._dispatcher.set_gating_mode(cmd.mode)
                    self._journal.gating_mode_change(
                        cmd.mode.value,
                        duration_minutes=cmd.duration_minutes,
                    )

                else:
                    logger.warning("Ignored unknown command type: %s", type(cmd))

        except asyncio.CancelledError:
            return None
        except Exception as exc:
            logger.exception("Send loop failed")
            await self._emit(SessionError(message=str(exc), fatal=True))
            return self._classify_disconnect(exc)

    async def _audio_send_loop(self, session: Any, mic: Microphone) -> None:
        """Stream microphone PCM to the model."""
        try:
            async for chunk in mic.chunks():
                if self._shutdown_event.is_set():
                    break
                if not self._mic_gate_open:
                    continue
                try:
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=chunk,
                            mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                        )
                    )
                except Exception as exc:
                    logger.warning("Audio send failed: %s", exc)
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.exception("Audio send loop failed")
            await self._emit(SessionError(message=str(exc), fatal=False))

    async def _video_send_loop(self, session: Any, screen: ScreenCapture) -> None:
        """Stream screen JPEG frames to the model."""
        consecutive_errors = 0
        try:
            while not self._shutdown_event.is_set():
                try:
                    frame = await screen.read_frame()
                    await session.send_realtime_input(
                        video=types.Blob(data=frame, mime_type="image/jpeg")
                    )
                    consecutive_errors = 0
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    consecutive_errors += 1
                    logger.warning(
                        "Video send failed (%d/%d): %s",
                        consecutive_errors,
                        _MAX_VIDEO_SEND_ERRORS,
                        exc,
                    )
                    if consecutive_errors >= _MAX_VIDEO_SEND_ERRORS:
                        logger.error(
                            "Too many video send errors; stopping screen share."
                        )
                        break
                    await asyncio.sleep(1.0)
        except Exception as exc:
            logger.exception("Video send loop failed")
            await self._emit(SessionError(message=str(exc), fatal=False))

    async def _meter_loop(self) -> None:
        """Poll media levels and emit ``AudioLevel`` events."""
        try:
            while True:
                if self._shutdown_event.is_set():
                    return

                if self._mic is not None:
                    await self._emit(
                        AudioLevel(source="mic", level=self._mic.level)
                    )
                if self._speaker is not None:
                    await self._emit(
                        AudioLevel(source="speaker", level=self._speaker.level)
                    )

                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=_METER_INTERVAL_SECONDS,
                    )
        except asyncio.CancelledError:
            pass

    def _make_respond_callback(
        self, session: Any
    ) -> Callable[[str, str, ToolResult], Awaitable[bool]]:
        """Return the callback used by ``ToolDispatcher`` to reply to tool calls."""

        async def respond(call_id: str, name: str, result: ToolResult) -> bool:
            await session.send_tool_response(
                function_responses=[
                    types.FunctionResponse(
                        id=call_id,
                        name=name,
                        response={"result": result.message},
                    )
                ]
            )
            return True

        return respond

    def _classify_disconnect(self, exc: BaseException) -> DisconnectReason:
        """Map an exception to a structured disconnect reason."""
        detail = str(exc)
        if isinstance(exc, google.genai.errors.APIError):
            status = getattr(exc, "status", None)
            code = getattr(exc, "code", None)
            if status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
                return DisconnectReason("auth_error", detail)
            if status == "NOT_FOUND":
                return DisconnectReason("not_found", detail)
            if status == "RESOURCE_EXHAUSTED" or code == 429:
                return DisconnectReason("quota", detail)
            if status in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED", "UNKNOWN"):
                return DisconnectReason("transient", detail)

        lowered = detail.lower()
        if "resumption handle" in lowered and "rejected" in lowered:
            return DisconnectReason("rejected_handle", detail)

        if isinstance(exc, (ConnectionError, OSError, TimeoutError, asyncio.TimeoutError)):
            return DisconnectReason("transient", detail)

        return DisconnectReason("transient", detail)

    def _parse_time_left(self, time_left: str | None) -> int | None:
        """Parse a duration string such as ``'60s'`` into seconds."""
        if time_left is None:
            return None
        match = re.search(r"\d+", time_left)
        if match is None:
            return None
        return int(match.group())

    async def send_client_content(
        self, turns: list[types.Content], turn_complete: bool = True
    ) -> None:
        """Forward client content to the active API session."""
        if self._session is None:
            raise RuntimeError("not connected")
        await self._session.send_client_content(
            turns=turns, turn_complete=turn_complete
        )

    async def send_tool_response(self, call_id: str, name: str, result: ToolResult) -> None:
        """Send a tool response to the active API session."""
        if self._session is None:
            raise RuntimeError("not connected")
        await self._session.send_tool_response(
            function_responses=[
                types.FunctionResponse(
                    id=call_id,
                    name=name,
                    response={"result": result.message},
                )
            ]
        )

    async def _emit(self, event: Any) -> None:
        await self._event_queue.put(event)

    def _record_usage(self, event: UsageUpdate) -> None:
        """Persist non-zero usage modalities as incremental history rows."""
        recorded_at = datetime.now(UTC).isoformat()
        modalities = {
            "prompt": event.prompt_tokens,
            "cached": event.cached_tokens,
            "completion": event.completion_tokens,
            "audio": event.audio_tokens,
            "video": event.video_tokens,
        }
        for modality, tokens in modalities.items():
            if tokens:
                self._store.record_usage(
                    self._run_id, modality, tokens, event.estimated_usd, recorded_at
                )

    async def _stop_media(self) -> None:
        """Stop microphone, speaker, and screen capture if the session owns them."""
        if self._screen is not None and self._owns_screen:
            try:
                await self._screen.stop()
            except Exception as exc:
                logger.warning("Screen stop failed: %s", exc)
            self._screen = None

        if self._mic is not None and self._owns_mic:
            try:
                await self._mic.stop()
            except Exception as exc:
                logger.warning("Mic stop failed: %s", exc)
            self._mic = None

        if self._speaker is not None and self._owns_speaker:
            try:
                await self._speaker.stop()
            except Exception as exc:
                logger.warning("Speaker stop failed: %s", exc)
            self._speaker = None

    def _audio_output_available(self) -> bool:
        try:
            return len(devices.list_output_devices()) > 0
        except Exception:
            return False

    def _audio_input_available(self) -> bool:
        try:
            return len(devices.list_input_devices()) > 0
        except Exception:
            return False


# Re-export so callers can import the reason type from this module as well.
__all__ = ["LiveSession"]
