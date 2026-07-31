"""WebSocket session orchestration for the Gemini Live API.

A ``LiveSession`` owns a single Google Live API connection, the media send/receive
loops, and the command dispatcher. It is intentionally a single-socket object;
resumption and reconnect logic belong to Phase 3.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any

import google.genai as genai
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
    ) -> None:
        self._settings = settings
        self._command_queue = command_queue
        self._event_queue = event_queue
        self._journal = journal
        self._store = store
        self._run_id = run_id
        self._epoch = epoch

        self._turn_state = TurnState()
        self._dispatcher: ToolDispatcher | None = None

        self._mic: Microphone | None = None
        self._speaker: Speaker | None = None
        self._screen: ScreenCapture | None = None

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
            tools=list(build_function_declarations())
        )

        await self._emit(ConnectionStateChanged(ConnectionState.CONNECTING))

        try:
            async with client.aio.live.connect(
                model=self._settings.model,
                config=connect_config,
            ) as session:
                await self._emit(ConnectionStateChanged(ConnectionState.LIVE))
                self._journal.connection_event("connected", self._settings.model)

                reason = await self._run_session(session, policy)
                return reason

        except Exception as exc:
            logger.exception("Session failed")
            await self._emit(
                ConnectionStateChanged(
                    ConnectionState.ERROR,
                    detail=str(exc),
                )
            )
            return DisconnectReason("error", str(exc))
        finally:
            await self._emit(ConnectionStateChanged(ConnectionState.DISCONNECTED))
            self._journal.connection_event("disconnected", "")

    async def _run_session(
        self, session: Any, policy: PolicyEngine
    ) -> DisconnectReason:
        """Start media and the concurrent send/receive loops."""
        self._dispatcher = ToolDispatcher(
            policy=policy,
            journal=self._journal,
            event_queue=self._event_queue,
            run_id=self._run_id,
            epoch=self._epoch,
        )
        self._dispatcher.set_respond_callback(self._make_respond_callback(session))

        try:
            if self._audio_output_available():
                self._speaker = Speaker(device=self._settings.output_device)
                await self._speaker.start()

            if self._settings.mode != "text" and self._audio_input_available():
                self._mic = Microphone(device=self._settings.input_device)
                await self._mic.start()

            if self._settings.share_screen:
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
                tasks.add(
                    asyncio.create_task(
                        self._video_send_loop(session, self._screen),
                        name="video_send",
                    )
                )
            if self._mic is not None or self._speaker is not None:
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
                    reason = DisconnectReason("error", str(exc))
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

                    server_content = response.server_content
                    if server_content:
                        events = self._turn_state.consume_server_content(server_content)
                        for event in events:
                            await self._emit(event)

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
                        await self._emit(
                            UsageUpdate(
                                prompt_tokens=getattr(usage, "prompt_token_count", 0),
                                cached_tokens=getattr(
                                    usage, "cached_content_token_count", 0
                                ),
                                completion_tokens=getattr(
                                    usage, "completion_token_count", 0
                                ),
                                audio_tokens=getattr(usage, "audio_token_count", 0),
                                video_tokens=getattr(usage, "video_token_count", 0),
                                total_tokens=getattr(usage, "total_token_count", 0),
                            )
                        )

        except asyncio.CancelledError:
            return None
        except Exception as exc:
            logger.exception("Receive loop failed")
            await self._emit(SessionError(message=str(exc), fatal=True))
            return DisconnectReason("error", str(exc))

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
                            await self._screen.start()
                            asyncio.create_task(
                                self._video_send_loop(session, self._screen),
                                name="video_send_dynamic",
                            )
                    else:
                        if self._screen is not None:
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
            return DisconnectReason("error", str(exc))

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
    ) -> Callable[[str, str, ToolResult], Awaitable[None]]:
        """Return the callback used by ``ToolDispatcher`` to reply to tool calls."""

        async def respond(call_id: str, name: str, result: ToolResult) -> None:
            await session.send_tool_response(
                function_responses=[
                    types.FunctionResponse(
                        id=call_id,
                        name=name,
                        response={"result": result.message},
                    )
                ]
            )

        return respond

    async def _emit(self, event: Any) -> None:
        await self._event_queue.put(event)

    async def _stop_media(self) -> None:
        """Stop microphone, speaker, and screen capture if they were started."""
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
