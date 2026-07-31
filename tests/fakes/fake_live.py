"""Fake Google GenAI Live API for deterministic supervisor/session tests.

This fake mimics ``client.aio.live.connect(...) async with ...`` and the
resulting session object so ``core.session.LiveSession`` and the Phase 3
supervisor can be exercised without a real API key or network.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any

from google.genai import types


def _server_content(
    *,
    model_text: str | None = None,
    input_text: str | None = None,
    output_text: str | None = None,
    turn_complete: bool = False,
    interrupted: bool = False,
) -> types.LiveServerContent:
    parts: list[types.Part] = []
    if model_text:
        parts.append(types.Part(text=model_text))
    return types.LiveServerContent(
        model_turn=types.Content(parts=parts) if parts else None,
        input_transcription=types.Transcription(text=input_text)
        if input_text is not None
        else None,
        output_transcription=types.Transcription(text=output_text)
        if output_text is not None
        else None,
        turn_complete=turn_complete,
        interrupted=interrupted,
    )


def _message(**parts: Any) -> types.LiveServerMessage:
    return types.LiveServerMessage(**parts)


class FakeSessionError(Exception):
    """Raised by the fake to simulate a server-side or transport failure."""


class FakeLiveSession:
    """A single fake Live API session.

    The class is designed to be used as the value yielded by
    ``FakeLive.connect(...).__aenter__()``. It exposes the same public methods as
    the SDK session object:

    - ``receive()`` — async generator of ``types.LiveServerMessage``
    - ``send_client_content(...)``
    - ``send_realtime_input(...)``
    - ``send_tool_response(...)``

    Tests can script the server side through the ``say()``, ``request_tool()``,
    ``go_away()``, ``grant_resumption()``, etc. helpers, and inspect the client
    side through ``pop_outbox()``.
    """

    def __init__(
        self,
        model: str,
        config: types.LiveConnectConfig,
        *,
        preloaded_messages: list[types.LiveServerMessage] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self._inbox: asyncio.Queue[types.LiveServerMessage | None] = asyncio.Queue()
        self._outbox: asyncio.Queue[Any] = asyncio.Queue()
        self._closed = False

        self._drop_next_send = False
        self._drop_next_receive = False
        self._rejected_fc_ids: set[str] = set()

        if preloaded_messages:
            for msg in preloaded_messages:
                self._inbox.put_nowait(msg)

    # ------------------------------------------------------------------
    # SDK-compatible public API
    # ------------------------------------------------------------------
    async def __aenter__(self) -> FakeLiveSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._closed = True
        with contextlib.suppress(asyncio.QueueFull):
            self._inbox.put_nowait(None)

    async def receive(self) -> AsyncGenerator[types.LiveServerMessage, None]:
        """Yield server messages until the fake session is closed."""
        while True:
            if self._closed:
                break
            if self._drop_next_receive:
                self._drop_next_receive = False
                raise FakeSessionError("connection dropped")
            msg = await self._inbox.get()
            if msg is None:
                break
            if self._drop_next_receive:
                self._drop_next_receive = False
                raise FakeSessionError("connection dropped")
            yield msg

    async def send_client_content(
        self,
        *,
        turns: list[types.Content] | None = None,
        turn_complete: bool = False,
    ) -> None:
        self._check_send()
        await self._outbox.put(
            types.LiveClientContent(
                turns=turns or [],
                turn_complete=turn_complete,
            )
        )

    async def send_realtime_input(
        self,
        *,
        audio: types.Blob | None = None,
        video: types.Blob | None = None,
        **kwargs: Any,
    ) -> None:
        self._check_send()
        await self._outbox.put(
            types.LiveClientRealtimeInput(
                audio=audio,
                video=video,
                **kwargs,
            )
        )

    async def send_tool_response(
        self,
        *,
        function_responses: list[types.FunctionResponse],
    ) -> None:
        self._check_send()
        for fr in function_responses:
            if fr.id in self._rejected_fc_ids:
                raise FakeSessionError(f"function call id {fr.id!r} was rejected")
        await self._outbox.put(
            types.LiveClientToolResponse(
                function_responses=function_responses,
            )
        )

    # ------------------------------------------------------------------
    # Scripting helpers (server -> client)
    # ------------------------------------------------------------------
    def emit(self, message: types.LiveServerMessage) -> None:
        """Enqueue a raw server message."""
        self._inbox.put_nowait(message)

    def say(
        self,
        text: str,
        *,
        turn_complete: bool = False,
        input_text: str | None = None,
    ) -> None:
        """Enqueue a server content message containing ``text``."""
        self.emit(
            _message(
                server_content=_server_content(
                    model_text=text,
                    input_text=input_text,
                    turn_complete=turn_complete,
                )
            )
        )

    def request_tool(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        """Enqueue a tool call request from the server."""
        self.emit(
            _message(
                tool_call=types.LiveServerToolCall(
                    function_calls=[
                        types.FunctionCall(
                            id=call_id,
                            name=name,
                            args=args,
                        )
                    ]
                )
            )
        )

    def cancel_tool(self, call_ids: list[str]) -> None:
        """Enqueue a tool-call cancellation from the server."""
        self.emit(
            _message(
                tool_call_cancellation=types.LiveServerToolCallCancellation(
                    ids=call_ids
                )
            )
        )

    def go_away(self, time_left: str) -> None:
        """Enqueue a ``go_away`` message from the server."""
        self.emit(
            _message(
                go_away=types.LiveServerGoAway(time_left=time_left),
            )
        )

    def grant_resumption(
        self,
        handle: str,
        *,
        last_consumed_client_message_index: int = 0,
    ) -> None:
        """Enqueue a session-resumption update from the server."""
        self.emit(
            _message(
                session_resumption_update=types.LiveServerSessionResumptionUpdate(
                    new_handle=handle,
                    resumable=True,
                    last_consumed_client_message_index=last_consumed_client_message_index,
                )
            )
        )

    def usage(self, **metadata: Any) -> None:
        """Enqueue a usage metadata update."""
        self.emit(_message(usage_metadata=types.UsageMetadata(**metadata)))

    def close(self) -> None:
        """Signal the receive loop to end gracefully."""
        self._inbox.put_nowait(None)

    # ------------------------------------------------------------------
    # Failure scripting helpers
    # ------------------------------------------------------------------
    def drop_next_send(self) -> None:
        """Make the next client send raise ``FakeSessionError``."""
        self._drop_next_send = True

    def drop_next_receive(self) -> None:
        """Make the next server receive raise ``FakeSessionError``."""
        self._drop_next_receive = True

    def reject_fc_id(self, call_id: str) -> None:
        """Make ``send_tool_response`` raise for the given function call id."""
        self._rejected_fc_ids.add(call_id)

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------
    def pop_outbox(self) -> list[Any]:
        """Drain and return every client message captured by the fake."""
        items: list[Any] = []
        while not self._outbox.empty():
            items.append(self._outbox.get_nowait())
        return items

    def _check_send(self) -> None:
        if self._closed:
            raise FakeSessionError("session is closed")
        if self._drop_next_send:
            self._drop_next_send = False
            raise FakeSessionError("connection dropped")


class FakeLive:
    """Replacement for ``client.aio.live`` in tests.

    Use it by monkeypatching ``genai.Client`` so that its constructor returns a
    ``FakeClient`` wrapping this instance:

    .. code-block:: python

        from unittest.mock import patch
        from tests.fakes.fake_live import FakeClient, FakeLive

        fake_live = FakeLive()
        with patch("google.genai.Client", return_value=FakeClient(fake_live)):
            ...
    """

    def __init__(self) -> None:
        self._sessions: list[FakeLiveSession] = []
        self._rejected_handles: set[str] = set()
        self._next_handle: str | None = None
        self._next_connect_error: BaseException | None = None
        self._on_connect: (
            Callable[[FakeLiveSession], Coroutine[Any, Any, None]] | None
        ) = None

    @property
    def sessions(self) -> list[FakeLiveSession]:
        """All sessions created by this fake, in connection order."""
        return list(self._sessions)

    @property
    def latest_session(self) -> FakeLiveSession | None:
        """The most recently created session, or ``None``."""
        return self._sessions[-1] if self._sessions else None

    def reject_handle(self, handle: str) -> None:
        """Reject reconnects that present ``handle`` as a resumption handle."""
        self._rejected_handles.add(handle)

    def grant_next_handle(self, handle: str) -> None:
        """Grant ``handle`` to the next successfully created session."""
        self._next_handle = handle

    def fail_next_connect(self, exc: BaseException) -> None:
        """Make the next ``connect`` call raise ``exc``."""
        self._next_connect_error = exc

    def on_connect(
        self, callback: Callable[[FakeLiveSession], Coroutine[Any, Any, None]] | None
    ) -> None:
        """Register a coroutine invoked after each new session is created."""
        self._on_connect = callback

    @contextlib.asynccontextmanager
    async def connect(
        self,
        *,
        model: str,
        config: types.LiveConnectConfig | None = None,
    ) -> AsyncGenerator[FakeLiveSession, None]:
        """Open a new fake session (async context manager)."""
        if self._next_connect_error is not None:
            exc = self._next_connect_error
            self._next_connect_error = None
            raise exc

        resumption = getattr(config, "session_resumption", None)
        handle = getattr(resumption, "handle", None) if resumption else None
        if handle is not None and handle in self._rejected_handles:
            raise FakeSessionError(f"resumption handle {handle!r} was rejected")

        session = FakeLiveSession(model=model, config=config or types.LiveConnectConfig())
        if self._next_handle is not None:
            session.grant_resumption(self._next_handle)
            self._next_handle = None

        self._sessions.append(session)

        if self._on_connect is not None:
            await self._on_connect(session)

        try:
            yield session
        finally:
            session._closed = True


class _FakeAio:
    def __init__(self, live: FakeLive) -> None:
        self.live = live


class FakeClient:
    """Minimal stand-in for ``google.genai.Client``.

    Only the attributes used by this codebase are provided: ``aio.live``.
    """

    def __init__(self, live: FakeLive) -> None:
        self.aio = _FakeAio(live=live)


def make_fake_client_factory(live: FakeLive) -> Callable[..., FakeClient]:
    """Return a callable that can be patched over ``google.genai.Client``.

    Example
    -------
    ``monkeypatch.setattr("google.genai.Client", make_fake_client_factory(live))``
    """

    def factory(*args: Any, **kwargs: Any) -> FakeClient:
        return FakeClient(live=live)

    return factory


def connect_config_with_handle(handle: str | None) -> types.LiveConnectConfig:
    """Build a config that requests the given resumption handle.

    Convenience helper for tests that do not want to import the full
    ``AppSettings`` machinery.
    """
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(parts=[types.Part(text="fake")]),
        tools=[],
        session_resumption=types.SessionResumptionConfig(handle=handle),
    )
