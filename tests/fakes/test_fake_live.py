"""Unit tests for the fake Gemini Live SDK."""

from __future__ import annotations

import pytest
from google.genai import types

from tests.fakes.fake_live import (
    FakeLive,
    FakeLiveSession,
    FakeSessionError,
    connect_config_with_handle,
)


async def test_session_yields_preloaded_messages() -> None:
    session = FakeLiveSession(
        model="test",
        config=types.LiveConnectConfig(),
        preloaded_messages=[
            types.LiveServerMessage(
                server_content=types.LiveServerContent(
                    model_turn=types.Content(parts=[types.Part(text="hello")])
                )
            )
        ],
    )

    session.close()
    messages = []
    async with session:
        async for msg in session.receive():
            messages.append(msg)

    assert len(messages) == 1
    sc = messages[0].server_content
    assert sc is not None
    mt = sc.model_turn
    assert mt is not None
    parts = mt.parts
    assert parts is not None
    assert parts[0].text == "hello"


async def test_say_helper() -> None:
    session = FakeLiveSession(model="test", config=types.LiveConnectConfig())
    session.say("hi", turn_complete=True)
    session.close()

    messages = []
    async with session:
        async for msg in session.receive():
            messages.append(msg)

    assert len(messages) == 1
    content = messages[0].server_content
    assert content is not None
    model_turn = content.model_turn
    assert model_turn is not None
    parts = model_turn.parts
    assert parts is not None
    assert parts[0].text == "hi"
    assert content.turn_complete is True


async def test_request_tool_and_send_tool_response() -> None:
    session = FakeLiveSession(model="test", config=types.LiveConnectConfig())
    session.request_tool("fc-1", "read_file", {"path": "/tmp/foo.txt"})

    async with session:
        async for msg in session.receive():
            tool_call = msg.tool_call
            assert tool_call is not None
            function_calls = tool_call.function_calls
            assert function_calls is not None
            fc = function_calls[0]
            assert fc.id == "fc-1"
            assert fc.name == "read_file"
            assert fc.args == {"path": "/tmp/foo.txt"}
            break

        await session.send_tool_response(
            function_responses=[
                types.FunctionResponse(id="fc-1", name="read_file", response={"result": "ok"})
            ]
        )

    out = session.pop_outbox()
    assert len(out) == 1
    assert out[0].function_responses[0].id == "fc-1"


async def test_rejected_fc_id_raises() -> None:
    session = FakeLiveSession(model="test", config=types.LiveConnectConfig())
    session.reject_fc_id("fc-2")

    with pytest.raises(FakeSessionError, match="fc-2"):
        await session.send_tool_response(
            function_responses=[
                types.FunctionResponse(id="fc-2", name="read_file", response={"result": "ok"})
            ]
        )


async def test_drop_next_receive() -> None:
    session = FakeLiveSession(model="test", config=types.LiveConnectConfig())
    session.say("hello")
    session.drop_next_receive()

    async with session:
        with pytest.raises(FakeSessionError):
            async for _msg in session.receive():
                pass


async def test_drop_next_send() -> None:
    session = FakeLiveSession(model="test", config=types.LiveConnectConfig())
    session.drop_next_send()

    with pytest.raises(FakeSessionError):
        await session.send_client_content(
            turns=[types.Content(parts=[types.Part(text="hi")])], turn_complete=True
        )


async def test_live_connect_rejects_handle() -> None:
    live = FakeLive()
    live.reject_handle("old-handle")

    with pytest.raises(FakeSessionError):
        async with live.connect(
            model="test", config=connect_config_with_handle("old-handle")
        ):
            pass


async def test_live_connect_grants_handle() -> None:
    live = FakeLive()
    live.grant_next_handle("new-handle")

    async with live.connect(model="test", config=connect_config_with_handle(None)) as session:
        session.close()
        messages = [msg async for msg in session.receive()]

    assert len(messages) == 1
    update = messages[0].session_resumption_update
    assert update is not None
    assert update.new_handle == "new-handle"
    assert update.resumable is True


async def test_live_tracks_sessions() -> None:
    live = FakeLive()
    async with live.connect(model="test") as session:
        assert live.latest_session is session
        assert len(live.sessions) == 1


async def test_go_away() -> None:
    session = FakeLiveSession(model="test", config=types.LiveConnectConfig())
    session.go_away("60s")
    session.close()

    async with session:
        async for msg in session.receive():
            go_away = msg.go_away
            assert go_away is not None
            assert go_away.time_left == "60s"
            break
