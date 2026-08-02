"""Phase 3 Supervisor integration tests.

These tests exercise the ``Supervisor`` reconnect loop, resumption-handle
management, GoAway handling, and delivery of tool results that finish while the
underlying WebSocket is being replaced.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from google.genai import types
from pydantic import BaseModel

import core.dispatcher as dispatcher_module
from audit.journal import Journal
from core.commands import Disconnect, SendText
from core.events import (
    ConnectionState,
    ConnectionStateChanged,
    ContextReset,
    PartialTranscript,
    ToolCallReceived,
    ToolResultSent,
    TurnComplete,
)
from core.supervisor import Supervisor
from persist.store import Store
from settings.settings import AppSettings
from tests.fakes.fake_live import (
    FakeLive,
    FakeLiveSession,
    FakeSessionError,
    make_fake_client_factory,
)
from tools.registry import ToolResult


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def settings(workspace: Path) -> AppSettings:
    return AppSettings(
        model="gemini-3.1-flash-live-preview",
        mode="text",
        working_dir=workspace,
        workspace_root=workspace,
        share_screen=False,
    )


@pytest.fixture
def event_queue() -> asyncio.Queue[Any]:
    return asyncio.Queue()


@pytest.fixture
def command_queue() -> asyncio.Queue[Any]:
    return asyncio.Queue()


@pytest.fixture
def journal(tmp_path: Path) -> Generator[Journal, None, None]:
    journal = Journal(run_id="test-run", log_dir=tmp_path)
    yield journal
    journal._close()


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(db_path=tmp_path / "state.db")


async def _wait_for_event(
    queue: asyncio.Queue[Any],
    buffer: list[Any],
    predicate: Any,
    timeout_seconds: float = 2.0,
) -> Any:
    """Drain ``queue`` into ``buffer`` until ``predicate`` matches an event.

    Non-matching events are kept in ``buffer`` for later assertions.
    """
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while True:
        while not queue.empty():
            buffer.append(queue.get_nowait())
        for index, event in enumerate(buffer):
            if predicate(event):
                return buffer.pop(index)
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("Timeout waiting for event")
        await asyncio.sleep(0.01)


async def drain(queue: asyncio.Queue[Any]) -> list[Any]:
    items: list[Any] = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def _patch_for_supervisor(monkeypatch: Any, live: FakeLive) -> None:
    """Apply the standard monkeypatches used by every Supervisor test."""
    monkeypatch.setattr("google.genai.Client", make_fake_client_factory(live))
    monkeypatch.setattr("settings.secrets.get_api_key", lambda: "fake-api-key")
    monkeypatch.setattr("media.devices.list_input_devices", lambda: [])
    monkeypatch.setattr("media.devices.list_output_devices", lambda: [])
    monkeypatch.setattr("core.supervisor.random.uniform", lambda a, b: 0)


@pytest.mark.asyncio
async def test_simple_text_round_trip(
    settings: AppSettings,
    command_queue: asyncio.Queue[Any],
    event_queue: asyncio.Queue[Any],
    journal: Journal,
    store: Store,
    monkeypatch: Any,
) -> None:
    """A single text exchange runs end-to-end through the Supervisor."""
    live = FakeLive()

    async def on_connect(session: FakeLiveSession) -> None:
        session.say("Hello from the fake model.", turn_complete=True)

    live.on_connect(on_connect)
    _patch_for_supervisor(monkeypatch, live)

    supervisor = Supervisor(
        settings=settings,
        command_queue=command_queue,
        event_queue=event_queue,
        journal=journal,
        store=store,
        run_id="test-run",
    )
    task = asyncio.create_task(supervisor.run())

    events: list[Any] = []
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ConnectionStateChanged)
        and e.state == ConnectionState.LIVE,
    )
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, PartialTranscript) and "fake model" in e.text,
    )

    await command_queue.put(SendText(text="Hi fake model"))
    await asyncio.sleep(0.05)
    await command_queue.put(Disconnect())

    reason = await task
    assert reason.code == "user_request"

    assert len(live.sessions) == 1
    outbox = live.sessions[0].pop_outbox()
    client_contents = [o for o in outbox if isinstance(o, types.LiveClientContent)]
    assert any(
        o.turns[0].parts[0].text == "Hi fake model"
        for o in client_contents
        if o.turns and o.turns[0].parts
    )


@pytest.mark.asyncio
async def test_socket_drop_mid_turn_resumes_and_continues(
    settings: AppSettings,
    command_queue: asyncio.Queue[Any],
    event_queue: asyncio.Queue[Any],
    journal: Journal,
    store: Store,
    monkeypatch: Any,
) -> None:
    """A dropped socket mid-turn reconnects with the same resumption handle."""
    live = FakeLive()
    live.grant_next_handle("handle-1")

    async def on_connect(session: FakeLiveSession) -> None:
        if len(live.sessions) == 1:
            session.say("Hello part 1", turn_complete=False)

            async def _drop_after_first() -> None:
                await asyncio.sleep(0.1)
                session.drop_next_receive()
                session.say("ignored", turn_complete=True)

            asyncio.create_task(_drop_after_first())
            live.grant_next_handle("handle-1")
        else:
            session.say(" part 2", turn_complete=True)

    live.on_connect(on_connect)
    _patch_for_supervisor(monkeypatch, live)

    supervisor = Supervisor(
        settings=settings,
        command_queue=command_queue,
        event_queue=event_queue,
        journal=journal,
        store=store,
        run_id="test-run",
    )
    task = asyncio.create_task(supervisor.run())

    events: list[Any] = []

    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ConnectionStateChanged)
        and e.state == ConnectionState.LIVE,
    )
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ConnectionStateChanged)
        and e.state == ConnectionState.RECONNECTING,
    )
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ConnectionStateChanged)
        and e.state == ConnectionState.LIVE,
    )
    completed = await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, TurnComplete)
        and "Hello part 1 part 2" in e.text,
    )
    assert completed.text == "Hello part 1 part 2"

    assert len(live.sessions) == 2
    resumption = live.sessions[1].config.session_resumption
    assert resumption is not None
    assert resumption.handle == "handle-1"

    await command_queue.put(Disconnect())
    reason = await task
    assert reason.code == "user_request"


@pytest.mark.asyncio
async def test_go_away_emits_draining_and_reconnects_with_handle(
    settings: AppSettings,
    command_queue: asyncio.Queue[Any],
    event_queue: asyncio.Queue[Any],
    journal: Journal,
    store: Store,
    monkeypatch: Any,
) -> None:
    """GoAway triggers DRAINING -> RECONNECTING -> LIVE with resumption."""
    live = FakeLive()
    live.grant_next_handle("handle-1")

    async def on_connect(session: FakeLiveSession) -> None:
        if len(live.sessions) == 1:
            session.go_away("60s")
            live.grant_next_handle("handle-1")
        else:
            session.say("Welcome back", turn_complete=True)

    live.on_connect(on_connect)
    _patch_for_supervisor(monkeypatch, live)

    supervisor = Supervisor(
        settings=settings,
        command_queue=command_queue,
        event_queue=event_queue,
        journal=journal,
        store=store,
        run_id="test-run",
    )
    task = asyncio.create_task(supervisor.run())

    events: list[Any] = []

    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ConnectionStateChanged)
        and e.state == ConnectionState.LIVE,
    )
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ConnectionStateChanged)
        and e.state == ConnectionState.DRAINING,
    )
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ConnectionStateChanged)
        and e.state == ConnectionState.RECONNECTING,
    )
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ConnectionStateChanged)
        and e.state == ConnectionState.LIVE,
    )
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, PartialTranscript) and "Welcome back" in e.text,
    )

    assert len(live.sessions) == 2
    resumption = live.sessions[1].config.session_resumption
    assert resumption is not None
    assert resumption.handle == "handle-1"

    await command_queue.put(Disconnect())
    reason = await task
    assert reason.code == "user_request"


@pytest.mark.asyncio
async def test_tool_result_finished_during_reconnect_is_delivered(
    settings: AppSettings,
    command_queue: asyncio.Queue[Any],
    event_queue: asyncio.Queue[Any],
    journal: Journal,
    store: Store,
    monkeypatch: Any,
) -> None:
    """A tool that completes while the socket is down is delivered to the new session.

    The new session rejects the original function-call id, so the Supervisor falls
    back to a system note containing the tool result.
    """

    class FakeToolArgs(BaseModel):
        delay: float = 0.0

    tool_done = asyncio.Event()

    async def fake_tool(args: FakeToolArgs, *, call_id: str) -> ToolResult:
        await tool_done.wait()
        return ToolResult(True, "done")

    monkeypatch.setattr(
        dispatcher_module,
        "_TOOL_TABLE",
        {"fake_tool": (fake_tool, FakeToolArgs)},
    )

    live = FakeLive()

    async def on_connect(session: FakeLiveSession) -> None:
        if len(live.sessions) == 1:
            session.request_tool("fc-1", "fake_tool", {})

            async def _drop_after_first() -> None:
                await asyncio.sleep(0.1)
                session.drop_next_receive()
                session.say("ignored", turn_complete=True)

            asyncio.create_task(_drop_after_first())
        else:
            session.reject_fc_id("fc-1")
            tool_done.set()

    live.on_connect(on_connect)
    _patch_for_supervisor(monkeypatch, live)

    supervisor = Supervisor(
        settings=settings,
        command_queue=command_queue,
        event_queue=event_queue,
        journal=journal,
        store=store,
        run_id="tool-run",
    )
    task = asyncio.create_task(supervisor.run())

    events: list[Any] = []

    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ToolCallReceived) and e.call_id == "fc-1",
    )
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ConnectionStateChanged)
        and e.state == ConnectionState.RECONNECTING,
    )
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ConnectionStateChanged)
        and e.state == ConnectionState.LIVE,
    )
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ToolResultSent) and e.call_id == "fc-1",
    )

    assert len(live.sessions) == 2
    outbox = live.sessions[1].pop_outbox()
    notes = [
        o
        for o in outbox
        if isinstance(o, types.LiveClientContent)
        and o.turns
        and o.turns[0].parts
        and o.turns[0].parts[0].text is not None
        and "tool result for fake_tool" in o.turns[0].parts[0].text
    ]
    assert notes, "Expected a fallback system note for the tool result"

    await command_queue.put(Disconnect())
    reason = await task
    assert reason.code == "user_request"


@pytest.mark.asyncio
async def test_rejected_resumption_handle_emits_context_reset(
    settings: AppSettings,
    command_queue: asyncio.Queue[Any],
    event_queue: asyncio.Queue[Any],
    journal: Journal,
    store: Store,
    monkeypatch: Any,
) -> None:
    """A rejected resumption handle is deleted and the Supervisor reconnects fresh."""
    now = datetime.now(UTC)
    store.save_resumption_handle(
        "bad-handle",
        "run",
        now.isoformat(),
        (now + timedelta(hours=2)).isoformat(),
    )

    live = FakeLive()
    live.reject_handle("bad-handle")

    async def on_connect(session: FakeLiveSession) -> None:
        session.say("Fresh start", turn_complete=True)

    live.on_connect(on_connect)
    _patch_for_supervisor(monkeypatch, live)

    supervisor = Supervisor(
        settings=settings,
        command_queue=command_queue,
        event_queue=event_queue,
        journal=journal,
        store=store,
        run_id="run",
    )
    task = asyncio.create_task(supervisor.run())

    events: list[Any] = []

    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ContextReset) and "resumption handle rejected" in e.reason,
    )
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, ConnectionStateChanged)
        and e.state == ConnectionState.LIVE,
    )
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, PartialTranscript) and "Fresh start" in e.text,
    )

    assert len(live.sessions) == 1
    resumption = live.sessions[0].config.session_resumption
    assert resumption is not None
    assert resumption.handle is None
    assert store.get_active_resumption_handle("run") is None

    await command_queue.put(Disconnect())
    reason = await task
    assert reason.code == "user_request"


@pytest.mark.asyncio
async def test_failure_budget_returns_failed(
    settings: AppSettings,
    command_queue: asyncio.Queue[Any],
    event_queue: asyncio.Queue[Any],
    journal: Journal,
    store: Store,
    monkeypatch: Any,
) -> None:
    """After too many transient failures the Supervisor gives up and returns FAILED."""
    live = FakeLive()
    original_connect = live.connect
    attempts = 0

    @contextlib.asynccontextmanager
    async def failing_connect(
        *, model: str, config: types.LiveConnectConfig | None = None
    ) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts <= 7:
            raise FakeSessionError(f"connection dropped attempt {attempts}")
        async with original_connect(model=model, config=config) as session:
            yield session

    monkeypatch.setattr(live, "connect", failing_connect)

    async def on_connect(session: FakeLiveSession) -> None:
        session.say("Finally", turn_complete=True)

    live.on_connect(on_connect)
    _patch_for_supervisor(monkeypatch, live)

    supervisor = Supervisor(
        settings=settings,
        command_queue=command_queue,
        event_queue=event_queue,
        journal=journal,
        store=store,
        run_id="budget-run",
    )
    task = asyncio.create_task(supervisor.run())

    reason = await asyncio.wait_for(task, timeout=5)

    assert reason.code == "failed"
    assert "failure budget" in reason.detail.lower()
    assert attempts == 7
    assert any(
        isinstance(e, ConnectionStateChanged) and e.state == ConnectionState.FAILED
        for e in await drain(event_queue)
    )


@pytest.mark.asyncio
async def test_run_lifecycle_turns_and_usage_persisted(
    settings: AppSettings,
    command_queue: asyncio.Queue[Any],
    event_queue: asyncio.Queue[Any],
    journal: Journal,
    store: Store,
    monkeypatch: Any,
) -> None:
    """Phase 6: runs, turns, and usage land in SQLite incrementally."""
    live = FakeLive()

    async def on_connect(session: FakeLiveSession) -> None:
        session.say("persisted turn", turn_complete=True)
        session.usage(
            prompt_token_count=100,
            response_token_count=20,
            total_token_count=120,
        )

    live.on_connect(on_connect)
    _patch_for_supervisor(monkeypatch, live)

    supervisor = Supervisor(
        settings=settings,
        command_queue=command_queue,
        event_queue=event_queue,
        journal=journal,
        store=store,
        run_id="persist-run",
    )
    task = asyncio.create_task(supervisor.run())

    events: list[Any] = []
    await _wait_for_event(
        event_queue,
        events,
        lambda e: isinstance(e, TurnComplete) and "persisted turn" in e.text,
    )
    await asyncio.sleep(0.05)  # let the usage row flush
    await command_queue.put(Disconnect())
    reason = await asyncio.wait_for(task, timeout=5)
    assert reason.code == "user_request"

    run = store.get_run("persist-run")
    assert run is not None
    assert run["model"] == "gemini-3.1-flash-live-preview"
    assert run["ended_at"] is not None

    turns = store.get_turns("persist-run")
    assert any("persisted turn" in text for _role, text, _epoch in turns)

    usage = store.get_usage("persist-run")
    by_modality = {modality: tokens for modality, tokens, _usd, _at in usage}
    assert by_modality.get("prompt") == 100
    assert by_modality.get("completion") == 20
