"""Unit tests for ``core.dispatcher``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Generator
from pathlib import Path
from typing import Any

import pytest

from audit.journal import Journal
from core.dispatcher import ToolDispatcher
from core.events import (
    ToolApprovalRequested,
    ToolCallCancelled,
    ToolCallReceived,
    ToolResultSent,
)
from policy import PolicyEngine
from tools._state import set_policy_engine, set_workspace_root
from tools.registry import ToolResult


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def policy(workspace: Path) -> PolicyEngine:
    return PolicyEngine(
        allow_roots=[workspace],
        deny_globs=[],
        never_allow_regex=[],
    )


@pytest.fixture
def event_queue() -> asyncio.Queue[Any]:
    return asyncio.Queue()


@pytest.fixture
def journal(tmp_path: Path) -> Generator[Journal, None, None]:
    journal = Journal(run_id="test-run", log_dir=tmp_path)
    yield journal
    journal._close()


@pytest.fixture
def dispatcher(
    policy: PolicyEngine,
    event_queue: asyncio.Queue[Any],
    journal: Journal,
) -> ToolDispatcher:
    set_policy_engine(policy)
    set_workspace_root(policy._allow_roots[0])
    return ToolDispatcher(
        policy=policy,
        journal=journal,
        event_queue=event_queue,
        run_id="test-run",
    )


async def drain(queue: asyncio.Queue[Any]) -> list[Any]:
    items: list[Any] = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def _capture_callback(
    responses: list[tuple[str, str, ToolResult]],
) -> Callable[[str, str, ToolResult], Awaitable[bool]]:
    async def respond(call_id: str, name: str, result: ToolResult) -> bool:
        responses.append((call_id, name, result))
        return True

    return respond


@pytest.mark.asyncio
async def test_unknown_tool(
    dispatcher: ToolDispatcher, event_queue: asyncio.Queue[Any]
) -> None:
    responses: list[tuple[str, str, ToolResult]] = []
    dispatcher.set_respond_callback(_capture_callback(responses))

    await dispatcher.submit("call-1", "not_a_tool", {})
    events = await drain(event_queue)

    assert any(
        isinstance(e, ToolCallReceived) and e.call_id == "call-1" for e in events
    )
    result_event = [e for e in events if isinstance(e, ToolResultSent)][0]
    assert result_event.ok is False
    assert "Unknown tool" in result_event.result
    assert len(responses) == 1
    assert responses[0][2].ok is False


@pytest.mark.asyncio
async def test_invalid_arguments(
    dispatcher: ToolDispatcher, event_queue: asyncio.Queue[Any]
) -> None:
    responses: list[tuple[str, str, ToolResult]] = []
    dispatcher.set_respond_callback(_capture_callback(responses))

    # read_file requires a ``path`` argument.
    await dispatcher.submit("call-2", "read_file", {})
    events = await drain(event_queue)

    result_event = [e for e in events if isinstance(e, ToolResultSent)][0]
    assert result_event.ok is False
    assert "Invalid arguments" in result_event.result


@pytest.mark.asyncio
async def test_policy_denies_path_outside_root(
    dispatcher: ToolDispatcher,
    event_queue: asyncio.Queue[Any],
    workspace: Path,
) -> None:
    responses: list[tuple[str, str, ToolResult]] = []
    dispatcher.set_respond_callback(_capture_callback(responses))

    outside = workspace.parent / "outside.txt"
    await dispatcher.submit("call-3", "read_file", {"path": str(outside)})
    events = await drain(event_queue)

    result_event = [e for e in events if isinstance(e, ToolResultSent)][0]
    assert result_event.ok is False
    assert "Policy denied" in result_event.result


@pytest.mark.asyncio
async def test_approval_flow(
    dispatcher: ToolDispatcher,
    event_queue: asyncio.Queue[Any],
    workspace: Path,
) -> None:
    responses: list[tuple[str, str, ToolResult]] = []
    dispatcher.set_respond_callback(_capture_callback(responses))

    target = workspace / "approved.txt"
    task = asyncio.create_task(
        dispatcher.submit(
            "call-4",
            "write_file",
            {"path": str(target), "content": "hello"},
        )
    )

    # Yield so the dispatcher reaches the approval wait.
    await asyncio.sleep(0)
    events = await drain(event_queue)
    assert any(
        isinstance(e, ToolApprovalRequested) and e.call_id == "call-4"
        for e in events
    )

    dispatcher.approve("call-4")
    await task

    events = await drain(event_queue)
    result_event = [e for e in events if isinstance(e, ToolResultSent)][0]
    assert result_event.ok is True
    assert target.read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_cancel_pending(
    dispatcher: ToolDispatcher,
    event_queue: asyncio.Queue[Any],
    workspace: Path,
) -> None:
    target = workspace / "cancelled.txt"
    task = asyncio.create_task(
        dispatcher.submit(
            "call-5",
            "write_file",
            {"path": str(target), "content": "hello"},
        )
    )

    await asyncio.sleep(0)
    dispatcher.cancel(["call-5"])
    await task

    events = await drain(event_queue)
    assert any(
        isinstance(e, ToolCallCancelled) and "call-5" in e.call_ids for e in events
    )
    # The file should not have been written because approval was cancelled.
    assert not target.exists()


@pytest.mark.asyncio
async def test_approval_timeout(
    dispatcher: ToolDispatcher,
    event_queue: asyncio.Queue[Any],
    workspace: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr("core.dispatcher.APPROVAL_TIMEOUT_SECONDS", 0.01)

    responses: list[tuple[str, str, ToolResult]] = []
    dispatcher.set_respond_callback(_capture_callback(responses))

    target = workspace / "timeout.txt"
    await dispatcher.submit(
        "call-6",
        "write_file",
        {"path": str(target), "content": "hello"},
    )
    events = await drain(event_queue)

    result_event = [e for e in events if isinstance(e, ToolResultSent)][0]
    assert result_event.ok is False
    assert "approval_timeout" in result_event.result
    assert not target.exists()
