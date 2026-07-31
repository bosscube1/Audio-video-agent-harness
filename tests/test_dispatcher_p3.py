"""Phase 3 tests for ``core.dispatcher``.

Covers tool-call cancellation, orphaned-result handling, and the extended
approval/denial vocabulary.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from audit.journal import Journal
from core.dispatcher import ToolDispatcher
from core.events import ToolCallCancelled, ToolResultSent
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
    journal = Journal(run_id="p3-run", log_dir=tmp_path)
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
        run_id="p3-run",
    )


async def drain(queue: asyncio.Queue[Any]) -> list[Any]:
    items: list[Any] = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def _journal_records(journal: Journal) -> list[dict[str, Any]]:
    journal._close()
    with open(journal._path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.mark.asyncio
async def test_cancel_calls_cancel_shell(
    dispatcher: ToolDispatcher,
    monkeypatch: Any,
) -> None:
    cancelled: list[str] = []

    async def fake_cancel_shell(call_id: str) -> bool:
        cancelled.append(call_id)
        return True

    monkeypatch.setattr("core.dispatcher.shell_tools.cancel_shell", fake_cancel_shell)

    dispatcher.cancel(["shell-1", "shell-2"])
    await asyncio.sleep(0)

    assert "shell-1" in cancelled
    assert "shell-2" in cancelled


@pytest.mark.asyncio
async def test_deny_all_pending_denies_pending_approvals(
    dispatcher: ToolDispatcher,
    event_queue: asyncio.Queue[Any],
    workspace: Path,
) -> None:
    target = workspace / "denied_all.txt"
    task = asyncio.create_task(
        dispatcher.submit(
            "call-dap",
            "write_file",
            {"path": str(target), "content": "hello"},
        )
    )

    await asyncio.sleep(0)
    dispatcher.deny_all_pending("connection_lost")
    await task

    events = await drain(event_queue)
    cancelled = [e for e in events if isinstance(e, ToolCallCancelled)]
    results = [e for e in events if isinstance(e, ToolResultSent)]

    assert any(e.call_ids == ["call-dap"] and e.reason == "connection_lost" for e in cancelled)
    assert any(e.call_id == "call-dap" and "connection_lost" in e.result for e in results)
    assert not target.exists()


@pytest.mark.asyncio
async def test_callback_false_produces_orphaned_journal_record(
    dispatcher: ToolDispatcher,
    event_queue: asyncio.Queue[Any],
    journal: Journal,
) -> None:
    async def failing_respond(call_id: str, name: str, result: ToolResult) -> bool:
        return False

    dispatcher.set_respond_callback(failing_respond)
    await dispatcher.submit("call-orphan-false", "not_a_tool", {})
    await asyncio.sleep(0)

    events = await drain(event_queue)
    result_event = [e for e in events if isinstance(e, ToolResultSent)][0]
    assert result_event.orphaned is True

    records = _journal_records(journal)
    result_records = [r for r in records if r.get("type") == "tool_result"]
    assert result_records[0].get("orphaned") is True


@pytest.mark.asyncio
async def test_callback_raises_produces_orphaned_result(
    dispatcher: ToolDispatcher,
    event_queue: asyncio.Queue[Any],
) -> None:
    async def raising_respond(call_id: str, name: str, result: ToolResult) -> bool:
        raise RuntimeError("upstream disconnected")

    dispatcher.set_respond_callback(raising_respond)
    await dispatcher.submit("call-orphan-raise", "not_a_tool", {})
    await asyncio.sleep(0)

    events = await drain(event_queue)
    result_event = [e for e in events if isinstance(e, ToolResultSent)][0]
    assert result_event.orphaned is True
