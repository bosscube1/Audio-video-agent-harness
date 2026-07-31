"""Tests for Phase 3 session-lifetime persistence helpers in ``persist.store``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from persist.store import Store


def _iso(offset: timedelta) -> str:
    return (datetime.now(UTC) + offset).isoformat()


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(db_path=tmp_path / "test.db")


def test_get_active_resumption_handle_returns_latest_unexpired(store: Store) -> None:
    run_id = "run-1"
    store.start_run(run_id, _iso(timedelta()), "model", "mode", "/tmp")

    store.save_resumption_handle("old-handle", run_id, _iso(timedelta(minutes=-10)), None)
    store.save_resumption_handle("new-handle", run_id, _iso(timedelta(minutes=-5)), None)

    assert store.get_active_resumption_handle(run_id) == "new-handle"


def test_expired_handles_are_pruned_and_not_returned(store: Store) -> None:
    run_id = "run-1"
    store.start_run(run_id, _iso(timedelta()), "model", "mode", "/tmp")

    now = datetime.now(UTC)
    expired = (now - timedelta(minutes=5)).isoformat()
    valid = (now + timedelta(minutes=5)).isoformat()

    store.save_resumption_handle("expired-handle", run_id, expired, expired)
    store.save_resumption_handle("valid-handle", run_id, expired, valid)

    assert store.get_active_resumption_handle(run_id, now=now) == "valid-handle"
    latest = store.get_latest_resumption_handle(run_id)
    assert latest is not None
    assert latest[0] == "valid-handle"


def test_delete_resumption_handle_removes_specific_handle(store: Store) -> None:
    run_id = "run-1"
    store.start_run(run_id, _iso(timedelta()), "model", "mode", "/tmp")

    store.save_resumption_handle("handle-a", run_id, _iso(timedelta(minutes=-5)), None)
    store.save_resumption_handle("handle-b", run_id, _iso(timedelta(minutes=-5)), None)

    store.delete_resumption_handle("handle-a")

    assert store.get_active_resumption_handle(run_id) == "handle-b"


def test_get_turns_returns_order_and_epoch(store: Store) -> None:
    run_id = "run-1"
    store.start_run(run_id, _iso(timedelta()), "model", "mode", "/tmp")

    store.append_turn("t1", run_id, "user", "hello", _iso(timedelta(minutes=1)), 0)
    store.append_turn("t2", run_id, "model", "hi", _iso(timedelta(minutes=2)), 1)
    store.append_turn("t3", run_id, "user", "bye", _iso(timedelta(minutes=3)), 2)

    turns = store.get_turns(run_id, limit=2)
    assert turns == [
        ("user", "hello", 0),
        ("model", "hi", 1),
    ]


def test_get_recent_tool_results_returns_completed_rows(store: Store) -> None:
    run_id = "run-1"
    store.start_run(run_id, _iso(timedelta()), "model", "mode", "/tmp")

    store.record_tool_call(
        "tc-1", run_id, "t1", "read_file", {"path": "a.txt"}, {"content": "A"}, True, False, 0
    )
    store.record_tool_call(
        "tc-2", run_id, "t1", "read_file", {"path": "b.txt"}, None, False, False, 1
    )
    store.record_tool_call(
        "tc-3", run_id, "t1", "read_file", {"path": "c.txt"}, {"content": "C"}, True, False, 2
    )

    results = store.get_recent_tool_results(run_id)
    assert len(results) == 2
    assert results[0][:2] == ("tc-1", "read_file")
    assert results[0][2] is True
    assert results[0][3] == '{"content":"A"}'
    assert results[0][4] == 0
    assert results[1][0] == "tc-3"
