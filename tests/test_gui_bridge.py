"""Tests for the PySide6 -> asyncio bridge.

The bridge starts a ``Supervisor`` on a QThread and forwards events to the main
thread via a Qt signal. These tests use the fake Live harness so no API key or
network is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.bridge import Bridge
from core.events import (
    ConnectionState,
    ConnectionStateChanged,
    PartialTranscript,
)
from settings.settings import AppSettings
from tests.fakes.fake_live import FakeLive, FakeLiveSession, make_fake_client_factory


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
def patched_dirs(monkeypatch: Any, tmp_path: Path) -> None:
    """Redirect user data/log directories into the test tmp_path."""
    monkeypatch.setattr(
        "platformdirs.user_data_dir",
        lambda appname, appauthor=None, **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        "platformdirs.user_log_dir",
        lambda appname, appauthor=None, **kwargs: str(tmp_path),
    )


def _patch_dependencies(monkeypatch: Any, live: FakeLive) -> None:
    """Apply the standard patches needed to run the bridge offline."""
    monkeypatch.setattr("google.genai.Client", make_fake_client_factory(live))
    monkeypatch.setattr("settings.secrets.get_api_key", lambda: "fake-key")
    monkeypatch.setattr("media.devices.list_input_devices", lambda: [])
    monkeypatch.setattr("media.devices.list_output_devices", lambda: [])


def test_bridge_starts_emits_events_and_stops_cleanly(
    qtbot: Any,
    monkeypatch: Any,
    settings: AppSettings,
    patched_dirs: None,
) -> None:
    """The bridge reaches LIVE, receives a model message, and exits on Disconnect."""
    live = FakeLive()

    async def on_connect(session: FakeLiveSession) -> None:
        session.say("Hello from the fake model.", turn_complete=True)

    live.on_connect(on_connect)
    _patch_dependencies(monkeypatch, live)

    bridge = Bridge(settings, run_id="bridge-test")
    events: list[Any] = []
    bridge.event_emitted.connect(events.append)
    bridge.start()

    def _live_seen() -> bool:
        return any(
            isinstance(e, ConnectionStateChanged) and e.state == ConnectionState.LIVE
            for e in events
        )

    qtbot.waitUntil(_live_seen, timeout=5000)

    def _transcript_seen() -> bool:
        return any(
            isinstance(e, PartialTranscript) and "fake model" in e.text
            for e in events
        )

    qtbot.waitUntil(_transcript_seen, timeout=5000)

    bridge.stop()
    bridge.wait(5000)
    assert not bridge.isRunning()

    assert any(
        isinstance(e, ConnectionStateChanged) and e.state == ConnectionState.LIVE
        for e in events
    )


def test_bridge_buffers_commands_sent_before_the_loop_is_ready(
    qtbot: Any,
    monkeypatch: Any,
    settings: AppSettings,
    patched_dirs: None,
) -> None:
    """Commands sent immediately after construction must not be dropped.

    Regression: the GUI fires SetGatingMode/SetMicGate/SetShareScreen right
    after start(), while the QThread's asyncio loop does not exist yet.
    Previously send_command silently returned; now they are buffered and
    drained into the queue when run() starts.
    """
    from core.commands import Disconnect

    live = FakeLive()
    _patch_dependencies(monkeypatch, live)

    bridge = Bridge(settings, run_id="bridge-buffer-test")
    reasons: list[str] = []
    bridge.finished_with_reason.connect(reasons.append)

    # Sent before start(): the loop cannot exist yet, so this must buffer.
    bridge.send_command(Disconnect())
    bridge.start()

    assert bridge.wait(8000)
    # The signal was emitted from the bridge thread; let Qt deliver it.
    qtbot.waitUntil(lambda: bool(reasons), timeout=2000)
    assert reasons == ["user_request"]
