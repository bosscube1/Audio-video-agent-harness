"""Tests for the PySide6 main window lifecycle.

These tests drive the GUI with pytest-qt and use the fake Live harness so no
API key or network is required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PySide6.QtCore import Qt

from app.main_window import MainWindow
from core.events import ConnectionState
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
    """Apply the standard patches needed to run the main window offline."""
    monkeypatch.setattr("google.genai.Client", make_fake_client_factory(live))
    monkeypatch.setattr("settings.secrets.get_api_key", lambda: "fake-key")
    monkeypatch.setattr("media.devices.list_input_devices", lambda: [])
    monkeypatch.setattr("media.devices.list_output_devices", lambda: [])


def test_main_window_connect_and_disconnect(
    qtbot: Any,
    monkeypatch: Any,
    settings: AppSettings,
    patched_dirs: None,
) -> None:
    """The main window can connect to the fake API, see a model reply, and disconnect."""
    live = FakeLive()

    async def on_connect(session: FakeLiveSession) -> None:
        session.say("Hello GUI", turn_complete=True)

    live.on_connect(on_connect)
    _patch_dependencies(monkeypatch, live)

    window = MainWindow(settings)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    assert window._status_label.text() == "Disconnected"
    assert window._connect_btn.text() == "Connect"

    qtbot.mouseClick(window._connect_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: window._status_label.text() == ConnectionState.LIVE.value,
        timeout=5000,
    )

    qtbot.waitUntil(
        lambda: "Hello GUI" in window._transcript.toPlainText(),
        timeout=5000,
    )

    qtbot.mouseClick(window._connect_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: window._connect_btn.text() == "Connect",
        timeout=5000,
    )

    window.close()
    assert not window._bridge


def test_main_window_tool_approval_card(
    qtbot: Any,
    monkeypatch: Any,
    settings: AppSettings,
    patched_dirs: None,
) -> None:
    """A tool approval card appears and can be approved."""
    live = FakeLive()

    async def on_connect(session: FakeLiveSession) -> None:
        session.request_tool("fc-1", "run_command", {"command": "echo hello"})

    live.on_connect(on_connect)
    _patch_dependencies(monkeypatch, live)

    window = MainWindow(settings)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    qtbot.mouseClick(window._connect_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: window._status_label.text() == ConnectionState.LIVE.value,
        timeout=5000,
    )

    qtbot.waitUntil(lambda: "fc-1" in window._tool_cards, timeout=5000)
    card = window._tool_cards["fc-1"]
    card._approve_btn.click()

    qtbot.waitUntil(
        lambda: "OK" in card._status.text() or "FAILED" in card._status.text(),
        timeout=5000,
    )

    window.close()


def test_main_window_reconnects_on_go_away(
    qtbot: Any,
    monkeypatch: Any,
    settings: AppSettings,
    patched_dirs: None,
) -> None:
    """The GUI survives a server-initiated reconnect (GoAway)."""
    live = FakeLive()
    live.grant_next_handle("handle-1")

    async def on_connect(session: FakeLiveSession) -> None:
        if len(live.sessions) == 1:
            session.say("First epoch", turn_complete=False)
            session.go_away("60s")
            live.grant_next_handle("handle-1")
        else:
            session.say("Resumed", turn_complete=True)

    live.on_connect(on_connect)
    _patch_dependencies(monkeypatch, live)

    window = MainWindow(settings)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    qtbot.mouseClick(window._connect_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: window._status_label.text() == ConnectionState.LIVE.value,
        timeout=5000,
    )

    qtbot.waitUntil(
        lambda: "First epoch" in window._transcript.toPlainText(),
        timeout=5000,
    )

    # The server initiates a GoAway; the Supervisor should reconnect and the
    # second epoch should deliver the resumed text.
    qtbot.waitUntil(
        lambda: "Resumed" in window._transcript.toPlainText(),
        timeout=5000,
    )

    # Status should have returned to live after the reconnect.
    assert window._status_label.text() == ConnectionState.LIVE.value

    qtbot.mouseClick(window._connect_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: window._connect_btn.text() == "Connect",
        timeout=5000,
    )

    window.close()
    assert not window._bridge
