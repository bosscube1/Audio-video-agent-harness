"""Tests for Phase 5 GUI surfaces: usage panel, activity feed, mic state
machine, screen-share banner, tray behavior, and keyboard accessibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSystemTrayIcon

from app.bridge import Bridge
from app.main_window import MainWindow, _MicState
from core.events import ConnectionState, ConnectionStateChanged, UsageUpdate
from settings.settings import AppSettings
from tests.fakes.fake_live import FakeLive, FakeLiveSession, make_fake_client_factory


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


def _make_settings(workspace: Path, **overrides: Any) -> AppSettings:
    base: dict[str, Any] = {
        "model": "gemini-3.1-flash-live-preview",
        "mode": "text",
        "working_dir": workspace,
        "workspace_root": workspace,
        "share_screen": False,
        "minimize_to_tray": False,
    }
    base.update(overrides)
    return AppSettings(**base)


@pytest.fixture
def settings(workspace: Path) -> AppSettings:
    return _make_settings(workspace)


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
    monkeypatch.setattr("google.genai.Client", make_fake_client_factory(live))
    monkeypatch.setattr("settings.secrets.get_api_key", lambda: "fake-key")
    monkeypatch.setattr("media.devices.list_input_devices", lambda: [])
    monkeypatch.setattr("media.devices.list_output_devices", lambda: [])


def _activity_text(window: MainWindow) -> str:
    return "\n".join(
        window._activity.item(i).text() for i in range(window._activity.count())
    )


# -- Usage panel --


def test_usage_panel_updates_on_usage_update(
    qtbot: Any, settings: AppSettings, patched_dirs: None
) -> None:
    window = MainWindow(settings)
    qtbot.addWidget(window)

    assert window._usage_label.text() == "0 tokens"

    window._on_event(
        UsageUpdate(
            prompt_tokens=1200,
            completion_tokens=300,
            total_tokens=1500,
            estimated_usd=0.0021,
        )
    )

    assert "1500 tokens" in window._usage_label.text()
    assert "1200" in window._usage_label.text()
    assert "$0.0021" in window._usage_label.text()
    window.close()


# -- Activity feed --


def test_activity_feed_logs_connection_changes(
    qtbot: Any, settings: AppSettings, patched_dirs: None
) -> None:
    window = MainWindow(settings)
    qtbot.addWidget(window)

    window._on_event(ConnectionStateChanged(state=ConnectionState.CONNECTING))
    window._on_event(
        ConnectionStateChanged(state=ConnectionState.LIVE, detail="epoch 1")
    )

    text = _activity_text(window)
    assert "Connection: connecting" in text
    assert "Connection: live — epoch 1" in text
    window.close()


def test_activity_feed_logs_tool_cycle(
    qtbot: Any, monkeypatch: Any, settings: AppSettings, patched_dirs: None
) -> None:
    """A full tool approval cycle leaves entries in the activity feed."""
    live = FakeLive()

    async def on_connect(session: FakeLiveSession) -> None:
        session.request_tool("fc-9", "run_command", {"command": "echo hi"})

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
    qtbot.waitUntil(lambda: "fc-9" in window._tool_cards, timeout=5000)
    window._tool_cards["fc-9"]._approve_btn.click()
    qtbot.waitUntil(
        lambda: "Tool result: run_command" in _activity_text(window),
        timeout=5000,
    )

    text = _activity_text(window)
    assert "Approval requested: run_command" in text
    # The fake harness may report OK or FAILED depending on the sandboxed
    # shell environment; what matters is that the result was logged.
    assert "Tool result: run_command ->" in text

    qtbot.mouseClick(window._connect_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._connect_btn.text() == "Connect", timeout=5000)
    window.close()


# -- Mic state machine --


def test_mic_chip_off_in_text_mode(
    qtbot: Any, monkeypatch: Any, settings: AppSettings, patched_dirs: None
) -> None:
    """Text mode never shows the mic as live, even when connected."""
    live = FakeLive()

    async def on_connect(session: FakeLiveSession) -> None:
        session.say("hi", turn_complete=True)

    live.on_connect(on_connect)
    _patch_dependencies(monkeypatch, live)

    window = MainWindow(settings)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    assert window._mic_state == _MicState.OFF
    assert window._mic_chip.text() == "Mic: off"

    qtbot.mouseClick(window._connect_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: window._status_label.text() == ConnectionState.LIVE.value,
        timeout=5000,
    )
    assert window._mic_state == _MicState.OFF

    qtbot.mouseClick(window._connect_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._connect_btn.text() == "Connect", timeout=5000)
    window.close()


def test_mic_chip_transitions_with_mute_toggle(
    qtbot: Any, settings: AppSettings, patched_dirs: None
) -> None:
    """The chip follows the mute toggle while a voice-capable session is live."""
    window = MainWindow(settings)
    qtbot.addWidget(window)

    # Simulate a live, voice-capable session without real audio devices.
    class _StubBridge:
        def send_command(self, command: Any) -> None:
            pass

    window._mode_combo.setCurrentText("Both")
    window._on_event(ConnectionStateChanged(state=ConnectionState.LIVE))
    window._bridge = cast(Bridge, _StubBridge())
    window._refresh_mic_state()
    assert window._mic_chip.text() == "Mic: live"

    window._mute_btn.setChecked(True)
    assert window._mic_chip.text() == "Mic: muted"
    assert window._mute_btn.text() == "Unmute mic"

    window._mute_btn.setChecked(False)
    assert window._mic_chip.text() == "Mic: live"

    window._bridge = None
    window._refresh_mic_state()
    assert window._mic_state == _MicState.OFF
    window.close()


# -- Screen-share banner --


def test_share_banner_follows_connect_settings(
    qtbot: Any, monkeypatch: Any, workspace: Path, patched_dirs: None
) -> None:
    """The banner appears when connecting with share-screen on, hides after."""
    live = FakeLive()

    async def on_connect(session: FakeLiveSession) -> None:
        session.say("hi", turn_complete=True)

    live.on_connect(on_connect)
    _patch_dependencies(monkeypatch, live)

    settings = _make_settings(workspace, share_screen=True)
    window = MainWindow(settings)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    assert not window._share_banner.isVisible()

    qtbot.mouseClick(window._connect_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: window._status_label.text() == ConnectionState.LIVE.value,
        timeout=5000,
    )
    assert window._sharing_active
    assert window._share_banner.isVisible()

    qtbot.mouseClick(window._connect_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._connect_btn.text() == "Connect", timeout=5000)
    qtbot.waitUntil(lambda: not window._sharing_active, timeout=5000)
    assert not window._share_banner.isVisible()
    window.close()


# -- Tray --


def test_close_hides_to_tray_when_enabled(
    qtbot: Any, monkeypatch: Any, workspace: Path, patched_dirs: None
) -> None:
    """With minimize-to-tray on and a tray present, close hides the window."""
    settings = _make_settings(workspace, minimize_to_tray=True)
    window = MainWindow(settings)
    qtbot.addWidget(window)

    # Force a tray stand-in regardless of platform tray availability.
    class _FakeTray:
        def __init__(self) -> None:
            self.hidden = False

        def isVisible(self) -> bool:
            return True

        def showMessage(self, *args: Any, **kwargs: Any) -> None:
            pass

        def hide(self) -> None:
            self.hidden = True

    fake_tray = _FakeTray()
    window._tray = cast(QSystemTrayIcon, fake_tray)
    window._tray_check.setChecked(True)

    window.show()
    qtbot.waitExposed(window)
    window.close()

    assert not window.isVisible()
    assert window._bridge is None

    # A forced quit closes for real even with the tray active.
    window._force_quit = True
    window.close()
    assert fake_tray.hidden


def test_close_quits_normally_when_tray_disabled(
    qtbot: Any, settings: AppSettings, patched_dirs: None
) -> None:
    """Default path (tray off / unavailable) still quits cleanly."""
    window = MainWindow(settings)
    qtbot.addWidget(window)
    window._tray = None
    window.show()
    qtbot.waitExposed(window)
    window.close()
    assert not window.isVisible()


# -- Keyboard accessibility --


def test_tool_card_buttons_have_mnemonics(
    qtbot: Any, monkeypatch: Any, settings: AppSettings, patched_dirs: None
) -> None:
    live = FakeLive()

    async def on_connect(session: FakeLiveSession) -> None:
        session.request_tool("fc-k", "write_file", {"path": "a.txt", "content": "hi"})

    live.on_connect(on_connect)
    _patch_dependencies(monkeypatch, live)

    window = MainWindow(settings)
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    qtbot.mouseClick(window._connect_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: "fc-k" in window._tool_cards, timeout=5000)

    card = window._tool_cards["fc-k"]
    assert card._approve_btn.text() == "&Approve"
    assert card._deny_btn.text() == "&Deny"
    assert card._cancel_btn.text() == "&Cancel"

    # Resolve the card before disconnecting to keep the teardown simple.
    card._deny_btn.click()
    qtbot.waitUntil(
        lambda: "Denied" in card._status.text() or "OK" in card._status.text()
        or "FAILED" in card._status.text() or "Cancel" in card._status.text(),
        timeout=5000,
    )

    qtbot.mouseClick(window._connect_btn, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window._connect_btn.text() == "Connect", timeout=5000)
    window.close()


# -- Theme --


def test_theme_stylesheet_and_object_names(
    qtbot: Any, settings: AppSettings, patched_dirs: None
) -> None:
    """The navy/cyan glass theme is wired to the expected widgets."""
    from app.theme import MAIN_QSS

    # Palette sanity: navy base + cyan accent + glass fill present.
    assert "#0A1430" in MAIN_QSS
    assert "#22D3EE" in MAIN_QSS
    assert "rgba(148, 197, 255, 0.07)" in MAIN_QSS

    window = MainWindow(settings)
    qtbot.addWidget(window)

    assert window._settings_widget.objectName() == "glassPanel"
    assert window._connect_btn.objectName() == "accentButton"
    assert window._status_label.objectName() == "statusChip"
    assert window._share_banner.objectName() == "shareBanner"
    assert window._tool_container.objectName() == "toolContainer"
    # The central widget must stay transparent so the window gradient shows.
    assert window.centralWidget().objectName() == "centralRoot"
    window.close()
