"""Minimal PySide6 main window for the Gemini Live Agent.

Consumes the same frozen event/command vocabulary as the headless driver. The
asyncio Supervisor lives on a ``Bridge`` QThread; the main window only touches
Qt objects and ``AgentEvent`` / ``AgentCommand`` dataclasses.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStyle,
    QSystemTrayIcon,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.bridge import Bridge
from app.theme import DANGER, MAIN_QSS, SUCCESS, TEXT_MUTED, apply_glass_shadow
from core.commands import (
    ApproveTool,
    CancelTool,
    DenyTool,
    GatingMode,
    SendText,
    SetGatingMode,
    SetMicGate,
    SetShareScreen,
)
from core.events import (
    AgentEvent,
    AudioLevel,
    ConnectionState,
    ConnectionStateChanged,
    ContextReset,
    PartialTranscript,
    SessionError,
    SessionExpiring,
    ToolApprovalRequested,
    ToolCallCancelled,
    ToolCallReceived,
    ToolResultSent,
    TranscriptSource,
    TurnComplete,
    UsageUpdate,
)
from media import devices as media_devices
from obs.logging_setup import get_run_id, setup_logging
from settings.secrets import get_api_key, set_api_key
from settings.settings import AppSettings

_AVAILABLE_VOICES = ("Puck", "Charon", "Aoede", "Fenrir", "Kore")
_MODES = ("text", "voice", "both")
_METER_INTERVAL_MS = 33

logger = logging.getLogger(__name__)


class _MicState(StrEnum):
    """GUI-side microphone state machine (voice UX)."""

    OFF = "off"
    LIVE = "live"
    MUTED = "muted"


_MIC_CHIP_STYLES = {
    _MicState.OFF: f"color: {TEXT_MUTED};",
    _MicState.LIVE: f"color: {SUCCESS}; font-weight: bold;",
    _MicState.MUTED: f"color: {DANGER}; font-weight: bold;",
}

_MIC_CHIP_TEXT = {
    _MicState.OFF: "Mic: off",
    _MicState.LIVE: "Mic: live",
    _MicState.MUTED: "Mic: muted",
}


class _ToolCard(QFrame):
    """Small widget that asks the user to approve/deny/cancel a tool call."""

    def __init__(self, event: ToolApprovalRequested, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.call_id = event.call_id
        self.setObjectName("glassCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setAccessibleName(f"Tool approval request: {event.name}")
        apply_glass_shadow(self, blur=18, dy=4)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        layout.addWidget(QLabel(f"Tool: <b>{event.name}</b>"))
        layout.addWidget(QLabel(f"Args: {event.args}"))
        if event.allow_text:
            layout.addWidget(QLabel(f"Policy: {event.allow_text}"))

        self._status = QLabel("Pending approval...")
        layout.addWidget(self._status)

        btn_layout = QHBoxLayout()
        # Mnemonics make the full approve/deny cycle keyboard-only.
        self._approve_btn = QPushButton("&Approve")
        self._approve_btn.setAccessibleName(f"Approve {event.name}")
        self._deny_btn = QPushButton("&Deny")
        self._deny_btn.setAccessibleName(f"Deny {event.name}")
        self._cancel_btn = QPushButton("&Cancel")
        self._cancel_btn.setAccessibleName(f"Cancel {event.name}")
        btn_layout.addWidget(self._approve_btn)
        btn_layout.addWidget(self._deny_btn)
        btn_layout.addWidget(self._cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def set_result(self, message: str) -> None:
        """Disable controls and show the resolved outcome."""
        self._status.setText(message)
        self._approve_btn.setEnabled(False)
        self._deny_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)


class MainWindow(QMainWindow):
    """Primary GUI window for the Gemini Live Agent."""

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._bridge: Bridge | None = None
        self._last_source: TranscriptSource | None = None
        self._tool_cards: dict[str, _ToolCard] = {}
        self._mic_muted = False
        self._mic_state = _MicState.OFF
        self._sharing_active = False
        self._force_quit = False
        self._tray: QSystemTrayIcon | None = None
        self._tray_message_shown = False

        self.setWindowTitle("Gemini Live Agent")
        self.setMinimumSize(720, 520)
        self._build_ui()
        self._populate_devices()
        self._load_settings_to_ui()
        self._setup_tray()

    # -- UI construction --

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralRoot")
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)

        # Status bar + mic state chip + connect button
        top_layout = QHBoxLayout()
        self._status_label = QLabel("Disconnected")
        self._status_label.setObjectName("statusChip")
        self._mic_chip = QLabel(_MIC_CHIP_TEXT[_MicState.OFF])
        self._mic_chip.setStyleSheet(_MIC_CHIP_STYLES[_MicState.OFF])
        self._mic_chip.setAccessibleName("Microphone state")
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.setObjectName("accentButton")
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        top_layout.addWidget(self._status_label)
        top_layout.addWidget(self._mic_chip)
        top_layout.addStretch()
        top_layout.addWidget(self._connect_btn)
        main_layout.addLayout(top_layout)

        # Settings panel
        self._settings_widget = QWidget()
        self._settings_widget.setObjectName("glassPanel")
        apply_glass_shadow(self._settings_widget)
        settings_layout = QGridLayout(self._settings_widget)
        settings_layout.setSpacing(6)
        main_layout.addWidget(self._settings_widget)

        row = 0
        self._model_edit = QLineEdit()
        settings_layout.addWidget(QLabel("Model:"), row, 0)
        settings_layout.addWidget(self._model_edit, row, 1)

        row += 1
        self._voice_combo = QComboBox()
        self._voice_combo.addItems(_AVAILABLE_VOICES)
        settings_layout.addWidget(QLabel("Voice:"), row, 0)
        settings_layout.addWidget(self._voice_combo, row, 1)

        row += 1
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([m.title() for m in _MODES])
        settings_layout.addWidget(QLabel("Mode:"), row, 0)
        settings_layout.addWidget(self._mode_combo, row, 1)

        row += 1
        self._share_screen_check = QCheckBox("Share screen")
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.1, 30.0)
        self._fps_spin.setDecimals(1)
        self._fps_spin.setSuffix(" fps")
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(self._share_screen_check)
        fps_layout.addWidget(QLabel("FPS:"))
        fps_layout.addWidget(self._fps_spin)
        fps_layout.addStretch()
        settings_layout.addLayout(fps_layout, row, 0, 1, 2)

        row += 1
        self._input_device_combo = QComboBox()
        self._input_device_combo.addItem("(default)", None)
        settings_layout.addWidget(QLabel("Input device:"), row, 0)
        settings_layout.addWidget(self._input_device_combo, row, 1)

        row += 1
        self._output_device_combo = QComboBox()
        self._output_device_combo.addItem("(default)", None)
        settings_layout.addWidget(QLabel("Output device:"), row, 0)
        settings_layout.addWidget(self._output_device_combo, row, 1)

        row += 1
        self._yolo_check = QCheckBox("Skip approvals (yolo)")
        settings_layout.addWidget(self._yolo_check, row, 0, 1, 2)

        row += 1
        self._tray_check = QCheckBox("Minimize to tray on close")
        settings_layout.addWidget(self._tray_check, row, 0, 1, 2)

        row += 1
        self._system_prompt_edit = QPlainTextEdit()
        self._system_prompt_edit.setPlaceholderText(
            "Extra instructions for the model (optional). Appended to the "
            "built-in system prompt; applies on the next connect."
        )
        self._system_prompt_edit.setMaximumHeight(72)
        self._system_prompt_edit.setAccessibleName("Custom system prompt")
        system_prompt_label = QLabel("System prompt:")
        settings_layout.addWidget(
            system_prompt_label, row, 0, Qt.AlignmentFlag.AlignTop
        )
        settings_layout.addWidget(self._system_prompt_edit, row, 1)

        row += 1
        self._wake_word_check = QCheckBox("Wake-word gating (voice)")
        self._wake_word_edit = QLineEdit()
        self._wake_word_edit.setPlaceholderText("e.g. Bongo")
        self._wake_word_edit.setAccessibleName("Wake word")
        self._wake_word_edit.setMaximumWidth(180)
        wake_layout = QHBoxLayout()
        wake_layout.addWidget(self._wake_word_check)
        wake_layout.addWidget(QLabel("Wake word:"))
        wake_layout.addWidget(self._wake_word_edit)
        wake_layout.addStretch()
        settings_layout.addLayout(wake_layout, row, 0, 1, 2)

        row += 1
        self._working_dir_edit = QLineEdit()
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse_working_dir)
        wd_layout = QHBoxLayout()
        wd_layout.addWidget(self._working_dir_edit)
        wd_layout.addWidget(browse_btn)
        settings_layout.addWidget(QLabel("Working dir:"), row, 0)
        settings_layout.addLayout(wd_layout, row, 1)

        row += 1
        save_settings_btn = QPushButton("Save settings")
        save_settings_btn.clicked.connect(self._on_save_settings)
        settings_layout.addWidget(save_settings_btn, row, 1)

        # Screen-share indicator banner (visible only while sharing + connected)
        self._share_banner = QLabel("● Sharing screen — the model can see your screen")
        self._share_banner.setObjectName("shareBanner")
        self._share_banner.setAccessibleName("Screen sharing indicator")
        self._share_banner.setVisible(False)
        main_layout.addWidget(self._share_banner)

        # Transcript + activity feed tabs
        self._tabs = QTabWidget()

        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setPlaceholderText("Transcript will appear here...")
        self._transcript.setAccessibleName("Conversation transcript")
        self._tabs.addTab(self._transcript, "Transcript")

        self._activity = QListWidget()
        self._activity.setAccessibleName("Activity feed")
        self._tabs.addTab(self._activity, "Activity")

        main_layout.addWidget(self._tabs, stretch=1)

        # Tool approvals
        self._tool_scroll = QScrollArea()
        self._tool_scroll.setWidgetResizable(True)
        self._tool_container = QWidget()
        self._tool_container.setObjectName("toolContainer")
        self._tool_layout = QVBoxLayout(self._tool_container)
        self._tool_layout.addStretch()
        self._tool_scroll.setWidget(self._tool_container)
        self._tool_scroll.setMaximumHeight(180)
        self._tool_scroll.setVisible(False)
        main_layout.addWidget(self._tool_scroll)

        # Audio meters
        meter_layout = QHBoxLayout()
        self._mic_meter = QProgressBar()
        self._mic_meter.setRange(0, 100)
        self._mic_meter.setTextVisible(True)
        self._mic_meter.setFormat("Mic: %p%")
        self._speaker_meter = QProgressBar()
        self._speaker_meter.setRange(0, 100)
        self._speaker_meter.setTextVisible(True)
        self._speaker_meter.setFormat("Speaker: %p%")
        meter_layout.addWidget(self._mic_meter)
        meter_layout.addWidget(self._speaker_meter)
        main_layout.addLayout(meter_layout)

        # Input bar + mic mute
        bottom_layout = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("Type a message and press Enter...")
        self._input_edit.setAccessibleName("Message input")
        self._input_edit.returnPressed.connect(self._on_send_clicked)
        self._send_btn = QPushButton("Send")
        self._send_btn.clicked.connect(self._on_send_clicked)
        self._mute_btn = QPushButton("Mute mic")
        self._mute_btn.setCheckable(True)
        self._mute_btn.setAccessibleName("Toggle microphone mute")
        self._mute_btn.toggled.connect(self._on_mute_toggled)
        bottom_layout.addWidget(self._input_edit)
        bottom_layout.addWidget(self._send_btn)
        bottom_layout.addWidget(self._mute_btn)
        main_layout.addLayout(bottom_layout)

        # Usage panel in the status bar (updated on UsageUpdate events)
        self._usage_label = QLabel("0 tokens")
        self._usage_label.setAccessibleName("Token usage")
        self.statusBar().addPermanentWidget(self._usage_label)

        # Meter timer
        self._meter_timer = QTimer(self)
        self._meter_timer.setInterval(_METER_INTERVAL_MS)
        self._meter_timer.timeout.connect(self._update_meters)
        self._meter_timer.start()

        # Menu bar shortcut for exit
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        self.menuBar().addAction(exit_action)

        # Ctrl+M toggles the mic gate from anywhere in the window.
        mute_shortcut = QShortcut(QKeySequence("Ctrl+M"), self)
        mute_shortcut.activated.connect(self._mute_btn.toggle)

    def _populate_devices(self) -> None:
        """Fill the input/output device combos from sounddevice."""
        try:
            for device in media_devices.list_input_devices():
                name = device.get("name")
                if name:
                    self._input_device_combo.addItem(name, name)
        except Exception as exc:
            logger.warning("Failed to enumerate input devices: %s", exc)

        try:
            for device in media_devices.list_output_devices():
                name = device.get("name")
                if name:
                    self._output_device_combo.addItem(name, name)
        except Exception as exc:
            logger.warning("Failed to enumerate output devices: %s", exc)

    def _load_settings_to_ui(self) -> None:
        """Populate controls from the initial AppSettings."""
        self._model_edit.setText(self._settings.model)
        self._voice_combo.setCurrentText(self._settings.voice)
        self._mode_combo.setCurrentText(self._settings.mode.title())
        self._share_screen_check.setChecked(self._settings.share_screen)
        self._fps_spin.setValue(self._settings.screen_fps)
        self._yolo_check.setChecked(self._settings.yolo)
        self._tray_check.setChecked(self._settings.minimize_to_tray)
        self._system_prompt_edit.setPlainText(self._settings.system_prompt)
        self._wake_word_check.setChecked(self._settings.wake_word_enabled)
        self._wake_word_edit.setText(self._settings.wake_word)
        self._working_dir_edit.setText(str(self._settings.working_dir))
        if self._settings.input_device:
            self._input_device_combo.setCurrentText(self._settings.input_device)
        if self._settings.output_device:
            self._output_device_combo.setCurrentText(self._settings.output_device)

    # -- Settings handling --

    def _build_settings(self) -> AppSettings:
        """Construct a validated AppSettings from the current UI values."""
        working_dir = Path(self._working_dir_edit.text().strip() or ".")
        return AppSettings(
            model=self._model_edit.text().strip(),
            voice=self._voice_combo.currentText(),
            mode=self._mode_combo.currentText().lower(),
            working_dir=working_dir,
            workspace_root=working_dir,
            share_screen=self._share_screen_check.isChecked(),
            screen_fps=self._fps_spin.value(),
            input_device=self._input_device_combo.currentData(),
            output_device=self._output_device_combo.currentData(),
            yolo=self._yolo_check.isChecked(),
            minimize_to_tray=self._tray_check.isChecked(),
            system_prompt=self._system_prompt_edit.toPlainText().strip(),
            wake_word_enabled=self._wake_word_check.isChecked(),
            wake_word=self._wake_word_edit.text().strip() or "Bongo",
            headless=False,
            debug=self._settings.debug,
        )

    @Slot()
    def _on_save_settings(self) -> None:
        try:
            self._build_settings().save()
        except Exception as exc:
            QMessageBox.warning(self, "Settings error", str(exc))

    @Slot()
    def _on_browse_working_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select working directory",
            self._working_dir_edit.text() or str(Path.cwd()),
        )
        if path:
            self._working_dir_edit.setText(path)

    # -- System tray --

    def _setup_tray(self) -> None:
        """Create the tray icon when the platform supports it."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.info("System tray not available; minimize-to-tray disabled")
            return

        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self._tray = QSystemTrayIcon(icon, self)
        self._tray.setToolTip("Gemini Live Agent")

        menu = QMenu()
        show_action = menu.addAction("Show / Hide")
        show_action.triggered.connect(self._on_tray_show_hide)
        mute_action = menu.addAction("Mute mic")
        mute_action.triggered.connect(self._mute_btn.toggle)
        menu.addSeparator()
        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(self._on_tray_quit)
        self._tray.setContextMenu(menu)

        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _on_tray_show_hide(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.showNormal()
            self.activateWindow()

    def _on_tray_quit(self) -> None:
        self._force_quit = True
        self.close()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._on_tray_show_hide()

    # -- Activity feed / usage / mic chip / share banner --

    def _add_activity(self, text: str) -> None:
        """Append a timestamped line to the activity feed."""
        stamp = datetime.now().strftime("%H:%M:%S")
        self._activity.addItem(f"{stamp}  {text}")
        self._activity.scrollToBottom()

    def _update_usage(self, event: UsageUpdate) -> None:
        parts = [f"{event.total_tokens} tokens"]
        detail = f"in {event.prompt_tokens} / out {event.completion_tokens}"
        if event.cached_tokens:
            detail += f" / cached {event.cached_tokens}"
        parts.append(detail)
        if event.estimated_usd:
            parts.append(f"~${event.estimated_usd:.4f}")
        self._usage_label.setText("  ·  ".join(parts))
        self._usage_label.setToolTip(
            f"audio {event.audio_tokens} · video {event.video_tokens}"
        )

    def _set_mic_state(self, state: _MicState) -> None:
        if state == self._mic_state:
            return
        self._mic_state = state
        self._mic_chip.setText(_MIC_CHIP_TEXT[state])
        self._mic_chip.setStyleSheet(_MIC_CHIP_STYLES[state])
        self._add_activity(_MIC_CHIP_TEXT[state])

    def _refresh_mic_state(self) -> None:
        """Recompute the mic chip from connection state, mode, and mute flag."""
        connected = self._bridge is not None and self._status_label.text() in (
            ConnectionState.LIVE.value,
            ConnectionState.DRAINING.value,
        )
        wants_mic = self._mode_combo.currentText().lower() != "text"
        if not connected or not wants_mic:
            self._set_mic_state(_MicState.OFF)
        elif self._mic_muted:
            self._set_mic_state(_MicState.MUTED)
        else:
            self._set_mic_state(_MicState.LIVE)

    def _update_share_banner(self) -> None:
        self._share_banner.setVisible(self._sharing_active)

    # -- Bridge lifecycle --

    @Slot()
    def _on_connect_clicked(self) -> None:
        if self._bridge is not None:
            self._bridge.stop()
            return

        try:
            settings = self._build_settings()
            settings.save()
        except Exception as exc:
            QMessageBox.warning(self, "Settings error", str(exc))
            return

        self._bridge = Bridge(settings, run_id=get_run_id())
        self._bridge.event_emitted.connect(self._on_event)
        self._bridge.finished_with_reason.connect(self._on_bridge_finished)
        self._bridge.start()

        self._connect_btn.setText("Disconnect")
        self._settings_widget.setEnabled(False)

        self._sharing_active = settings.share_screen
        self._update_share_banner()
        self._add_activity(f"Connecting (model {settings.model})...")

        # Reflect UI toggles in the freshly started core.
        if settings.yolo:
            self._bridge.send_command(SetGatingMode(mode=GatingMode.YOLO))
        self._bridge.send_command(SetMicGate(open=not self._mic_muted))
        self._bridge.send_command(
            SetShareScreen(
                enabled=settings.share_screen,
                monitor=settings.screen_monitor,
            )
        )

    @Slot(str)
    def _on_bridge_finished(self, reason: str) -> None:
        """Clean up when the Supervisor loop exits."""
        if self._bridge is not None:
            self._bridge.wait()
            self._bridge.deleteLater()
            self._bridge = None
        self._status_label.setText(f"Disconnected ({reason})")
        self._connect_btn.setText("Connect")
        self._settings_widget.setEnabled(True)
        self._mic_meter.setValue(0)
        self._speaker_meter.setValue(0)
        self._sharing_active = False
        self._update_share_banner()
        self._refresh_mic_state()
        self._add_activity(f"Disconnected ({reason})")

    # -- Event handling --

    @Slot(object)
    def _on_event(self, event: AgentEvent) -> None:
        if isinstance(event, ConnectionStateChanged):
            self._status_label.setText(event.state.value)
            self._add_activity(
                f"Connection: {event.state.value}"
                + (f" — {event.detail}" if event.detail else "")
            )
            if event.detail:
                self._status_label.setToolTip(event.detail)
            if event.state == ConnectionState.DISCONNECTED:
                self._connect_btn.setText("Connect")
                self._settings_widget.setEnabled(True)
            elif event.state == ConnectionState.LIVE:
                self._connect_btn.setText("Disconnect")
                self._settings_widget.setEnabled(False)
            self._refresh_mic_state()
        elif isinstance(event, PartialTranscript):
            self._append_partial(event.text, event.source)
        elif isinstance(event, TurnComplete):
            self._append_system(f"[TURN COMPLETE] {event.source.value}")
            self._last_source = None
        elif isinstance(event, ToolCallReceived):
            self._add_activity(f"Tool call: {event.name}")
        elif isinstance(event, ToolApprovalRequested):
            self._add_tool_card(event)
            self._add_activity(f"Approval requested: {event.name}")
        elif isinstance(event, ToolResultSent):
            self._update_tool_card(event)
            self._add_activity(
                f"Tool result: {event.name} -> {'OK' if event.ok else 'FAILED'}"
            )
        elif isinstance(event, ToolCallCancelled):
            self._update_tool_card_cancelled(event)
            self._add_activity(f"Tool cancelled: {event.reason}")
        elif isinstance(event, ContextReset):
            self._append_system(f"[RESET] {event.reason}")
            self._add_activity(f"Context reset: {event.reason}")
        elif isinstance(event, SessionExpiring):
            notice = (
                f"Session expiring in {event.seconds}s; reconnecting..."
                if event.seconds is not None
                else "Session expiring; reconnecting..."
            )
            self._append_system(f"[EXPIRING] {notice}")
            self._add_activity(notice)
        elif isinstance(event, SessionError):
            self._add_activity(f"Error: {event.message}")
            if event.fatal:
                QMessageBox.critical(self, "Session error", event.message)
            else:
                self._append_system(f"[ERROR] {event.message}")
        elif isinstance(event, UsageUpdate):
            self._update_usage(event)
        elif isinstance(event, AudioLevel):
            # Levels are polled via _update_meters; ignore pushed events.
            pass

    def _append_partial(self, text: str, source: TranscriptSource) -> None:
        if self._last_source != source:
            self._insert_at_end("\n")
            prefix = "[USER] " if source == TranscriptSource.USER else "[MODEL] "
            self._insert_at_end(prefix)
            self._last_source = source
        self._insert_at_end(text)
        self._transcript.ensureCursorVisible()

    def _append_system(self, text: str) -> None:
        self._insert_at_end(f"\n{text}\n")
        self._transcript.ensureCursorVisible()
        self._last_source = None

    def _insert_at_end(self, text: str) -> None:
        cursor = self._transcript.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._transcript.setTextCursor(cursor)
        self._transcript.insertPlainText(text)

    # -- Tool cards --

    def _add_tool_card(self, event: ToolApprovalRequested) -> None:
        self._tool_scroll.setVisible(True)
        card = _ToolCard(event)
        card._approve_btn.clicked.connect(
            lambda _checked=False, call_id=event.call_id: self._approve_tool(call_id)
        )
        card._deny_btn.clicked.connect(
            lambda _checked=False, call_id=event.call_id: self._deny_tool(call_id)
        )
        card._cancel_btn.clicked.connect(
            lambda _checked=False, call_id=event.call_id: self._cancel_tool(call_id)
        )
        self._tool_cards[event.call_id] = card
        self._tool_layout.insertWidget(0, card)

    @Slot()
    def _approve_tool(self, call_id: str) -> None:
        if self._bridge is not None:
            self._bridge.send_command(ApproveTool(call_id=call_id))

    @Slot()
    def _deny_tool(self, call_id: str) -> None:
        if self._bridge is not None:
            self._bridge.send_command(DenyTool(call_id=call_id))

    @Slot()
    def _cancel_tool(self, call_id: str) -> None:
        if self._bridge is not None:
            self._bridge.send_command(CancelTool(call_ids=[call_id]))

    def _update_tool_card(self, event: ToolResultSent) -> None:
        card = self._tool_cards.get(event.call_id)
        if card is None:
            return
        marker = "OK" if event.ok else "FAILED"
        message = f"{marker}: {event.result}"
        if event.orphaned:
            message += " (orphaned)"
        card.set_result(message)
        self._append_system(f"[TOOL] {event.name} -> {message}")

    def _update_tool_card_cancelled(self, event: ToolCallCancelled) -> None:
        for call_id in event.call_ids:
            card = self._tool_cards.get(call_id)
            if card is None:
                continue
            card.set_result(f"Cancelled: {event.reason}")

    # -- Input and meters --

    @Slot()
    def _on_send_clicked(self) -> None:
        text = self._input_edit.text().strip()
        if not text or self._bridge is None:
            return
        self._bridge.send_command(SendText(text=text))
        self._input_edit.clear()

    @Slot(bool)
    def _on_mute_toggled(self, checked: bool) -> None:
        self._mic_muted = checked
        self._mute_btn.setText("Unmute mic" if checked else "Mute mic")
        if self._bridge is not None:
            self._bridge.send_command(SetMicGate(open=not checked))
        self._refresh_mic_state()

    @Slot()
    def _update_meters(self) -> None:
        if self._bridge is None:
            self._mic_meter.setValue(0)
            self._speaker_meter.setValue(0)
            return
        self._mic_meter.setValue(int(self._bridge.mic_level * 100))
        self._speaker_meter.setValue(int(self._bridge.speaker_level * 100))

    # -- Window close --

    def closeEvent(self, event: Any) -> None:
        # Minimize to the tray instead of quitting when the setting is on.
        if (
            not self._force_quit
            and self._tray_check.isChecked()
            and self._tray is not None
            and self._tray.isVisible()
        ):
            self.hide()
            if not self._tray_message_shown:
                self._tray_message_shown = True
                self._tray.showMessage(
                    "Gemini Live Agent",
                    "Still running in the tray. Use Quit in the tray menu to exit.",
                    QSystemTrayIcon.MessageIcon.Information,
                    3000,
                )
            event.ignore()
            return

        if self._bridge is not None:
            self._bridge.stop()
            self._bridge.wait()
            self._bridge.deleteLater()
            self._bridge = None
        if self._tray is not None:
            self._tray.hide()
        event.accept()


def run_gui(settings: AppSettings) -> int:
    """Create the QApplication, prompt for an API key if needed, and run the GUI."""
    app = QApplication(sys.argv)
    app.setStyleSheet(MAIN_QSS)

    if get_api_key() is None:
        key, ok = QInputDialog.getText(
            None,
            "API key required",
            "No Google API key found. Enter your API key:",
            QLineEdit.EchoMode.Password,
        )
        if not ok or not key:
            QMessageBox.critical(None, "Error", "An API key is required to continue.")
            return 1
        set_api_key(key)

    setup_logging(run_id=get_run_id(), debug=settings.debug)

    window = MainWindow(settings)
    window.show()
    return app.exec()
