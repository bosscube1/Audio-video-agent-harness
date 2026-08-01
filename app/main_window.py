"""Minimal PySide6 main window for the Gemini Live Agent.

Consumes the same frozen event/command vocabulary as the headless driver. The
asyncio Supervisor lives on a ``Bridge`` QThread; the main window only touches
Qt objects and ``AgentEvent`` / ``AgentCommand`` dataclasses.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, Slot
from PySide6.QtGui import QAction, QTextCursor
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
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.bridge import Bridge
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
    ToolApprovalRequested,
    ToolCallCancelled,
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


class _ToolCard(QFrame):
    """Small widget that asks the user to approve/deny/cancel a tool call."""

    def __init__(self, event: ToolApprovalRequested, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.call_id = event.call_id
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        layout.addWidget(QLabel(f"Tool: <b>{event.name}</b>"))
        layout.addWidget(QLabel(f"Args: {event.args}"))
        if event.allow_text:
            layout.addWidget(QLabel(f"Policy: {event.allow_text}"))

        self._status = QLabel("Pending approval...")
        layout.addWidget(self._status)

        btn_layout = QHBoxLayout()
        self._approve_btn = QPushButton("Approve")
        self._deny_btn = QPushButton("Deny")
        self._cancel_btn = QPushButton("Cancel")
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

        self.setWindowTitle("Gemini Live Agent")
        self.setMinimumSize(720, 520)
        self._build_ui()
        self._populate_devices()
        self._load_settings_to_ui()

    # -- UI construction --

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)

        # Status bar + connect button
        top_layout = QHBoxLayout()
        self._status_label = QLabel("Disconnected")
        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        top_layout.addWidget(self._status_label)
        top_layout.addStretch()
        top_layout.addWidget(self._connect_btn)
        main_layout.addLayout(top_layout)

        # Settings panel
        self._settings_widget = QWidget()
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

        # Transcript
        self._transcript = QTextEdit()
        self._transcript.setReadOnly(True)
        self._transcript.setPlaceholderText("Transcript will appear here...")
        main_layout.addWidget(self._transcript, stretch=1)

        # Tool approvals
        self._tool_scroll = QScrollArea()
        self._tool_scroll.setWidgetResizable(True)
        self._tool_container = QWidget()
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
        self._input_edit.returnPressed.connect(self._on_send_clicked)
        self._send_btn = QPushButton("Send")
        self._send_btn.clicked.connect(self._on_send_clicked)
        self._mute_btn = QPushButton("Mute mic")
        self._mute_btn.setCheckable(True)
        self._mute_btn.toggled.connect(self._on_mute_toggled)
        bottom_layout.addWidget(self._input_edit)
        bottom_layout.addWidget(self._send_btn)
        bottom_layout.addWidget(self._mute_btn)
        main_layout.addLayout(bottom_layout)

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

    # -- Event handling --

    @Slot(object)
    def _on_event(self, event: AgentEvent) -> None:
        if isinstance(event, ConnectionStateChanged):
            self._status_label.setText(event.state.value)
            if event.detail:
                self._status_label.setToolTip(event.detail)
            if event.state == ConnectionState.DISCONNECTED:
                self._connect_btn.setText("Connect")
                self._settings_widget.setEnabled(True)
            elif event.state == ConnectionState.LIVE:
                self._connect_btn.setText("Disconnect")
                self._settings_widget.setEnabled(False)
        elif isinstance(event, PartialTranscript):
            self._append_partial(event.text, event.source)
        elif isinstance(event, TurnComplete):
            self._append_system(f"[TURN COMPLETE] {event.source.value}")
            self._last_source = None
        elif isinstance(event, ToolApprovalRequested):
            self._add_tool_card(event)
        elif isinstance(event, ToolResultSent):
            self._update_tool_card(event)
        elif isinstance(event, ToolCallCancelled):
            self._update_tool_card_cancelled(event)
        elif isinstance(event, ContextReset):
            self._append_system(f"[RESET] {event.reason}")
        elif isinstance(event, SessionError):
            if event.fatal:
                QMessageBox.critical(self, "Session error", event.message)
            else:
                self._append_system(f"[ERROR] {event.message}")
        elif isinstance(event, UsageUpdate):
            # Intentionally not displayed in the minimal UI.
            pass
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
        if self._bridge is not None:
            self._bridge.stop()
            self._bridge.wait()
            self._bridge.deleteLater()
            self._bridge = None
        event.accept()


def run_gui(settings: AppSettings) -> int:
    """Create the QApplication, prompt for an API key if needed, and run the GUI."""
    app = QApplication(sys.argv)

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
