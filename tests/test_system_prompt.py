"""Tests for the customizable system prompt (settings field + GUI box)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from google.genai import types

from app.main_window import MainWindow
from settings.settings import AppSettings


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


def _instruction_text(settings: AppSettings) -> str:
    config = settings.as_connect_config(tools=[])
    instruction = config.system_instruction
    assert isinstance(instruction, types.Content)
    assert instruction.parts is not None
    return "".join(
        part.text or "" for part in instruction.parts if isinstance(part, types.Part)
    )


# -- Settings / connect config --


def test_default_system_prompt_leaves_template_unchanged(
    workspace: Path,
) -> None:
    settings = _make_settings(workspace)
    text = _instruction_text(settings)
    assert "Additional Instructions" not in text
    assert "Your Tools" in text  # built-in template still intact


def test_custom_system_prompt_is_appended(workspace: Path) -> None:
    settings = _make_settings(
        workspace, system_prompt="Always answer in pirate speak."
    )
    text = _instruction_text(settings)
    assert "Your Tools" in text  # built-in template preserved
    assert "## Additional Instructions" in text
    assert "Always answer in pirate speak." in text
    assert text.index("Your Tools") < text.index("Additional Instructions")


def test_whitespace_only_system_prompt_is_ignored(workspace: Path) -> None:
    settings = _make_settings(workspace, system_prompt="   \n  ")
    assert "Additional Instructions" not in _instruction_text(settings)


def test_system_prompt_round_trips_through_settings_file(
    monkeypatch: Any, tmp_path: Path, workspace: Path
) -> None:
    monkeypatch.setattr(
        "settings.settings.user_config_dir",
        lambda appname, appauthor=None, **kwargs: str(tmp_path),
    )
    settings = _make_settings(workspace, system_prompt="Be terse.")
    settings.save()

    loaded = AppSettings.load(
        working_dir=workspace, workspace_root=workspace
    )
    assert loaded.system_prompt == "Be terse."


# -- GUI box --


def test_gui_loads_system_prompt_into_box(
    qtbot: Any, workspace: Path
) -> None:
    settings = _make_settings(workspace, system_prompt="Be terse.")
    window = MainWindow(settings)
    qtbot.addWidget(window)

    assert window._system_prompt_edit.toPlainText() == "Be terse."


def test_gui_box_round_trips_into_built_settings(
    qtbot: Any, workspace: Path
) -> None:
    settings = _make_settings(workspace)
    window = MainWindow(settings)
    qtbot.addWidget(window)

    assert window._system_prompt_edit.toPlainText() == ""

    window._system_prompt_edit.setPlainText("  Speak like a robot.  ")
    built = window._build_settings()
    assert built.system_prompt == "Speak like a robot."
