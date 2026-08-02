"""Tests for main.py entry-point helpers (frozen-app workspace handling)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from main import _guard_frozen_workspace, _resolve_working_dir
from settings.settings import AppSettings


def test_resolve_working_dir_passes_through_explicit_path(tmp_path: Path) -> None:
    assert _resolve_working_dir(str(tmp_path)) == tmp_path


def test_resolve_working_dir_defaults_to_cwd_when_not_frozen(
    monkeypatch: Any,
) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert _resolve_working_dir(".") == Path(".")


def test_resolve_working_dir_defaults_to_documents_when_frozen(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    resolved = _resolve_working_dir(".")

    assert resolved == tmp_path / "Documents" / "GeminiLiveAgent"
    assert resolved.is_dir()  # created on demand


def test_guard_frozen_workspace_redirects_install_dir_workspace(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """A persisted Program Files workspace must be replaced when frozen."""
    install_dir = tmp_path / "Program Files" / "GeminiLiveAgent"
    install_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(install_dir / "gemini-live-agent.exe"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    settings = AppSettings(
        working_dir=install_dir,
        workspace_root=install_dir,
    )
    guarded = _guard_frozen_workspace(settings)

    assert guarded.working_dir == tmp_path / "Documents" / "GeminiLiveAgent"
    assert guarded.workspace_root == tmp_path / "Documents" / "GeminiLiveAgent"


def test_guard_frozen_workspace_keeps_user_workspace(
    monkeypatch: Any, tmp_path: Path
) -> None:
    install_dir = tmp_path / "Program Files" / "GeminiLiveAgent"
    install_dir.mkdir(parents=True)
    user_ws = tmp_path / "somewhere-else"
    user_ws.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(install_dir / "gemini-live-agent.exe"))

    settings = AppSettings(working_dir=user_ws, workspace_root=user_ws)
    guarded = _guard_frozen_workspace(settings)

    assert guarded.working_dir == user_ws


def test_guard_is_a_noop_when_not_frozen(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    settings = AppSettings(working_dir=tmp_path, workspace_root=tmp_path)
    assert _guard_frozen_workspace(settings) is settings
