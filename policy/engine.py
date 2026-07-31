"""Path and command policy enforcement.

No PySide6 here.
"""

from __future__ import annotations

import ctypes
import fnmatch
import os
import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from core.commands import GatingMode

if TYPE_CHECKING:
    from collections.abc import Iterable


class PathDeniedError(Exception):
    """Raised when ``resolve_path`` rejects a supplied path."""


class DecisionKind(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class Decision:
    kind: DecisionKind
    reason: str = ""

    def is_allowed(self) -> bool:
        return self.kind == DecisionKind.ALLOW

    def is_confirm(self) -> bool:
        return self.kind == DecisionKind.CONFIRM

    def is_denied(self) -> bool:
        return self.kind == DecisionKind.DENY


def Allow() -> Decision:  # noqa: N802
    return Decision(DecisionKind.ALLOW)


def Confirm(reason: str) -> Decision:  # noqa: N802
    return Decision(DecisionKind.CONFIRM, reason)


def Deny(reason: str) -> Decision:  # noqa: N802
    return Decision(DecisionKind.DENY, reason)


# Tools that touch the filesystem.
_PATH_FIELDS: dict[str, list[str]] = {
    "read_file": ["path"],
    "write_file": ["path"],
    "list_directory": ["path"],
    "delete_file": ["path"],
    "run_command": ["working_dir"],
}

_WRITE_TOOLS = {"write_file", "delete_file"}
_DESTRUCTIVE_TOOLS = {"write_file", "delete_file", "run_command"}

# Name patterns that are always off-limits, regardless of location.
_NAME_DENY_GLOBS = (".env*", "*.pem", "*.key", "id_rsa*", "credentials*")


def _long_path(path: Path) -> Path:
    r"""Expand Windows 8.3 / ``\\?\`` path forms when possible."""
    if sys.platform != "win32":
        return path
    try:
        buf = ctypes.create_unicode_buffer(65536)
        result = ctypes.windll.kernel32.GetLongPathNameW(str(path), buf, 65536)
        if result and result < 65536 and buf.value:
            return Path(buf.value)
    except OSError:
        pass
    return path


def _path_key(path: Path) -> str:
    """Return a consistently-cased POSIX key for comparisons."""
    return _long_path(path).as_posix().lower()


def _is_under(child: Path, root: Path) -> bool:
    """Case-insensitive ``is_relative_to`` for Windows, exact elsewhere."""
    child_key = _path_key(child)
    root_key = _path_key(root)
    return child_key == root_key or child_key.startswith(root_key + "/")


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(text, pat) or fnmatch.fnmatch(text.lower(), pat) for pat in patterns)


def _immutable_dirs() -> list[Path]:
    """Return directories that must never be modified by tools."""
    import platformdirs

    return [
        Path(platformdirs.user_config_dir("GeminiLiveAgent", appauthor=False)),
        Path(sys.executable).resolve().parent,
        Path(platformdirs.user_log_dir("GeminiLiveAgent", appauthor=False)) / "journal",
    ]


class PolicyEngine:
    """Enforces allow-roots, deny-globs, name restrictions, and gating mode."""

    def __init__(
        self,
        allow_roots: list[Path],
        deny_globs: list[str],
        never_allow_regex: list[str],
    ) -> None:
        self._allow_roots = [_long_path(r.resolve(strict=False)) for r in allow_roots]
        self._deny_globs = deny_globs
        self._never_allow = [re.compile(p) for p in never_allow_regex]
        self._immutable_roots = [_long_path(p.resolve(strict=False)) for p in _immutable_dirs()]
        self._gating_mode = GatingMode.GUARDED
        self._workspace_root = self._allow_roots[0] if self._allow_roots else Path.cwd()

    def set_workspace(self, root: Path) -> None:
        """Set the default workspace root used during ``evaluate``."""
        self._workspace_root = _long_path(root.resolve(strict=False))

    def set_gating_mode(self, mode: GatingMode) -> None:
        """Toggle between guarded (confirm destructive) and yolo modes."""
        self._gating_mode = mode

    def _deny_by_name(self, path: Path) -> bool:
        """Check immutable name-based deny rules."""
        name = path.name
        posix = path.as_posix()
        return (
            _matches_any(name, _NAME_DENY_GLOBS)
            or _matches_any(posix, _NAME_DENY_GLOBS)
            or (any(part == ".git" for part in path.parts) and name == "config")
        )

    def _is_immutable(self, path: Path) -> bool:
        return any(_is_under(path, root) for root in self._immutable_roots)

    def resolve_path(self, raw: str, workspace_root: Path, *, for_write: bool = False) -> Path:
        r"""Resolve and validate a user-supplied path.

        Steps:
        1. Expand environment variables and ``~``.
        2. Relative paths are joined with ``workspace_root``.
        3. Resolve (for writes, resolve the parent so the target itself can be new).
        4. Normalize ``\\?\`` / 8.3 forms on Windows.
        5. Allow only if under one of the configured ``allow_roots``.
        6. Apply deny globs.
        7. Apply name-based deny patterns (``.env*``, ``*.pem``, etc.).
        """
        expanded = os.path.expandvars(os.path.expanduser(raw))
        p = Path(expanded)
        if not p.is_absolute():
            p = workspace_root / p

        p = (
            p.parent.resolve(strict=False) / p.name
            if for_write
            else p.resolve(strict=False)
        )

        p = _long_path(p)

        if not any(_is_under(p, root) for root in self._allow_roots):
            raise PathDeniedError(f"path escapes allowed roots: {p}")

        if _matches_any(p.as_posix(), self._deny_globs):
            raise PathDeniedError(f"path matches deny glob: {p}")

        if self._deny_by_name(p):
            raise PathDeniedError(f"path matches denied name pattern: {p}")

        return p

    def evaluate(self, name: str, args: BaseModel) -> Decision:
        """Return an allow/confirm/deny decision for a tool invocation."""
        # Never-allow regex applies to paths and, for run_command, the command text.
        texts: list[str] = []
        for field in _PATH_FIELDS.get(name, []):
            value = getattr(args, field, None)
            if value:
                texts.append(str(value))
        if name == "run_command":
            command = getattr(args, "command", "")
            if command:
                texts.append(str(command))

        for text in texts:
            for pattern in self._never_allow:
                if pattern.search(text):
                    return Deny(f"matches forbidden pattern: {pattern.pattern}")

        # Validate path fields.
        for field in _PATH_FIELDS.get(name, []):
            value = getattr(args, field, None)
            if value is None or str(value) == "":
                continue
            try:
                resolved = self.resolve_path(
                    str(value), self._workspace_root, for_write=(name in _WRITE_TOOLS)
                )
            except PathDeniedError as exc:
                return Deny(str(exc))

            if name in _WRITE_TOOLS and self._is_immutable(resolved):
                return Deny(f"immutable path: {resolved}")

        # Destructive tools require confirmation unless yolo mode is active.
        if name in _DESTRUCTIVE_TOOLS and self._gating_mode != GatingMode.YOLO:
            return Confirm("destructive tool requires user confirmation")

        return Allow()
