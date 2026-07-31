"""Filesystem tool implementations.

No PySide6 here.
"""

from __future__ import annotations

import tools._state as _state
from policy import PathDeniedError
from tools.registry import (
    DeleteFileArgs,
    ListDirectoryArgs,
    ReadFileArgs,
    ToolResult,
    WriteFileArgs,
)

MAX_READ_BYTES = 16 * 1024  # 16 KB


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.1f} GB"


def read_file(args: ReadFileArgs) -> ToolResult:
    """Read a file, optionally constrained to a line range."""
    decision = _state._policy.evaluate("read_file", args)
    if decision.is_denied():
        return ToolResult(False, f"Policy denied: {decision.reason}")

    try:
        path = _state._policy.resolve_path(args.path, _state._workspace_root, for_write=False)
    except PathDeniedError as exc:
        return ToolResult(False, str(exc))

    if not path.is_file():
        return ToolResult(False, f"Error: Not a file: {path}")

    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return ToolResult(False, f"Error reading file: {exc}")

    total_lines = len(lines)
    start = max(1, args.start_line or 1) - 1
    end = min(total_lines, args.end_line or total_lines)

    if args.start_line is not None or args.end_line is not None:
        selected = lines[start:end]
        header = f"[Lines {start + 1}-{end} of {total_lines} total]\n"
    else:
        selected = lines
        header = f"[{total_lines} lines, {path.stat().st_size} bytes]\n"

    content = "".join(selected)
    if len(content) > MAX_READ_BYTES:
        content = (
            content[:MAX_READ_BYTES]
            + f"\n\n… (truncated — file is {len(content):,} bytes total)"
        )

    return ToolResult(True, header + content)


def write_file(args: WriteFileArgs) -> ToolResult:
    """Write content to a file, creating parent directories as needed."""
    decision = _state._policy.evaluate("write_file", args)
    if decision.is_denied():
        return ToolResult(False, f"Policy denied: {decision.reason}")

    try:
        path = _state._policy.resolve_path(args.path, _state._workspace_root, for_write=True)
    except PathDeniedError as exc:
        return ToolResult(False, str(exc))

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(args.content)
        return ToolResult(True, f"Successfully wrote {len(args.content):,} bytes to {path}")
    except OSError as exc:
        return ToolResult(False, f"Error writing file: {exc}")


def list_directory(args: ListDirectoryArgs) -> ToolResult:
    """List the contents of a directory."""
    decision = _state._policy.evaluate("list_directory", args)
    if decision.is_denied():
        return ToolResult(False, f"Policy denied: {decision.reason}")

    try:
        path = _state._policy.resolve_path(args.path, _state._workspace_root, for_write=False)
    except PathDeniedError as exc:
        return ToolResult(False, str(exc))

    if not path.is_dir():
        return ToolResult(False, f"Error: Not a directory: {path}")

    try:
        items = sorted(path.iterdir())
    except PermissionError as exc:
        return ToolResult(False, f"Error: Permission denied: {path} ({exc})")

    entries: list[str] = []
    for item in items:
        try:
            if item.is_dir():
                child_count: int | str
                try:
                    child_count = sum(1 for _ in item.iterdir())
                except PermissionError:
                    child_count = "?"
                entries.append(f"[DIR]  {item.name}/ ({child_count} items)")
            else:
                size = item.stat().st_size
                entries.append(f"[FILE] {item.name} ({_human_size(size)})")
        except (PermissionError, OSError):
            entries.append(f"[!]    {item.name} (access denied)")

    header = f"Directory: {path}\n{'-' * 50}\n"
    if not entries:
        return ToolResult(True, header + "(empty directory)")
    return ToolResult(True, header + "\n".join(entries))


def delete_file(args: DeleteFileArgs) -> ToolResult:
    """Delete a file or an empty directory."""
    decision = _state._policy.evaluate("delete_file", args)
    if decision.is_denied():
        return ToolResult(False, f"Policy denied: {decision.reason}")

    try:
        path = _state._policy.resolve_path(args.path, _state._workspace_root, for_write=True)
    except PathDeniedError as exc:
        return ToolResult(False, str(exc))

    try:
        if path.is_file():
            path.unlink()
            return ToolResult(True, f"Successfully deleted file: {path}")
        if path.is_dir():
            path.rmdir()  # Only succeeds for empty directories.
            return ToolResult(True, f"Successfully deleted empty directory: {path}")
        return ToolResult(False, f"Error: Path not found: {path}")
    except OSError as exc:
        return ToolResult(False, f"Error deleting: {exc}")
