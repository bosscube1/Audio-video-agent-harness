"""Tool implementations and declarations for the Gemini Live Agent.

No PySide6 here.
"""

from __future__ import annotations

from tools._state import set_gating_mode, set_policy, set_workspace
from tools.fs import delete_file, list_directory, read_file, write_file
from tools.registry import (
    ALL_TOOLS,
    DeleteFileArgs,
    ListDirectoryArgs,
    ReadFileArgs,
    RunCommandArgs,
    ToolResult,
    ToolSpec,
    WriteFileArgs,
    build_function_declarations,
)
from tools.shell import run_command

__all__ = [
    "ALL_TOOLS",
    "DeleteFileArgs",
    "ListDirectoryArgs",
    "ReadFileArgs",
    "RunCommandArgs",
    "ToolResult",
    "ToolSpec",
    "WriteFileArgs",
    "build_function_declarations",
    "delete_file",
    "list_directory",
    "read_file",
    "run_command",
    "set_gating_mode",
    "set_policy",
    "set_workspace",
    "write_file",
]


def _redact_result(result: ToolResult) -> ToolResult:
    """Convenience helper for callers that want redacted tool results."""
    from policy import redact

    return ToolResult(result.ok, redact(result.message))
