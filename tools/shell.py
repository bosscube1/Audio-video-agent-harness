"""Shell command tool implementation.

No PySide6 here.
"""

from __future__ import annotations

import asyncio
import sys

import tools._state as _state
from policy import PathDeniedError
from tools.registry import RunCommandArgs, ToolResult

MAX_CMD_BYTES = 8 * 1024  # 8 KB
MAX_TIMEOUT_SECONDS = 300


def _clamp_timeout(value: int | None) -> int:
    if value is None:
        return 30
    return max(1, min(value, MAX_TIMEOUT_SECONDS))


def _truncate(text: str, limit: int) -> str:
    if len(text) > limit:
        return text[:limit] + "\n… (truncated)"
    return text


async def run_command(args: RunCommandArgs) -> ToolResult:
    """Run a PowerShell command (Windows) or shell command (other platforms)."""
    decision = _state._policy.evaluate("run_command", args)
    if decision.is_denied():
        return ToolResult(False, f"Policy denied: {decision.reason}")

    cwd = _state._workspace_root
    if args.working_dir:
        try:
            cwd = _state._policy.resolve_path(
                args.working_dir, _state._workspace_root, for_write=False
            )
        except PathDeniedError as exc:
            return ToolResult(False, str(exc))
        if not cwd.is_dir():
            return ToolResult(False, f"Error: Not a directory: {cwd}")

    timeout = _clamp_timeout(args.timeout)

    try:
        if sys.platform == "win32":
            proc = await asyncio.create_subprocess_exec(
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                args.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                args.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            return ToolResult(
                False,
                f"Error: Command timed out after {timeout}s.\nCommand: {args.command}",
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""

        parts: list[str] = []
        if stdout:
            parts.append(f"[stdout]\n{_truncate(stdout, MAX_CMD_BYTES)}")
        if stderr:
            parts.append(f"[stderr]\n{_truncate(stderr, MAX_CMD_BYTES)}")

        exit_info = f"[exit code: {proc.returncode}]"
        if not parts:
            return ToolResult(True, f"Command completed with no output. {exit_info}")

        return ToolResult(True, "\n".join(parts) + f"\n{exit_info}")
    except OSError as exc:
        return ToolResult(False, f"Error executing command: {exc}")
