"""Local tool implementations for the Gemini Live Agent.

Each function here is a tool the model can invoke. The google-genai SDK
auto-generates the JSON schema for function declarations from the type
hints and docstrings, so keep them accurate.
"""

import asyncio
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Optional

# ── Output limits (prevent flooding the model's context) ──
MAX_READ_BYTES = 16 * 1024  # 16 KB
MAX_CMD_BYTES = 8 * 1024  # 8 KB

# ── Working directory ──
# Relative paths the model passes are resolved against this, not the process
# cwd. Set once at startup from AgentConfig.working_dir (see --working-dir).
_WORKING_DIR = os.getcwd()


def set_working_dir(path: str) -> None:
    """Set the base directory used to resolve relative tool paths."""
    global _WORKING_DIR
    _WORKING_DIR = os.path.abspath(path)


def get_working_dir() -> str:
    """Return the current base directory for relative tool paths."""
    return _WORKING_DIR


def _resolve(path: str) -> str:
    """Resolve a model-supplied path against the working directory."""
    expanded = os.path.expanduser(path)
    # os.path.join returns `expanded` unchanged when it is already absolute.
    return os.path.abspath(os.path.join(_WORKING_DIR, expanded))


# ─────────────────────────────────────────────
# File System Tools
# ─────────────────────────────────────────────


def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """Read the contents of a file. Optionally specify start_line and end_line (1-indexed, inclusive) to read a specific range of lines."""
    try:
        path = _resolve(path)
        if not os.path.exists(path):
            return f"Error: File not found: {path}"
        if not os.path.isfile(path):
            return f"Error: Not a file: {path}"

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)

        # Apply line range if specified
        if start_line is not None or end_line is not None:
            start = max(1, start_line or 1) - 1  # Convert to 0-indexed
            end = min(total_lines, end_line or total_lines)
            lines = lines[start:end]
            header = f"[Lines {start + 1}-{end} of {total_lines} total]\n"
        else:
            header = f"[{total_lines} lines, {os.path.getsize(path)} bytes]\n"

        content = "".join(lines)
        if len(content) > MAX_READ_BYTES:
            content = (
                content[:MAX_READ_BYTES]
                + f"\n\n… (truncated — file is {len(content):,} bytes total)"
            )

        return header + content
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories if they don't exist. Overwrites existing files."""
    try:
        path = _resolve(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        return f"Successfully wrote {len(content):,} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def list_directory(path: str = ".") -> str:
    """List the contents of a directory, showing file types and sizes."""
    try:
        path = _resolve(path)
        if not os.path.exists(path):
            return f"Error: Directory not found: {path}"
        if not os.path.isdir(path):
            return f"Error: Not a directory: {path}"

        try:
            items = sorted(os.listdir(path))
        except PermissionError:
            return f"Error: Permission denied: {path}"

        entries = []
        for item in items:
            full_path = os.path.join(path, item)
            try:
                if os.path.isdir(full_path):
                    try:
                        child_count = len(os.listdir(full_path))
                    except PermissionError:
                        child_count = "?"
                    entries.append(f"[DIR]  {item}/ ({child_count} items)")
                else:
                    size = os.path.getsize(full_path)
                    size_str = _human_size(size)
                    entries.append(f"[FILE] {item} ({size_str})")
            except (PermissionError, OSError):
                entries.append(f"[!]    {item} (access denied)")

        header = f"Directory: {path}\n{'-' * 50}\n"
        if not entries:
            return header + "(empty directory)"
        return header + "\n".join(entries)
    except Exception as e:
        return f"Error listing directory: {e}"


def delete_file(path: str) -> str:
    """Delete a file or an empty directory."""
    try:
        path = _resolve(path)
        if not os.path.exists(path):
            return f"Error: Path not found: {path}"
        if os.path.isdir(path):
            os.rmdir(path)  # Only removes empty directories for safety
            return f"Successfully deleted empty directory: {path}"
        else:
            os.remove(path)
            return f"Successfully deleted file: {path}"
    except OSError as e:
        return f"Error deleting: {e}"


# ─────────────────────────────────────────────
# Shell Tool
# ─────────────────────────────────────────────


async def run_command(
    command: str,
    working_dir: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """Execute a shell command and return its output (stdout + stderr combined). Timeout is in seconds (default 30)."""
    try:
        cwd = _resolve(working_dir) if working_dir else _WORKING_DIR

        if sys.platform == "win32":
            # The system instruction tells the model it is driving PowerShell,
            # so run PowerShell rather than cmd.exe (what create_subprocess_shell
            # would use here).
            proc = await asyncio.create_subprocess_exec(
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return f"Error: Command timed out after {timeout}s.\nCommand: {command}"

        output_parts = []
        if stdout:
            text = stdout.decode("utf-8", errors="replace")
            if len(text) > MAX_CMD_BYTES:
                text = text[:MAX_CMD_BYTES] + "\n… (stdout truncated)"
            output_parts.append(f"[stdout]\n{text}")
        if stderr:
            text = stderr.decode("utf-8", errors="replace")
            if len(text) > MAX_CMD_BYTES:
                text = text[:MAX_CMD_BYTES] + "\n… (stderr truncated)"
            output_parts.append(f"[stderr]\n{text}")

        exit_info = f"[exit code: {proc.returncode}]"

        if not output_parts:
            return f"Command completed with no output. {exit_info}"

        return "\n".join(output_parts) + f"\n{exit_info}"
    except Exception as e:
        return f"Error executing command: {e}"


# ─────────────────────────────────────────────
# Web Search Tool
# ─────────────────────────────────────────────


def web_search(query: str) -> str:
    """Search the web for information and return summarized results."""
    try:
        # Primary: try googlesearch-python if installed
        try:
            from googlesearch import search as gsearch  # type: ignore

            results = []
            for i, url in enumerate(gsearch(query, num_results=5)):
                results.append(f"{i + 1}. {url}")
            if results:
                return f"Search results for '{query}':\n" + "\n".join(results)
            return f"No results found for: {query}"
        except ImportError:
            pass

        # Fallback: DuckDuckGo instant answer API (no key required)
        encoded = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1"
        req = urllib.request.Request(
            url, headers={"User-Agent": "GeminiLiveAgent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        parts: list[str] = []
        if data.get("Abstract"):
            parts.append(f"Summary: {data['Abstract']}")
            parts.append(
                f"Source: {data.get('AbstractSource', 'N/A')} "
                f"({data.get('AbstractURL', '')})"
            )

        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and "Text" in topic:
                parts.append(f"• {topic['Text'][:300]}")

        if parts:
            return "\n".join(parts)
        return (
            f"No detailed results found for: {query}. "
            "Try rephrasing your search query."
        )
    except Exception as e:
        return f"Error searching: {e}"


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

# Map of tool name → callable (used for dispatch in agent.py)
TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "delete_file": delete_file,
    "run_command": run_command,
    "web_search": web_search,
}

# List of callables for the SDK to auto-generate function declaration schemas
ALL_TOOLS = [read_file, write_file, list_directory, delete_file, run_command, web_search]


# ── Helpers ──

def _human_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.1f} GB"
