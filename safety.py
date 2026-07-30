"""Safety confirmation layer for destructive operations."""

from typing import Any, Awaitable, Callable, Dict

from rich.box import ASCII
from rich.console import Console
from rich.panel import Panel

console = Console()

# Tools that require user confirmation before execution
DESTRUCTIVE_TOOLS = {"write_file", "delete_file", "run_command"}


class SafetyGuard:
    """Prompts the user for confirmation before executing destructive operations."""

    def __init__(self, yolo: bool = False):
        self.yolo = yolo

    async def confirm(
        self,
        tool_name: str,
        args: Dict[str, Any],
        read_line: Callable[[], Awaitable[str]],
    ) -> bool:
        """
        Ask the user to confirm a destructive operation.

        `read_line` must come from the agent's single stdin reader — reading
        stdin directly here would race the text input loop for the same line.

        Returns True if the user approves, False otherwise.
        In --yolo mode, always returns True without prompting.
        """
        if self.yolo:
            return True

        # Format the operation details for display
        details_lines = []
        for key, value in args.items():
            val_str = str(value)
            if len(val_str) > 300:
                val_str = val_str[:300] + "... (truncated)"
            details_lines.append(f"  [cyan]{key}[/cyan]: {val_str}")

        details = "\n".join(details_lines)

        panel = Panel(
            f"[bold yellow]{tool_name}[/bold yellow]\n{details}",
            title="[!] Gemini wants to execute",
            border_style="yellow",
            box=ASCII,
            padding=(1, 2),
        )
        console.print(panel)

        # \[ escapes the bracket so Rich does not parse [y/N] as markup
        console.print("[bold]Allow? \\[y/N]: [/bold]", end="")
        response = (await read_line()).strip().lower()

        approved = response in ("y", "yes")
        if approved:
            console.print("[green]  [OK] Approved[/green]")
        else:
            console.print("[red]  [X] Denied[/red]")

        return approved
