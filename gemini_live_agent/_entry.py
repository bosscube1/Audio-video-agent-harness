"""CLI entry point shared by `python -m gemini_live_agent` and the console script.

Guards UTF-8 stdout/stderr reconfiguration so importing this package (e.g.
under pytest) does not mutate global streams.
"""

import os
import sys
import traceback


def _reconfigure_stream(stream: object) -> None:
    """Reconfigure a stream to UTF-8 when invoked as a real CLI entry point."""
    if (
        "PYTEST_CURRENT_TEST" not in os.environ
        and hasattr(stream, "reconfigure")
    ):
        stream.reconfigure(encoding="utf-8", errors="replace")


def _prepare_windows_console() -> None:
    """Enable VT100 and UTF-8 on Windows; called only at CLI entry time."""
    os.system("")
    _reconfigure_stream(sys.stdout)
    _reconfigure_stream(sys.stderr)


def run() -> None:
    """Run the legacy CLI entry point."""
    if sys.platform == "win32":
        _prepare_windows_console()

    from main import main  # noqa: E402

    main()


def run_with_error_handling() -> None:
    """Run the CLI and print exceptions on failure."""
    try:
        run()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
