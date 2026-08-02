"""Entry point for the Gemini Live Agent.

Usage:
    python main.py                              # GUI (default)
    python main.py --headless                   # Terminal mode (text + voice)
    python main.py --headless --mode text       # Terminal, text-only
    python main.py --mode voice --voice Kore    # GUI, voice-only with Kore
    python main.py --yolo                       # Skip safety confirmations
    python main.py --working-dir C:\\Projects    # Set working directory
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from headless import run_headless
from settings import AppSettings, get_api_key, migrate_dotenv, set_api_key

AVAILABLE_VOICES = ("Puck", "Charon", "Aoede", "Fenrir", "Kore")
DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Puck"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Gemini 3.1 Live — Agentic harness with local computer access",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py                              Start in text + voice mode
  python main.py --mode text                  Text-only (no microphone)
  python main.py --mode voice --voice Kore    Voice-only with the Kore voice
  python main.py --yolo                       Skip all safety confirmations
  python main.py --working-dir C:\\Projects    Set the agent's working directory
  python main.py --share-screen               Let the model see your screen
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["text", "voice", "both"],
        default="both",
        help="Interaction mode (default: both)",
    )
    parser.add_argument(
        "--voice",
        choices=AVAILABLE_VOICES,
        default=DEFAULT_VOICE,
        help=f"Voice for audio responses (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Skip confirmation prompts for destructive operations (write, delete, run)",
    )
    parser.add_argument(
        "--working-dir",
        default=".",
        help="Agent's working directory for resolving relative paths (default: .)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model identifier (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--share-screen",
        action="store_true",
        help="Stream your screen to the model so it can see what you see",
    )
    parser.add_argument(
        "--screen-fps",
        type=float,
        default=1.0,
        help="Screen capture frame rate (default: 1.0)",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Monitor to share: 0 for all, 1+ for a single screen (default: 1)",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run in headless terminal mode instead of the GUI (default: False)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )

    return parser.parse_args()


def _prompt_for_api_key() -> None:
    """Prompt the user for an API key and store it in the OS credential store."""
    print(
        "No Google API key found. Please paste your API key and press Enter:",
        file=sys.stderr,
    )
    key = input().strip()
    if not key:
        raise ValueError("API key cannot be empty")
    set_api_key(key)


def main() -> int:
    """Parse args, build settings, ensure an API key, and run the agent."""
    args = parse_args()

    migrate_dotenv()
    if args.headless and get_api_key() is None:
        _prompt_for_api_key()

    working_dir = Path(args.working_dir)

    settings = AppSettings.load(
        mode=args.mode,
        voice=args.voice,
        yolo=args.yolo,
        working_dir=working_dir,
        workspace_root=working_dir,
        model=args.model,
        share_screen=args.share_screen,
        screen_fps=args.screen_fps,
        screen_monitor=args.monitor,
        headless=args.headless,
        debug=args.debug,
    )

    if settings.headless:
        return asyncio.run(run_headless(settings))

    from app.main_window import run_gui

    return run_gui(settings)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception:
        import traceback

        print("\n[ERROR] Fatal error occurred:")
        traceback.print_exc()
        sys.exit(1)
