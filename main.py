"""Entry point for the Gemini Live Agent.

Usage:
    python main.py                              # Text + voice (default)
    python main.py --mode text                  # Text-only
    python main.py --mode voice --voice Kore    # Voice-only with Kore
    python main.py --yolo                       # Skip safety confirmations
    python main.py --working-dir C:\\Projects    # Set working directory
"""

import argparse
import asyncio
import os
import sys

# Force UTF-8 encoding and enable VT100 ANSI processing in Windows Command Prompt
if sys.platform == "win32":
    os.system("")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import AVAILABLE_VOICES, DEFAULT_MODEL, DEFAULT_VOICE, AgentConfig


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

    return parser.parse_args()


def main() -> None:
    """Parse args, build config, launch agent."""
    args = parse_args()

    config = AgentConfig(
        mode=args.mode,
        voice=args.voice,
        yolo=args.yolo,
        working_dir=args.working_dir,
        model=args.model,
        share_screen=args.share_screen,
        screen_fps=args.screen_fps,
        monitor=args.monitor,
    )

    # Import here so --help works even without dependencies installed
    from agent import GeminiLiveAgent

    agent = GeminiLiveAgent(config)

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception:
        import traceback
        print("\n[ERROR] Fatal error occurred:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
