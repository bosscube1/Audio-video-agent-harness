"""Configuration and environment setup for the Gemini Live Agent."""

import os
from pathlib import Path

# Load .env file from the project directory (if it exists)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv is optional; fall back to real env vars
import sys
import platform
from dataclasses import dataclass, field
from typing import Optional

from google import genai
from google.genai import types


AVAILABLE_VOICES = ["Puck", "Charon", "Aoede", "Fenrir", "Kore"]
DEFAULT_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Puck"

SYSTEM_INSTRUCTION = """\
You are a powerful AI assistant with direct access to the user's local computer. \
You can read and write files, execute shell commands, list directories, delete files, \
and search the web.

## Your Environment
- Operating System: {os_info}
- Working Directory: {working_dir}
- Shell: {shell}
{screen_note}
## Your Tools
1. **read_file(path, start_line?, end_line?)** — Read file contents, optionally a specific line range (1-indexed, inclusive).
2. **write_file(path, content)** — Write or overwrite a file. Creates parent directories automatically.
3. **list_directory(path)** — List directory contents with file types and sizes.
4. **delete_file(path)** — Delete a file or empty directory.
5. **run_command(command, working_dir?, timeout?)** — Execute a shell command and return stdout + stderr.
6. **web_search(query)** — Search the web and return summarized results.

## Guidelines
- When editing files, first read the file to understand its current contents, then write the complete updated version.
- For shell commands, prefer specific targeted commands over broad ones.
- When creating files, always create any necessary parent directories.
- If a task requires multiple steps, plan ahead and execute them in logical order.
- Report what you've done after completing each operation.
- If a command fails, analyze the error and suggest or try a fix.
- Be concise in your responses during voice conversations.
- Paths can be relative (resolved against your working directory) or absolute.
"""

SCREEN_NOTE = """\
- Screen Sharing: ACTIVE — you receive a live video feed of the user's screen.

When the user says "this", "that", "here", or asks what they are looking at, \
check the screen feed before asking them to describe it. The feed updates at a \
low frame rate, so it may lag a moment behind their actions."""


@dataclass
class AgentConfig:
    """Runtime configuration for the Gemini Live Agent."""

    mode: str = "both"  # "text", "voice", or "both"
    voice: str = DEFAULT_VOICE
    yolo: bool = False
    working_dir: str = field(default_factory=os.getcwd)
    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    share_screen: bool = False
    screen_fps: float = 1.0
    monitor: int = 1

    def __post_init__(self):
        # Resolve working directory to absolute path
        self.working_dir = os.path.abspath(self.working_dir)

        # Get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.environ.get("GOOGLE_API_KEY")

        # ── Validate ──
        if not self.api_key:
            print("\n[ERROR] GOOGLE_API_KEY is not set.")
            print("   Option 1:  Create a .env file with:  GOOGLE_API_KEY=your-key")
            print("   Option 2:  set GOOGLE_API_KEY=your-api-key")
            print("   Get a key at: https://aistudio.google.com/apikey\n")
            sys.exit(1)

        if self.voice not in AVAILABLE_VOICES:
            print(
                f"\n[ERROR] Invalid voice '{self.voice}'. "
                f"Available: {', '.join(AVAILABLE_VOICES)}\n"
            )
            sys.exit(1)

        if self.mode not in ("text", "voice", "both"):
            print(f"\n[ERROR] Invalid mode '{self.mode}'. Choose: text, voice, both\n")
            sys.exit(1)

        if not os.path.isdir(self.working_dir):
            print(f"\n[ERROR] Working directory does not exist: {self.working_dir}\n")
            sys.exit(1)

        if self.share_screen and self.screen_fps <= 0:
            print(f"\n[ERROR] --screen-fps must be greater than 0 (got {self.screen_fps})\n")
            sys.exit(1)

    # ── Builders ──

    def build_genai_client(self) -> genai.Client:
        """Create a Google GenAI client."""
        return genai.Client(api_key=self.api_key)

    def build_live_config(self, tools: list) -> types.LiveConnectConfig:
        """Build the LiveConnectConfig for the session."""
        # The Live API returns a single modality per session, and this agent
        # always uses AUDIO. Transcription (below) is what makes the model's
        # replies readable in the terminal — without it, text mode is silent.
        modalities = ["AUDIO"]

        # Build system instruction with environment info
        os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
        shell = (
            "PowerShell"
            if platform.system() == "Windows"
            else os.environ.get("SHELL", "/bin/bash")
        )

        instruction_text = SYSTEM_INSTRUCTION.format(
            os_info=os_info,
            working_dir=self.working_dir,
            shell=shell,
            screen_note=SCREEN_NOTE if self.share_screen else "",
        )

        config = types.LiveConnectConfig(
            response_modalities=modalities,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=instruction_text)]
            ),
            tools=tools,
            # Transcribe the model's audio so replies are visible in the terminal,
            # and transcribe mic input so the user can see what was heard.
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            # Enable context window compression for unlimited session duration
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(target_tokens=16000)
            ),
        )

        return config
