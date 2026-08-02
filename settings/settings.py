"""Pydantic-settings based application configuration."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any

from google.genai import types
from platformdirs import user_config_dir
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_AVAILABLE_VOICES = ("Puck", "Charon", "Aoede", "Fenrir", "Kore")
_MODES = ("text", "voice", "both")

_SYSTEM_INSTRUCTION_TEMPLATE = """\
You are a powerful AI assistant with direct access to the user's local computer. \
You can read and write files, execute shell commands, list directories, delete files, \
and search the web.

## Your Environment
- Operating System: {os_info}
- Working Directory: {working_dir}
- Workspace Root: {workspace_root}
- Shell: {shell}
{screen_note}

## Your Tools
1. **read_file(path, start_line?, end_line?)** — Read file contents, optionally a specific \
line range (1-indexed, inclusive).
2. **write_file(path, content)** — Write or overwrite a file. Creates parent directories \
automatically.
3. **list_directory(path)** — List directory contents with file types and sizes.
4. **delete_file(path)** — Delete a file or empty directory.
5. **run_command(command, working_dir?, timeout?)** — Execute a shell command and return stdout \
+ stderr.
6. **web_search(query)** — Search the web and return summarized results.

## Guidelines
- When editing files, first read the file to understand its current contents, then write the \
complete updated version.
- For shell commands, prefer specific targeted commands over broad ones.
- When creating files, always create any necessary parent directories.
- If a task requires multiple steps, plan ahead and execute them in logical order.
- Report what you've done after completing each operation.
- If a command fails, analyze the error and suggest or try a fix.
- Be concise in your responses during voice conversations.
- Paths can be relative (resolved against your working directory) or absolute.
- Destructive operations (deleting or overwriting files, running commands that mutate state) \
require explicit user approval before you execute them.
"""

_SCREEN_NOTE = """\
- Screen Sharing: ACTIVE — you receive a live video feed of the user's screen.

When the user says "this", "that", "here," or asks what they are looking at, \
check the screen feed before asking them to describe it. The feed updates at a \
low frame rate, so it may lag a moment behind their actions."""


class AppSettings(BaseSettings):
    """Runtime configuration for the Gemini Live Agent."""

    model_config = SettingsConfigDict(
        env_prefix="GEMINI_",
        extra="forbid",
    )

    model: str = "gemini-3.1-flash-live-preview"
    fallback_model: str | None = "gemini-2.5-flash-native-audio"
    voice: str = "Puck"
    mode: str = "both"  # text, voice, both
    working_dir: Path = Field(default_factory=Path.cwd)
    workspace_root: Path = Field(default_factory=Path.cwd)
    api_key: str | None = None  # legacy env override
    share_screen: bool = False
    screen_fps: float = 1.0
    screen_monitor: int = 1
    input_device: str | None = None  # explicit sounddevice input name
    output_device: str | None = None  # explicit sounddevice output name
    half_duplex: bool = True
    vad_prefix_padding_ms: int = 300
    vad_silence_duration_ms: int = 800
    yolo: bool = False
    headless: bool = True
    debug: bool = False
    minimize_to_tray: bool = True
    # Extra instructions appended to the built-in system prompt. Empty = default.
    system_prompt: str = ""

    @field_validator("voice")
    @classmethod
    def _validate_voice(cls, value: str) -> str:
        if value not in _AVAILABLE_VOICES:
            msg = f"Invalid voice '{value}'. Choose one of: {', '.join(_AVAILABLE_VOICES)}"
            raise ValueError(msg)
        return value

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        if value not in _MODES:
            msg = f"Invalid mode '{value}'. Choose one of: {', '.join(_MODES)}"
            raise ValueError(msg)
        return value

    @field_validator("screen_fps")
    @classmethod
    def _validate_screen_fps(cls, value: float) -> float:
        if value <= 0:
            msg = f"screen_fps must be greater than 0 (got {value})"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _resolve_and_validate_paths(self) -> AppSettings:
        self.working_dir = self.working_dir.expanduser().resolve()
        if not self.working_dir.is_dir():
            msg = f"Working directory does not exist or is not a directory: {self.working_dir}"
            raise ValueError(msg)

        self.workspace_root = self.workspace_root.expanduser().resolve()
        if not self.workspace_root.is_dir():
            msg = f"Workspace root does not exist or is not a directory: {self.workspace_root}"
            raise ValueError(msg)

        return self

    def save(self) -> None:
        """Persist user-editable settings to the settings JSON file.

        API keys are never written to disk; headless/debug are runtime CLI flags
        and are also excluded.
        """
        settings_dir = Path(user_config_dir("GeminiLiveAgent", appauthor=False))
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings_file = settings_dir / "settings.json"

        data = self.model_dump(exclude={"api_key", "headless", "debug"})
        settings_file.write_text(
            json.dumps(data, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, **overrides: Any) -> AppSettings:
        """Load settings from the user settings file, env vars, and optional overrides.

        Precedence (lowest to highest): defaults < settings.json < env vars < overrides.
        """
        settings_dir = Path(user_config_dir("GeminiLiveAgent", appauthor=False))
        settings_file = settings_dir / "settings.json"

        file_values: dict[str, Any] = {}
        if settings_file.is_file():
            try:
                raw = json.loads(settings_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                msg = f"Settings file is not valid JSON: {settings_file}"
                raise ValueError(msg) from exc
            if not isinstance(raw, dict):
                msg = f"Settings file must contain a JSON object: {settings_file}"
                raise ValueError(msg)
            file_values = raw

        # Apply file values only when the corresponding env var is not set, so
        # env vars override the settings file. Init kwargs override everything.
        env_prefix = cls.model_config.get("env_prefix", "")
        effective: dict[str, Any] = {}
        for key, value in file_values.items():
            if key in overrides:
                continue
            env_name = f"{env_prefix}{key.upper()}"
            if env_name not in os.environ:
                effective[key] = value

        effective.update(overrides)
        return cls(**effective)

    def as_connect_config(
        self, tools: list[Any], *, resumption_handle: str | None = None
    ) -> types.LiveConnectConfig:
        """Build the LiveConnectConfig for a Gemini Live session."""
        os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
        shell = (
            "PowerShell"
            if platform.system() == "Windows"
            else os.environ.get("SHELL", "/bin/bash")
        )

        instruction_text = _SYSTEM_INSTRUCTION_TEMPLATE.format(
            os_info=os_info,
            working_dir=self.working_dir,
            workspace_root=self.workspace_root,
            shell=shell,
            screen_note=_SCREEN_NOTE if self.share_screen else "",
        )
        if self.system_prompt.strip():
            instruction_text += (
                "\n## Additional Instructions\n" + self.system_prompt.strip() + "\n"
            )

        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
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
            # Server-side VAD. START_OF_ACTIVITY_INTERRUPTS makes the model stop
            # generating as soon as the user starts speaking, instead of
            # finishing its turn and queueing the user's speech behind it.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    prefix_padding_ms=self.vad_prefix_padding_ms,
                    silence_duration_ms=self.vad_silence_duration_ms,
                ),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            ),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(target_tokens=16000)
            ),
            session_resumption=types.SessionResumptionConfig(handle=resumption_handle),
        )
