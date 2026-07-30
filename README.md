# ✦ Gemini Live Agent

An agentic harness that wraps **Gemini 3.1 Live** (Google's real-time multimodal API) with local computer access — file editing, shell commands, directory browsing, and web search — over a bidirectional WebSocket connection.

Talk to it by **typing** or **speaking**. It talks back.

---

## Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **Google AI API key** — get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **Audio device** (optional) — a microphone and speaker for voice mode

### 2. Install Dependencies

```bash
cd gemini-live-agent
pip install -r requirements.txt
```

### 3. Set Your API Key

Copy `.env.example` to `.env` and put your key in it (`.env` is gitignored):

```
GOOGLE_API_KEY=your-api-key-here
```

Or use the environment instead:

```powershell
# PowerShell, current session
$env:GOOGLE_API_KEY = "your-api-key-here"

# Or set it permanently:
[System.Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", "your-api-key-here", "User")
```

### 4. Run

```bash
python main.py
```

---

## Usage

```
python main.py [OPTIONS]
```

| Flag              | Default                          | Description                                        |
|:------------------|:---------------------------------|:---------------------------------------------------|
| `--mode`          | `both`                           | `text`, `voice`, or `both`                         |
| `--voice`         | `Puck`                           | `Puck`, `Charon`, `Aoede`, `Fenrir`, or `Kore`    |
| `--yolo`          | off                              | Skip safety confirmations for destructive ops      |
| `--working-dir`   | `.`                              | Working directory for relative file paths          |
| `--model`         | `gemini-3.1-flash-live-preview`  | Gemini model identifier                            |
| `--share-screen`  | off                              | Stream your screen so the model can see it         |
| `--screen-fps`    | `1.0`                            | Screen capture frame rate                          |
| `--monitor`       | `1`                              | Which screen to share (`0` = all monitors)         |

### Examples

```bash
# Text-only mode (no microphone needed)
python main.py --mode text

# Voice-only with the Kore voice
python main.py --mode voice --voice Kore

# Full access, no safety prompts
python main.py --yolo

# Work in a specific project directory
python main.py --working-dir C:\Users\Me\Projects\myapp

# Let it see your screen while you talk
python main.py --share-screen
```

---

## Screen Sharing

`--share-screen` streams your desktop to the model as JPEG frames, so you can
say "what's this error?" or "what am I looking at?" instead of describing it.

Frames are captured at 1 fps and downscaled to 1024 px wide before encoding —
roughly 60 KB per frame. Screen content changes slowly next to speech, so a
higher `--screen-fps` costs bandwidth and context without helping the model
read the screen. Raise it only for genuinely moving content.

On a multi-monitor setup, `--monitor 2` shares the second screen and
`--monitor 0` shares all of them stitched together.

**Everything on the shared screen goes to Google** for as long as the session
runs, including whatever is in the background — password managers, private
messages, other people's data. The banner shows `Screen: ON` while it is
active.

### In-Session Commands

| Command          | Action                |
|:-----------------|:----------------------|
| `exit` / `quit`  | End the session       |
| *speak freely*   | Voice input (in voice mode) |

---

## Tools

The agent has access to six local tools:

| Tool               | What it does                                      | Needs confirmation? |
|:-------------------|:--------------------------------------------------|:-------------------:|
| `read_file`        | Read file contents (with optional line range)     | No                  |
| `write_file`       | Create or overwrite a file                        | **Yes**             |
| `list_directory`   | List directory contents with types and sizes      | No                  |
| `delete_file`      | Delete a file or empty directory                  | **Yes**             |
| `run_command`      | Execute a shell command — PowerShell on Windows, `sh` elsewhere (with timeout) | **Yes** |
| `web_search`       | Search the web and return results                 | No                  |

### Safety Confirmations

Destructive operations (`write_file`, `delete_file`, `run_command`) show an interactive prompt:

```
╭─ ⚠️  Gemini wants to execute ──────────────────╮
│                                                 │
│  run_command                                    │
│    command: echo "Hello, World!"                │
│    working_dir: C:\Users\Me\Projects            │
│                                                 │
│  Allow? [y/N]:                                  │
╰─────────────────────────────────────────────────╯
```

Use `--yolo` to skip all confirmations (at your own risk).

---

## Architecture

```
User ↔ [Text CLI / Microphone+Speaker]
              ↕
    GeminiLiveAgent (asyncio)
      ├── _receive_loop    ← model transcript / audio / tool calls
      ├── _stdin_loop      → user text input + safety answers
      ├── _audio_send_loop → microphone PCM
      └── _video_send_loop → screen frames (--share-screen)
              ↕
    Gemini 3.1 Live API (WebSocket)
              ↕
    Local Tool Execution
      ├── read_file / write_file / list_directory / delete_file
      ├── run_command (async subprocess)
      └── web_search (DuckDuckGo / googlesearch)
```

### Audio Specs

| Direction | Sample Rate | Bit Depth | Channels | Format |
|:----------|:------------|:----------|:---------|:-------|
| Input     | 16 kHz      | 16-bit    | Mono     | PCM    |
| Output    | 24 kHz      | 16-bit    | Mono     | PCM    |

---

## Troubleshooting

### "No audio device detected"
The app will automatically fall back to text-only mode. To use voice, ensure your microphone and speakers are connected and recognized by your OS.

### "GOOGLE_API_KEY environment variable is not set"
Set the key as shown in the Quick Start section above.

### Session disconnects after ~10 minutes
Context window compression is enabled by default, which should allow unlimited sessions. If you experience disconnects, try reducing the conversation length or restarting.

### Command timeout
The default timeout for `run_command` is 30 seconds. The model can request a longer timeout via the `timeout` parameter.

---

## File Structure

```
gemini-live-agent/
├── main.py           # Entry point & CLI argument parsing
├── agent.py          # Core agent loop (WebSocket, tool dispatch)
├── audio.py          # Microphone capture & speaker playback
├── screen.py         # Screen capture for --share-screen
├── config.py         # Configuration, API key, system instruction
├── safety.py         # Interactive confirmation prompts
├── tools.py          # Local tool implementations
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## License

MIT
