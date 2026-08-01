# Handoff & Continuation Plan

## Current State

- **Repository:** `C:\Users\Hp\projects\gemini-live-agent`
- **Remote:** `https://github.com/bosscube1/Audio-video-agent-harness.git`
- **Branch:** `Kimi-V2`
- **Latest commit:** `e9632d9` — `feat(media): barge-in — stop playback the moment the server interrupts` *(pushed to origin)*
- **Phases complete:** 1 (foundation), 2 (headless core extraction), 3 (session lifetime / resumption / barge-in)
- **Phase ready to start:** 4 (GUI shell on a QThread)

## How to run right now

```powershell
cd C:\Users\Hp\projects\gemini-live-agent
.venv\Scripts\python -m gemini_live_agent --help

# Text mode
.venv\Scripts\python -m gemini_live_agent --mode text

# Voice mode
.venv\Scripts\python -m gemini_live_agent --mode voice

# With screen share
.venv\Scripts\python -m gemini_live_agent --mode both --share-screen
```

The first run migrates `GOOGLE_API_KEY` from `.env` into the Windows Credential Manager. If no key exists, it prompts for one.

## What's working

- **Packaging:** `pyproject.toml`, locked `requirements*.txt`, `uv` venv, ruff/mypy/pytest configs. PySide6 and pytest-qt are already declared as dependencies.
- **Logging:** JSON Lines rotating logs + stderr, `run_id` on every record.
- **Core vocabulary:** frozen dataclasses in `core/events.py` / `core/commands.py`.
- **Single-session headless driver:** connects, sends text, streams mic/screen, plays speaker, dispatches tools.
- **Tool layer:** explicit pydantic schemas, no docstring introspection, `web_search` deleted.
- **Policy layer:** Windows path confinement, deny globs, sensitive-name rules, gating mode, outbound redaction.
- **Audit + persistence:** append-only JSONL journal, SQLite store schema.
- **Media layer:** explicit device selection, bounded mic queue, speaker generation-counter flush fix, and barge-in stream abort on `server_content.interrupted`.
- **Entry points:** `python -m gemini_live_agent`, `gemini-live-agent` console script, and `python main.py` all work.
- **Phase 3 session lifetime:** `core/supervisor.py` reconnect loop, resumption-handle save/expire/seed, GoAway handling, classified backoff, failure budgets, in-flight tool-result delivery across reconnect.
- **Single-epoch driver:** `core/session.py` injects `ToolDispatcher`, `TurnState`, and media objects so the Supervisor can reuse them across epochs.
- **Resumption support:** `settings/settings.py` builds `LiveConnectConfig` with `SessionResumptionConfig`; `persist/store.py` persists handles and seeds prior turns on fresh reconnects.
- **Tool cancellation:** `core/dispatcher.py` can kill running shells via `tools/shell.cancel_shell`; cancelled ids never receive a `FunctionResponse`.
- **Barge-in:** `Speaker.flush()` aborts and restarts the PortAudio stream on interruption; `LiveSession` calls `flush()` before handling `server_content.interrupted`; `AppSettings` enables server-side VAD/activity detection so the server cuts off the model turn instead of queueing user speech behind it.
- **Fake Live harness:** `tests/fakes/fake_live.py` scripts socket drops, GoAway, rejected handles, rejected `fc_id`s, tool-call cancellation, and barge-in interruption.
- **Headless driver resilience:** `headless.py` drives `Supervisor` instead of a single `LiveSession`.

## Key design decisions locked in

- **Single process, asyncio core.** GUI will live on a `QThread` with Qt signals in, `loop.call_soon_threadsafe` out. `core/**` never imports PySide6; `app/**` never imports asyncio primitives.
- **Frozen dataclasses across the thread boundary.** No shared mutable state.
- **Meters polled, not pushed.** `AudioLevel` events exist but the GUI should poll `Microphone.level` / `Speaker.level` at ~30 Hz.
- **Transcription coalesced to 80 ms in core.** `TurnState` emits at most one `PartialTranscript` per source per window.
- **Typed text via `send_client_content`.** Verified working on `gemini-3.1-flash-live-preview` with `turn_complete=True`.
- **Supervisor owns the long-lived conversation.** `LiveSession` is a single-epoch object; the Supervisor creates one per reconnect and reuses the dispatcher, turn state, and media.
- **Resumption handles are persisted with a 110-minute TTL.** `Store` prunes expired handles and can seed prior turns as client content on fresh reconnects.
- **Tool results are delivered across reconnects.** If the original `fc_id` is rejected, the result is reinjected as a system-note client turn.
- **GoAway causes a hard reconnect, not overlapped.** Media stays alive; the old socket is closed and a new one is opened with the saved handle.

## Immediate next: Phase 4 — GUI shell on a QThread

The headless harness is now reconnect-resilient and supports barge-in. The next high-leverage phase is a minimal PySide6 GUI that consumes the same event/command vocabulary.

`main.py` already accepts `--headless` / `--no-headless` and passes `headless` through `AppSettings`, but the non-headless path raises `ValueError("GUI mode is not yet implemented; run with --headless")`. Implement the GUI path.

### Files to create / change

- `app/bridge.py` — NEW. Qt thread bridge: `QThread` runs the asyncio Supervisor; Qt signals in, `loop.call_soon_threadsafe` out.
- `app/main_window.py` — NEW. Minimal window: connection state, transcript view, tool approval cards, audio level indicators, settings toggle.
- `app/__init__.py` — NEW. Package marker.
- `core/supervisor.py` — TWEAK if needed. Ensure events and commands cross the thread boundary cleanly (they are already frozen dataclasses).
- `main.py` — REPLACE. When `settings.headless` is false, launch the GUI instead of `run_headless`.
- `pyproject.toml` — VERIFY. PySide6 and pytest-qt are already declared; confirm the wheel target includes `.` and `gemini_live_agent` plus `app/` if it becomes a separate package (or keep `app/` under the root package layout).

### Implementation checklist

1. **Bridge first.** Create a `Bridge` that starts `Supervisor.run()` in a `QThread` and forwards `AgentEvent` objects to the main thread as Qt signals. Provide a slot that accepts `AgentCommand` objects and calls `loop.call_soon_threadsafe(queue.put_nowait, ...)`.
2. **Main window.** Display `ConnectionState`, `PartialTranscript`/`TurnComplete`, `ToolApprovalRequested` cards, `ToolResultSent`, `ContextReset`, and `SessionError`. Poll audio levels via `Microphone.level` / `Speaker.level` at ~30 Hz.
3. **Command wiring.** Connect GUI actions to `AgentCommand` objects pushed into the bridge's command queue: `SendText`, `ApproveTool`, `DenyTool`, `CancelTool`, `Disconnect`, `SetGatingMode`, `SetMicGate`, `SetShareScreen`.
4. **Settings persistence.** Load/save `settings.json` via the existing `AppSettings.load()` path; expose the most common knobs (model, voice, mode, share-screen, input/output devices, yolo).
5. **Headless still works.** Keep `run_headless` as the default; GUI is opt-in via `--no-headless`.
6. **Tests.** Add `tests/test_gui_bridge.py` or `tests/test_gui_lifecycle.py` using `pytest-qt` to verify the bridge starts, emits events, and forwards commands without deadlocking. Add a smoke test that `--no-headless` instantiates the GUI window in a non-interactive QApplication and exits cleanly.

### Phase 4 verification gate

- GUI starts without an API key and prompts for one.
- Text-mode conversation survives a reconnect in the GUI (use the fake harness for unit tests; real socket for a short smoke test).
- Tool approval card appears, approve/deny/cancel buttons work, and results update the transcript.
- `ruff check .`, `mypy .`, and `pytest -q` all pass.
- `--headless` still runs the terminal driver exactly as before.

### Risks / known issues

- **Old `scratch/` directory** at `C:\Users\Hp\.gemini\antigravity\scratch\gemini-live-agent` still exists. Remove it manually if still locked.
- **`.env` file still exists** in the repo root. It is ignored by git, but `settings.secrets.migrate_dotenv()` already moved the key into the credential store. Decide whether to delete the file or keep it as a dev-only convenience.
- **Directory/package name still has a hyphen.** `python -m gemini_live_agent` works because of the `gemini_live_agent/` package, but a proper layout will be needed before PyInstaller packaging.
- **Live 30-minute wall test** from the Phase 3 verification gate has not been run yet. Run it before declaring Phase 3 fully validated in production.
- **`.claude/` worktree cleanup** was attempted: the `claude/adoring-wright-c18ce1` worktree metadata was removed from git, but the empty `.claude/worktrees/adoring-wright-c18ce1` directory could not be deleted because Windows reports it as "Device or resource busy" (likely held by a shell or explorer). Reboot or close the holding process, then delete `.claude/` manually.

## Cut list (still valid)

Do not start these yet:

- Overlapped reconnect
- History browser GUI
- Diagnostics panel (log file only for now)
- Policy editor GUI (hand-edit JSON)
- Voice-announced approvals
- `tools/jobs.py` deferred-job table

## Running checks

```powershell
cd C:\Users\Hp\projects\gemini-live-agent
.venv\Scripts\ruff check .
.venv\Scripts\mypy .
.venv\Scripts\pytest -q
```

## Open questions

1. **Does `fc_id` survive real resumption?** The fallback (system-note reinjection) is implemented and tested; a live socket still needs to confirm the happy path.
2. **VB-Audio Virtual Cable loopback guard** currently warns. Change to hard-refuse if desired.
3. **Close button minimize-to-tray?** Deferred to GUI Phase 5.
4. **Should the model be told the policy rules explicitly?** Currently only the workspace root and destructive-op approval are mentioned in the system instruction.
5. **Single `.exe` vs. Inno Setup?** Deferred to Phase 6 packaging.
6. **Searchable transcript across runs?** Left out.

## Recommended first Phase 4 task

Create `app/bridge.py`: a `QThread` that owns the `asyncio` event loop and `Supervisor`, emits `AgentEvent` objects as Qt signals, and accepts `AgentCommand` objects via a slot that calls `loop.call_soon_threadsafe(queue.put_nowait, ...)`. This is the highest-leverage piece; once the bridge works, the rest of the GUI is just widgets consuming signals.
