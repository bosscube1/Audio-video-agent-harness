# Handoff & Continuation Plan

## Current State

- **Repository:** `C:\Users\Hp\projects\gemini-live-agent`
- **Remote:** `https://github.com/bosscube1/Audio-video-agent-harness.git`
- **Branch:** `Kimi-V2`
- **Latest commit:** *(update after next commit)* — Phase 5 GUI control surfaces & voice UX
- **Phases complete:** 1 (foundation), 2 (headless core extraction), 3 (session lifetime / resumption / barge-in), 4 (GUI shell on a QThread), 5 (GUI control surfaces + voice UX polish), 6 (persistence wiring + packaging)
- **Next:** manual Phase 6 gate with real hardware (see below), then cut-list candidates

## Phase 6 additions

- **Wheel fixed.** `pyproject.toml` hatch target replaced `packages = [".", ...]` with an explicit `include` list; verified `python -m build` produces a wheel whose install imports every package cleanly.
- **Run lifecycle persisted.** `Supervisor.run()` calls `store.start_run()` / `end_run()`; `start_run` is now idempotent (`INSERT OR IGNORE`).
- **Usage persisted.** `LiveSession` records per-modality usage rows. **Bug fix:** genai 2.x `UsageMetadata` has no `completion_token_count` / `audio_token_count` / `video_token_count` — completion now reads `response_token_count`, audio/video come from `prompt_tokens_details` / `response_tokens_details`. These were previously always 0.
- **New accessors:** `Store.get_usage(run_id)`, `Store.get_run(run_id)`.
- **PyInstaller onedir build.** `packaging/gemini-live-agent.spec` (+ generated `icon.ico`, `version.txt`). Hidden imports for `mss` and `_sounddevice_data` PortAudio DLLs; heavy PySide6 modules excluded. Console kept on so `--headless` works.
- **Smoke test passing:** `dist/gemini-live-agent/gemini-live-agent.exe --help` exits 0 from a clean directory; bundle is 153 MB.

## Phase 5 additions (GUI)

- **Usage panel** — status-bar label fed by `UsageUpdate` (total / in / out / cached tokens, est. cost).
- **Activity feed** — second tab logging connection changes, tool calls, approvals, results, resets, errors with timestamps.
- **Mic state machine** — `_MicState` OFF/LIVE/MUTED chip next to the status label; `Ctrl+M` mute shortcut; follows connection state, mode, and mute toggle.
- **System tray** — `QSystemTrayIcon` with Show/Hide, Mute, Quit; close button minimizes to tray when `minimize_to_tray` is enabled (new persisted `AppSettings` field); tray Quit forces a real exit.
- **Screen-share indicator** — red banner while a share-screen session is connected.
- **Accessibility** — `&Approve`/`&Deny`/`&Cancel` mnemonics on tool cards, accessible names on transcript/feed/meters/input.
- **Tests** — `tests/test_gui_phase5.py` (9 tests): usage panel, activity feed, mic chip transitions, share banner, tray hide/quit, card mnemonics.

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

## Immediate next: manual Phase 6 gate + Inno Setup

Phase 6 code work is done (wheel fixed, run/turn/usage persistence wired,
PyInstaller onedir build passing its smoke test). What remains needs real
hardware and the real API:

### Manual Phase 6 gate

- Built exe launches on a clean path with no `.env` and no source tree,
  prompts for the API key, completes a voice turn, and writes state under
  `%LOCALAPPDATA%`.
- Rebuild command: `.venv\Scripts\pyinstaller packaging\gemini-live-agent.spec --clean --noconfirm`

### Inno Setup installer

`packaging/installer.iss` is drafted (per-machine install under Program Files,
Start Menu + optional desktop shortcut, post-install launch). To produce the
installer, install Inno Setup 6 and run:

```
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
```

Output lands in `dist\installer\gemini-live-agent-setup-0.2.0.exe`. Inno Setup
is not installed on this machine yet, so the script has not been compile-tested.

### Live wall tests still pending

- **Phase 3 gate:** 30-minute real session with `--share-screen`, crossing the
  ~10-minute connection wall repeatedly, recording reconnect count and gaps.
- **Phase 5 gate:** 20-minute streaming session while dragging the window and
  opening a native file dialog — zero audio dropouts; keyboard-only
  approve/deny; screen-reader pass.

### Risks / known issues

- **Old `scratch/` directory** at `C:\Users\Hp\.gemini\antigravity\scratch\gemini-live-agent` still exists. Remove it manually if still locked.
- **`.env` file still exists** in the repo root. It is ignored by git, but `settings.secrets.migrate_dotenv()` already moved the key into the credential store. Decide whether to delete the file or keep it as a dev-only convenience.
- **Directory/package name still has a hyphen.** No longer blocking: the flat layout builds fine under PyInstaller (Phase 6 smoke test passed). Only relevant if the project ever moves to a `src/` layout.
- **Live 30-minute wall test** from the Phase 3 verification gate has not been run yet. Run it before declaring Phase 3 fully validated in production.
- **`.claude/` worktree cleanup** was attempted: the `claude/adoring-wright-c18ce1` worktree metadata was removed from git, but the empty `.claude/worktrees/adoring-wright-c18ce1` directory could not be deleted because Windows reports it as "Device or resource busy" (likely held by a shell or explorer). Reboot or close the holding process, then delete `.claude/` manually.
- **Global hotkeys** (system-wide mute/push-to-talk) were not implemented in Phase 5 — no cross-platform hotkey dependency is declared. Revisit after packaging.

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
3. **Close button minimize-to-tray?** Resolved in Phase 5 — tray icon with Show/Hide, Mute, Quit; close minimizes to tray (persisted `minimize_to_tray` setting).
4. **Should the model be told the policy rules explicitly?** Currently only the workspace root and destructive-op approval are mentioned in the system instruction.
5. **Single `.exe` vs. Inno Setup?** Leaning onedir + Inno Setup per Context.txt; decide during Phase 6 packaging.
6. **Searchable transcript across runs?** Left out; Phase 6's incremental SQLite history is the prerequisite.

## Recommended first Phase 6 task

Run `python -m build` and inspect the produced wheel. Confirm every runtime
package (`app/`, `core/`, `media/`, `tools/`, `policy/`, `audit/`, `obs/`,
`persist/`, `settings/`, `gemini_live_agent/`) is included, and fix the
hatch `packages` target before writing the PyInstaller spec.
