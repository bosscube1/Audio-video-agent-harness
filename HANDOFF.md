# Handoff & Continuation Plan

## Current State

- **Repository:** `C:\Users\Hp\projects\gemini-live-agent`
- **Remote:** `https://github.com/bosscube1/Audio-video-agent-harness.git`
- **Branch:** `Kimi-V2`
- **Latest commit:** `ad3cebd` — `feat(phase2): headless core rewrite — session, media, tools, policy, audit`
- **Phases complete:** 1 (foundation) and 2 (headless core extraction)
- **Phase ready to start:** 3 (session lifetime / resumption)

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

- **Packaging:** `pyproject.toml`, locked `requirements*.txt`, `uv` venv, ruff/mypy/pytest configs.
- **Logging:** JSON Lines rotating logs + stderr, `run_id` on every record.
- **Core vocabulary:** frozen dataclasses in `core/events.py` / `core/commands.py`.
- **Single-session headless driver:** connects, sends text, streams mic/screen, plays speaker, dispatches tools.
- **Tool layer:** explicit pydantic schemas, no docstring introspection, `web_search` deleted.
- **Policy layer:** Windows path confinement, deny globs, sensitive-name rules, gating mode, outbound redaction.
- **Audit + persistence:** append-only JSONL journal, SQLite store schema.
- **Media layer:** explicit device selection, bounded mic queue, speaker generation-counter flush fix.
- **Entry points:** `python -m gemini_live_agent`, `gemini-live-agent` console script, and `python main.py` all work.

## Key design decisions already locked in

- **Single process, asyncio core.** GUI will live on a `QThread` with Qt signals in, `loop.call_soon_threadsafe` out. `core/**` never imports PySide6; `app/**` never imports asyncio primitives.
- **Frozen dataclasses across the thread boundary.** No shared mutable state.
- **Meters polled, not pushed.** `AudioLevel` events exist but the GUI should poll `Microphone.level` / `Speaker.level` at ~30 Hz.
- **Transcription coalesced to 80 ms in core.** `TurnState` emits at most one `PartialTranscript` per source per window.
- **Typed text via `send_client_content`.** Verified working on `gemini-3.1-flash-live-preview` with `turn_complete=True`. This answers the Phase 2 open question; the Phase 3 reinjection fallback can use the same path.

## Immediate next: Phase 3 — Session Lifetime

Phase 3 is the highest-leverage reliability work. The goal is to survive the ~10-minute WebSocket connection wall via resumption + proactive reconnect.

### Files to create / change

- `core/supervisor.py` — NEW. The conversation that survives sockets.
- `core/session.py` — MODERATE. Add `session_epoch`, resumption handle plumbing, disconnect classification.
- `persist/store.py` — ADD. Save/expire resumption handles.
- `core/events.py` — ADD. `SessionExpiring`, `ContextReset` already exist; add `Reconnecting` if needed.
- `tests/fakes/fake_live.py` — NEW. Highest-leverage test asset.
- `tests/test_supervisor.py` — NEW.

### Implementation checklist

1. **Fake Live session harness first.**
   - Scriptable: drop socket, send `go_away`, reject handle, reject `fc_id`, send `tool_call_cancellation`.
   - Must expose the same interface as `client.aio.live.connect(...) async with ...` so `LiveSession` can be tested against it.

2. **Day-1 live experiments.**
   - Does `fc_id` survive a resumption reconnect? Run once against a real socket before building on the answer.
   - Confirm `send_client_content` reinjection after reconnect lands as restored history, not a fresh user turn.

3. **Supervisor state machine.**
   - States: `IDLE -> CONNECTING -> LIVE -> {DRAINING, RECONNECTING} -> CONNECTING -> LIVE`, `CLOSING -> CLOSED`, `FAILED`.
   - Each socket generation gets a monotonic `session_epoch` stamped on every event/tool-call/audit record.

4. **Resumption.**
   - First connect: `session_resumption=types.SessionResumptionConfig(handle=None)`.
   - Accept `new_handle` only when `resumable is True`.
   - Persist handle to SQLite; expire locally at 110 min.

5. **GoAway handling.**
   - On `go_away.time_left`, transition to `DRAINING`, emit `SessionExpiring(seconds)`, stop accepting new tool calls, proactively reconnect.
   - Hard cut: close, reconnect with handle, resume. Overlapped reconnect is explicitly Phase 5 cut-list.

6. **Classified backoff.**
   - Full jitter: `random.uniform(0, min(30, 0.5 * 2**attempt))`.
   - Failure budget: 6 connects in 5 min -> `FAILED`.
   - Auth/not-found -> no retry; quota -> base 5s cap 120s budget 3; transient -> base 0.5s cap 30s budget 6; rejected handle -> drop handle, one fresh reconnect, then transient.

7. **In-flight tools across reconnect.**
   - Journal-before-execute invariant already holds in dispatcher.
   - `pending_approval` -> auto-deny, reason `connection_lost`.
   - `running` -> let finish, journal, mark `orphaned`.
   - On reconnect, try `send_tool_response` with original `fc_id`. On rejection, reinject as client content system note.
   - No valid handle -> reconnect fresh, seed from SQLite, emit visible `ContextReset`.

8. **`tool_call_cancellation` handling.**
   - Queued tool -> drop.
   - Awaiting approval -> dismiss card.
   - Running shell -> `proc.kill()`; file ops -> let finish.
   - Never send `FunctionResponse` for a cancelled id.

### Phase 3 verification gate

From the spec:

> Against `FakeLiveSession` — socket drop mid-turn resumes and the conversation continues; GoAway drains and reconnects with < 1s audible gap; a `running` tool at drop time completes, is journaled, and its result reaches the model post-reconnect; rejected handle produces a visible `ContextReset`. Then live: a 30-minute real session with `--share-screen` on, crossing the ~10-minute connection wall repeatedly.

### Risks / known issues

- **Old `scratch/` directory** at `C:\Users\Hp\.gemini\antigravity\scratch\gemini-live-agent` could not be deleted because this session had it locked. Remove it manually or wait until the session releases it.
- **`python -m gemini_live_agent`** works because a `gemini_live_agent/` package exists. The directory name still has a hyphen; this is fine for local dev but will need a proper package layout before PyInstaller packaging.
- **`.env` file still exists** in the repo root. `settings.secrets.migrate_dotenv()` moved the key into the credential store, but the file itself was left for Phase 4 deletion.
- **Policy immutable dirs** currently use `GeminiLiveAgent` casing. Ensure this matches wherever `platformdirs` is called.

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

## Open questions from the original spec

1. **VB-Audio Virtual Cable default input** — explicit device selection is now implemented; the loopback guard is implemented in `media/devices.py`. Decision: it currently warns. If you want it to hard-refuse, change `detect_loopback()` callers.
2. **Does `fc_id` survive resumption?** Phase 3 must answer this live before building on it. Fallback (client-content reinjection) already planned.
3. **Close button minimize-to-tray?** Deferred to GUI Phase 5.
4. **Should the model be told the policy?** Currently the system instruction mentions the workspace root and that destructive ops need approval. Consider adding allow-roots/command rules if denied calls become noisy.
5. **Single `.exe` vs. Inno Setup?** Deferred to Phase 6 packaging.
6. **Searchable transcript across runs?** Left out.

## Recommended first Phase 3 task

Build `tests/fakes/fake_live.py`. It is the highest-leverage test asset; you cannot reliably exercise drop/GoAway/rejected-handle/cancellation against the live preview API on demand.
