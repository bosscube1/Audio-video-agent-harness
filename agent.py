"""Core agent loop — connects to Gemini Live, dispatches tools, manages audio.

This is the heart of the application. It opens a WebSocket session to the
Gemini Live API and runs three concurrent async tasks:

  1. _receive_loop  — listens for model text / audio / tool-call responses
  2. _stdin_loop    — sole reader of stdin; routes each line to either the
                      model as a turn, or to a pending safety confirmation
  3. _audio_send_loop — streams microphone PCM to the model in real time
  4. _video_send_loop — streams screen frames when --share-screen is on
"""

import asyncio
import inspect
import sys
from typing import Optional

from google.genai import types
from rich.box import ASCII
from rich.console import Console
from rich.panel import Panel

from audio import INPUT_SAMPLE_RATE, MicrophoneStream, SpeakerStream, check_audio_available
from config import AgentConfig
from safety import DESTRUCTIVE_TOOLS, SafetyGuard
from screen import ScreenCapture
from tools import ALL_TOOLS, TOOL_FUNCTIONS, set_working_dir

console = Console()

EXIT_COMMANDS = ("exit", "quit", "/exit", "/quit")


class GeminiLiveAgent:
    """Agentic harness wrapping the Gemini 3.1 Live API with local computer tools."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = config.build_genai_client()
        self.safety = SafetyGuard(yolo=config.yolo)
        self.mic: Optional[MicrophoneStream] = None
        self.speaker: Optional[SpeakerStream] = None
        self.screen: Optional[ScreenCapture] = None
        self._running = False
        # Set once so relative paths in tool calls resolve against --working-dir
        set_working_dir(config.working_dir)
        # Pending safety confirmation waiting on the next line of stdin
        self._confirm_waiter: Optional[asyncio.Future] = None
        # Transcript display state
        self._model_line_open = False
        self._user_transcript = ""

    # ─────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────

    async def run(self) -> None:
        """Connect to the Gemini Live API and run the interactive agent loop."""
        self._running = True
        use_mic = self.config.mode in ("voice", "both")
        use_speaker = check_audio_available()  # Always play audio if hardware exists

        # Graceful fallback if audio hardware is missing
        if not use_speaker:
            console.print(
                "[yellow][!] No audio device detected. "
                "Audio playback disabled.[/yellow]"
            )
            use_mic = False

        # Build the session configuration
        live_config = self.config.build_live_config(tools=ALL_TOOLS)

        self._print_banner(use_mic)

        try:
            async with self.client.aio.live.connect(
                model=self.config.model,
                config=live_config,
            ) as session:
                console.print("[bold green][OK] Connected to Gemini Live API[/bold green]\n")

                tasks: list[asyncio.Task] = []

                # Always run the receive loop (plays audio if speaker available)
                tasks.append(
                    asyncio.create_task(self._receive_loop(session, use_speaker), name="receive")
                )

                # Single stdin reader. Runs in every mode: even voice-only
                # needs it to answer safety confirmations.
                tasks.append(
                    asyncio.create_task(self._stdin_loop(session), name="stdin")
                )

                # Set up speaker for audio playback (model always responds with audio)
                if use_speaker:
                    self.speaker = SpeakerStream()
                    await self.speaker.start()

                # Set up mic only in voice/both mode
                if use_mic:
                    self.mic = MicrophoneStream()
                    await self.mic.start()
                    tasks.append(
                        asyncio.create_task(self._audio_send_loop(session), name="audio_send")
                    )

                # Screen sharing
                if self.config.share_screen:
                    self.screen = ScreenCapture(
                        fps=self.config.screen_fps,
                        monitor=self.config.monitor,
                    )
                    await self.screen.start()
                    tasks.append(
                        asyncio.create_task(self._video_send_loop(session), name="video_send")
                    )

                # Run until any task exits (e.g. user types 'exit')
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                # Tear down remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Propagate exceptions from finished tasks
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        raise exc

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted by user.[/dim]")
        except Exception as e:
            console.print(f"\n[bold red]Connection error: {e}[/bold red]")
            raise
        finally:
            self._running = False
            if self.screen:
                await self.screen.stop()
            if self.mic:
                await self.mic.stop()
            if self.speaker:
                await self.speaker.stop()
            console.print("[dim]Session ended.[/dim]")

    # ─────────────────────────────────────────
    # Receive loop
    # ─────────────────────────────────────────

    async def _receive_loop(self, session, use_audio: bool) -> None:
        """Listen for and handle all server messages (text, audio, tool calls)."""
        try:
            while self._running:
                async for response in session.receive():
                    sc = response.server_content
                    if sc:
                        # What the model heard from the microphone
                        if sc.input_transcription and sc.input_transcription.text:
                            self._user_transcript += sc.input_transcription.text

                        # The model replies in audio only, so its transcription
                        # is the readable output. It arrives in small chunks.
                        if sc.output_transcription and sc.output_transcription.text:
                            self._flush_user_transcript()
                            self._open_model_line()
                            console.print(
                                sc.output_transcription.text, end="", highlight=False
                            )

                        model_turn = sc.model_turn
                        if model_turn:
                            for part in model_turn.parts:
                                if part.text:
                                    self._flush_user_transcript()
                                    self._open_model_line()
                                    console.print(part.text, end="", highlight=False)
                                if part.inline_data and use_audio and self.speaker:
                                    self.speaker.write(part.inline_data.data)

                        # Barge-in: user spoke while model was speaking
                        if sc.interrupted:
                            self._close_model_line()
                            console.print("[dim](interrupted)[/dim]")
                            if self.speaker:
                                self.speaker.flush()

                        # End of model turn
                        if sc.turn_complete:
                            self._close_model_line()
                            console.print()  # visual separator

                    # ── Tool calls ──
                    if response.tool_call:
                        self._close_model_line()
                        await self._handle_tool_calls(session, response.tool_call)

                # session.receive() returns at the end of each turn; the outer
                # while re-enters it for the next one. A genuinely closed
                # connection raises here rather than returning, so this does
                # not spin.

        except asyncio.CancelledError:
            return
        except Exception as e:
            console.print(f"[bold red]Receive error: {e}[/bold red]")
            raise

    # ─────────────────────────────────────────
    # Tool dispatch
    # ─────────────────────────────────────────

    async def _handle_tool_calls(self, session, tool_call) -> None:
        """Execute tool calls locally and send results back to the model."""
        responses: list[types.FunctionResponse] = []

        for fc in tool_call.function_calls:
            name: str = fc.name
            args: dict = fc.args or {}
            call_id: str = fc.id

            console.print(
                f"[dim][TOOL] {name}({_format_args(args)})[/dim]"
            )

            # Safety gate for destructive operations
            if name in DESTRUCTIVE_TOOLS and not self.config.yolo:
                approved = await self.safety.confirm(name, args, self._read_stdin_line)
                if not approved:
                    responses.append(
                        types.FunctionResponse(
                            id=call_id,
                            name=name,
                            response={"result": "Operation denied by user."},
                        )
                    )
                    continue

            # Execute the tool
            try:
                func = TOOL_FUNCTIONS.get(name)
                if func is None:
                    result = f"Error: Unknown tool '{name}'"
                elif inspect.iscoroutinefunction(func):
                    result = await func(**args)
                else:
                    # Run synchronous tools in an executor to keep the loop responsive
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, lambda f=func, a=args: f(**a))

                console.print(f"[dim green]  [OK] {name} completed[/dim green]")
            except Exception as e:
                result = f"Error executing {name}: {e}"
                console.print(f"[dim red]  [X] {name} failed: {e}[/dim red]")

            responses.append(
                types.FunctionResponse(
                    id=call_id,
                    name=name,
                    response={"result": result},
                    scheduling="WHEN_IDLE",
                )
            )

        # Send all responses back in one batch
        if responses:
            await session.send_tool_response(function_responses=responses)

    # ─────────────────────────────────────────
    # Stdin loop
    # ─────────────────────────────────────────

    async def _stdin_loop(self, session) -> None:
        """Read stdin and route each line to a pending confirmation or the model.

        This is the only place stdin is read. A safety confirmation registers a
        waiter via `_read_stdin_line`; the next line goes to it instead of being
        sent to the model as a turn.
        """
        loop = asyncio.get_running_loop()
        try:
            while self._running:
                line: str = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:  # EOF
                    break
                user_input = line.strip()

                # A safety prompt is waiting for this line
                waiter = self._confirm_waiter
                if waiter is not None and not waiter.done():
                    waiter.set_result(user_input)
                    continue

                if not user_input:
                    continue

                if user_input.lower() in EXIT_COMMANDS:
                    console.print("[dim]Goodbye![/dim]")
                    self._running = False
                    return

                # Voice-only mode: keyboard is for confirmations and exit only
                if self.config.mode == "voice":
                    continue

                await session.send_client_content(
                    turns=[
                        types.Content(
                            role="user",
                            parts=[types.Part(text=user_input)],
                        )
                    ],
                    turn_complete=True,
                )

        except (EOFError, asyncio.CancelledError):
            return

    async def _read_stdin_line(self) -> str:
        """Claim the next line of stdin (used by the safety confirmation prompt)."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._confirm_waiter = future
        try:
            return await future
        finally:
            self._confirm_waiter = None

    # ─────────────────────────────────────────
    # Audio send loop
    # ─────────────────────────────────────────

    async def _audio_send_loop(self, session) -> None:
        """Capture microphone PCM and stream it to the model in real time."""
        while self._running and self.mic:
            try:
                pcm_bytes = await self.mic.read_chunk()
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=pcm_bytes,
                        # The Live API requires the sample rate in the mime type
                        mime_type=f"audio/pcm;rate={INPUT_SAMPLE_RATE}",
                    )
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                console.print(f"[bold red]Audio send error: {e}[/bold red]")
                await asyncio.sleep(0.1)

    # ─────────────────────────────────────────
    # Video send loop
    # ─────────────────────────────────────────

    async def _video_send_loop(self, session) -> None:
        """Stream screen frames to the model as realtime video."""
        while self._running and self.screen:
            try:
                frame = await self.screen.read_frame()
                await session.send_realtime_input(
                    video=types.Blob(data=frame, mime_type="image/jpeg")
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                console.print(f"[bold red]Screen send error: {e}[/bold red]")
                await asyncio.sleep(1)

    # ─────────────────────────────────────────
    # UI helpers
    # -----------------------------------------

    def _open_model_line(self) -> None:
        """Print the 'Gemini:' prefix once per turn, before streamed output."""
        if not self._model_line_open:
            console.print("[bold cyan]Gemini:[/bold cyan] ", end="")
            self._model_line_open = True

    def _close_model_line(self) -> None:
        """Terminate a streamed model line, if one is open."""
        if self._model_line_open:
            console.print()
            self._model_line_open = False

    def _flush_user_transcript(self) -> None:
        """Print what the model heard from the mic, once the turn is over."""
        if self._user_transcript.strip():
            console.print(
                f"[dim]You (voice): {self._user_transcript.strip()}[/dim]",
                highlight=False,
            )
        self._user_transcript = ""

    def _print_banner(self, use_audio: bool) -> None:
        """Print the startup banner with session info."""
        if use_audio and self.config.mode == "both":
            mode_str = "Voice + Text"
        elif use_audio:
            mode_str = "Voice only"
        else:
            mode_str = "Text only"

        safety_str = (
            "[red]OFF (--yolo)[/red]" if self.config.yolo else "[green]ON[/green]"
        )

        if self.config.share_screen:
            target = "all monitors" if self.config.monitor == 0 else f"monitor {self.config.monitor}"
            screen_str = f"[yellow]ON[/yellow] ({target}, {self.config.screen_fps:g} fps)"
        else:
            screen_str = "[dim]off[/dim]"

        banner = (
            f"[bold]Model:[/bold]      {self.config.model}\n"
            f"[bold]Mode:[/bold]       {mode_str}\n"
            f"[bold]Voice:[/bold]      {self.config.voice}\n"
            f"[bold]Safety:[/bold]     {safety_str}\n"
            f"[bold]Screen:[/bold]     {screen_str}\n"
            f"[bold]Work Dir:[/bold]   {self.config.working_dir}\n"
            f"\n[dim]Type [bold]exit[/bold] to quit | speak freely in voice mode[/dim]"
        )

        console.print(
            Panel(
                banner,
                title="[bold magenta]>> Gemini Live Agent <<[/bold magenta]",
                border_style="magenta",
                box=ASCII,
                padding=(1, 2),
            )
        )


def _format_args(args: dict) -> str:
    """Format tool arguments for compact log display."""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 80:
            v_str = v_str[:80] + "..."
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)
