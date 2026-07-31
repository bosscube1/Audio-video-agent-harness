"""Execute tool calls off the receive path.

``ToolDispatcher`` owns policy evaluation, user approval, execution, timeout,
and cancellation of Gemini function calls. It never blocks the receive loop:
each call is submitted as a separate asyncio task by the session.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import tools.fs as fs_tools
import tools.shell as shell_tools
from audit.journal import Journal
from core.commands import GatingMode
from core.events import (
    AgentEvent,
    ToolApprovalRequested,
    ToolCallCancelled,
    ToolCallReceived,
    ToolResultSent,
)
from policy import PolicyEngine, redact
from policy.engine import DecisionKind
from tools.registry import (
    DeleteFileArgs,
    ListDirectoryArgs,
    ReadFileArgs,
    RunCommandArgs,
    ToolResult,
    WriteFileArgs,
)

APPROVAL_TIMEOUT_SECONDS = 60.0

# Map tool names to their implementation and pydantic argument model.
_TOOL_TABLE: dict[
    str,
    tuple[
        Callable[[Any], ToolResult] | Callable[[Any], Awaitable[ToolResult]],
        type[ReadFileArgs | WriteFileArgs | ListDirectoryArgs | DeleteFileArgs | RunCommandArgs],
    ],
] = {
    "read_file": (fs_tools.read_file, ReadFileArgs),
    "write_file": (fs_tools.write_file, WriteFileArgs),
    "list_directory": (fs_tools.list_directory, ListDirectoryArgs),
    "delete_file": (fs_tools.delete_file, DeleteFileArgs),
    "run_command": (shell_tools.run_command, RunCommandArgs),
}

RespondCallback = Callable[[str, str, ToolResult], Awaitable[None]]


class ToolDispatcher:
    """Async tool-call executor with policy gating and cancellation.

    Parameters
    ----------
    policy:
        Active policy engine for this session.
    journal:
        Audit journal; ``tool_attempt`` is written before any side effect.
    event_queue:
        Queue where core events are published for the GUI.
    run_id:
        Run identifier forwarded to the journal.
    epoch:
        Resumption epoch forwarded to the journal.
    """

    def __init__(
        self,
        policy: PolicyEngine,
        journal: Journal,
        event_queue: asyncio.Queue[AgentEvent],
        run_id: str,
        epoch: int = 0,
    ) -> None:
        self._policy = policy
        self._journal = journal
        self._event_queue = event_queue
        self._run_id = run_id
        self._epoch = epoch
        self._respond: RespondCallback | None = None

        # Pending user confirmations.
        self._approval_events: dict[str, asyncio.Event] = {}
        self._approval_results: dict[str, bool] = {}

        # Cancelled ids still in flight. Running file ops are allowed to finish,
        # but no FunctionResponse is sent for these ids.
        self._cancelled_ids: set[str] = set()

    def set_respond_callback(self, callback: RespondCallback) -> None:
        """Set the callback used to send ``FunctionResponse`` objects upstream."""
        self._respond = callback

    def set_gating_mode(self, mode: GatingMode) -> None:
        """Forward a gating-mode change to the active policy engine."""
        self._policy.set_gating_mode(mode)

    async def submit(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        """Execute or queue a single tool call.

        This method runs off the receive loop (typically as a task) so that
        blocking or waiting-for-approval tools do not stall message handling.
        """
        if self._consume_cancellation(call_id):
            return

        self._journal.tool_attempt(call_id, name, args, epoch=self._epoch)
        await self._emit(ToolCallReceived(call_id=call_id, name=name, args=args))

        # Resolve the tool implementation and argument schema.
        entry = _TOOL_TABLE.get(name)
        if entry is None:
            result = ToolResult(False, f"Unknown tool '{name}'")
            await self._finish(call_id, name, result)
            return

        func, schema = entry

        # Validate arguments before any side effect or policy decision.
        try:
            validated_args = schema.model_validate(args)
        except Exception as exc:  # pydantic.ValidationError
            result = ToolResult(False, f"Invalid arguments for {name}: {exc}")
            await self._finish(call_id, name, result)
            return

        # Policy check. The tool implementations perform their own deny checks
        # as a safety net; we evaluate here so denials become clean UI events.
        decision = self._policy.evaluate(name, validated_args)

        if decision.kind == DecisionKind.DENY:
            result = ToolResult(False, f"Policy denied: {decision.reason}")
            await self._finish(call_id, name, result)
            return

        if decision.kind == DecisionKind.CONFIRM:
            await self._emit(
                ToolApprovalRequested(
                    call_id=call_id,
                    name=name,
                    args=args,
                    allow_text=decision.reason,
                )
            )
            approved = await self._wait_for_approval(call_id)
            if not approved:
                if self._consume_cancellation(call_id):
                    return
                reason = "approval_timeout"
                if call_id in self._cancelled_ids:
                    reason = "cancelled"
                    self._cancelled_ids.discard(call_id)
                elif call_id in self._approval_results:
                    # Explicit deny from the user.
                    reason = "user_denied"
                result = ToolResult(False, f"Tool not executed: {reason}")
                await self._finish(call_id, name, result)
                return

        # Allowed or explicitly approved: execute the tool.
        try:
            if asyncio.iscoroutinefunction(func):
                raw_result = await func(validated_args)
            else:
                raw_result = await asyncio.to_thread(func, validated_args)
        except Exception as exc:  # pragma: no cover - defensive
            raw_result = ToolResult(False, f"Error executing {name}: {exc}")

        # Redact before it leaves the sandbox.
        result = ToolResult(raw_result.ok, redact(raw_result.message))
        await self._finish(call_id, name, result)

    def approve(self, call_id: str) -> None:
        """Approve a pending tool call."""
        self._approval_results[call_id] = True
        event = self._approval_events.pop(call_id, None)
        if event is not None:
            event.set()

    def deny(self, call_id: str, reason: str = "user_denied") -> None:
        """Deny a pending tool call."""
        self._approval_results[call_id] = False
        event = self._approval_events.pop(call_id, None)
        if event is not None:
            event.set()

    def cancel(self, call_ids: list[str]) -> None:
        """Cancel the given tool call ids.

        Pending approvals are dropped immediately. Running shell processes are
        not killed in Phase 2 (NotImplemented). Running file operations are
        allowed to finish, but no ``FunctionResponse`` is ever sent for a
        cancelled id.
        """
        for call_id in call_ids:
            already_pending = call_id in self._approval_events
            if already_pending:
                self._approval_results[call_id] = False
                event = self._approval_events.pop(call_id)
                event.set()

            self._cancelled_ids.add(call_id)
            # Emit once, even if the id was not known to the dispatcher.
            asyncio.create_task(
                self._emit(
                    ToolCallCancelled(call_ids=[call_id], reason="cancelled_by_user")
                )
            )

    async def _wait_for_approval(self, call_id: str) -> bool:
        """Wait for approve/deny/cancel, with a timeout.

        Returns ``True`` if the call was explicitly approved.
        """
        event = asyncio.Event()
        self._approval_events[call_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=APPROVAL_TIMEOUT_SECONDS)
        except TimeoutError:
            self._approval_events.pop(call_id, None)
            return False

        self._approval_events.pop(call_id, None)
        return self._approval_results.pop(call_id, False)

    async def _finish(self, call_id: str, name: str, result: ToolResult) -> None:
        """Journal, emit, and respond with a completed tool result."""
        self._journal.tool_result(
            call_id,
            name,
            result.ok,
            result.message,
            epoch=self._epoch,
        )
        await self._emit(
            ToolResultSent(
                call_id=call_id,
                name=name,
                ok=result.ok,
                result=result.message,
            )
        )

        if self._consume_cancellation(call_id):
            return

        respond = self._respond
        if respond is not None:
            await respond(call_id, name, result)

    def _consume_cancellation(self, call_id: str) -> bool:
        """Return True and remove the id if it has been cancelled."""
        if call_id in self._cancelled_ids:
            self._cancelled_ids.discard(call_id)
            return True
        return False

    async def _emit(self, event: AgentEvent) -> None:
        await self._event_queue.put(event)
