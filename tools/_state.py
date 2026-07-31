"""Internal module-level state shared by the tool implementations.

Kept separate from ``tools.__init__`` to avoid circular imports.
"""

from __future__ import annotations

from pathlib import Path

from core.commands import GatingMode
from policy import PolicyEngine

_workspace_root: Path = Path.cwd()
_policy: PolicyEngine = PolicyEngine(
    allow_roots=[_workspace_root],
    deny_globs=[],
    never_allow_regex=[],
)
_policy.set_workspace(_workspace_root)


def set_workspace(path: str | Path) -> None:
    """Change the working directory used to resolve relative tool paths."""
    global _workspace_root
    _workspace_root = Path(path).resolve()
    _policy.set_workspace(_workspace_root)
    if _workspace_root not in _policy._allow_roots:
        _policy._allow_roots.append(_workspace_root)


def set_policy(engine: PolicyEngine) -> None:
    """Replace the active policy engine."""
    global _policy
    _policy = engine
    _policy.set_workspace(_workspace_root)


def set_gating_mode(mode: GatingMode) -> None:
    """Set the gating mode on the active policy engine."""
    _policy.set_gating_mode(mode)


# Aliases that match the naming convention used by the rest of the codebase.
set_policy_engine = set_policy
set_workspace_root = set_workspace
