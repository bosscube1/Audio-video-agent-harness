"""Redact secrets and credentials from tool output before it leaves the sandbox."""

from __future__ import annotations

import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}"), "[REDACTED: Google API key]"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED: OpenAI-style key]"),
    (re.compile(r"ghp_[A-Za-z0-9]{36}"), "[REDACTED: GitHub personal access token]"),
    (
        re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
        "[REDACTED: GitHub fine-grained token]",
    ),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED: AWS access key ID]"),
    (
        re.compile(
            r"-----BEGIN [A-Z ]+-----\n[\s\S]*?\n-----END [A-Z ]+-----",
            re.MULTILINE,
        ),
        "[REDACTED: PEM block]",
    ),
]


def redact(text: str) -> str:
    """Scrub common secrets from ``text``."""
    scrubbed = text
    for pattern, replacement in _PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)
    return scrubbed
