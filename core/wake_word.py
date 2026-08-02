"""Wake-word detection for voice gating.

The matcher answers one narrow question: does this ASR transcript contain the
configured wake word? It is deliberately tolerant — ASR routinely mangles short
names — because a false negative (ignoring the user) is cheaper than a false
positive (interrupting a conversation), but the word must still be *present*
in some form; the model's system prompt handles the harder question of whether
the utterance is actually directed at the agent.
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    """Lowercase and collapse every non-alphanumeric run to a single space."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def _levenshtein(a: str, b: str) -> int:
    """Classic edit distance, O(min(len)) space — tokens here are short."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def contains_wake_word(text: str, wake_word: str) -> bool:
    """Return True if ``text`` appears to contain ``wake_word``.

    Matching rules, in order:
    1. Normalized substring — handles "hey bongo", "thanks, Bongo!", and
       multi-word wake phrases.
    2. Per-token fuzzy match — a single-token wake word matches any transcript
       token within edit distance 1 (ASR near-misses like "bonggo" or "bingo").
       Fuzzy matching is disabled for wake words shorter than 4 characters,
       where one edit changes the word too easily.
    """
    normalized_text = _normalize(text)
    normalized_wake = _normalize(wake_word)
    if not normalized_text or not normalized_wake:
        return False

    if normalized_wake in normalized_text:
        return True

    if " " in normalized_wake or len(normalized_wake) < 4:
        return False

    return any(
        _levenshtein(token, normalized_wake) <= 1
        for token in normalized_text.split()
        # Skip tokens whose length differs by more than the allowed distance.
        if abs(len(token) - len(normalized_wake)) <= 1
    )
