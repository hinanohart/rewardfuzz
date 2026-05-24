"""Defenses for the (optional) LLM judge against being gamed by the candidate it inspects.

A candidate is adversarial by construction, so it may contain prompt-injection text aimed at the
judge. Two helpers cooperate before a candidate is embedded in a judge prompt: ``sanitize``
truncates over-long text and redacts role markers, while ``looks_injected`` flags direct
reward-injection phrasing so the caller can fall back to the (un-gameable) structural verdict.
"""

from __future__ import annotations

import re

_ROLE_MARKERS = re.compile(
    r"(?im)^\s*(system|assistant|user)\s*:|<<\s*system\s*>>|</?s>|\[INST\]|\[/INST\]"
)
_INJECTION = re.compile(
    r"(?i)(ignore (all )?previous|disregard (the )?above|assign full reward|reward\s*=\s*1)"
)

MAX_LEN = 4000


def looks_injected(text: str) -> bool:
    return bool(_INJECTION.search(text))


def sanitize(text: str) -> str:
    """Make candidate text safe to embed in a judge prompt."""

    text = _ROLE_MARKERS.sub("[redacted-role-marker]", text)
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN] + "\n...[truncated]..."
    return text
