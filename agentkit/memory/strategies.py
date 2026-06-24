"""Memory extraction strategies (§5.2).

Heuristic, LLM-free extractors so long-term memory populates with zero infra. The
same hooks accept an LLM-backed extractor later (Mem0/provider) without changing
the read/write paths.
"""

from __future__ import annotations

import re

from ..types import Message

_PREF = re.compile(
    r"\b(i (?:prefer|like|love|hate|want|need|use|always|never)\b.*)", re.IGNORECASE
)


def extract_semantic(messages: list[Message]) -> list[str]:
    """Atomic facts: the salient user/assistant utterances from a transcript."""
    facts: list[str] = []
    for m in messages:
        if m.role in ("user", "assistant") and m.content:
            text = m.content.strip()
            if len(text) >= 3:
                facts.append(f"{m.role}: {text}")
    return facts


def extract_preferences(messages: list[Message]) -> list[str]:
    """Typed preference statements (e.g. 'I prefer Python')."""
    prefs: list[str] = []
    for m in messages:
        if m.role == "user" and m.content:
            for match in _PREF.findall(m.content):
                prefs.append(match.strip().rstrip("."))
    return prefs


def build_summary(messages: list[Message], *, max_chars: int = 1000) -> str:
    """Rolling session summary: a compact concatenation of the transcript."""
    lines = [f"{m.role}: {m.content}" for m in messages if m.content]
    text = " | ".join(lines)
    return text[:max_chars]
