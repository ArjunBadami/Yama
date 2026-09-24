"""Small text utilities shared by dataset builders."""

from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, sentence) using the same splitter, with char offsets."""
    spans: list[tuple[int, int, str]] = []
    pos = 0
    for s in split_sentences(text):
        idx = text.find(s, pos)
        if idx < 0:
            # whitespace normalisation may have changed the string; fall back
            idx = pos
        spans.append((idx, idx + len(s), s))
        pos = idx + len(s)
    return spans
