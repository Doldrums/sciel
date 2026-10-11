"""Per-block character budget of the generation prompt; the task and the parent program are never trimmed."""

from __future__ import annotations

CAPS = {"inspirations": 8000, "directive": 8000, "lessons": 3000, "artifacts": 4000}
RETRY_TAIL = 2000


def fit(name: str, text: str) -> str:
    cap = CAPS[name]
    if len(text) <= cap:
        return text
    note = f"\n[{len(text) - cap} characters of this block omitted by the prompt budget]"
    return text[:cap - len(note)] + note


def tail(text: str, limit: int = RETRY_TAIL) -> str:
    if len(text) <= limit:
        return text
    return f"[first {len(text) - limit} characters omitted]\n" + text[-limit:]
