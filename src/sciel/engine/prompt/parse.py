"""Extract candidate code and the C2 bet from an LLM response."""

from __future__ import annotations

import re

from ..types import Bet

# Consume every complete fenced block before selecting Python. Otherwise a
# skipped JSON opener lets its closing fence consume the next Python opener.
_FENCE = re.compile(r"```([^\r\n`]*)\r?\n(.*?)```", re.DOTALL)
_BET = re.compile(r"BET:\s*\{.*?mechanism\s*:\s*(?P<q1>[\"'])(?P<mech>.*?)(?P=q1)"
                  r".*?q\s*:\s*(?P<q>[0-9]*\.?[0-9]+)", re.DOTALL)


def extract_code(response: str) -> str:
    matches = [body for language, body in _FENCE.findall(response)
               if language.strip() in ("", "python")]
    if matches:
        return max(matches, key=len).strip() + "\n"
    return response.strip() + "\n"


def has_code_block(response: str) -> bool:
    return any(language.strip() in ("", "python") for language, _ in _FENCE.findall(response))


def parse_bet(response: str) -> Bet | None:
    m = _BET.search(response)
    if not m:
        return None
    try:
        q = float(m.group("q"))
    except ValueError:
        return None
    q = min(1.0, max(0.0, q))
    return Bet(mechanism=m.group("mech").strip(), q=q)


def extract_card(text: str, limit: int = 200) -> str:
    """Strategy card: the prose the LLM wrote BEFORE its first code fence,
    compressed to one line. Free -- no extra call; rides on every response."""
    head = text.split("```", 1)[0]
    line = " ".join(head.split())
    return line[:limit]
