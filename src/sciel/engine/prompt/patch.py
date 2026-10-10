"""SEARCH/REPLACE edits applied exact, then whitespace-normalized, then by a unique fuzzy window."""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import re

_BLOCK = re.compile(r"<{5,9} ?SEARCH[^\n]*\n(.*?)\n?={5,9}[^\n]*\n(.*?)\n?>{5,9} ?REPLACE", re.DOTALL)
FUZZY_MIN = 0.9
FUZZY_MARGIN = 0.05


@dataclass
class PatchResult:
    code: str
    blocks: int
    placements: list = field(default_factory=list)

    @property
    def applied(self) -> int:
        return sum(p["match"] != "failed" for p in self.placements)

    def summary(self) -> dict:
        kinds = [p["match"] for p in self.placements]
        return {"blocks": self.blocks, "applied": self.applied,
                "exact": kinds.count("exact"), "whitespace": kinds.count("whitespace"),
                "fuzzy": kinds.count("fuzzy"), "failed": kinds.count("failed"),
                "min_ratio": min((p["ratio"] for p in self.placements if "ratio" in p), default=None)}


def parse_edits(response: str) -> list[tuple[str, str]]:
    return [(search, replace) for search, replace in _BLOCK.findall(response) if search.strip()]


def _indent(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def _reindent(replace: list[str], found: list[str], search: list[str]) -> list[str]:
    """Shift replacement lines by the indent difference between the located text and SEARCH."""
    def first(lines):
        return next((_indent(l) for l in lines if l.strip()), "")
    have, wrote = first(found), first(search)
    if have == wrote:
        return replace
    out = []
    for line in replace:
        if line.strip() and line.startswith(wrote):
            out.append(have + line[len(wrote):])
        elif line.strip() and not wrote:
            out.append(have + line)
        else:
            out.append(line)
    return out


def _locate_lines(lines: list[str], search: list[str]):
    """Unique window equal to SEARCH after whitespace normalization, else a unique fuzzy window."""
    n = len(search)
    if n == 0 or n > len(lines):
        return None
    key = [" ".join(l.split()) for l in search]
    hits = [i for i in range(len(lines) - n + 1)
            if [" ".join(l.split()) for l in lines[i:i + n]] == key]
    if len(hits) == 1:
        return hits[0], "whitespace", 1.0
    if hits:
        return None
    target = "\n".join(l.strip() for l in search)
    scored = sorted(((difflib.SequenceMatcher(None, target, "\n".join(l.strip() for l in lines[i:i + n])).ratio(), i)
                     for i in range(len(lines) - n + 1)), reverse=True)
    best, idx = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    if best >= FUZZY_MIN and best - runner >= FUZZY_MARGIN:
        return idx, "fuzzy", round(best, 4)
    return None


def _apply_one(source: str, search: str, replace: str):
    if source.count(search) == 1:
        return source.replace(search, replace, 1), {"match": "exact"}
    lines = source.split("\n")
    search_lines, replace_lines = search.split("\n"), replace.split("\n")
    found = _locate_lines(lines, search_lines)
    if found is None:
        return source, {"match": "failed", "reason": "ambiguous" if source.count(search) > 1 else "not_found"}
    start, kind, ratio = found
    span = lines[start:start + len(search_lines)]
    patched = lines[:start] + _reindent(replace_lines, span, search_lines) + lines[start + len(search_lines):]
    return "\n".join(patched), {"match": kind, "ratio": ratio, "line": start + 1}


def apply_edits(source: str, edits: list[tuple[str, str]]) -> PatchResult:
    code, placements = source, []
    for search, replace in edits:
        code, placement = _apply_one(code, search, replace)
        placements.append(placement)
    if not code.endswith("\n"):
        code += "\n"
    return PatchResult(code=code, blocks=len(edits), placements=placements)
