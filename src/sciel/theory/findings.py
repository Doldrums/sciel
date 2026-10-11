"""Knowledge delivered to program generation: findings of research cells and lessons from island events.

Every item reaches every island only as a randomized arm judged by the ledger; proven items become culture.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from . import schools

CANDIDATES = 4
NONE_SHARE = 0.4
MAX_CULTURE = 3
MAX_CHARS = 600
_NUMBER = re.compile(r"(?<![\w.])-?(?:\d+\.\d+(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])")
_WORD = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
SIMILAR = 0.6


def _tokens(text: str) -> set:
    return {"#" if w[0].isdigit() else w for w in _WORD.findall(text.lower()) if len(w) > 2 or w[0].isdigit()}


def _similar(a: str, b: str) -> bool:
    x, y = _tokens(a), _tokens(b)
    return bool(x and y) and len(x & y) / len(x | y) >= SIMILAR


def _round(text: str) -> str:
    return _NUMBER.sub(lambda m: f"{float(m.group()):.4g}", text)


_HEDGE = re.compile(r"\b(guess|not run|n't run|untested|cannot run|can't run|estimate of the score)\b", re.IGNORECASE)


def lesson(mechanism: str, before: float, after: float) -> str:
    """A measured improvement: the mechanism the generator named, without its hedges, and the scores it moved."""
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", " ".join(mechanism.split())) if x and not _HEDGE.search(x)]
    return f"{' '.join(sentences[:2])} Measured: {before:.4g} -> {after:.4g}."


def fresh(items: list) -> str:
    """Conclusions the current episode just reached, for the generation step that follows it."""
    lines = [f"- {_round(' '.join(item['text'].split()))}"[:MAX_CHARS] for item in items if item.get("receipt_id")]
    return ("\nFindings this research cell just reached (measured by tools, not by the task evaluator):\n"
            + "\n".join(lines) + "\n") if lines else ""


@dataclass
class Finding:
    id: str
    text: str
    cell_id: str | None
    island: int
    receipt_id: str | None
    born: int
    status: str = "trial"
    kind: str = "finding"
    created: int = 1
    delivered: int = 0
    islands: list = field(default_factory=list)
    local: dict = field(default_factory=dict)

    def card(self) -> str:
        if self.kind == "lesson":
            return f"- {self.text} (strategy of a program that improved island {self.island})"[:MAX_CHARS]
        source = f" [receipt {self.receipt_id[:12]}]" if self.receipt_id else ""
        return f"- {self.text} (research cell on island {self.island}{source})"[:MAX_CHARS]


class Findings:
    def __init__(self):
        self.pool: list[Finding] = []

    def add(self, items: list, iteration: int) -> list[Finding]:
        added = []
        for item in items:
            if item.get("kind", "finding") == "finding" and not item.get("receipt_id"):
                continue
            text = _round(" ".join(item["text"].split()))
            same = next((f for f in self.pool if f.kind == item.get("kind", "finding") and f.status != "retired"
                         and _similar(f.text, text)), None)
            if same is not None:
                same.created += 1
                same.text, same.receipt_id = text, item.get("receipt_id") or same.receipt_id
                continue
            finding = Finding(id=f"f{len(self.pool) + 1}", text=text, cell_id=item.get("cell_id"),
                              island=int(item["island_id"]), receipt_id=item.get("receipt_id"), born=iteration,
                              kind=item.get("kind", "finding"))
            self.pool.append(finding)
            added.append(finding)
        return added

    def culture(self, island: int) -> list[Finding]:
        shown = [f for f in self.pool if f.status == "culture" or f in schools.local_culture(self.pool, island)]
        return [f for f in shown if f.local.get(str(island)) != "retired"][-MAX_CULTURE:]

    def distribution(self, island: int) -> dict:
        trials = schools.trials(self.pool, island)[:CANDIDATES]
        if not trials:
            return {"none": 1.0}
        return {"none": NONE_SHARE} | {f.id: (1 - NONE_SHARE) / len(trials) for f in trials}

    def deliver(self, chosen: str, island: int) -> str:
        culture = self.culture(island)
        shown = culture + [f for f in self.pool if f.id == chosen and f not in culture]
        for finding in shown:
            finding.delivered += 1
            if island not in finding.islands:
                finding.islands.append(island)
        if not shown:
            return ""
        return "\nKnowledge from this run (cell findings are measured by tools, not by the task evaluator):\n" + "\n".join(
            f.card() for f in shown) + "\n"

    def judge(self, theory) -> list[Finding]:
        changed = []
        for finding in self.pool:
            status = schools.verdict(theory, "finding", finding.id)
            if status is not None and finding.status != status:
                finding.status = status
                changed.append(finding)
        return changed

    def counters(self) -> list[dict]:
        return [{"id": f.id, "kind": f.kind, "status": f.status, "created": f.created, "delivered": f.delivered,
                 "islands": len(f.islands)} for f in self.pool]

    def top(self, k: int = 5) -> list[str]:
        ranked = [f for f in self.pool if f.status == "culture"] + [f for f in reversed(self.pool) if f.status == "trial"]
        return [f.text for f in ranked[:k]]

    def state(self) -> dict:
        return {"pool": [asdict(f) for f in self.pool]}

    def load_state(self, data: dict) -> None:
        self.pool = [Finding(**f) for f in data["pool"]]
