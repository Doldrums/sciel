"""Assumptions about the task that the generator's best programs rest on, made explicit so they can be tested.

Each assumption is extracted from the code and the forecast mechanisms of the top programs, carries a counterfactual
("if false, the program should change X") and the code units that encode it. Reliance comes from knocking those
units out; truth comes from a registered probe whose predicted outcomes differ under the assumption and its negation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from ..evidence.change_audit import unit_catalog
from .findings import _similar

INTERVAL = 50
TOP = 3
MAX_OPEN = 8
EXCERPT = 2500


@dataclass
class Assumption:
    id: str
    text: str
    counterfactual: str
    units: dict
    programs: list
    born: int
    weight: float = 0.0
    status: str = "untested"
    reliance: float | None = None
    probe: dict | None = None
    receipt_id: str | None = None
    evidence: list = field(default_factory=list)


def _excerpt(code: str) -> str:
    return code if len(code) <= EXCERPT else code[:EXCERPT] + "\n# ..."


def _array(text: str) -> list:
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


class Assumptions:
    def __init__(self):
        self.items: list[Assumption] = []
        self.last_request = -INTERVAL
        self.pending: str | None = None

    def due(self, iteration: int) -> bool:
        return iteration - self.last_request >= INTERVAL and sum(a.status == "untested" for a in self.items) < MAX_OPEN

    def request(self, iteration: int, *, task: str, programs: list) -> str:
        self.last_request = iteration
        shown = "\n\n".join(
            f"Program {p.id} (score {p.score:.4f}); its author's stated mechanism: {p.mechanism or 'none'}\n"
            f"Code units: {', '.join(u['id'] for u in unit_catalog(p.code)[1:]) or 'none'}\n"
            f"```python\n{_excerpt(p.code)}\n```" for p in programs)
        return ("You audit an automated program search.\n\n"
                f"Task:\n{task}\n\nBest programs:\n{shown}\n\n"
                "List the premises about the TASK or its DATA that these programs rely on, that the task description "
                "above does NOT state, and that could be false (for example what a column means, which inputs carry "
                "signal, which structure the optimum has, which cases dominate the score). Do not restate anything the "
                "task description says, and ignore implementation choices. Reply with a JSON array, at most 4 items, "
                "each "
                '{"assumption": "<one testable sentence>", "counterfactual": "<if false, the program should change '
                'what>", "units": {"<program id>": ["<code unit that encodes it>"]}}.')

    def adopt(self, reply: str, programs: list, iteration: int) -> list[Assumption]:
        known = {p.id: p for p in programs}
        added = []
        for item in _array(reply):
            text = " ".join(str(item.get("assumption", "")).split())[:400]
            claimed = item.get("units") if isinstance(item.get("units"), dict) else {}
            units = {pid: [u for u in names if isinstance(u, str)
                           and u in {x["id"] for x in unit_catalog(known[pid].code)[1:]}]
                     for pid, names in claimed.items() if pid in known and isinstance(names, list)}
            units = {pid: names for pid, names in units.items() if names}
            if not text or not units:
                continue
            same = next((a for a in self.items if _similar(a.text, text)), None)
            if same is not None:
                same.units.update(units)
                same.programs = sorted(set(same.programs) | set(units))
                continue
            assumption = Assumption(id=f"a{len(self.items) + 1}", text=text,
                                    counterfactual=" ".join(str(item.get("counterfactual", "")).split())[:400],
                                    units=units, programs=sorted(units), born=iteration)
            self.items.append(assumption)
            added.append(assumption)
        return added

    def weigh(self, db) -> None:
        """Weight = scores of the programs that still rest on the assumption, relative to the population best."""
        best = db.pop_best if db.pop_best not in (None, float("-inf")) and db.pop_best != 0 else 1.0
        for a in self.items:
            a.weight = sum(db._all[pid].score / abs(best) for pid in a.programs if pid in db._all)

    def open(self) -> list[Assumption]:
        return sorted((a for a in self.items if a.status == "untested"), key=lambda a: -a.weight)

    def get(self, assumption_id: str) -> Assumption | None:
        return next((a for a in self.items if a.id == assumption_id), None)

    def cards(self) -> list[str]:
        return [f"- [{a.status}] {a.text}" + (f" (evidence: {a.evidence[-1]})" if a.evidence else "")
                for a in self.items if a.status in ("supported", "refuted")]

    def state(self) -> dict:
        return {"items": [asdict(a) for a in self.items], "last_request": self.last_request, "pending": self.pending}

    def load_state(self, data: dict) -> None:
        self.items = [Assumption(**a) for a in data["items"]]
        self.last_request = int(data["last_request"])
        self.pending = None
