"""Second order: the procedures that turn evidence into decisions are versioned bundles, varied and judged like any arm."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

PROPOSE_INTERVAL = 50
MAX_TRIALS = 2
NONE_SHARE = 0.5
STEP = 0.4
BOUNDS = {"shift": (0.5, 16.0), "mix": (0.05, 0.9), "base_share": (0.1, 0.9)}


@dataclass
class Procedure:
    id: str
    shift: float = 4.0
    mix: float = 0.3
    base_share: float = 0.5
    parent: str | None = None
    born: int = 0
    status: str = "trial"

    def params(self) -> dict:
        return {"shift": self.shift, "mix": self.mix, "base_share": self.base_share}


class Procedures:
    def __init__(self):
        self.pool: list[Procedure] = [Procedure(id="p0", status="culture")]
        self.last_proposal = 0

    def live(self) -> list[Procedure]:
        return [p for p in self.pool if p.status != "retired"]

    def get(self, procedure_id: str) -> Procedure:
        return next(p for p in self.pool if p.id == procedure_id)

    def distribution(self) -> dict:
        trials = [p for p in self.live() if p.status == "trial"]
        incumbents = [p for p in self.live() if p.status == "culture"]
        if not trials:
            return {p.id: 1 / len(incumbents) for p in incumbents}
        return ({p.id: NONE_SHARE / len(incumbents) for p in incumbents}
                | {p.id: (1 - NONE_SHARE) / len(trials) for p in trials})

    def propose(self, iteration: int, rng) -> Procedure | None:
        if (iteration - self.last_proposal < PROPOSE_INTERVAL
                or sum(p.status == "trial" for p in self.pool) >= MAX_TRIALS):
            return None
        self.last_proposal = iteration
        incumbents = [p for p in self.live() if p.status == "culture"]
        parent = incumbents[rng.randrange(len(incumbents))]
        values = {}
        for name, (low, high) in BOUNDS.items():
            moved = getattr(parent, name) * math.exp(rng.gauss(0.0, STEP))
            values[name] = round(min(max(moved, low), high), 3)
        child = Procedure(id=f"p{len(self.pool)}", parent=parent.id, born=iteration, **values)
        self.pool.append(child)
        return child

    def judge(self, theory) -> list[Procedure]:
        changed = []
        for claim in theory.claims:
            procedure = next((p for p in self.pool if p.id == claim.action), None)
            if claim.decision != "procedure" or procedure is None or not claim.decided:
                continue
            status = "culture" if claim.effect > 0 else "retired"
            last_incumbent = sum(p.status == "culture" for p in self.pool) == 1 and procedure.status == "culture"
            if procedure.status != status and not (status == "retired" and last_incumbent):
                procedure.status = status
                changed.append(procedure)
        return changed

    def state(self) -> dict:
        return {"pool": [asdict(p) for p in self.pool], "last_proposal": self.last_proposal}

    def load_state(self, data: dict) -> None:
        self.pool = [Procedure(**p) for p in data["pool"]]
        self.last_proposal = int(data["last_proposal"])
