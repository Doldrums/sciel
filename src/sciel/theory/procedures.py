"""Second order: the procedures that turn evidence into decisions are versioned bundles, varied and judged like any arm."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from . import schools

PROPOSE_INTERVAL = 50
MAX_TRIALS = 2
NONE_SHARE = 0.5
STEP = 0.4
BOUNDS = {"shift": (0.5, 16.0), "mix": (0.05, 0.9), "base_share": (0.1, 0.9), "theorist_interval": (10.0, 100.0),
          "uncertainty_scale": (0.5, 2.0), "voi_scale": (0.25, 4.0)}


@dataclass
class Procedure:
    id: str
    shift: float = 4.0
    mix: float = 0.3
    base_share: float = 0.5
    theorist_interval: float = 25.0
    uncertainty_scale: float = 1.0
    voi_scale: float = 1.0
    parent: str | None = None
    born: int = 0
    status: str = "trial"
    local: dict = field(default_factory=dict)

    def params(self) -> dict:
        return {name: getattr(self, name) for name in BOUNDS}


class Procedures:
    def __init__(self):
        self.pool: list[Procedure] = [Procedure(id="p0", status="culture")]
        self.last_proposal = 0

    def live(self) -> list[Procedure]:
        return [p for p in self.pool if p.status != "retired"]

    def get(self, procedure_id: str) -> Procedure:
        return next(p for p in self.pool if p.id == procedure_id)

    def distribution(self, island: int) -> dict:
        trials = schools.trials(self.pool, island)
        incumbents = [p for p in schools.available(self.pool, island)
                      if p.status == "culture" or p.local.get(str(island)) == "culture"] or [
                      p for p in self.pool if p.status == "culture"][:1]
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
        for procedure in self.pool:
            status = schools.verdict(theory, "procedure", procedure.id)
            if status is None:
                continue
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
