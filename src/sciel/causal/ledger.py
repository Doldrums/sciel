"""Journaled decisions joined to lineage outcomes: an island event by the child or its descendants within a horizon."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

HORIZON = 20


@dataclass
class Row:
    decision: str
    iteration: int
    island: int
    context: str
    distribution: dict
    choice: str
    propensity: float
    child: str | None
    outcome: float | None = None
    state: str | None = None
    reward: float = 0.0
    scope: str = "lineage"
    population: list = field(default_factory=list)
    gain: float = 0.0


def context_of(decision_context: dict) -> str:
    phase = decision_context.get("phase", 0)
    progress = decision_context.get("progress") or 0.0
    return f"phase={phase}|third={min(int(progress * 3), 2)}"


class Ledger:
    def __init__(self, horizon: int = HORIZON):
        self.horizon = horizon
        self.rows: list[Row] = []
        self.parent: dict[str, str] = {}
        self.now = 0
        self.path_best: dict[str, float] = {}

    def path_gain(self, parent: str | None, child: str | None, score: float | None, origin: float) -> float:
        """Rise of the best score on the root-to-child path, on a log scale above the starting program's score;
        it telescopes along every lineage, so a copy adds nothing and siblings never take each other's credit."""
        before = self.path_best.get(parent, origin) if parent is not None else origin
        if child is None or score is None:
            return 0.0
        after = max(before, score)
        self.path_best[child] = after
        scale = max(abs(origin), 1e-9)
        potential = lambda b: math.log1p(max(b - origin, 0.0) / scale)
        return potential(after) - potential(before)

    def add(self, *, iteration: int, island: int, decision_context: dict, child: str | None,
            parent: str | None, event: bool, gain: float = 0.0) -> None:
        self.now = max(self.now, iteration)
        context = context_of(decision_context)
        decided = int(decision_context.get("iteration_no", iteration))
        for record in decision_context.get("decisions", []):
            self.rows.append(Row(decision=record["decision"], iteration=decided, island=island,
                                 context=context, distribution=dict(record["distribution"]),
                                 choice=str(record["choice"]), propensity=float(record["propensity"]),
                                 child=child, outcome=None if child is not None else 0.0,
                                 state=decision_context.get("parent_id"), reward=float(event),
                                 population=list(decision_context.get("population", [])), gain=gain))
        if child is not None and parent is not None:
            self.parent[child] = parent
        if event and child is not None:
            self._credit(child, iteration)
        if event:
            for row in self.rows:
                if (row.scope == "island" and row.outcome is None and row.island == island
                        and iteration - row.iteration <= self.horizon):
                    row.outcome = 1.0
        self._expire(iteration)

    def add_island(self, *, iteration: int, island: int, context: str, record: dict) -> None:
        """A decision whose outcome is any island event within the horizon, not a lineage event."""
        self.now = max(self.now, iteration)
        self.rows.append(Row(decision=record["decision"], iteration=iteration, island=island, context=context,
                             distribution=dict(record["distribution"]), choice=str(record["choice"]),
                             propensity=float(record["propensity"]), child=None, scope="island"))

    def _lineage(self, program: str) -> set:
        seen, node = set(), program
        while node is not None and node not in seen and len(seen) <= self.horizon:
            seen.add(node)
            node = self.parent.get(node)
        return seen

    def _credit(self, child: str, iteration: int) -> None:
        ancestors = self._lineage(child)
        for row in self.rows:
            if row.outcome is None and row.child in ancestors and iteration - row.iteration <= self.horizon:
                row.outcome = 1.0

    def _expire(self, iteration: int) -> None:
        for row in self.rows:
            if row.outcome is None and iteration - row.iteration > self.horizon:
                row.outcome = 0.0

    def settled(self, row: Row) -> bool:
        """A row enters inference only once its outcome is fixed for good: a lineage row's own event at once, an island
        row at the end of its horizon, whether or not an event came earlier."""
        return row.scope == "lineage" or (row.outcome is not None and self.now - row.iteration > self.horizon)

    def labeled(self, decision: str) -> list[Row]:
        return [r for r in self.rows if r.decision == decision and r.outcome is not None]

    def state(self) -> dict:
        return {"horizon": self.horizon, "rows": [asdict(r) for r in self.rows],
                "parent": dict(self.parent), "now": self.now, "path_best": dict(self.path_best)}

    def load_state(self, data: dict) -> None:
        self.horizon = int(data["horizon"])
        self.rows = [Row(**r) for r in data["rows"]]
        self.parent = dict(data["parent"])
        self.now = int(data["now"])
        self.path_best = dict(data["path_best"])
