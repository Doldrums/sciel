"""Journaled decisions joined to lineage outcomes: an island event by the child or its descendants within a horizon."""

from __future__ import annotations

from dataclasses import asdict, dataclass

HORIZON = 20
MAX_ROWS = 20000


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


def context_of(decision_context: dict) -> str:
    phase = decision_context.get("phase", 0)
    progress = decision_context.get("progress") or 0.0
    return f"phase={phase}|third={min(int(progress * 3), 2)}"


class Ledger:
    def __init__(self, horizon: int = HORIZON):
        self.horizon = horizon
        self.rows: list[Row] = []
        self.parent: dict[str, str] = {}

    def add(self, *, iteration: int, island: int, decision_context: dict, child: str | None,
            parent: str | None, event: bool) -> None:
        context = context_of(decision_context)
        for record in decision_context.get("decisions", []):
            self.rows.append(Row(decision=record["decision"], iteration=iteration, island=island,
                                 context=context, distribution=dict(record["distribution"]),
                                 choice=str(record["choice"]), propensity=float(record["propensity"]),
                                 child=child, outcome=None if child is not None else 0.0))
        if child is not None and parent is not None:
            self.parent[child] = parent
        if event and child is not None:
            self._credit(child, iteration)
        self._expire(iteration)
        del self.rows[:-MAX_ROWS]

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

    def labeled(self, decision: str) -> list[Row]:
        return [r for r in self.rows if r.decision == decision and r.outcome is not None]

    def state(self) -> dict:
        return {"horizon": self.horizon, "rows": [asdict(r) for r in self.rows],
                "parent": dict(self.parent)}

    def load_state(self, data: dict) -> None:
        self.horizon = int(data["horizon"])
        self.rows = [Row(**r) for r in data["rows"]]
        self.parent = dict(data["parent"])
