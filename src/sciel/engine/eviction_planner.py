"""Who chooses the eviction victim is a randomized arm: the score rule or the world model's lowest lineage value."""

from __future__ import annotations

from ..causal import context_of


class EvictionPlanner:
    def __init__(self, engine):
        self.engine = engine

    def __call__(self, island: int, removable: list):
        engine = self.engine
        context = context_of({"phase": engine.schedule.phase_code(), "progress": engine._progress()})
        values = {p.id: engine.swm.value(p.id, context, "operator", engine.op_bandit.arms) for p in removable}
        if any(v is None for v in values.values()):
            return removable[0]
        distribution = engine.experiments.allocate("eviction_planner", context,
            engine.theory.tilt("eviction_planner", context, {"rule": 0.5, "swm": 0.5}))
        choice, record = engine.decisions.pick("eviction_planner", distribution)
        engine.ledger.add_island(iteration=engine.iteration_no, island=island, context=context, record=record)
        return min(removable, key=lambda p: values[p.id]) if choice == "swm" else removable[0]
