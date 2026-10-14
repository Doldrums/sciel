"""Research programmes: a theorist strategy grows on its own protected island next to a control programme of the
same age, as clones mature in a germinal centre before they compete outside.

Both programmes start from scratch (no champion code, no inspirations) and then refine only their own lineage. They
take no part in successive-halving cuts or migration. After K steps the strategy programme is compared with its
control (W = 1 if its best is higher); a winner continues to K2 steps and graduates when its best ranks in the upper
half of the scheduled islands. W feeds an e-process about the theorist as a source of approaches.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np

from ..causal.sequential import mixture
from ..components.selection.novelty_chain import Selected
from .types import Snapshot

K = 10
K2 = 30
SHARE = 0.1
QUEUE = 2
ALPHA = 0.05
STRATEGY_BRIEF = ("\nRESEARCH PROGRAMME: design a new program for this task built on this approach, from scratch; "
                  "do not reuse the current designs:\n{text}\n")
CONTROL_BRIEF = ("\nRESEARCH PROGRAMME: design a new program for this task from an approach family that none of the "
                 "current programs use, from scratch; do not reuse the current designs.\n")
REFINE_BRIEF = ("\nRESEARCH PROGRAMME: keep this program's approach and improve it within that approach; do not "
                "switch to a different design.\n")


@dataclass
class Programme:
    id: int
    pair: int
    kind: str
    strategy: str
    brief: str
    island: int
    anchor: str
    steps: int = 0
    stage: int = 1
    best: float | None = None
    status: str = "open"
    opened: int = 0


class Incubator:
    def __init__(self):
        self.programmes: list[Programme] = []
        self.outcomes: list[int] = []
        self.prior: list[int] = []
        self.next_pair = 0
        self.queue: list[dict] = []

    def free(self) -> bool:
        return not any(p.status == "open" for p in self.programmes)

    def holds(self, island: int) -> bool:
        return any(p.island == island and p.status != "promoted" for p in self.programmes)

    def closed(self) -> list[int]:
        return [p.island for p in self.programmes if p.status == "closed"]

    def owner(self, island: int) -> Programme | None:
        return next((p for p in self.programmes if p.island == island and p.status == "open"), None)

    def open(self, engine, strategy) -> list[Programme]:
        """Open a strategy programme and its control on two new islands; while a pair runs, the strategy waits among
        the QUEUE newest."""
        remaining = (engine.budget - engine.B_evals) if engine.budget is not None else 10 ** 6
        if remaining < 2 * K:
            return []
        if not self.free():
            if all(q["id"] != strategy.id for q in self.queue):
                self.queue = (self.queue + [{"id": strategy.id, "text": strategy.text}])[-QUEUE:]
            return []
        pair = self.next_pair
        self.next_pair += 1
        opened = []
        for kind, brief in (("strategy", STRATEGY_BRIEF.format(text=strategy.text)), ("control", CONTROL_BRIEF)):
            island = engine._setup_island(register=False)
            anchor, _, _ = engine._copy_baseline(island)
            programme = Programme(id=len(self.programmes), pair=pair, kind=kind, strategy=strategy.id, brief=brief,
                                  island=island, anchor=anchor.id, opened=engine.iteration_no)
            self.programmes.append(programme)
            opened.append(programme)
        return opened

    def open_next(self, engine) -> list[Programme]:
        if not self.free() or not self.queue:
            return []
        from types import SimpleNamespace
        return self.open(engine, SimpleNamespace(**self.queue.pop()))

    def next_island(self, engine, busy=()) -> int | None:
        """The open programme with the fewest steps, while open programmes stay within their share of the steps
        taken since they opened."""
        open_ = [p for p in self.programmes if p.status == "open"]
        if not open_:
            return None
        since = min(p.opened for p in open_)
        if sum(p.steps for p in open_) > SHARE * (engine.iteration_no - since) + 1:
            return None
        open_ = [p for p in open_ if p.island not in busy and p.steps < (K if p.stage == 1 else K2)]
        return min(open_, key=lambda p: (p.steps, p.id)).island if open_ else None

    def _lineage(self, engine, programme: Programme) -> list:
        return [p for p in engine.db.islands[programme.island].programs.values() if p.id != programme.anchor]

    def prepare(self, engine, island: int, iteration_no: int) -> dict:
        """A programme step: from scratch at its first step, then a refinement of the best of its own lineage."""
        programme = self.owner(island)
        lineage = self._lineage(engine, programme)
        anchor = engine.db.get(programme.anchor)
        if lineage:
            parent = max(lineage, key=lambda p: p.score)
            op, directive, code, score = "diff", REFINE_BRIEF + programme.brief, parent.code, parent.score
        else:
            parent, op, directive, code, score = anchor, "full_rewrite", programme.brief, anchor.code, anchor.score
        model = engine.llm.choose_model(engine.rng.stream("prompt"))
        decision = dict(requested=op, effective=op, selection="programme", fallback_reason=None, on_policy=False)
        snap = Snapshot(island=island, op=op, model=model, iteration_no=iteration_no, parent_id=parent.id,
                        parent_code=code, parent_score=score, directive=directive,
                        code_hashes=engine.db.code_hashes(island), escalation_seed=engine.rng.derive_seed("escalation"),
                        task_prompt=engine.task_prompt, bet_enabled=True,
                        required_entrypoints=engine.required_entrypoints, operator_decision=decision)
        context = {"phase": engine.schedule.phase_code(), "progress": engine._progress(), "parent_id": parent.id,
                   "iteration_no": iteration_no, "pop_best": engine.db.pop_best, "baseline": score, "decisions": [],
                   "programme": {"id": programme.id, "step": programme.steps}}
        return {"op": op, "procedure": engine._procedure.id, "snap": snap,
                "selected": Selected(parent.id, [], 1.0, {"selection": "programme"}),
                "op_prop": None, "op_stat": {"programme": programme.id}, "model": model, "model_prop": 1.0,
                "injected_ids": [], "mode": "base", "mode_prop": 1.0, "eff_mode": "base",
                "phase": engine.schedule.phase_code(), "decision_context": context, "mode_on_policy": False,
                "preset": None, "preset_prop": None, "theorist": None, "assumptions": None,
                "assumption_programs": [], "assay": None, "block": None}

    def commit(self, engine, plan: dict) -> list[dict]:
        """Count a committed programme step; settle the pair at its stage boundaries."""
        tag = plan["decision_context"].get("programme")
        programme = self.programmes[tag["id"]] if tag else None
        if programme is None or programme.status != "open":
            return []
        programme.steps += 1
        lineage = self._lineage(engine, programme)
        programme.best = max((p.score for p in lineage), default=None)
        pair = [p for p in self.programmes if p.pair == programme.pair]
        strategy = next(p for p in pair if p.kind == "strategy")
        control = next(p for p in pair if p.kind == "control")
        events = []
        if strategy.stage == 1 and strategy.steps >= K and control.steps >= K and control.status == "open":
            s, c = strategy.best, control.best
            s_value = -np.inf if s is None else s
            c_value = -np.inf if c is None else c
            won = s_value > c_value or (s_value == c_value and random.Random(programme.pair).random() < 0.5)
            self.outcomes.append(int(won))
            control.status = "closed"
            if won:
                strategy.stage = 2
            else:
                strategy.status = "closed"
            events.append({"pair": programme.pair, "strategy": strategy.strategy, "won": won,
                           "strategy_best": s, "control_best": c, **self.evidence()})
        if strategy.stage == 2 and strategy.status == "open" and strategy.steps >= K2:
            scheduled = sorted((engine.db.island_best(i) for i in engine.schedule.survivors), reverse=True)
            bar = scheduled[(len(scheduled) - 1) // 2] if scheduled else -np.inf
            graduated = strategy.best is not None and strategy.best >= bar
            strategy.status = "promoted" if graduated else "closed"
            if graduated:
                engine.schedule.register_island(strategy.island)
                engine.schedule.dispatched[strategy.island] = min(engine.schedule.dispatched.values(), default=0)
            events.append({"pair": programme.pair, "strategy": strategy.strategy, "graduated": graduated,
                           "strategy_best": strategy.best, "bar": bar})
        return events

    def evidence(self) -> dict:
        """Mixture-of-bets e-process on 2W-1: is a theorist programme more likely than its control to end higher?"""
        outcomes = self.prior + self.outcomes
        if not outcomes:
            return {"pairs": 0, "e": 1.0}
        s = mixture(2.0 * np.array(outcomes, dtype=float) - 1.0, 1.0, ALPHA)
        return {"pairs": s["n"], "e": s["e"], "win_rate": float(np.mean(outcomes)), "inherited": len(self.prior)}

    def state(self) -> dict:
        return {"programmes": [asdict(p) for p in self.programmes], "outcomes": self.outcomes,
                "next_pair": self.next_pair, "queue": self.queue}

    def load_state(self, data: dict) -> None:
        self.programmes = [Programme(**p) for p in data["programmes"]]
        self.outcomes = [int(o) for o in data["outcomes"]]
        self.next_pair = int(data["next_pair"])
        self.queue = [dict(q) for q in data["queue"]]
