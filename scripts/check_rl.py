"""Synthetic check of the step policy on a simulated lineage tree with a known best operator-origin combination.

Children improve their path best with a probability that depends on the operator and on starting from scratch,
with an interaction (a full rewrite pays only from scratch). The logged policy is a drifting mixture. The check
reports how the fitted policy moves each decision, its KL to the logging policy, and that nothing moves when the
outcome ignores the decisions.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sciel.causal.ledger import Ledger
from sciel.rl import StepPolicy

STEPS = 1500


def rate(operator: str, origin: str, effect: bool) -> float:
    if not effect:
        return 0.08
    return {("diff", "parent"): 0.12, ("full_rewrite", "parent"): 0.02,
            ("diff", "fresh"): 0.04, ("full_rewrite", "fresh"): 0.20}[(operator, origin)]


def simulate(effect: bool, seed: int):
    rng = random.Random(seed)
    ledger = Ledger()
    programs = {"p0": SimpleNamespace(score=0.5, gen=0)}
    for t in range(STEPS):
        tilt = 0.3 + 0.4 * ((t // 300) % 2)
        operators = {"diff": tilt, "full_rewrite": 1 - tilt}
        operators = {a: 0.9 * p + 0.05 for a, p in operators.items()}
        operator = rng.choices(list(operators), weights=list(operators.values()))[0]
        origins = {"parent": 0.8, "fresh": 0.2}
        origin = rng.choices(list(origins), weights=list(origins.values()))[0]
        parent = rng.choice(list(programs)[-50:])
        child = f"p{len(programs)}"
        improved = rng.random() < rate(operator, origin, effect)
        start = 0.5 if origin == "fresh" else programs[parent].score
        score = start * (1.1 if improved else 1.0)
        programs[child] = SimpleNamespace(score=score, gen=programs[parent].gen + 1)
        records = [{"decision": "operator", "distribution": operators, "choice": operator,
                    "propensity": operators[operator]},
                   {"decision": "origin", "distribution": origins, "choice": origin, "propensity": origins[origin]}]
        gain = ledger.path_gain(None if origin == "fresh" else parent, child, score, 0.5)
        ledger.add(iteration=t, island=t % 4, decision_context={"phase": 0, "progress": t / STEPS, "parent_id": parent,
                   "iteration_no": t, "decisions": records}, child=child, parent=parent, event=improved, gain=gain)
    best = max(p.score for p in programs.values())
    db = SimpleNamespace(_all=programs, pop_best=best)
    return SimpleNamespace(ledger=ledger, db=db, swm=SimpleNamespace(ready=False), iteration_no=STEPS,
                           strategies=SimpleNamespace(pool=[]), findings=SimpleNamespace(pool=[]))


def probe(policy, engine) -> dict:
    """The operator choice, then the origin choice after each operator: a full rewrite pays only from scratch."""
    context, origins = "phase=0|third=1", {"parent": 0.8, "fresh": 0.2}
    out = {"operator": policy.steer(engine, "operator", "p10", context, {"diff": 0.5, "full_rewrite": 0.5}, [])}
    for operator in ("diff", "full_rewrite"):
        earlier = [{"decision": "operator", "choice": operator}]
        out[f"origin after {operator}"] = policy.steer(engine, "origin", "p10", context, origins, earlier)
    return out


def main():
    out = {}
    for effect in (True, False):
        engine = simulate(effect, 0)
        policy = StepPolicy()
        report = policy.fit(engine, 0)
        out["effect" if effect else "null"] = {
            "moved": report["moved"],
            "decisions": {d: {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
                          for d, r in report.get("decisions", {}).items()},
            "steer": probe(policy, engine)}
    print(json.dumps(out, indent=2, default=lambda v: round(float(v), 4)))


if __name__ == "__main__":
    main()
