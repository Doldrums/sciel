"""Assumption knockouts: the code units that encode an assumption are rewritten to a neutral variant, and the drop
in score is compared with knockouts of as many random other units, all measured with one seed outside the
population.
"""

from __future__ import annotations

import ast
import random

from ..evidence.change_audit import unit_catalog
from .prompt.parse import extract_code

NULLS = 2
PROMPT = ("Rewrite ONLY these code units of the program below: {units}. {reason} Replace their logic with the "
          "simplest neutral behaviour that keeps the same names and signatures and still returns valid values "
          "(constants, identity, uniform weights, unchanged inputs). Leave every other line exactly as it is and "
          "return the whole program in one python block.\n\n```python\n{code}\n```")


def plan(engine, iteration_no: int) -> dict | None:
    """The heaviest untested assumption without a measured reliance, on its best program still in the population."""
    for assumption in engine.assumptions.open():
        if assumption.reliance is not None:
            continue
        programs = [engine.db._all[pid] for pid in assumption.units if pid in engine.db._all]
        if not programs:
            continue
        program = max(programs, key=lambda p: p.score)
        targets = assumption.units[program.id]
        others = [u["id"] for u in unit_catalog(program.code)[1:] if u["id"] not in targets]
        if len(others) < len(targets):
            continue
        seed = engine.rng.derive_seed("sibling") % 2 ** 31
        rng = random.Random(seed)
        return {"assumption": assumption.id, "text": assumption.text, "program": program.id, "code": program.code,
                "targets": targets, "nulls": [rng.sample(others, len(targets)) for _ in range(NULLS)], "seed": seed,
                "iteration": iteration_no}
    return None


def _neutral(llm, model: str, code: str, units: list, reason: str):
    """A neutral rewrite of the units; an unparseable reply or one that leaves the program unchanged is no knockout."""
    from .llm.client import accounted_cost
    reply = llm.generate(PROMPT.format(units=", ".join(units), reason=reason, code=code), model)
    variant = extract_code(reply.text)
    try:
        same = ast.dump(ast.parse(variant)) == ast.dump(ast.parse(code))
    except SyntaxError:
        variant, same = None, False
    return (None if same else variant), accounted_cost(reply)


def execute(engine, spec: dict, llm, model: str) -> dict:
    """Parent, knockout and null knockouts, each measured once with the same seed."""
    cost, scores, measured = 0.0, {}, 1
    scores["parent"] = engine._measure(spec["code"], purpose="knockout", seed=spec["seed"])
    variant, spent = _neutral(llm, model, spec["code"], spec["targets"],
                              f"They encode the assumption: {spec['text']}.")
    cost += spent
    scores["target"] = engine._measure(variant, purpose="knockout", seed=spec["seed"]) if variant else None
    measured += variant is not None
    scores["nulls"] = []
    for units in spec["nulls"]:
        variant, spent = _neutral(llm, model, spec["code"], units, "")
        cost += spent
        scores["nulls"].append(engine._measure(variant, purpose="knockout", seed=spec["seed"]) if variant else None)
        measured += variant is not None
    return {"scores": scores, "cost": cost, "measured": measured}


def reliance(scores: dict) -> float | None:
    """Drop from knocking the assumption out, beyond the mean drop from random knockouts of the same size. A knockout
    that fails, or collapses a nonzero score to exactly zero, killed the program trivially and measures nothing."""
    parent, target = scores["parent"], scores["target"]
    if parent is None:
        return None
    alive = lambda s: s is not None and not (s == 0 and parent != 0)
    nulls = [s for s in scores["nulls"] if alive(s)]
    if not alive(target) or not nulls:
        return None
    scale = max(abs(parent), 1e-9)
    return ((parent - target) - sum(parent - s for s in nulls) / len(nulls)) / scale
