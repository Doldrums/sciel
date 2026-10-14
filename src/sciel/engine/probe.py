"""Registered probes: the program is rewritten as it should be if the assumption were false, and the outcomes are
predicted in advance to differ (worse if the assumption holds, better if it is false). Both versions are measured on
the same two seeds outside the population; only a consistent difference decides.
"""

from __future__ import annotations

import ast
from types import SimpleNamespace

from .prompt.parse import extract_code

RELIANCE = 0.05
SEEDS = 2
TOLERANCE = 1e-9
PROMPT = ("Rewrite the program below as it should be if this assumption about the task were FALSE: {text}. "
          "{counterfactual} Keep everything that does not depend on the assumption. Return the whole program in "
          "one python block.\n\n```python\n{code}\n```")


def plan(engine) -> dict | None:
    for assumption in engine.assumptions.open():
        if assumption.reliance is None:
            continue
        if assumption.reliance < RELIANCE:
            assumption.status = "unused"
            continue
        program = max((engine.db._all[pid] for pid in assumption.units if pid in engine.db._all),
                      key=lambda p: p.score, default=None)
        if program is None:
            continue
        seed = engine.rng.derive_seed("sibling") % 2 ** 31
        return {"assumption": assumption.id, "text": assumption.text, "counterfactual": assumption.counterfactual,
                "program": program.id, "code": program.code, "seeds": [seed + k for k in range(SEEDS)],
                "prediction": {"holds": "variant scores lower on every seed",
                               "false": "variant scores higher on every seed"}}
    return None


def execute(engine, spec: dict, llm, model: str, out: dict) -> None:
    """A rewrite that does not parse or leaves the program unchanged is no probe and is not measured."""
    from .llm.client import accounted_cost
    reply = llm.generate(PROMPT.format(text=spec["text"], counterfactual=spec["counterfactual"], code=spec["code"]),
                         model)
    out["cost"] += accounted_cost(reply)
    variant = extract_code(reply.text)
    try:
        if ast.dump(ast.parse(variant)) == ast.dump(ast.parse(spec["code"])):
            variant = None
    except SyntaxError:
        variant = None
    out["parent"], out["variant"] = [None] * len(spec["seeds"]), [None] * len(spec["seeds"])
    if variant is None:
        return
    for k, seed in enumerate(spec["seeds"]):
        out["parent"][k] = engine._measure(spec["code"], purpose="probe", seed=seed)
        out["variant"][k] = engine._measure(variant, purpose="probe", seed=seed)
        out["measured"] += 2


def verdict(scores: dict) -> str:
    pairs = list(zip(scores["parent"], scores["variant"]))
    if any(p is None or v is None for p, v in pairs):
        return "undecided"
    if all(v > p + TOLERANCE for p, v in pairs):
        return "refuted"
    if all(v < p - TOLERANCE for p, v in pairs):
        return "supported"
    return "undecided"


def brief(assumption) -> SimpleNamespace:
    """A refuted assumption becomes the brief of a research programme built on its negation."""
    return SimpleNamespace(id=assumption.id, text=(f"It is NOT true that {assumption.text} "
                                                   f"{assumption.counterfactual} Design around the corrected fact."))
