"""Assays of the generator's assumptions, one at a time: a knockout to measure reliance, then a registered probe for
an assumption the programs rely on. In a parallel wave an assay is its own job, so its rewrites and measurements
never hold an ordinary generation step; a serial run carries it on a step. Assays spend at most SHARE of the budget,
and an assay whose model call fails is retried once before the assumption counts as unmeasurable.
"""

from __future__ import annotations

from . import knockout, probe
from .llm.errors import LLMError

SHARE = 0.1
EVALS = 4
ATTEMPTS = 2


def plan(engine, iteration_no: int, room: int | None) -> dict | None:
    """The next assay, if none is pending, the budget share allows it and `room` evaluations are left for it."""
    if engine.assumptions.pending or (room is not None and room < EVALS):
        return None
    total = engine.budget if engine.budget is not None else engine.B_evals
    if engine.assumptions.spent + EVALS > SHARE * total:
        return None
    spec = knockout.plan(engine, iteration_no)
    kind = "knockout"
    if spec is None:
        spec, kind = probe.plan(engine), "probe"
    if spec is None:
        return None
    engine.assumptions.pending = spec["assumption"]
    return {"kind": kind, "spec": spec}


def execute(engine, job: dict, llm, model: str) -> dict:
    run = knockout.execute if job["kind"] == "knockout" else probe.execute
    out = {"cost": 0.0, "measured": 0}
    try:
        run(engine, job["spec"], llm, model, out)
    except LLMError as error:
        out["error"] = type(error).__name__
    return out


def commit(engine, job: dict, measured: dict) -> None:
    spec = job["spec"]
    engine.B_evals += measured.get("measured", 0)
    engine.assumptions.spent += measured.get("measured", 0)
    engine.assumptions.pending = None
    assumption = engine.assumptions.get(spec["assumption"])
    if measured.get("error"):
        assumption.attempts += 1
        if assumption.attempts >= ATTEMPTS:
            assumption.status = "unmeasurable"
        engine.log.decision(policy=job["kind"], action="failed", iteration_no=engine.iteration_no,
                            assumption=assumption.id, attempts=assumption.attempts, error=measured["error"],
                            evaluations=measured.get("measured", 0))
        return
    if job["kind"] == "knockout":
        scores = measured.get("scores") or {"parent": None, "target": None, "nulls": []}
        if scores["parent"] is not None:
            engine.assumptions.baselines.setdefault(spec["program"], [spec["seed"], scores["parent"]])
        assumption.reliance = knockout.reliance(scores)
        if assumption.reliance is None:
            assumption.status = "unmeasurable"
        engine.log.decision(policy="knockout", action="measured", iteration_no=engine.iteration_no,
                            assumption=spec["assumption"], program=spec["program"], targets=spec["targets"],
                            nulls=spec["nulls"], scores=scores, reliance=assumption.reliance,
                            evaluations=measured.get("measured", 0))
        return
    scores = {"parent": measured.get("parent", [None]), "variant": measured.get("variant", [None])}
    assumption.status = probe.verdict(scores)
    assumption.probe = {**spec["prediction"], "seeds": spec["seeds"], "program": spec["program"]}
    assumption.evidence.append(f"parent {scores['parent']} vs variant {scores['variant']}")
    engine.log.decision(policy="probe", action=assumption.status, iteration_no=engine.iteration_no,
                        assumption=assumption.id, text=assumption.text, scores=scores, program=spec["program"],
                        evaluations=measured.get("measured", 0))
    if assumption.status == "refuted":
        for programme in engine.incubator.open(engine, probe.brief(assumption)):
            engine.log.decision(policy="programme", action="open", iteration_no=engine.iteration_no,
                                programme=programme.id, kind=programme.kind, island=programme.island,
                                strategy=programme.strategy)
