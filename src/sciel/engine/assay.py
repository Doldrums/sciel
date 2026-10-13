"""Assays of the generator's assumptions, one at a time: a knockout to measure reliance, then a registered probe for
an assumption the programs rely on. In a parallel wave an assay is its own job, so its rewrites and measurements
never hold an ordinary generation step; a serial run carries it on a step.
"""

from __future__ import annotations

from ..campaign.budget import CampaignStopped
from . import knockout, probe


def plan(engine, iteration_no: int) -> dict | None:
    if engine.assumptions.pending:
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
    try:
        return run(engine, job["spec"], llm, model)
    except CampaignStopped:
        raise
    except Exception as error:
        return {"error": type(error).__name__, "cost": 0.0, "measured": 0}


def commit(engine, job: dict, measured: dict) -> None:
    spec = job["spec"]
    engine.B_evals += measured.get("measured", 0)
    engine.assumptions.pending = None
    assumption = engine.assumptions.get(spec["assumption"])
    if job["kind"] == "knockout":
        scores = measured.get("scores") or {"parent": None, "target": None, "nulls": []}
        assumption.reliance = knockout.reliance(scores)
        if assumption.reliance is None:
            assumption.status = "unmeasurable"
        engine.log.decision(policy="knockout", action="measured", iteration_no=engine.iteration_no,
                            assumption=spec["assumption"], program=spec["program"], targets=spec["targets"],
                            nulls=spec["nulls"], scores=scores, reliance=assumption.reliance,
                            evaluations=measured.get("measured", 0), error=measured.get("error"))
        return
    scores = {"parent": measured.get("parent", [None]), "variant": measured.get("variant", [None])}
    assumption.status = probe.verdict(scores)
    assumption.probe = {**spec["prediction"], "seeds": spec["seeds"], "program": spec["program"]}
    assumption.evidence.append(f"parent {scores['parent']} vs variant {scores['variant']}")
    engine.log.decision(policy="probe", action=assumption.status, iteration_no=engine.iteration_no,
                        assumption=assumption.id, text=assumption.text, scores=scores, program=spec["program"],
                        evaluations=measured.get("measured", 0), error=measured.get("error"))
    if assumption.status == "refuted":
        for programme in engine.incubator.open(engine, probe.brief(assumption)):
            engine.log.decision(policy="programme", action="open", iteration_no=engine.iteration_no,
                                programme=programme.id, kind=programme.kind, island=programme.island,
                                strategy=programme.strategy)
