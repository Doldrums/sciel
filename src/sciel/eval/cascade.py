"""Cascade driver: smoke -> full -> confirm. Lives in the engine; task ships stages."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .facade import EvalResult
from .sandbox import run_stage


@dataclass
class CascadeOutcome:
    result: EvalResult
    raw_first_score: float
    reached: str
    confirmed: bool = False


def cascade_evaluate(task, program_path: str, record_candidate: bool = False,
                     python: str | None = None) -> CascadeOutcome:
    smoke = run_stage(task.evaluator_path, "smoke", program_path,
                      timeout=task.smoke_timeout, cpu_seconds=task.cpu_seconds,
                      mem_bytes=task.mem_bytes, python=python)
    if not smoke.ok:
        return CascadeOutcome(result=smoke, raw_first_score=smoke.score,
                              reached="smoke")

    full = run_stage(task.evaluator_path, "full", program_path,
                     timeout=task.full_timeout, cpu_seconds=task.cpu_seconds,
                     mem_bytes=task.mem_bytes, python=python)
    raw_first = full.score
    if not full.ok:
        return CascadeOutcome(result=full, raw_first_score=raw_first,
                              reached="full")

    if record_candidate and task.stochastic:
        conf = run_stage(task.evaluator_path, "confirm", program_path,
                         timeout=task.confirm_timeout, n=task.confirm_n,
                         cpu_seconds=task.cpu_seconds, mem_bytes=task.mem_bytes,
                         python=python)
        trials = conf.cases.get("n_trials", 0.0)
        enough_samples = (trials >= max(2, task.confirm_n) and float(trials).is_integer())
        if conf.ok and conf.confirmed_sampled and enough_samples:
            return CascadeOutcome(result=conf, raw_first_score=raw_first,
                                  reached="confirm", confirmed=True)
        if conf.ok:
            conf = replace(conf, ok=False, score=0.0,
                           artifacts={**conf.artifacts,
                                      "error_type": "InsufficientConfirmation",
                                      "error": "confirm did not provide all requested repeated samples"})
        return CascadeOutcome(result=conf, raw_first_score=raw_first,
                              reached="confirm")

    return CascadeOutcome(result=full, raw_first_score=raw_first, reached="full")
