"""Cascade driver: smoke -> full -> confirm. Lives in the engine; task ships stages."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, replace

from .facade import EvalResult
from .sandbox import run_stage


@dataclass
class CascadeOutcome:
    result: EvalResult
    raw_first_score: float
    reached: str
    confirmed: bool = False
    # Host-only evidence travels through the explicit native WAL adapter. Keep
    # the established dataclass wire shape used by isolated method bundles.
    oracle_evidence: InitVar[list | None] = None

    def __post_init__(self, oracle_evidence):
        self.oracle_evidence = oracle_evidence


def cascade_evaluate(task, program_path: str, record_candidate: bool = False,
                     python: str | None = None, oracle_context=None) -> CascadeOutcome:
    evidence = [] if oracle_context is not None else None

    def run(stage):
        kwargs = dict(timeout=getattr(task, stage + '_timeout'), cpu_seconds=task.cpu_seconds,
                      mem_bytes=task.mem_bytes, python=python)
        if stage == 'confirm':
            kwargs['n'] = task.confirm_n
        if oracle_context is None:
            return run_stage(task.evaluator_path, stage, program_path, **kwargs)
        from .oracle_evidence import prepare_stage, collect_stage
        directory, invocation, environment = prepare_stage(oracle_context, task, stage)
        try:
            result = run_stage(task.evaluator_path, stage, program_path, oracle_environment=environment, **kwargs)
        except BaseException as exc:
            failed = EvalResult(ok=False, score=0., artifacts={'error_type': type(exc).__name__})
            collect_stage(directory, invocation, failed)
            raise
        result, retained = collect_stage(directory, invocation, result)
        evidence.append(retained)
        return result

    smoke = run('smoke')
    if not smoke.ok:
        return CascadeOutcome(result=smoke, raw_first_score=smoke.score,
                              reached="smoke", oracle_evidence=evidence)

    full = run('full')
    raw_first = full.score
    if not full.ok:
        return CascadeOutcome(result=full, raw_first_score=raw_first,
                              reached="full", oracle_evidence=evidence)

    if record_candidate and task.stochastic:
        conf = run('confirm')
        trials = conf.cases.get("n_trials", 0.0)
        enough_samples = (trials >= max(2, task.confirm_n) and float(trials).is_integer())
        if conf.ok and conf.confirmed_sampled and enough_samples:
            return CascadeOutcome(result=conf, raw_first_score=raw_first,
                                  reached="confirm", confirmed=True, oracle_evidence=evidence)
        if conf.ok:
            conf = replace(conf, ok=False, score=0.0,
                           artifacts={**conf.artifacts,
                                      "error_type": "InsufficientConfirmation",
                                      "error": "confirm did not provide all requested repeated samples"})
        return CascadeOutcome(result=conf, raw_first_score=raw_first,
                              reached="confirm", oracle_evidence=evidence)

    return CascadeOutcome(result=full, raw_first_score=raw_first, reached="full", oracle_evidence=evidence)
