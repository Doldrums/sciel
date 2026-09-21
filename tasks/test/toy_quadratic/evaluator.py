"""Deterministic toy evaluator for identity/resume tests: pure function of run_search."""

from sciel.candidate_modules import load_candidate

TARGET = (-1.704, 0.678)


def _load(program_path, name):
    return load_candidate(program_path, name)


def _score(program_path, name):
    mod = _load(program_path, name)
    if not hasattr(mod, "run_search"):
        raise AttributeError("missing run_search")
    result = mod.run_search()
    if not isinstance(result, tuple) or len(result) < 2:
        raise ValueError("run_search must return (x, y[, value])")
    x, y = float(result[0]), float(result[1])
    dist = ((x - TARGET[0]) ** 2 + (y - TARGET[1]) ** 2) ** 0.5
    return 1.0 / (1.0 + dist)


def smoke(program_path):
    try:
        s = _score(program_path, "toy_smoke")
    except Exception as e:
        return {"ok": False, "score": 0.0, "cases": {"combined_score": 0.0},
                "artifacts": {"error": str(e)}}
    return {"ok": True, "score": s, "cases": {"combined_score": s}, "artifacts": {}}


def full(program_path):
    return smoke(program_path)


def confirm(program_path, n=1):
    return smoke(program_path)
