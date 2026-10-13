"""Subprocess entrypoint: import a task evaluator, run one stage, write JSON.

Stdlib only. Invoked by sandbox.py; never imported by the engine in-process.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


def _load(path):
    # The script entry point need not have an editable installation or cwd=src.
    # Pin imports to the package containing this runner, including in spawn.
    package_root = str(Path(__file__).resolve().parents[2])
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from sciel.candidate_modules import load_candidate
    return load_candidate(path, "task_evaluator")


def _resolve_stage(mod, stage):
    if hasattr(mod, stage):
        return getattr(mod, stage)
    aliases = {
        "smoke": ["evaluate_stage1", "smoke"],
        "full": ["evaluate", "evaluate_stage2", "full"],
        "confirm": ["confirm", "evaluate"],
    }
    for name in aliases.get(stage, []):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"evaluator has no stage '{stage}'")


import math


def _numeric(d):
    """Metrics, coerced to finite floats. Non-numeric entries are dropped.

    isinstance(v, float) is not enough. numpy.float64 subclasses float and survived,
    but numpy.float32 does not and was silently discarded -- an evaluator built on
    float32 arrays reported a perfectly good child as score 0.0 with ok=True, and the
    child then entered the archive as a parent. Anything with __float__ is accepted,
    and non-finite values are dropped rather than passed on: a NaN score reaches
    db.add, where int(score * grid_bins) raises and takes the run with it."""
    out = {}
    for k, v in d.items():
        if (isinstance(v, (bool, str, bytes))
                or getattr(getattr(v, "dtype", None), "kind", None) == "b"):
            continue
        try:
            f = float(v)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(f):
            out[str(k)] = f
    return out


def _normalize(result):
    # _runner is also executed as a script, so avoid a relative import here.
    # EvalResult's public JSON contract preserves its score, status and artifacts.
    if (callable(getattr(result, "to_json", None))
            and all(hasattr(result, key) for key in ("ok", "score", "cases", "artifacts"))):
        result = result.to_json()
    metrics = None
    artifacts = {}
    explicit_score = None
    has_explicit_score = False
    # An absent legacy status may be inferred; a present malformed status must
    # not be converted by Python truthiness (including None and numpy.bool_).
    if isinstance(result, dict):
        has_explicit_ok = "ok" in result
        explicit_ok = result.get("ok")
    else:
        has_explicit_ok = hasattr(result, "ok")
        explicit_ok = getattr(result, "ok", None)
    if has_explicit_ok and type(explicit_ok) is not bool:
        raise ValueError("evaluator result ok must be boolean when present")
    if hasattr(result, "metrics"):
        metrics = dict(result.metrics)
        artifacts = dict(getattr(result, "artifacts", {}) or {})
    elif isinstance(result, dict):
        if "metrics" in result:
            metrics = dict(result["metrics"])
            artifacts = dict(result.get("artifacts", {}) or {})
        elif "score" in result or "cases" in result or "ok" in result:
            has_explicit_score = "score" in result
            explicit_score = result.get("score")
            metrics = dict(result.get("cases", {}) or {})
            artifacts = dict(result.get("artifacts", {}) or {})
            # an evaluator may report the failure at the top level rather than inside
            # cases/artifacts; that used to be ignored entirely on this branch
            for key in ("error", "error_type", "traceback"):
                if key in result and key not in artifacts:
                    artifacts[key] = result[key]
        else:
            metrics = dict(result)
    else:
        raise TypeError(f"unsupported evaluator return {type(result)}")
    cases = _numeric(metrics)
    bad_score = None
    # A score key that was present but did not survive coercion is a failure, not a
    # zero: a string, None, NaN or inf all used to become score 0.0 with ok=True, and
    # the child entered the population as a valid program.
    if "combined_score" in metrics and "combined_score" not in cases:
        bad_score = f"combined_score is not a finite number: {metrics['combined_score']!r}"
    if has_explicit_score:
        numeric_score = _numeric({"score": explicit_score})
        if "score" in numeric_score:
            score = numeric_score["score"]
        else:
            score, bad_score = 0.0, f"score is not a finite number: {explicit_score!r}"
    elif "combined_score" in cases:
        score = cases["combined_score"]
    else:
        score = 0.0
        bad_score = "evaluator result has no numeric score or combined_score"
    ok = explicit_ok if has_explicit_ok else ("error" not in metrics)
    # An `error` key means the evaluator is reporting a failure, whichever branch
    # produced the payload. The explicit-score branch used to look for it only inside
    # `cases`, so a result carrying both `score` and `error` was recorded as a valid
    # program: the failure never reached n_eval_fail and the repair operator was
    # handed nothing to repair.
    if "error" in metrics or "error" in artifacts:
        ok = False
    if bad_score is not None:
        ok = False
        artifacts = dict(artifacts)
        artifacts.setdefault("error_type", "NonFiniteScore")
        artifacts.setdefault("error", bad_score)
    art = {str(k): (v if isinstance(v, str) else str(v))
           for k, v in artifacts.items()}
    return {"ok": ok, "score": score, "cases": cases, "artifacts": art,
            "stderr": str(result.get("stderr", "")) if isinstance(result, dict) else "",
            "cost": _numeric({"cost": result.get("cost", 0.0)}).get("cost", 0.0)
                    if isinstance(result, dict) else 0.0}


class _ResourceSetupError(RuntimeError):
    def __init__(self, code, resource):
        self.code, self.resource = code, resource
        super().__init__(code + ": " + resource)


def _apply_limits(cpu_seconds, mem_bytes):
    """Apply requested limits before imports; refusal is never candidate quality."""
    if cpu_seconds is None and mem_bytes is None:
        return
    if os.name != "posix":
        raise _ResourceSetupError("unsupported_platform", "platform")
    try:
        import resource
    except ImportError:
        raise _ResourceSetupError("resource_unavailable", "platform") from None
    for name, value, constant in (("cpu_seconds", cpu_seconds, "RLIMIT_CPU"),
                                   ("mem_bytes", mem_bytes, "RLIMIT_AS")):
        if value is None:
            continue
        kind = getattr(resource, constant, None)
        apply = getattr(resource, "setrlimit", None)
        if kind is None or not callable(apply):
            raise _ResourceSetupError("resource_unavailable", name)
        hard = value + 1 if name == "cpu_seconds" else value
        try:
            apply(kind, (value, hard))
        except (ValueError, OSError, OverflowError):
            raise _ResourceSetupError("setrlimit_failed", name) from None


def _seed(seed):
    """Paired measurements share the process-wide random streams a candidate draws from without its own seed."""
    import random
    random.seed(seed)
    try:
        import numpy
    except ImportError:
        return
    numpy.random.seed(seed % 2 ** 32)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluator_path")
    parser.add_argument("stage")
    parser.add_argument("program_path")
    parser.add_argument("out_path")
    parser.add_argument("n", type=int, nargs="?")
    parser.add_argument("--cpu-seconds", type=int)
    parser.add_argument("--mem-bytes", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv[1:])
    evaluator_path, stage, program_path, out_path, n = (
        args.evaluator_path, args.stage, args.program_path, args.out_path, args.n)
    started = time.time()
    setup_complete = False
    try:
        _apply_limits(args.cpu_seconds, args.mem_bytes)
        if args.seed is not None:
            _seed(args.seed)
        setup_complete = True
        mod = _load(evaluator_path)
        fn = _resolve_stage(mod, stage)
        # `n` goes only to a function the evaluator actually named `confirm`. Resolving
        # "confirm" can alias to `evaluate(program_path, timeout=60)`, and _wants_n
        # only counts parameters, so confirm_n used to be handed to whatever that
        # second parameter happened to be.
        pass_n = (stage == "confirm" and n is not None
                  and getattr(mod, "confirm", None) is fn and _wants_n(fn))
        result = fn(program_path, n) if pass_n else fn(program_path)
        payload = _normalize(result)
        # Whether re-sampling HAPPENED is the evaluator's to report, not something to
        # infer from a signature: every stub `def confirm(p, n=3): return full(p)`
        # ignores n, yet was flagged as a confirmed multi-sample and its single fresh
        # draw replaced the full score in the journal.
        trials = payload["cases"].get("n_trials")
        payload["confirmed_sampled"] = bool(
            pass_n and payload["ok"] and trials is not None
            and trials.is_integer() and trials >= max(2, n))
        if pass_n and trials is None:
            payload["artifacts"]["confirm_note"] = (
                "confirm() did not report n_trials, so the result is not treated as "
                "re-sampled; have it return cases['n_trials']")
    except Exception as e:
        if not setup_complete:
            # This phase has not imported the evaluator. The unisolated parent
            # still treats the marker conservatively, never as zero-use proof.
            payload = {"schema": "sciel-native-evaluator-setup-failure/1",
                "code": e.code if isinstance(e, _ResourceSetupError) else "setup_exception",
                "resource": e.resource if isinstance(e, _ResourceSetupError) else "unknown",
                "requested_limits": {"cpu_seconds": args.cpu_seconds, "mem_bytes": args.mem_bytes},
                "execution": "unknown", "resource_usage": "unknown",
                "reported_wall_seconds": max(0.0, time.time() - started)}
        else:
            payload = {"ok": False, "score": 0.0, "cases": {},
                       "artifacts": {"error_type": type(e).__name__,
                                     "traceback": traceback.format_exc()[:4000]},
                       "stderr": str(e)}
    if "schema" not in payload:
        payload["timing"] = time.time() - started
    with open(out_path, "w") as f:
        json.dump(payload, f)


def _wants_n(fn):
    try:
        import inspect
        return len(inspect.signature(fn).parameters) >= 2
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main(sys.argv)
