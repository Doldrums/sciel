"""Opt-in black-box task: 50 oracle observations, trusted best over queried points.

Candidate code runs only in the pinned container, never in this evaluator. The
public entry point is run_search(oracle), with oracle.query(x, y), bounds and
query_budget. Local surrogate computation is allowed. Historical Function scores
and task identities are unrelated to this version.
"""

import importlib.util
import json
import os
from pathlib import Path
import sys


_ASSETS = Path(__file__).resolve().parent / "assets"
_SPEC = importlib.util.spec_from_file_location("sciel_hidden_oracle_runtime", _ASSETS / "oracle_runtime.py")
runtime = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runtime
_SPEC.loader.exec_module(runtime)


def _count(n):
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 100:
        raise ValueError("trial count must be an integer between 1 and 100")
    return n


def _evaluate(program_path, n, *, sealed=None, image=None):
    n = _count(n)
    image = image or os.environ.get("SCIEL_ORACLE_IMAGE", "")
    # Preflight before generating private data or importing any candidate code.
    binding = runtime.verify_runtime_image(image)
    with open(program_path, "rb") as stream:
        source = stream.read(runtime.MAX_SOURCE_BYTES + 1)
    if not 0 < len(source) <= runtime.MAX_SOURCE_BYTES:
        raise ValueError("candidate size outside allowed range")
    if sealed is None:
        instances, commitment = runtime.create_instances(n,
            private_dir=os.environ.get("SCIEL_ORACLE_PRIVATE_DIR"),
            commitment_dir=os.environ.get("SCIEL_ORACLE_COMMITMENT_DIR"))
    else:
        instances, commitment = runtime.load_sealed_instances(*sealed, n=n)
    # This public commitment is emitted before the first candidate starts. Its
    # fsynced record also survives an outer stage kill. No seed/parameters emitted.
    print(json.dumps({"oracle_commitment": commitment, "runtime": binding}), flush=True)
    trials = [runtime.run_container_trial(source, instance, image) for instance in instances]
    score = sum(t["score"] for t in trials) / n
    successful = sum(t["status"] == "ok" for t in trials)
    infrastructure = any(t["status"] in {"startup_timeout", "infrastructure_error"} for t in trials)
    valid = successful > 0 and not infrastructure
    return {"ok": valid,
            "score": score if not infrastructure else 0.,
            "cases": {"combined_score": score if not infrastructure else 0.,
                      "mean_best_query_score": score if valid else 0.,
                      "validity": float(valid),
                      "n_trials": float(n), "reliability_score": successful/n,
                      "objective_queries": float(sum(t["queries"] for t in trials)),
                      "objective_query_budget": float(runtime.QUERY_BUDGET)},
            "artifacts": {"oracle_contract": runtime.PROTOCOL,
                          "instance_commitment": json.dumps(commitment, sort_keys=True),
                          "runtime_binding": json.dumps(binding, sort_keys=True),
                          "trials": json.dumps(trials, sort_keys=True),
                          **({"error": "oracle runtime infrastructure failure",
                              "raw_trial_mean": score} if infrastructure else {})}}


def smoke(program_path):
    return _evaluate(program_path, 1)


def full(program_path):
    return _evaluate(program_path, 5)


def confirm(program_path, n=20):
    result = _evaluate(program_path, _count(n))
    if result["cases"]["reliability_score"] < 1:
        result = {**result, "ok": False, "score": 0.,
                  "cases": {**result["cases"], "combined_score": 0.,
                            "mean_best_query_score": 0., "validity": 0.}, "artifacts": {
            **result["artifacts"], "raw_trial_mean": result["artifacts"].get("raw_trial_mean", result["score"]),
            "error": "one or more confirmation trials failed"}}
    return result


def evaluate_sealed(program_path, instance_path, commitment_sha256, *, n, image=None):
    """Explicit final-only entry point; the normal search cascade never selects it.

    Caller freezes candidate sources before opening the common holdout. Its private
    path/commitment are not accepted from the candidate protocol or development env.
    """
    return _evaluate(program_path, _count(n), sealed=(instance_path, commitment_sha256), image=image)
