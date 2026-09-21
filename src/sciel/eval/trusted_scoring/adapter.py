"""Pure host-owned task scoring and output evidence. Call only after trusted executor verification.

No executor exists here: HostBinding is caller-owned context, not proof. Trial
aggregation consumes completed host results and cannot prove physical dispatch.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re

from . import circle, heil, beat
from .transport import decode


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


@dataclass(frozen=True)
class HostBinding:
    source_sha256: str
    evaluator_sha256: str
    invocation_id: str
    stage: str
    elapsed_seconds: float


def _binding(binding):
    if type(binding) is not HostBinding:
        raise ValueError("host binding required")
    for field, length in (("source_sha256", 64), ("evaluator_sha256", 64), ("invocation_id", 32)):
        value = getattr(binding, field)
        if type(value) is not str or re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is None:
            raise ValueError("invalid host binding")
    if type(binding.stage) is not str or binding.stage not in ("smoke", "full"):
        raise ValueError("invalid host stage")
    if type(binding.elapsed_seconds) is not float or not math.isfinite(binding.elapsed_seconds) \
            or binding.elapsed_seconds < 0:
        raise ValueError("invalid host elapsed time")


def _failure(kind, message):
    cases = {"combined_score": 0.0, **circle._BEHAVIOR_ZERO}
    cases.update({"sum_radii": 0.0, "target_ratio": 0.0, "validity": 0.0} if kind == "circle"
                 else {"min_area": 0.0, "min_area_normalized": 0.0, "validity": 0.0} if kind == "heil"
                 else {"success_prob": 0.0})
    return {"ok": False, "score": 0.0, "cases": cases,
            "artifacts": {"error": message, "traceback": ""}}


def score_payload(kind, payload, binding):
    _binding(binding)
    if type(kind) is not str or kind not in ("circle", "heil", "beat"):
        raise ValueError("unsupported adapter")
    try:
        values = decode(kind, payload)
        witness = {"source_sha256": binding.source_sha256,
                   "evaluator_sha256": binding.evaluator_sha256,
                   "invocation_id": binding.invocation_id, "stage": binding.stage,
                   "confirmation": None}
        if kind == "circle":
            centers, radii = values["centers"], values["radii"]
            result = circle.score(centers, radii)
            numeric = {"centers": centers.astype(float).tolist(), "radii": radii.astype(float).tolist()}
            witness.update(schema="sciel-native-circle-geometry/1", numeric=numeric,
                           numeric_sha256=hashlib.sha256(canonical(numeric).encode()).hexdigest(),
                           centers_dtype=centers.dtype.name, radii_dtype=radii.dtype.name,
                           native_validity=bool(result["cases"]["validity"]),
                           native_sum_radii=result["cases"]["sum_radii"], native_score=result["score"])
            artifacts = {"native_geometry": canonical(witness)}
            if binding.stage == "full":
                artifacts.update(eval_time=f"{binding.elapsed_seconds:.4f}", reported_sum=values["reported_sum"])
        elif kind == "heil":
            points = values["points"]
            result = heil.score(points)
            witness.update(schema="sciel-full-output/1", input_sha256=None,
                           seed_sha256=None, output={"points": points.tolist()})
            artifacts = {"eval_time": f"{binding.elapsed_seconds:.4f}", "full_output": canonical(witness)}
        else:
            result = beat.score(values["support"], values["weights"])
            artifacts = {"eval_time": f"{binding.elapsed_seconds:.4f}"}
        result["artifacts"] = artifacts
        return result
    except (ValueError, TypeError, OverflowError, FloatingPointError) as exc:
        return _failure(kind, str(exc))


def aggregate_completed(kind, runs, expected_trials):
    """No scheduling or early exit: caller must supply every completed trial."""
    if type(kind) is not str or kind not in ("circle", "heil", "beat"):
        raise ValueError("unsupported adapter")
    if type(expected_trials) is not int or expected_trials < 1 or type(runs) is not list \
            or len(runs) != expected_trials:
        raise ValueError("confirmation requires exactly all completed trial results")
    # These are host-produced results; child JSON is never passed here directly.
    for result in runs:
        if type(result) is not dict or any(type(k) is not str for k in result) \
                or set(result) != {"ok", "score", "cases", "artifacts"} \
                or type(result["ok"]) is not bool or type(result["score"]) is not float \
                or not math.isfinite(result["score"]) or type(result["cases"]) is not dict \
                or type(result["artifacts"]) is not dict:
            raise ValueError("invalid completed host result")
        if any(type(k) is not str or type(v) is not float or not math.isfinite(v)
               for k, v in result["cases"].items()) \
                or any(type(k) is not str or type(v) is not str
                       for k, v in result["artifacts"].items()):
            raise ValueError("completed result must contain primitive cases/artifacts")
    selected = next((i for i, result in enumerate(runs) if not result["ok"]), None)
    if selected is None:
        selected = min(range(expected_trials), key=lambda i: runs[i]["score"])
    worst = deepcopy(runs[selected])
    worst["cases"]["n_trials"] = float(expected_trials)
    if kind in ("circle", "heil"):
        selection = {"trials": expected_trials, "selected_trial": selected,
                     "selection": "first_failure_else_lowest_score"}
        worst["artifacts"]["confirmation_selection"] = canonical(selection)
        field = "native_geometry" if kind == "circle" else "full_output"
        if field in worst["artifacts"]:
            witness = json.loads(worst["artifacts"][field])
            witness["confirmation"] = selection
            worst["artifacts"][field] = canonical(witness)
    return worst
