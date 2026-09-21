"""Shared repeat promotion contracts and pure helpers."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import math


CONTRACT = "sciel-repeat-promotion/1"

SEED_CONTRACT = "sciel-repeat-seeds/1"

_LIMITS = ("timeout_seconds", "cpu_seconds", "mem_bytes", "certification_seconds")

_REQUIRED = {"seed_master", "max_comparisons", "max_draw_attempts", "max_wall_seconds", *_LIMITS}

_FIXED = {"draws_per_source": 3, "rule": "paired_dominance",
          "quality_loss_tolerance": 0, "require_confirmed": True}

class RepeatPromotionError(ValueError):
    """An identity, durability, or protocol violation: no further dispatch is safe."""

def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False

def validate_config(raw=None):
    if raw is None or raw == {}:
        return {"enabled": False}
    if not isinstance(raw, dict) or type(raw.get("enabled")) is not bool:
        raise ValueError("repeat_promotion.enabled must be an explicit boolean")
    if not raw["enabled"]:
        if set(raw) != {"enabled"}:
            raise ValueError("disabled repeat_promotion must contain only enabled")
        return {"enabled": False}
    if set(raw) - {"enabled", "runtime", *_REQUIRED, *_FIXED} or _REQUIRED - set(raw):
        raise ValueError("repeat_promotion has unknown or missing budget/limit fields")
    cfg = dict(_FIXED, **raw)
    runtime = deepcopy(raw.get("runtime", {"backend": "native"}))
    if (not isinstance(runtime, dict) or
            (runtime != {"backend": "native"} and not (
                set(runtime) == {"backend", "image"} and runtime["backend"] == "docker"
                and isinstance(runtime["image"], str) and runtime["image"].startswith("sha256:")
                and len(runtime["image"]) == 71 and all(c in "0123456789abcdef" for c in runtime["image"][7:])))):
        raise ValueError("repeat_promotion.runtime must be native or Docker with an immutable image ID")
    cfg["runtime"] = runtime
    for key, value in _FIXED.items():
        if type(cfg[key]) is not type(value) or cfg[key] != value:
            raise ValueError(f"repeat_promotion.{key} is fixed by {CONTRACT}")
    for key, minimum in (("seed_master", 0), ("max_comparisons", 0),
                         ("max_draw_attempts", 3), ("cpu_seconds", 1), ("mem_bytes", 1)):
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f"repeat_promotion.{key} must be an integer >= {minimum}")
    if cfg["seed_master"] >= 2**64:
        raise ValueError("repeat_promotion.seed_master must fit uint64")
    if cfg["max_draw_attempts"] > 3 + 6 * cfg["max_comparisons"]:
        raise ValueError("max_draw_attempts exceeds the fixed 3 + 6R schedule")
    for key in ("max_wall_seconds", "timeout_seconds", "certification_seconds"):
        if not _finite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"repeat_promotion.{key} must be finite and positive")
        cfg[key] = float(cfg[key])
    return cfg

def _fraction(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError("certificate must be an exact rational")
    if type(value["numerator"]) is not int or type(value["denominator"]) is not int or value["denominator"] <= 0:
        raise ValueError("invalid certificate rational")
    return Fraction(value["numerator"], value["denominator"])

def _rational(value):
    return {"numerator": value.numerator, "denominator": value.denominator}

def _numeric_capture(value):
    if not isinstance(value, dict):
        raise RepeatPromotionError("missing numeric geometry")
    centers, radii = value.get("centers"), value.get("radii")
    if (not isinstance(centers, list) or len(centers) != 26
            or any(not isinstance(c, list) or len(c) != 2 or not all(_finite(x) for x in c) for c in centers)
            or not isinstance(radii, list) or len(radii) != 26
            or not all(_finite(r) and r >= 0 for r in radii)):
        raise RepeatPromotionError("numeric geometry must contain finite 26x2 centers and 26 nonnegative radii")
    return {"centers": centers, "radii": radii}

def _valid(receipt):
    try:
        return (receipt.get("status") == "valid" and _finite(receipt.get("raw_sum"))
                and receipt["raw_sum"] >= 0 and _fraction(receipt.get("certified_sum")) >= 0)
    except (ValueError, TypeError, AttributeError):
        return False

def _aggregate(receipts):
    if len(receipts) != 3 or not all(_valid(r) for r in receipts):
        return None
    raw = [r["raw_sum"] for r in receipts]
    exact = [_fraction(r["certified_sum"]) for r in receipts]
    return {"raw": {"values": raw, "mean": math.fsum(raw) / 3, "minimum": min(raw), "maximum": max(raw)},
            "certified": {"values": [_rational(v) for v in exact],
                          "mean": _rational(sum(exact, Fraction()) / 3),
                          "minimum": _rational(min(exact)), "maximum": _rational(max(exact))}}

def bootstrap_decision(receipts):
    aggregate = _aggregate(receipts)
    return {"status": "confirmed" if aggregate is not None else "incomplete",
            "reason": "all_draws_valid" if aggregate is not None else "bootstrap_not_fully_valid",
            "aggregates": aggregate}

def paired_dominance(incumbent, challenger):
    """All three pairs must dominate in both independently recorded metrics."""
    a, b = _aggregate(incumbent), _aggregate(challenger)
    if a is None or b is None:
        return {"status": "incomplete", "reason": "batch_not_fully_valid",
                "aggregates": None, "raw_deltas": None, "certified_deltas": None}
    raw = [new["raw_sum"] - old["raw_sum"] for old, new in zip(incumbent, challenger)]
    exact = [_fraction(new["certified_sum"]) - _fraction(old["certified_sum"])
             for old, new in zip(incumbent, challenger)]
    dominates = all(d >= 0 for d in raw) and all(d >= 0 for d in exact)
    improves = any(d > 0 for d in raw) and any(d > 0 for d in exact)
    return {"status": "promote" if dominates and improves else "reject",
            "reason": "paired_dominance" if dominates and improves else
                      "observed_loss" if not dominates else "no_strict_improvement_in_both_metrics",
            "aggregates": {"incumbent": a, "challenger": b},
            "raw_deltas": raw, "certified_deltas": [_rational(d) for d in exact]}

def _source_sha(code):
    if not isinstance(code, str) or not code:
        raise RepeatPromotionError("source must be a nonempty string")
    return hashlib.sha256(code.encode("utf-8")).hexdigest()
