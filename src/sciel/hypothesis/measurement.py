"""Trusted task descriptions and exact returned observations for research prompts.

These functions do not evaluate code, select a score, convert units, or certify
the task's output. Declarations come from task metadata, never candidate text.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

from ..eval.native_geometry import (VIEW_SCHEMA as NATIVE_GEOMETRY_VIEW_SCHEMA,
    parse_artifact, validate_binding, validate_view, view_geometry)


SCHEMA = "sciel-research-measurement/1"
_CONTEXT_FIELDS = {"schema", "task_name", "task_hash", "primary_metric",
                   "score_source", "goal", "semantics", "sha256"}
_RATIO_FIELDS = {"kind", "score_units", "numerator_case", "numerator_units",
                 "denominator", "validity_case", "valid_value", "invalid_score",
                 "invalid_numerator_value", "invalid_numerator_semantics"}
_EVALUATION_FIELDS = {"ok", "score", "cases", "reached", "confirmed", "raw_first_score"}
_UNKNOWN = {"kind": "unknown", "score_units": None}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _require(type(value) in (int, float), f"{name} must be a finite JSON number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite, f"{name} must be a finite JSON number")
    return value  # Preserve the supplied binary float/int, without rounding.


def _text(value, name, *, nullable=False):
    _require((nullable and value is None) or
             (isinstance(value, str) and bool(value.strip()) and len(value) <= 256),
             f"invalid {name}")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def validate_research_metric(raw):
    """Validate optional task.yaml metadata; unsupported semantics fail explicitly.

    This first declared model is a validity-gated ratio. Its invalid numerator is
    a task penalty sentinel, not an observation of the physical quantity. No
    relationship is inferred from ``known_best``, task name or metric spelling.
    """
    if raw is None:
        return None
    _require(type(raw) is dict and set(raw) == _RATIO_FIELDS,
             "research_metric must contain exactly the validity-gated ratio fields")
    _require(raw["kind"] == "validity_gated_ratio", "unsupported research_metric kind")
    _require(raw["score_units"] == "dimensionless", "ratio score_units must be dimensionless")
    for key in ("numerator_case", "numerator_units", "validity_case"):
        _text(raw[key], key)
    _require(raw["numerator_case"] != raw["validity_case"],
             "numerator and validity case must differ")
    for key in ("denominator", "valid_value", "invalid_score", "invalid_numerator_value"):
        _number(raw[key], key)
    _require(raw["denominator"] > 0, "metric denominator must be positive")
    _require(raw["invalid_numerator_semantics"] == "penalty_sentinel_not_physical_measurement",
             "invalid numerator must be explicitly identified as a penalty sentinel")
    return copy.deepcopy(raw)


def metric_context(task=None):
    """Build a content-bound declaration; missing task semantics stay unknown."""
    declared = validate_research_metric(getattr(task, "research_metric", None))
    value = {"schema": SCHEMA,
             "task_name": getattr(task, "qualified_name", None) or None,
             "task_hash": getattr(task, "task_hash", None) or None,
             "primary_metric": getattr(task, "primary_metric", None) or None,
             "score_source": "EvalResult.score", "goal": "maximize",
             "semantics": declared if declared is not None else copy.deepcopy(_UNKNOWN)}
    value["sha256"] = _digest(value)
    return validate_metric_context(value)


def validate_metric_context(raw):
    """Validate and copy a descriptor, including its non-circular content hash."""
    _require(type(raw) is dict and set(raw) == _CONTEXT_FIELDS, "invalid metric context fields")
    _require(raw["schema"] == SCHEMA and raw["score_source"] == "EvalResult.score"
             and raw["goal"] == "maximize", "unsupported metric context contract")
    for key in ("task_name", "primary_metric"):
        _text(raw[key], key, nullable=True)
    _require(raw["task_hash"] is None or (isinstance(raw["task_hash"], str)
             and re.fullmatch(r"[0-9a-f]{64}", raw["task_hash"]) is not None), "invalid task_hash")
    if raw["semantics"] != _UNKNOWN:
        validate_research_metric(raw["semantics"])
        _require(all(raw[key] is not None for key in ("task_name", "task_hash", "primary_metric")),
                 "declared metric semantics require a bound task identity")
    else:
        _require(type(raw["semantics"]) is dict, "invalid unknown semantics")
    _require(isinstance(raw["sha256"], str) and raw["sha256"] ==
             _digest({key: value for key, value in raw.items() if key != "sha256"}),
             "metric context hash mismatch")
    return copy.deepcopy(raw)


def validate_cases(raw):
    """Copy every returned case exactly; reject malformed entries, never drop them."""
    _require(type(raw) is dict, "evaluation cases must be a dict")
    for key, value in raw.items():
        _text(key, "case name")
        _number(value, f"case {key}")
    return copy.deepcopy(raw)


def validate_evaluation(raw):
    """Validate a returned cascade result; None means no result was returned.

    Failure scores/cases are retained as diagnostics. Whether they are measured
    quality is controlled by the research outcome status, not by this serializer.
    """
    if raw is None:
        return None
    _require(type(raw) is dict and set(raw) in
             (_EVALUATION_FIELDS, _EVALUATION_FIELDS | {"native_geometry"}),
             "invalid evaluation fields")
    _require(type(raw["ok"]) is bool and type(raw["confirmed"]) is bool,
             "evaluation status flags must be bool")
    _number(raw["score"], "evaluation score")
    _number(raw["raw_first_score"], "raw_first_score")
    validate_cases(raw["cases"])
    _require(raw["reached"] in ("smoke", "full", "confirm"), "invalid cascade stage")
    _require(not raw["confirmed"] or (raw["ok"] and raw["reached"] == "confirm"),
             "confirmed evaluation must be a successful confirm result")
    result = copy.deepcopy(raw)
    if "native_geometry" in raw:
        result["native_geometry"] = validate_binding(raw["native_geometry"], raw)
    return result


def evaluation_observation(outcome):
    """Project an existing CascadeOutcome without importing or invoking a task."""
    try:
        result = outcome.result
        raw = {"ok": result.ok, "score": result.score, "cases": result.cases,
               "reached": outcome.reached, "confirmed": outcome.confirmed,
               "raw_first_score": outcome.raw_first_score}
        if "native_geometry" in getattr(result, "artifacts", {}):
            raw["native_geometry"] = parse_artifact(result.artifacts["native_geometry"])
    except AttributeError as exc:
        raise ValueError("expected a returned CascadeOutcome") from exc
    return validate_evaluation(raw)


def evaluation_view(raw):
    """Detached prompt projection; full geometry remains stored as evidence.

    Already projected views are validated as metadata, never promoted back to
    evidence. Only the schema-owned native_geometry slot is transformed.
    """
    result = copy.deepcopy(raw)
    if type(result) is dict and "native_geometry" in result:
        geometry = result["native_geometry"]
        result["native_geometry"] = (validate_view(geometry)
            if type(geometry) is dict and geometry.get("schema") == NATIVE_GEOMETRY_VIEW_SCHEMA
            else view_geometry(geometry))
    return result


def observation_view(record):
    """Project only an ordinary record's declared evaluation, not opaque data."""
    result = copy.deepcopy(record)
    if type(result) is dict and "evaluation" in result:
        result["evaluation"] = evaluation_view(result["evaluation"])
    if type(result) is dict and ('output_novelty' in result or 'output_admission' in result):
        from ..knowledge_base.eligibility import admission_view
        result['output_admission'] = admission_view(result)
        result.pop('output_novelty', None)
    return result
