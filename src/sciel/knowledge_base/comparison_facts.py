"""Prompt-only measurement scopes and signs, never new quality evidence.

The exact paired derivation and ordinary observed-comparison gates remain the
authority. This projection makes their distinct units and scopes explicit.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from ..hypothesis.observed_facts import observed_comparison
from .catalogs import canonical_json

SCHEMA = "sciel-comparison-facts/1"
MAX_CHARS = 6000
LIMITS = (
    "Directions label existing measurements, not additional evidence. positive/negative/zero "
    "mean challenger minus comparator for paired rows; raw physical sum and exact certified "
    "sum have separate signs. unknown is not zero. Ordinary controller evaluations and linked "
    "paired shared-seed groups are distinct measurement scopes, not the same measurement. "
    "Their origin link alone proves neither independent replication nor causal mechanism. "
    "Paired rows never rescore the ordinary forecast. Quality and lesson gates remain unchanged."
)


def identity():
    return {"schema": SCHEMA, "source_sha256": hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest(), "max_chars": MAX_CHARS}


def _sign(value):
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _raw_direction(row):
    value = row.get("raw_delta")
    if row.get("comparator_status") != "valid" or row.get("challenger_status") != "valid" or value is None:
        return "unknown"
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("comparison facts require a finite raw delta")
    return _sign(value)


def _certified_direction(row):
    value = row.get("certified_delta")
    if row.get("comparator_status") != "valid" or row.get("challenger_status") != "valid" or value is None:
        return "unknown"
    if (type(value) is not dict or set(value) != {"numerator", "denominator"}
            or type(value["numerator"]) is not int or type(value["denominator"]) is not int
            or value["denominator"] <= 0):
        raise ValueError("comparison facts require an exact certified ratio")
    # No float conversion, decimal formatting, tolerance, or rounding.
    return _sign(value["numerator"])


def _catalog(records):
    from .presentation import _comparison_view
    if type(records) is not list:
        raise ValueError("comparison facts require a record list")
    ordinary, paired, order = {}, {}, []
    for record in records:
        if type(record) is not dict:
            raise ValueError("comparison facts require record objects")
        ident = record.get("id")
        if type(ident) is not str or not ident or ident in order:
            raise ValueError("comparison facts require unique record IDs")
        order.append(ident)
        view = _comparison_view(record)
        if view is None:
            measured = observed_comparison(record)
            direction = measured["measurement"]["direction"]
            ordinary[ident] = {
                "scope": "ordinary_controller_evaluation",
                "controller_score_direction": {"improved": "positive", "worsened": "negative",
                    "unchanged": "zero", "unmeasured": "unknown"}[direction],
                "forecast_position": measured["forecast"]["position"],
            }
        else:
            paired[ident] = {
                "scope": "paired_shared_seed_group",
                "origin_experiment_id": view["origin_experiment_id"],
                "origin_record_present": False,
                "quality_outcome": view["derived"]["outcome"],
                "pairs": [{"seed": row["seed"],
                    "raw_physical_sum_direction": _raw_direction(row),
                    "exact_certified_sum_direction": _certified_direction(row)}
                    for row in view["derived"]["pairs"]],
            }
    for row in paired.values():
        row["origin_record_present"] = row["origin_experiment_id"] in ordinary
    return ordinary, paired, order


def _payload(ordinary, paired, order, selected, budget):
    value = {"schema": SCHEMA, "limits": LIMITS,
             "ordinary": {i: row for i, row in ordinary.items() if i in selected},
             "paired": {i: row for i, row in paired.items() if i in selected},
             "presented_ids": [i for i in order if i in selected],
             "omitted_ids": [i for i in order if i not in selected],
             "max_chars": budget, "rendered_chars": 0}
    while value["rendered_chars"] != len(canonical_json(value)):
        value["rendered_chars"] = len(canonical_json(value))
    return value


def comparison_facts(records):
    """Project all supplied records; the enclosing reflection budget applies."""
    ordinary, paired, order = _catalog(records)
    return _payload(ordinary, paired, order, set(order), None)


def bounded_comparison_facts(records, *, max_chars=MAX_CHARS):
    """Admit whole groups together with any supplied ordinary origin record."""
    if type(max_chars) is not int or max_chars < 0:
        raise ValueError("invalid comparison facts budget")
    ordinary, paired, order = _catalog(records)
    selected = set()
    # A paired row never labels a supplied origin as present while omitting its
    # scope. Scope linkage and all three seed signs are admitted as one unit.
    for ident, row in paired.items():
        trial = selected | {ident}
        if row["origin_record_present"]:
            trial.add(row["origin_experiment_id"])
        if _payload(ordinary, paired, order, trial, max_chars)["rendered_chars"] <= max_chars:
            selected = trial
    for ident in ordinary:
        trial = selected | {ident}
        if _payload(ordinary, paired, order, trial, max_chars)["rendered_chars"] <= max_chars:
            selected = trial
    value = _payload(ordinary, paired, order, selected, max_chars)
    return value if value["rendered_chars"] <= max_chars else None


def validate_comparison_facts(value, *, records, max_chars=None):
    """Re-derive from authoritative records, including exact admission budget."""
    expected = (comparison_facts(records) if max_chars is None else
                bounded_comparison_facts(records, max_chars=max_chars))
    if canonical_json(value) != canonical_json(expected):
        raise ValueError("comparison facts differ from recorded measurements")
    return expected
