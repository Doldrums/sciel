"""Re-derive known controlled edits without importing or executing candidate code.

An accepted proof establishes exact source bytes and the limited transformation,
not mechanism compliance in general, measured benefit, or causal attribution.
"""
from __future__ import annotations

import copy
import json


SCHEMA = "source-bound-edit/1"


def _canonical(value):
    # JSON encoding alone aliases tuples to lists and non-string dictionary keys
    # to strings. Accept only the finite, literal JSON types of stored receipts.
    def check(item):
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("controlled change keys must be strings")
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("controlled change must be literal JSON")
    try:
        check(value)
        return json.dumps(value, sort_keys=True, ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid controlled change JSON") from exc


def validate_controlled_change(proof, *, parent_source_sha256, proposed_sha256,
                              source_sha256, source_repaired, parent_code=None,
                              proposed_code=None, source_code=None):
    """Return a detached proof only if its full canonical value is re-derived.

    Hash bindings are required, even when full sources are absent in a retained
    evidence snapshot. Supplied source strings must also be byte-for-byte family
    members. Numeric/bool aliases, no-ops, extra fields and repair are rejected.
    """
    if type(proof) is not dict or proof.get("schema") != SCHEMA:
        raise ValueError("invalid controlled change schema")
    operation = proof.get("operation")
    from .circle_source_family import builder_for_operation
    builder = builder_for_operation(operation)
    if operation in ('circle_restart_body/1', 'circle_restart_body_traced/1', 'circle_restart_body_traced/2'):
        parent = builder.source_for(proof.get('parent_body'))
        built = builder.apply(parent, proof.get('target_body'))
    else:
        parent = builder.source_for(proof.get("parent_policy"))
        built = builder.apply(parent, proof.get("target_policy"))
    expected, child = built["proof"], built["code"]
    if _canonical(proof) != _canonical(expected):
        raise ValueError("controlled change differs from the exact builder proof")
    if (source_repaired is not False
            or type(parent_source_sha256) is not str
            or type(proposed_sha256) is not str or type(source_sha256) is not str
            or parent_source_sha256 != expected["parent_source_sha256"]
            or proposed_sha256 != expected["child_source_sha256"]
            or source_sha256 != expected["child_source_sha256"]):
        raise ValueError("controlled change source binding mismatch or repair")
    for supplied, known in ((parent_code, parent), (proposed_code, child), (source_code, child)):
        if supplied is not None and (type(supplied) is not str or supplied != known):
            raise ValueError("controlled change source bytes differ from the pinned family")
    return copy.deepcopy(expected)
