"""Two-entry wire codec; the Engine owns selection and experiment accounting.

The outer transport is atomic. Within a valid envelope, each complete entry is
validated independently using the existing forecast and compile-only body rules.
No provider response is rewritten and no candidate source is assembled or run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from . import circle_restart_body as builder
from .parsing import parse_hypothesis_value


CONTRACT = "circle_proposal_pair/1"
ENTRY_COUNT = 2
MAX_RESPONSE_BYTES = builder.MAX_RESPONSE_BYTES


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def _envelope(raw):
    if type(raw) is not str:
        return None
    try:
        if len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
            return None
        value = json.loads(raw, object_pairs_hook=_unique,
                           parse_constant=_reject_constant, parse_float=_finite_float)
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeEncodeError):
        return None
    if (type(value) is not dict or set(value) != {"candidates"}
            or type(value["candidates"]) is not list
            or len(value["candidates"]) != ENTRY_COUNT):
        return None
    return value["candidates"]


def _entry(index, value):
    entry = {"index": index, "proposal": None, "hypothesis": None,
             "error": "invalid_entry", "body_error": None}
    if type(value) is not dict or set(value) != {"hypothesis", "body"}:
        return entry
    hypothesis = parse_hypothesis_value(value["hypothesis"])
    body_error = None
    try:
        builder.validate_body(value["body"])
    except ValueError as exc:
        body_error = str(exc)
    error = ("invalid_hypothesis" if hypothesis is None else
             "invalid_body" if body_error is not None else None)
    entry.update(hypothesis=hypothesis, error=error, body_error=body_error)
    if error is None:
        entry["proposal"] = builder.BodyProposal(hypothesis, value["body"])
    return entry


def decode_response(raw):
    """Decode exact raw text, without selecting, repairing or evaluating.

    Invalid outer JSON, shape, size, duplicate keys or nonfinite numbers yield
    ``{'entries': [], 'error': 'invalid_json_envelope'}`` with no partial salvage.
    Otherwise error is None and entries contains exactly indices 0 and 1. Each
    slot has proposal, hypothesis, error and body_error. Wrong entry shape is
    invalid_entry; complete entries retain independent forecast/body errors and
    preserve a valid forecast even when their body is invalid. A BodyProposal
    retains the submitted body bytes, including optional final LF and CRLF.

    A syntactically valid proposal is not a selected or eligible child. Provider
    refusal/finish metadata, source admission, accounting and execution belong
    to future integration, not to this text-only codec.
    """
    values = _envelope(raw)
    if values is None:
        return {"entries": [], "error": "invalid_json_envelope"}
    return {"entries": [_entry(index, value) for index, value in enumerate(values)],
            "error": None}


def response_format():
    """Native strict schema embedding an exact copy of the single-entry schema."""
    item = copy.deepcopy(builder.response_format()["json_schema"]["schema"])
    return {"type": "json_schema", "json_schema": {
        "name": "circle_proposal_pair_v1", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "required": ["candidates"], "properties": {
                       "candidates": {"type": "array", "minItems": ENTRY_COUNT,
                                      "maxItems": ENTRY_COUNT, "items": item}}}}}


def response_instruction(*, interface=None):
    return (
        "Return one JSON object matching the supplied schema with exactly two candidates, "
        "ordered from your preferred candidate to your alternative. Each entry must contain "
        "its own complete five-field hypothesis and Python body; no Markdown or other fields. "
        "Both hypotheses forecast their own candidate-minus-parent normalized-score DELTA "
        "against the SAME separately declared measured parent, not against each other. "
        "The two entries are independent alternatives, never sequential edits or combined code. "
        "Subsequent selection chooses the first locally eligible entry in this order, "
        "with at most one child. Parsing alone does not establish eligibility: the Engine "
        "checks both alternatives against the same source history before selection. After "
        "evaluation starts, failure of the chosen candidate never triggers the other. Unselected and "
        "rejected alternatives remain unmeasured; confidence is not a selection weight. "
        "Generate numerical restart transformations as body statements, not function definitions "
        "or full programs. Top-level statements start in column 1; indent only nested suites. "
        "The compiler owns the function indentation. Use the typed array interface below; "
        "NumPy is bound as np and math as math. Numerical benefit is uncertain.\n"
        + json.dumps(builder.interface_description() if interface is None else interface, sort_keys=True,
                     separators=(",", ":"), allow_nan=False)
    )


def identity():
    """Detached source/schema identity for future opt-in integration bindings."""
    return {
        "contract": CONTRACT,
        "codec_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "builder": copy.deepcopy(builder.identity()),
        "response_format_sha256": hashlib.sha256(_canonical(response_format())).hexdigest(),
        "entry_count": ENTRY_COUNT,
        "max_response_bytes": MAX_RESPONSE_BYTES,
        "scope": "Two-entry transport and local validation identity; the Engine owns selection, accounting and measurement.",
    }
