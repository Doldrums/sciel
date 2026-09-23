"""Bind a native research response to a versioned, character-level loss policy.

The caller authenticates the checkpoint, trace, evaluation and ancestry first.
This module checks their supplied content bindings; it does not establish that
an arbitrary caller-written receipt is genuine or that a candidate is eligible.
It neither tokenizes nor changes any existing episode/dataset schema.
"""
from __future__ import annotations

from pathlib import Path
import re

from ..engine.prompt import parse as code_parser
from ..hypothesis import parsing as hypothesis_parser
from ..hypothesis.models import CONTRACT
from .types import canonical_bytes, content_hash, require_hash, require_identifier, sha256_bytes

POLICY = "native-research-code-suffix/1"
SCHEMA = "sciel-native-response-target/1"

# Deliberately narrower than native admission: one complete lowercase Python
# fence, no surrounding prose. Unsupported layouts are rejected, not repaired.
_FENCE_LINE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})", re.MULTILINE)
_RESPONSE = re.compile(
    r"(?P<prefix>[\s\S]*?)(?P<opening>^[ \t]*```python[ \t]*\r?\n)"
    r"(?P<body>[\s\S]*?)(?P<closing>^[ \t]*```[ \t]*)"
    r"(?P<trailing>(?:\r?\n[ \t\r\n]*)?)\Z", re.MULTILINE)
_MARKER = re.compile(r"\s*HYPOTHESIS_JSON[ \t]*:")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def bind_native_response_target(*, raw_response: str, evaluated_source: str,
                                accepted_hypothesis: dict,
                                validation_attempts: list[dict],
                                response_receipt: dict) -> dict:
    """Keep the exact response; mask its forecast, supervise code format + EOS.

    ``accepted_hypothesis`` is the native Hypothesis.to_dict() admission, not a
    hindsight forecast. ``validation_attempts`` is the committed iteration's
    complete attempt list. Unique source selection mirrors the native export
    gate, but successful evaluation, improvement and distinct-source minimums
    remain the caller's responsibility. EOS is a future tokenizer operation.
    """
    _need(type(raw_response) is str and bool(raw_response.strip()), "empty native response")
    _need(type(evaluated_source) is str and bool(evaluated_source.strip()), "empty evaluated source")
    _need(type(accepted_hypothesis) is dict, "missing accepted hypothesis")
    _need(type(validation_attempts) is list and all(type(a) is dict for a in validation_attempts),
          "validation attempts must be the committed attempt list")
    _need(type(response_receipt) is dict, "missing response receipt")
    target_hash = sha256_bytes(raw_response.encode("utf-8"))
    source_hash = sha256_bytes(evaluated_source.encode("utf-8"))
    attempts = [a for a in validation_attempts if a.get("disposition") == "evaluate"
                and a.get("candidate_code_sha256") == source_hash
                and a.get("extracted_code_sha256") == source_hash]
    _need(len(attempts) == 1, "source has no unique unrepaired evaluate attempt")
    attempt = attempts[0]
    call_id = require_identifier(attempt.get("call_id"), "native target call_id")
    _need(attempt.get("generation_returned") is True, "native generation did not return")
    require_hash(attempt.get("response_sha256"), "admitted response hash")
    _need(response_receipt.get("event") == "response"
          and response_receipt.get("call_id") == call_id
          and response_receipt.get("response_sha256") == target_hash
          and attempt["response_sha256"] == target_hash,
          "raw response differs from admitted call/response commitments")
    _need(response_receipt.get("response_error") is None, "native response has a transport error")

    hypothesis = hypothesis_parser.parse_hypothesis(raw_response)
    _need(hypothesis is not None, "response lacks a valid native hypothesis")
    _need(canonical_bytes(hypothesis.to_dict()) == canonical_bytes(accepted_hypothesis),
          "response hypothesis differs from pre-evaluation admission")
    _need(code_parser.extract_code(raw_response) == evaluated_source,
          "response does not extract to the exact evaluated source")
    envelope = _RESPONSE.fullmatch(raw_response)
    _need(envelope is not None and raw_response.count("```") == 2
          and len(_FENCE_LINE.findall(raw_response)) == 2,
          "unsupported layout: require one complete Python fence")
    prefix = envelope.group("prefix")
    _need(prefix.lstrip().startswith("{") or _MARKER.match(prefix) is not None,
          "unsupported prose before the native hypothesis")
    start = envelope.start("opening")
    spans = {name: list(envelope.span(name)) for name in ("opening", "body", "closing", "trailing")}
    binding = dict(schema=SCHEMA, policy=POLICY, call_id=call_id,
        validation_attempt_sha256=content_hash(attempt),
        response_receipt_sha256=content_hash(response_receipt),
        accepted_hypothesis_sha256=content_hash(accepted_hypothesis),
        parser=dict(contract=CONTRACT, source_files={
            "hypothesis/parsing.py": sha256_bytes(Path(hypothesis_parser.__file__).read_bytes()),
            "engine/prompt/parse.py": sha256_bytes(Path(code_parser.__file__).read_bytes())}),
        span_units="unicode-code-points-half-open",
        masked_assistant_prefix=[0, start], supervised_assistant_suffix=[start, len(raw_response)],
        format_spans=spans, supervised_suffix_sha256=sha256_bytes(raw_response[start:].encode("utf-8")),
        eos="append-one-supervised-token-after-exact-response",
        tokenization="unverified-character-boundaries")
    value = dict(target=raw_response, target_sha256=target_hash,
        evaluated_source=evaluated_source, source_sha256=source_hash, target_policy=binding)
    return dict(value, sha256=content_hash(value))


def validate_native_response_target(value: dict, **inputs) -> dict:
    """Re-derive policy, spans, parser and source/response bindings from inputs."""
    _need(type(value) is dict and canonical_bytes(value) == canonical_bytes(bind_native_response_target(**inputs)),
          "native target binding differs from authenticated inputs")
    return value
