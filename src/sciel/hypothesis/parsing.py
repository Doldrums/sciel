"""Strict, bounded parsing of explicit research JSON blocks.

A marker starts a line (optional indentation), followed by one unfenced JSON
object on that or the next line. Prose before the marker is allowed. Only
whitespace may follow the object, except that a hypothesis may be followed by
candidate code starting at the first Markdown fence. Arbitrary fences, prose-only
fallbacks, duplicate keys, unknown fields and partial lessons are rejected.
Hypotheses additionally support an unmarked JSON object at the response start
(optional leading whitespace), immediately followed by a Python code fence.
Unmarked objects without that fence or after prose are never inferred as forecasts.
Dedicated conclusion/reflection replies also accept one bare JSON object or one
sole lowercase json fence occupying the complete response, optionally preceded
by exactly the requested marker. Its schema is identical; surrounding prose,
mismatched or repeated markers and multiple fences are not repaired.
Lessons may additionally carry branch: a string ID for branch scope, null for global.
Parsing does not validate evidence membership or authorize KB mutations.
"""

from __future__ import annotations

import json
import math
import re

from .models import Conclusion, Hypothesis, Reflection
from .mechanism_review import MechanismReviewError, validate_review

MAX_RESPONSE_CHARS = 262144
MAX_BLOCK_CHARS = 32768
MAX_TEXT_CHARS = 2048
MAX_REASONING_CHARS = 4096
MAX_LESSONS = 8
MAX_EVIDENCE_IDS = 32
MAX_RETIRE_IDS = 32
MAX_ID_CHARS = 128
AUX_TRANSPORT = "sciel-aux-json-transport/3"

_FENCE = re.compile(r"^[ \t]*(?:`{3,}|~{3,})", re.MULTILINE)
_PYTHON_FENCE = re.compile(r"[ \t]*```python[ \t]*\r?(?:\n|$)")
_JSON_MARKER = re.compile(r"^[ \t]*([A-Z][A-Z0-9_]*_JSON)[ \t]*:", re.MULTILINE)
_SOLE_JSON_FENCE = re.compile(r"\s*```json\r?\n([\s\S]*?)\r?\n```\s*")
_MARKED_JSON_FENCE = re.compile(
    r"\s*([A-Z][A-Z0-9_]*_JSON)[ \t]*:\s*```json\r?\n([\s\S]*?)\r?\n```\s*")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


_DECODER = json.JSONDecoder(object_pairs_hook=_unique_object,
                            parse_constant=_reject_constant)


def _block(response, marker, *, candidate_code=False, bare_python=False, bare_object=False, json_fence=False):
    if type(response) is not str or len(response) > MAX_RESPONSE_CHARS:
        return None
    if json_fence:
        envelope = _SOLE_JSON_FENCE.fullmatch(response)
        marked = _MARKED_JSON_FENCE.fullmatch(response) if envelope is None else None
        if marked is not None and marked.group(1) != marker:
            return None
        if envelope is not None or marked is not None:
            response = envelope.group(1) if envelope is not None else marked.group(2)
            # The entire envelope is explicit; inner JSON bytes are not repaired.
            if not response.lstrip().startswith("{"):
                return None
    fence = _FENCE.search(response)
    if fence and not candidate_code:
        return None
    prefix = response[:fence.start()] if fence else response
    if bare_object and response.lstrip().startswith("{"):
        body = response.strip()
    elif bare_python and response.lstrip().startswith("{"):
        if fence is None or _PYTHON_FENCE.match(response, fence.start()) is None:
            return None
        body = prefix.strip()
    else:
        # A different protocol block before the requested one is not prose.
        # Candidate code is outside prefix; markers inside JSON strings are
        # escaped/on quoted lines, so neither is mistaken for a wire block.
        matches = list(_JSON_MARKER.finditer(prefix))
        if len(matches) != 1 or matches[0].group(1) != marker:
            return None
        body = prefix[matches[0].end():].strip()
    if not body.startswith("{") or len(body) > MAX_BLOCK_CHARS:
        return None
    try:
        value, end = _DECODER.raw_decode(body)
    except (ValueError, RecursionError):
        return None
    if body[end:].strip() or type(value) is not dict:
        return None
    return value


def _text(value, limit):
    return type(value) is str and 0 < len(value) <= limit and bool(value.strip())


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _ids(value, limit, *, nonempty=False):
    return (type(value) is list and (not nonempty or bool(value)) and
            len(value) <= limit and
            all(_text(item, MAX_ID_CHARS) and
                not any(char.isspace() or ord(char) < 32 for char in item)
                for item in value) and len(set(value)) == len(value))


def _lessons(value):
    if type(value) is not list or len(value) > MAX_LESSONS:
        return False
    required = {"rule", "polarity", "scope", "evidence_ids"}
    return all(type(item) is dict and
               required <= set(item) <= required | {"branch"} and
               _text(item["rule"], MAX_TEXT_CHARS) and
               item["polarity"] in ("positive", "negative") and
               item["scope"] in ("branch", "global") and
               _ids(item["evidence_ids"], MAX_EVIDENCE_IDS, nonempty=True) and
               ("branch" not in item or
                (item["scope"] == "branch" and _ids([item["branch"]], 1)) or
                (item["scope"] == "global" and item["branch"] is None))
               for item in value)


def parse_hypothesis(response: str) -> Hypothesis | None:
    """Read a forecast before any candidate fence; never invent missing values."""
    value = _block(response, "HYPOTHESIS_JSON", candidate_code=True, bare_python=True)
    return parse_hypothesis_value(value)


def parse_hypothesis_value(value) -> Hypothesis | None:
    """Validate five wire fields and the explicit optional restart predictions.

    This does not discover a hypothesis in arbitrary data or accept arbitrary fields.
    The legacy marker/fence transport remains owned by parse_hypothesis.
    """
    fields = {"hypothesis", "predicted_low", "predicted_high", "reasoning", "confidence"}
    if type(value) is not dict or set(value) not in (fields, fields | {'restart_predictions'}):
        return None
    if not (_text(value["hypothesis"], MAX_TEXT_CHARS) and
            _text(value["reasoning"], MAX_REASONING_CHARS) and
            all(_number(value[key]) for key in ("predicted_low", "predicted_high", "confidence")) and
            value["predicted_low"] <= value["predicted_high"] and
            0 <= value["confidence"] <= 1):
        return None
    predictions = None
    if 'restart_predictions' in value:
        from .restart_predictions import validate_spec
        try:
            predictions = validate_spec(value['restart_predictions'])
        except (ValueError, TypeError, OverflowError, RecursionError):
            return None
    return Hypothesis(value["hypothesis"].strip(), float(value["predicted_low"]),
                      float(value["predicted_high"]), value["reasoning"].strip(),
                      float(value["confidence"]), predictions)


def decode_conclusion(response: str, *, require_mechanism: bool = False,
                      require_typed: bool = False) -> dict:
    """Return the parsed conclusion or one bounded, deterministic diagnostic.

    Rejected text is never echoed and no transport/content is repaired. Binding
    against the original evidence remains separate from this shape decoder.
    """
    def rejected(code, path):
        return {'conclusion': None, 'error': {'code': code, 'path': path}}

    value = _block(response, "CONCLUSION_JSON", bare_object=True, json_fence=True)
    fields = {"verdict", "explanation", "lessons"}
    if type(require_mechanism) is not bool:
        return rejected('invalid_mechanism_requirement', 'require_mechanism')
    if type(require_typed) is not bool:
        return rejected('invalid_typed_requirement', 'require_typed')
    if value is None:
        return rejected('invalid_conclusion_transport', 'response')
    if set(value) not in (fields, fields | {"mechanism_review"}):
        return rejected('invalid_conclusion_fields', 'conclusion')
    if (require_mechanism or require_typed) and "mechanism_review" not in value:
        return rejected('missing_mechanism_review', 'mechanism_review')
    if not (value["verdict"] in ("CONFIRMED", "REFUTED", "INCONCLUSIVE") and
            _text(value["explanation"], MAX_REASONING_CHARS) and _lessons(value["lessons"])):
        return rejected('invalid_conclusion_payload', 'conclusion')
    mechanism = None
    if "mechanism_review" in value:
        try:
            mechanism = validate_review(value["mechanism_review"], require_typed=require_typed)
        except MechanismReviewError as exc:
            return {'conclusion': None, 'error': exc.diagnostic}
        except (ValueError, TypeError, RecursionError):
            return rejected('invalid_mechanism_review', 'mechanism_review')
    return {'conclusion': Conclusion(value["verdict"], value["explanation"].strip(), value["lessons"], mechanism),
            'error': None}


def parse_conclusion(response: str, *, require_mechanism: bool = False,
                     require_typed: bool = False) -> Conclusion | None:
    """Compatibility API: accept exactly the same legacy/current wire shapes."""
    return decode_conclusion(response, require_mechanism=require_mechanism,
                             require_typed=require_typed)['conclusion']


def decode_reflection(response: str, *, comparison_reviews: bool = False,
                      comparison_review_version: int = 1) -> dict:
    """Return the parsed reflection or a bounded, deterministic diagnostic.

    Rejected text is never echoed or repaired. Shape acceptance does not establish
    evidence binding, truth, or eligibility as knowledge-base lessons.
    """
    def rejected(code, path):
        return {'reflection': None, 'error': {'code': code, 'path': path}}

    if type(comparison_reviews) is not bool:
        return rejected('invalid_comparison_reviews_requirement', 'comparison_reviews')
    if type(comparison_review_version) is not int or comparison_review_version not in (1, 2, 3):
        return rejected('invalid_comparison_review_version', 'comparison_review_version')
    if comparison_review_version in (2, 3) and not comparison_reviews:
        return rejected('comparison_reviews_required', 'comparison_reviews')
    value = _block(response, "REFLECTION_JSON", bare_object=True, json_fence=True)
    fields = {"lessons", "retire_ids"}
    if comparison_reviews:
        fields.add("comparison_reviews")
    if value is None:
        return rejected('invalid_reflection_transport', 'response')
    if set(value) != fields:
        return rejected('invalid_reflection_fields', 'reflection')
    if not _lessons(value["lessons"]):
        return rejected('invalid_reflection_lessons', 'lessons')
    if not _ids(value["retire_ids"], MAX_RETIRE_IDS):
        return rejected('invalid_reflection_retire_ids', 'retire_ids')
    reviews = None
    if comparison_reviews:
        from .reasoned_review import ReasonedReviewError
        if comparison_review_version == 3:
            from .reasoned_review import validate_proposals
        elif comparison_review_version == 2:
            from .grounded_review import validate_proposals
        else:
            from .comparison_review import validate_proposals
        try:
            reviews = validate_proposals(value["comparison_reviews"])
        except ReasonedReviewError as exc:
            return {'reflection': None, 'error': exc.diagnostic}
        except (ValueError, TypeError, OverflowError, RecursionError):
            return rejected('invalid_comparison_reviews', 'comparison_reviews')
    return {'reflection': Reflection(value["lessons"], value["retire_ids"], reviews), 'error': None}


def parse_reflection(response: str, *, comparison_reviews: bool = False,
                     comparison_review_version: int = 1) -> Reflection | None:
    """Compatibility API: accept exactly the same legacy/current wire shapes."""
    return decode_reflection(response, comparison_reviews=comparison_reviews,
                             comparison_review_version=comparison_review_version)['reflection']
