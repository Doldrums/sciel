"""Bounded model assessments with typed evidence bindings, never causal proofs.

Code effects bind literal claims/sources; numerical forecasts bind an ordinary
record and its host-derived comparison. No code is executed and no semantic
equivalence, prose truth or claim coverage is inferred. Only fields actually
included in the ordinary conclusion evidence can be quoted.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .observed_facts import observed_comparison
from . import conclusion_source

LEGACY_SCHEMA = 'sciel-mechanism-review/1'
SCHEMA = 'sciel-mechanism-review/2'
GROUNDING_SCHEMA = 'sciel-mechanism-forecast-grounding/1'
EVIDENCE_KINDS = ('source_effect', 'forecast', 'unavailable')
FORECAST_TARGET = 'native_delta_interval'
MAX_REFERENCE_ID_CHARS = 128
MAX_ENTRIES = 4
MAX_SOURCE_QUOTES = 2
MAX_TEXT_CHARS = 512
MAX_REVIEW_CHARS = 4096  # Canonical escaped JSON, including field names.
ASSESSMENTS = ('directly_supported', 'contradicted', 'unknown')
SOURCE_PATHS = (
    ('controlled_change', 'parent_body'),
    ('controlled_change', 'target_body'),
    ('controlled_change', 'before', 'text'),
    ('controlled_change', 'after', 'text'),
    ('change_excerpt',),
)
SOURCE_FIELDS = tuple('.'.join(path) for path in SOURCE_PATHS)
SOURCE_QUOTE_POLICY = 'sciel-mechanism-source-quotation/1'
SOURCE_QUOTE_RULES = (
    'Copy each source quote from the exact field named in that quote. '
    'controlled_change.parent_body and target_body are operator bodies; '
    'controlled_change.before.text and after.text can be complete wrapped functions '
    'with different indentation. These fields are not interchangeable. Prefer short '
    'single-line literal substrings; use separate quotes when two lines are needed. '
    'For a multiline quote preserve every newline and space in the named field. '
    'Do not copy a body and label it as the wrapped function, normalize whitespace, '
    'or assume a matching quote in another field will be accepted. '
    'A matching but unrelated source substring does not support an assessment. '
    'Use source quotes for code effects and the observed comparison for forecast '
    'agreement; a return statement alone cannot verify a numeric forecast. '
    'Literal binding does not prove the assessment or causal explanation.'
)
_FIELDS = {'claim_quote', 'source_quotes', 'assessment', 'direct_effect', 'untested_downstream'}
_TYPED_FIELDS = _FIELDS | {'evidence_kind', 'forecast_ref'}
TYPED_RULES = (
    'Each entry has exactly claim_quote, source_quotes, assessment, direct_effect, '
    'untested_downstream, evidence_kind and forecast_ref. '
    'evidence_kind is source_effect, forecast or unavailable. For source_effect, '
    'claim_quote is a literal substring of original hypothesis.text or reasoning; '
    'forecast_ref=null. A non-unknown source_effect assessment requires exact source '
    'quotes; it remains an unverified model interpretation. For forecast, '
    'claim_quote=null and source_quotes=[]: do not certify arbitrary prose as a '
    'numeric forecast. forecast_ref must contain exactly experiment_id (this '
    'ordinary experiment ID) and target="native_delta_interval". This refers only '
    'to the supplied hypothesis predicted_low/high for the native controller score '
    'delta, never paired raw/certified sums or intermediate restart effects. '
    'Use OBSERVED_COMPARISON_JSON.forecast: position=inside requires assessment '
    'directly_supported, above/below requires contradicted, unavailable requires '
    'unknown. The host rederives these numerical facts; your free explanation and '
    'effect text remain unverified. For unavailable, forecast_ref=null, '
    'source_quotes=[] and assessment=unknown; quote the original claim literally '
    'when available, otherwise use claim_quote=null. A spans_zero interval is not '
    'negative. No source quote is needed or allowed for a forecast or unavailable entry.'
)


def source_quote_identity():
    return dict(policy=SOURCE_QUOTE_POLICY,
                rules_sha256=hashlib.sha256(SOURCE_QUOTE_RULES.encode()).hexdigest())


def identity():
    """Version both the new evidence contract and its deterministic projection."""
    return dict(schema=SCHEMA, legacy_schema=LEGACY_SCHEMA, grounding=GROUNDING_SCHEMA,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                typed_rules_sha256=hashlib.sha256(TYPED_RULES.encode()).hexdigest())


class MechanismReviewError(ValueError):
    """A fixed-code diagnostic; never include rejected model text."""

    def __init__(self, reason, *, code, path, actual=None, maximum=None):
        super().__init__(reason)
        self._diagnostic = {'code': code, 'path': path}
        if actual is not None:
            self._diagnostic['actual'] = actual
        if maximum is not None:
            self._diagnostic['maximum'] = maximum

    @property
    def diagnostic(self):
        return dict(self._diagnostic)


def _require(condition, reason, *, code, path, actual=None, maximum=None):
    if not condition:
        raise MechanismReviewError(reason, code=code, path=path, actual=actual, maximum=maximum)


def _text(value):
    return type(value) is str and 0 < len(value) <= MAX_TEXT_CHARS and bool(value.strip())


def _require_text(value, reason, path):
    _require(_text(value), reason,
             code='text_too_long' if type(value) is str and len(value) > MAX_TEXT_CHARS else 'invalid_text',
             path=path, actual=len(value) if type(value) is str else None, maximum=MAX_TEXT_CHARS)


def _identifier(value):
    return (type(value) is str and 0 < len(value) <= MAX_REFERENCE_ID_CHARS
            and not any(c.isspace() or ord(c) < 32 for c in value))


def is_typed(value):
    """Inspect an already validated homogeneous review, without upgrading legacy."""
    return bool(value and type(value[0]) is dict and 'evidence_kind' in value[0])


def validate_review(value, *, require_typed=False):
    """Validate shape only and return a detached copy; quotes retain exact bytes."""
    _require(type(require_typed) is bool, 'typed requirement must be boolean',
             code='invalid_typed_requirement', path='require_typed')
    _require(type(value) is list and 1 <= len(value) <= MAX_ENTRIES,
             'mechanism review requires one to four entries', code='invalid_entry_count',
             path='mechanism_review', actual=len(value) if type(value) is list else None, maximum=MAX_ENTRIES)
    typed = is_typed(value)
    _require(not require_typed or typed, 'typed mechanism review required',
             code='typed_mechanism_required', path='mechanism_review')
    for index, entry in enumerate(value):
        path = f'mechanism_review[{index}]'
        _require(type(entry) is dict and set(entry) == (_TYPED_FIELDS if typed else _FIELDS), 'invalid mechanism entry fields',
                 code='invalid_entry_fields', path=path)
        _require(type(entry['assessment']) is str and entry['assessment'] in ASSESSMENTS,
                 'invalid mechanism assessment', code='invalid_assessment', path=path+'.assessment')
        kind = entry.get('evidence_kind') if typed else None
        if typed:
            _require(type(kind) is str and kind in EVIDENCE_KINDS,
                     'invalid mechanism evidence kind', code='invalid_evidence_kind', path=path+'.evidence_kind')
            ref = entry['forecast_ref']
            if kind == 'forecast':
                _require(type(ref) is dict and set(ref) == {'experiment_id', 'target'}
                         and _identifier(ref['experiment_id']) and ref['target'] == FORECAST_TARGET,
                         'invalid forecast reference', code='invalid_forecast_ref', path=path+'.forecast_ref')
                _require(entry['claim_quote'] is None, 'forecast claim must use its reference, not prose',
                         code='forecast_claim_not_null', path=path+'.claim_quote')
            else:
                _require(ref is None, 'non-forecast reference must be null',
                         code='unexpected_forecast_ref', path=path+'.forecast_ref')
            if kind == 'unavailable':
                _require(entry['assessment'] == 'unknown', 'unavailable assessment must be unknown',
                         code='unavailable_assessment', path=path+'.assessment')
        if entry['claim_quote'] is None:
            _require(kind == 'forecast' or (entry['assessment'] == 'unknown' and kind != 'source_effect'),
                     'invalid mechanism claim quote',
                     code='invalid_null_claim', path=path+'.claim_quote')
        else:
            _require_text(entry['claim_quote'], 'invalid mechanism claim quote', path+'.claim_quote')
        for key in ('direct_effect', 'untested_downstream'):
            _require_text(entry[key], 'invalid mechanism effect text', path+'.'+key)
        quotes = entry['source_quotes']
        _require(type(quotes) is list and len(quotes) <= MAX_SOURCE_QUOTES
                 and ((not quotes) if kind in ('forecast', 'unavailable') else
                      (entry['assessment'] == 'unknown' or bool(quotes))),
                 'invalid mechanism source quote count', code='invalid_source_quote_count',
                 path=path+'.source_quotes', actual=len(quotes) if type(quotes) is list else None, maximum=MAX_SOURCE_QUOTES)
        for quote_index, quote in enumerate(quotes):
            quote_path = f'{path}.source_quotes[{quote_index}]'
            _require(type(quote) is dict and set(quote) == {'field', 'text'}, 'invalid mechanism source quote',
                     code='invalid_source_quote_fields', path=quote_path)
            _require(type(quote['field']) is str and quote['field'] in SOURCE_FIELDS + conclusion_source.SOURCE_FIELDS,
                     'invalid mechanism source quote', code='invalid_source_field', path=quote_path+'.field')
            _require_text(quote['text'], 'invalid mechanism source quote', quote_path+'.text')
    # Shape above contains only known string keys, strings, null and lists.
    # Keep this low-level parser dependency-free to avoid importing KB through
    # hypothesis.parsing before the hypothesis package has finished loading.
    encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(',', ':'))
    _require(len(encoded) <= MAX_REVIEW_CHARS, 'mechanism review exceeds total character limit',
             code='review_too_long', path='mechanism_review', actual=len(encoded), maximum=MAX_REVIEW_CHARS)
    return copy.deepcopy(value)


def source_fields(evidence):
    """Read only the finite, schema-owned paths visible in conclusion evidence."""
    _require(type(evidence) is dict, 'mechanism evidence must be a record',
             code='invalid_evidence', path='evidence')
    result = {}
    for path in SOURCE_PATHS:
        current = evidence
        for part in path:
            current = current.get(part) if type(current) is dict else None
        if type(current) is str and current.strip():
            result['.'.join(path)] = current
    if 'conclusion_source' in evidence:
        result.update(conclusion_source.fields(evidence['conclusion_source']))
    return result


def _forecast_assessment(comparison):
    position = comparison['forecast']['position']
    return ('directly_supported' if position == 'inside' else
            'contradicted' if position in ('above', 'below') else 'unknown')


def validate_binding(value, evidence, *, require_typed=False):
    """Bind literal claims/sources or the one explicit numerical forecast target.

    Literal matching never verifies source_effect assessments or free prose.
    Only a typed forecast assessment is checked against computed interval facts.
    """
    value = validate_review(value, require_typed=require_typed)
    fields = source_fields(evidence)
    hypothesis = evidence.get('hypothesis')
    claims = ([hypothesis.get(key) for key in ('text', 'reasoning')]
              if type(hypothesis) is dict else [])
    claims = [claim for claim in claims if type(claim) is str and claim.strip()]
    for index, entry in enumerate(value):
        path = f'mechanism_review[{index}]'
        if entry.get('evidence_kind') == 'forecast':
            ref = entry['forecast_ref']
            _require('schema' not in evidence and _identifier(evidence.get('id'))
                     and ref['experiment_id'] == evidence['id'],
                     'forecast reference does not match ordinary experiment',
                     code='forecast_ref_mismatch', path=path+'.forecast_ref')
            _require(entry['assessment'] == _forecast_assessment(observed_comparison(evidence)),
                     'forecast assessment conflicts with observed comparison',
                     code='forecast_assessment_mismatch', path=path+'.assessment')
            continue
        quote = entry['claim_quote']
        if claims:
            _require(type(quote) is str and any(quote in claim for claim in claims),
                     'mechanism claim quote does not match original hypothesis',
                     code='claim_quote_mismatch', path=path+'.claim_quote')
        else:
            _require(quote is None and entry['assessment'] == 'unknown',
                     'mechanism claim unavailable; null quote and unknown assessment required',
                     code='claim_unavailable', path=path+'.claim_quote')
        for quote_index, quote in enumerate(entry['source_quotes']):
            quote_path = f'{path}.source_quotes[{quote_index}]'
            _require(quote['field'] in fields, 'mechanism source field unavailable in evidence',
                     code='source_field_unavailable', path=quote_path+'.field')
            _require(quote['text'] in fields[quote['field']], 'mechanism source quote does not match evidence',
                     code='source_quote_mismatch', path=quote_path+'.text')
    return value


def forecast_grounding(value, evidence):
    """Return separate host facts for typed references, never certify model prose.

    Rebind against the caller's authoritative record on every presentation. Old
    untyped entries stay untyped: no retrospective claim classification is made.
    No grounding field is accepted from the provider or stored in its review.
    """
    checked = validate_binding(value, evidence)
    if not is_typed(checked):
        return None
    forecasts = []
    for index, entry in enumerate(checked):
        if entry['evidence_kind'] != 'forecast':
            continue
        comparison = observed_comparison(evidence)
        assessment = _forecast_assessment(comparison)
        forecasts.append(dict(entry_index=index, forecast_ref=copy.deepcopy(entry['forecast_ref']),
            observed_comparison=comparison, required_assessment=assessment,
            computed_verdict={'directly_supported': 'CONFIRMED', 'contradicted': 'REFUTED',
                              'unknown': 'INCONCLUSIVE'}[assessment]))
    return dict(schema=GROUNDING_SCHEMA, scope='ordinary_maximize_controller_score_only',
        interpretation_limits='Only the referenced numerical forecast agreement is rederived. '
            'All free model prose and source_effect assessments remain unverified; '
            'these are not new observations, mechanism proofs or quality promotion.',
        forecasts=forecasts)
