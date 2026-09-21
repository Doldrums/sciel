"""Host-issued, exact-artifact decisions about comparison review text delivery.

The policy is trusted host configuration, never a model response. Audit and
criteria hashes record provenance; they do not authenticate an issuer, inspect
an audit, prove prose true, or authorize an experiment. Callers still validate
cards against complete groups and authenticate their writers against raw journals.
Only ``identity`` reads a file; all admission helpers are pure.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re

from ..knowledge_base.catalogs import canonical_sha


POLICY_SCHEMA = 'sciel-review-text-policy/1'
DECISION_SCHEMA = 'sciel-review-text-decision/1'
SCOPE = 'comparison_review_text'
MAX_DECISIONS = 64
MAX_ISSUER_CHARS = 128
MAX_ID_CHARS = 128
_POLICY_FIELDS = {'schema', 'decisions'}
_BINDING_FIELDS = ('comparison_id', 'comparison_sha256',
                   'comparator_source_sha256', 'challenger_source_sha256')
_SHA_FIELDS = ('card_sha256', 'raw_response_sha256', 'comparison_sha256',
               'comparator_source_sha256', 'challenger_source_sha256',
               'policy_sha256', 'audit_sha256', 'criteria_sha256')
_DECISION_FIELDS = {'schema', 'scope', 'decision', 'comparison_id', 'issuer', *_SHA_FIELDS}
_CARD_SCHEMAS = (tuple(f'sciel-comparison-review-card/{version}' for version in (1, 2, 3))
                 + ('sciel-generated-paired-review-card/1', 'sciel-generated-paired-review-card/2',
                    'sciel-evaluated-study-review-card/1'))


def _require(condition, message):
    if not condition:
        raise ValueError('review text eligibility: ' + message)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _id(value):
    return (type(value) is str and 0 < len(value) <= MAX_ID_CHARS
            and not any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value))


def validate_policy(value):
    """Return a detached exact host policy; no implicit or conflicting decisions."""
    _require(type(value) is dict and set(value) == _POLICY_FIELDS,
             'invalid policy fields')
    _require(type(value['schema']) is str and value['schema'] == POLICY_SCHEMA,
             'unsupported policy schema')
    decisions = value['decisions']
    _require(type(decisions) is list and len(decisions) <= MAX_DECISIONS,
             'invalid decision list')
    seen = set()
    for decision in decisions:
        _require(type(decision) is dict and set(decision) == _DECISION_FIELDS,
                 'invalid decision fields')
        _require(type(decision['schema']) is str and decision['schema'] == DECISION_SCHEMA,
                 'unsupported decision schema')
        _require(type(decision['scope']) is str and decision['scope'] == SCOPE,
                 'unsupported decision scope')
        _require(type(decision['decision']) is str and decision['decision'] in ('accepted', 'rejected'),
                 'invalid decision status')
        for key in _SHA_FIELDS:
            _require(_sha(decision[key]), 'invalid ' + key)
        _require(_id(decision['comparison_id']), 'invalid comparison ID')
        issuer = decision['issuer']
        _require(type(issuer) is str and 0 < len(issuer) <= MAX_ISSUER_CHARS and bool(issuer.strip()),
                 'invalid issuer')
        _require(decision['card_sha256'] not in seen, 'duplicate card decision')
        seen.add(decision['card_sha256'])
    return copy.deepcopy(value)


def decision_for(card, policy, *, policy_sha256):
    """Resolve an exact stored card to detached host delivery metadata.

    A different card, including revised prose or writer, is pending. A matching
    card with mismatched redundant bindings is invalid even when rejected or
    stale. A different policy version never inherits acceptance. This checks
    binding fields, not the complete card/group or its raw writer authenticity.
    """
    policy = validate_policy(policy)
    _require(_sha(policy_sha256), 'invalid current policy digest')
    _require(type(card) is dict and type(card.get('schema')) is str
             and card['schema'] in _CARD_SCHEMAS, 'unsupported stored card schema')
    _require(_id(card.get('comparison_id')), 'invalid card comparison ID')
    for key in _BINDING_FIELDS[1:]:
        _require(_sha(card.get(key)), 'invalid card ' + key)
    writer = card.get('writer')
    _require(type(writer) is dict and set(writer) == {'phase', 'reference', 'raw_response_sha256'}
             and type(writer['phase']) is str and writer['phase'] == 'reflection'
             and type(writer['reference']) is int and writer['reference'] > 0
             and _sha(writer['raw_response_sha256']), 'invalid card writer')
    digest = canonical_sha(card)
    decision = next((item for item in policy['decisions'] if item['card_sha256'] == digest), None)
    if decision is None:
        return {'status': 'pending', 'decision_sha256': None}
    _require(all(decision[key] == card[key] for key in _BINDING_FIELDS)
             and decision['raw_response_sha256'] == writer['raw_response_sha256'],
             'decision binding differs from exact card')
    status = decision['decision'] if decision['policy_sha256'] == policy_sha256 else 'stale'
    return {'status': status, 'decision_sha256': canonical_sha(decision)}


def require_review_only(result):
    """Reject a whole parsed R with lesson effects before any caller mutation."""
    lessons, retired = getattr(result, 'lessons', None), getattr(result, 'retire_ids', None)
    _require(type(lessons) is list and not lessons and type(retired) is list and not retired,
             'review-only reflection requires empty lessons and retire_ids')


def identity():
    """Bind this admission contract to its schema shape and source bytes."""
    return dict(schema=POLICY_SCHEMA, decision_schema=DECISION_SCHEMA, scope=SCOPE,
        max_decisions=MAX_DECISIONS, max_issuer_chars=MAX_ISSUER_CHARS,
        max_id_chars=MAX_ID_CHARS,
        schema_sha256=canonical_sha({'policy_fields': sorted(_POLICY_FIELDS),
                                     'decision_fields': sorted(_DECISION_FIELDS)}),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
