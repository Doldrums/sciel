"""Grounded review v3 adds a reason for proposing a test or abstaining.

The nullable target and all factual/source gates are the unchanged v2 contract.
Reasons remain unverified prose; validating a card cannot authenticate its model
writer without the caller's raw journal, or authorize an experiment.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from . import grounded_review
from ..knowledge_base.catalogs import canonical_json

CARD_SCHEMA = 'sciel-comparison-review-card/3'
CATALOG_SCHEMA = grounded_review.CATALOG_SCHEMA
PRESENTATION_SCHEMA = 'sciel-comparison-review-grounded-presentation/2'
MAX_CLAIMS = grounded_review.MAX_CLAIMS
MAX_ORIGIN_CHARS = grounded_review.MAX_ORIGIN_CHARS
MAX_PROPOSALS = grounded_review.MAX_PROPOSALS
MAX_PROPOSAL_CHARS = grounded_review.MAX_PROPOSAL_CHARS
MAX_REASON_CHARS = 1024
MODEL_FIELDS = grounded_review.MODEL_FIELDS + ('next_hypothesis_reason',)
_FIELDS = {'comparison_id', 'comparison_sha256', *MODEL_FIELDS}
RULES = grounded_review.RULES + (
    ' next_hypothesis_reason is also unverified model prose, including factual assertions '
    'or claims that information is missing. A null next_hypothesis is an abstention, not '
    'a failed measurement or proof that no useful test exists. A non-null target is only '
    'a proposed test, not a guarantee it will pass. Neither decision nor its reason '
    'authorizes generation, evaluation, retry, lesson support or promotion.'
)


def _require(ok, message):
    if not ok:
        raise ValueError('reasoned comparison review: ' + message)


class ReasonedReviewError(ValueError):
    """A bounded reason-field diagnostic containing no rejected model text."""

    def __init__(self, index, value):
        super().__init__('reasoned comparison review: invalid next hypothesis reason')
        self._diagnostic = {
            'code': 'text_too_long' if type(value) is str and len(value) > MAX_REASON_CHARS else 'invalid_text',
            'path': f'comparison_reviews[{index}].next_hypothesis_reason',
            'maximum': MAX_REASON_CHARS,
        }
        if type(value) is str:
            self._diagnostic['actual'] = len(value)

    @property
    def diagnostic(self):
        return dict(self._diagnostic)


def _base_proposal(proposal):
    result = copy.deepcopy(proposal)
    result.pop('next_hypothesis_reason')
    return result


def validate_proposals(value):
    """Validate reasoned decisions, including the full list's character budget."""
    from .comparison_review import _text
    _require(type(value) is list and len(value) <= MAX_PROPOSALS, 'invalid proposal list')
    for index, proposal in enumerate(value):
        _require(type(proposal) is dict and set(proposal) == _FIELDS, 'invalid proposal fields')
        if not _text(proposal['next_hypothesis_reason'], MAX_REASON_CHARS):
            raise ReasonedReviewError(index, proposal['next_hypothesis_reason'])
    grounded_review.validate_proposals([_base_proposal(p) for p in value])
    _require(len(canonical_json(value)) <= MAX_PROPOSAL_CHARS, 'proposal character limit exceeded')
    return copy.deepcopy(value)


def source_catalog(group, *, records=None, seen=None, origin_snapshot=None):
    return grounded_review.source_catalog(group, records=records, seen=seen,
                                           origin_snapshot=origin_snapshot)


def build_card(proposal, group, writer, *, records=None, seen=None):
    proposal = validate_proposals([proposal])[0]
    result = grounded_review.build_card(_base_proposal(proposal), group, writer,
                                        records=records, seen=seen)
    result['schema'] = CARD_SCHEMA
    result['unverified_interpretation']['next_hypothesis_reason'] = proposal['next_hypothesis_reason']
    return result


def validate_card(card, group, *, seen=None):
    """Rebuild the bound facts; raw-writer authentication remains the caller's job."""
    _require(type(card) is dict and card.get('schema') == CARD_SCHEMA, 'unsupported card schema')
    text = card.get('unverified_interpretation')
    _require(type(text) is dict and set(text) == set(MODEL_FIELDS), 'invalid retained proposal')
    validate_proposals([dict(comparison_id=card.get('comparison_id'),
                             comparison_sha256=card.get('comparison_sha256'), **text)])
    base = copy.deepcopy(card)
    base['schema'] = grounded_review.CARD_SCHEMA
    reason = base['unverified_interpretation'].pop('next_hypothesis_reason')
    result = grounded_review.validate_card(base, group, seen=seen)
    result['schema'] = CARD_SCHEMA
    result['unverified_interpretation']['next_hypothesis_reason'] = reason
    _require(canonical_json(card) == canonical_json(result), 'retained card differs from bound sources')
    return result


def presentation_card(card):
    """Keep the reason in the wholly maskable interpretation, never in host facts."""
    result = grounded_review.presentation_card(card)
    result['schema'] = PRESENTATION_SCHEMA
    result['rules'] = RULES
    return result


def source_block(groups, *, records=None, seen=None):
    block = grounded_review.source_block(groups, records=records, seen=seen)
    return RULES + block[len(grounded_review.RULES):]


def json_schema():
    result = grounded_review.json_schema()
    proposal = result['items']
    proposal['properties']['next_hypothesis_reason'] = dict(type='string', description=(
        'Nonblank, at most 1024 characters, enforced locally. For a non-null target explain '
        'the evidence-linked rationale and uncertainty to test. For null explain the '
        'concrete obstacle or missing information. This reason is unverified prose; '
        'do not invent a target merely to avoid abstaining.'))
    proposal['required'].append('next_hypothesis_reason')
    result['description'] = ('At most 4 distinct comparison reviews and 16384 canonical JSON '
        'characters including reasons, enforced locally. ' + RULES)
    return result


def identity():
    return dict(card_schema=CARD_SCHEMA, catalog_schema=CATALOG_SCHEMA,
        presentation_schema=PRESENTATION_SCHEMA,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        grounded_policy=grounded_review.identity(), max_reason_chars=MAX_REASON_CHARS,
        max_proposal_chars=MAX_PROPOSAL_CHARS)
