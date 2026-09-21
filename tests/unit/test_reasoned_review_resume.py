"""Raw-writer and origin-aging regressions for explicit next-step reasons.

Uses saved facts and synthetic local journals; the already-validated foreign
base is inert. No model, evaluator, candidate or solver is executed.
"""
import copy

import pytest

from sciel.engine import external_research_resume as resume
from sciel.hypothesis import reasoned_review
from sciel.hypothesis.auxiliary_schema import identity, response_format
from sciel.knowledge_base.catalogs import canonical_sha
from test_grounded_review_resume import aged, actual, catalog, _append, _writer
from test_grounded_review_actual import proposed, GROUP


def append_reasoned(rows, proposal, through, card):
    start = len(rows)
    _append(rows, [proposal], through, {GROUP: canonical_sha(card)})
    request = next(row for row in rows[start:] if row['phase'] == 'aux_start')
    request['response_format'] = response_format('reflection', comparison_reviews=True, comparison_review_version=3)
    request['auxiliary_schema'] = identity(comparison_reviews=True, comparison_review_version=3)


def replace_card(aged, actual, *, origin_available):
    aged.c.config = {'comparison_reviews': True, 'comparison_review_version': 3}
    records = actual.records if origin_available else {}
    facts = reasoned_review.source_catalog(actual.group, records=records, seen=actual.seen)
    proposal = proposed(actual, facts['facts'])
    proposal.update(next_hypothesis=None,
                    next_hypothesis_reason='This inert refusal identifies no justified bounded source edit from these measurements.')
    card = reasoned_review.build_card(proposal, actual.group, _writer([proposal], 2),
                                     records=records, seen=actual.seen)
    aged.c._comparison_reviews = {GROUP: card}
    append_reasoned(aged.rows, proposal, 2, card)
    return card


@pytest.mark.parametrize('origin_available', [False, True])
def test_new_retained_writer_owns_its_admission_time_origin_and_reason(aged, actual, origin_available):
    card = replace_card(aged, actual, origin_available=origin_available)
    assert (card['origin_snapshot'] is not None) is origin_available
    assert aged.c.knowledge._evidence() == {}
    before = copy.deepcopy(card)
    resume.assert_authoring_journal(aged.c, aged.rows)
    assert aged.c._comparison_reviews[GROUP] == before
    assert aged.c.knowledge._evidence() == {}


def test_valid_shaped_reason_forgery_rejected_even_with_matching_state_digest(aged, actual):
    card = replace_card(aged, actual, origin_available=True)
    card['unverified_interpretation']['next_hypothesis_reason'] = 'A syntactically valid but unauthored replacement reason.'
    # Rehashing current memory does not change what the raw model reply authored.
    aged.rows[-1]['external_history_transition']['comparison_reviews'][GROUP] = canonical_sha(card)
    with pytest.raises(ValueError, match='no local admitted response'):
        resume.assert_authoring_journal(aged.c, aged.rows)


def test_aged_reasoned_origin_cannot_be_replaced_by_another_snapshot(aged, actual):
    card = replace_card(aged, actual, origin_available=True)
    card['origin_snapshot']['hypothesis']['reasoning'] = 'Forged historical reasoning.'
    aged.rows[-1]['external_history_transition']['comparison_reviews'][GROUP] = canonical_sha(card)
    with pytest.raises(ValueError, match='lifetime observation'):
        resume.assert_authoring_journal(aged.c, aged.rows)


@pytest.mark.parametrize('receiving,enabled,expected', [
    (1, True, 'receiving version 3'),
    (2, True, 'receiving version 3'),
    (3, False, 'enabled receiving policy'),
    (True, True, 'unsupported receiving'),
    (4, True, 'unsupported receiving'),
])
def test_empty_local_new_wire_still_requires_supported_receiving_policy(aged, receiving, enabled, expected):
    aged.rows.clear()
    aged.c.config = {'comparison_reviews': enabled, 'comparison_review_version': receiving}
    _append(aged.rows, [], 1, {})
    request = next(row for row in aged.rows if row['phase'] == 'aux_start')
    request['response_format'] = response_format('reflection', comparison_reviews=True, comparison_review_version=3)
    request['auxiliary_schema'] = identity(comparison_reviews=True, comparison_review_version=3)
    assert aged.c._comparison_reviews == {}
    with pytest.raises(ValueError, match=expected):
        resume.assert_authoring_journal(aged.c, aged.rows)


def test_empty_local_new_wire_is_valid_under_version_three(aged):
    aged.rows.clear()
    aged.c.config = {'comparison_reviews': True, 'comparison_review_version': 3}
    _append(aged.rows, [], 1, {})
    request = next(row for row in aged.rows if row['phase'] == 'aux_start')
    request['response_format'] = response_format('reflection', comparison_reviews=True, comparison_review_version=3)
    request['auxiliary_schema'] = identity(comparison_reviews=True, comparison_review_version=3)
    resume.assert_authoring_journal(aged.c, aged.rows)
    assert aged.c._comparison_reviews == {}


def test_evicted_new_card_cannot_hide_its_newer_historical_wire(aged, actual):
    aged.rows.clear()
    replace_card(aged, actual, origin_available=True)
    aged.c._comparison_reviews.clear()
    aged.c._review_groups = lambda: []
    aged.c.config = {'comparison_reviews': True, 'comparison_review_version': 2}
    with pytest.raises(ValueError, match='receiving version 3'):
        resume.assert_authoring_journal(aged.c, aged.rows)
