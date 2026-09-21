"""Actual saved facts with synthetic anchored local admission/aging histories.

The foreign reducer is replaced by an already-validated empty memory. The real
resume helper, versioned raw-response parser, card reconstruction and lifetime
digest check run on these detached fixtures. No Engine, provider or solver runs.
"""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.engine import external_research_resume as resume
from sciel.hypothesis import grounded_review
from sciel.hypothesis.auxiliary_schema import identity, response_format
from sciel.hypothesis.comparison_review import MAX_STORED
from sciel.knowledge_base.catalogs import canonical_sha
from test_grounded_review_actual import actual, catalog, proposed, GROUP


def _raw(proposals):
    return json.dumps(dict(lessons=[], retire_ids=[], comparison_reviews=proposals))


def _writer(proposals, through):
    return dict(phase='reflection', reference=through,
                raw_response_sha256=resume._digest(_raw(proposals)))


def _append(rows, proposals, through, retained):
    """Explicit already-accepted event receipts; this does not run admission."""
    raw = _raw(proposals)
    ids = [p['comparison_id'] for p in proposals]
    rows.extend([
        dict(event='research', phase='reflection_context', through=through,
             lesson_ids=[], comparison_review_source_ids=ids),
        dict(event='research', phase='aux_start', kind='reflection', reference=through,
             call_index=through, response_format=response_format('reflection',
                 comparison_reviews=True, comparison_review_version=2),
             auxiliary_schema=identity(comparison_reviews=True, comparison_review_version=2)),
        dict(event='research', phase='aux_result', kind='reflection', reference=through,
             call_index=through, response=raw, response_sha256=resume._digest(raw),
             response_truncated=False, structured_response_usable=True,
             finish_reason='stop', refusal=None, cost=0.0),
        dict(event='research', phase='reflection', through=through, status='parsed', error=None,
             applied={'lesson_ids': [], 'retired_ids': []}, comparison_review_ids=ids,
             external_history_transition=dict(schema='sciel-external-research-transition/1',
                 lessons_sha256=canonical_sha([]), next_lesson_id=1,
                 comparison_reviews=copy.deepcopy(retained))),
    ])


@pytest.fixture
def aged(actual, catalog, monkeypatch):
    original = NS(lessons=[], _next_lesson_id=1)
    monkeypatch.setattr(resume, 'validate', lambda *args, **kwargs: {
        'knowledge': original, 'comparison_reviews': {}})
    first = proposed(actual, catalog['facts'])
    old_card = grounded_review.build_card(first, actual.group, _writer([first], 1),
        records=actual.records, seen=actual.seen)
    assert old_card['host_facts']['facts']['restart:541090050:effective_locations_changed:verdict'] == 'CONFIRMED'
    rows = []
    _append(rows, [first], 1, {GROUP: canonical_sha(old_card)})
    # Ordinary origin has aged out, while the paired group and lifetime digest
    # remain. This fixture makes no new ordinary observation or exposure.
    c = NS(_external_history={'capsule': {}},
        config={'comparison_reviews': True, 'comparison_review_version': 2}, metric={},
        knowledge=NS(lessons=[], _next_lesson_id=1, _seen=copy.deepcopy(actual.seen),
                     _evidence=lambda: {}), _comparison_reviews={},
        _review_groups=lambda: [actual.group])
    c._retained_comparison_reviews = lambda: copy.deepcopy(c._comparison_reviews)
    return NS(c=c, rows=rows, old_card=old_card, first=first)


def _replace_with_unknown(aged, actual):
    missing = grounded_review.source_catalog(actual.group, records={}, seen=actual.seen)
    second = proposed(actual, missing['facts'])
    new_card = grounded_review.build_card(second, actual.group, _writer([second], 2),
        records={}, seen=actual.seen)
    assert new_card['origin_snapshot'] is None
    assert new_card['host_facts']['facts']['restart:541090050:effective_locations_changed:verdict'] == 'INCONCLUSIVE'
    aged.c._comparison_reviews = {GROUP: new_card}
    _append(aged.rows, [second], 2, {GROUP: canonical_sha(new_card)})
    return new_card


def test_latest_unknown_card_replaces_old_confirmed_card_after_origin_ages(aged, actual):
    current = _replace_with_unknown(aged, actual)
    resume.assert_authoring_journal(aged.c, aged.rows)
    assert aged.c._comparison_reviews == {GROUP: current}
    assert aged.c.knowledge._evidence() == {}


def test_capacity_evicted_card_does_not_need_aged_origin_to_resume(aged):
    retained = {GROUP: canonical_sha(aged.old_card)}
    # Four already-admitted batches displace the old card. Their groups have
    # subsequently aged out. Historical cards are synthetic receipt hashes:
    # only the still-retained real group's old writer could trigger this bug.
    for offset in range(0, MAX_STORED, 4):
        proposals = []
        for index in range(offset, offset + 4):
            ident = f'comparison:already_admitted_then_aged_{index}'
            proposal = copy.deepcopy(aged.first)
            proposal.update(comparison_id=ident, comparison_sha256='a' * 64,
                claims=[dict(fact_id='quality:outcome', value='mixed')], next_hypothesis=None)
            proposals.append(proposal)
            retained[ident] = canonical_sha({'synthetic_admitted_card': ident})
        retained = dict(list(retained.items())[-MAX_STORED:])
        _append(aged.rows, proposals, 2 + offset // 4, retained)
    assert GROUP not in retained and aged.c._comparison_reviews == {}
    resume.assert_authoring_journal(aged.c, aged.rows)


def test_current_card_without_matching_local_writer_is_rejected(aged, actual):
    current = _replace_with_unknown(aged, actual)
    current['writer']['raw_response_sha256'] = 'f' * 64
    # Even a matching current-state receipt must not replace raw-writer proof.
    aged.rows[-1]['external_history_transition']['comparison_reviews'][GROUP] = canonical_sha(current)
    with pytest.raises(ValueError, match='local admitted response'):
        resume.assert_authoring_journal(aged.c, aged.rows)
