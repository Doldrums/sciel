"""Reasoned proposal/abstention through the real controller with inert replies."""
import copy

import pytest

from sciel.engine.research import ResearchController
from sciel.hypothesis import reasoned_review
from sciel.knowledge_base.catalogs import canonical_json
from test_comparison_review_flow import fixture, restored, reflection_event
from test_comparison_review_text_flow import common_review
from test_grounded_review_flow import grounded_proposal
from test_research_measurement_state import begin_fixture
from test_research_paired_comparisons import decoded_context, proposal as lesson_proposal


ABSTENTION = 'The retained measurements do not identify which trial won; no bounded source change is specified by this inert fixture.'
RATIONALE = 'Test whether changing only reinsertion order affects the paired result; the mixed measurements do not establish its benefit.'


def reasoned_fixture(*, abstain=False, **settings):
    f = fixture(comparison_review_version=3, **settings)
    proposal = grounded_proposal(f.group, f.c.knowledge)
    proposal['next_hypothesis_reason'] = ABSTENTION if abstain else RATIONALE
    if abstain:
        proposal['next_hypothesis'] = None
    f.reply['comparison_reviews'] = [proposal]
    return f


@pytest.mark.parametrize('abstain', [False, True])
def test_explicit_decision_survives_checkpoint_without_creating_outcomes_or_dispatch(abstain):
    f = reasoned_fixture(abstain=abstain)
    before = f.c.knowledge.state()
    f.c.complete_step()
    assert reflection_event(f)['error'] is None
    assert f.calls[-1]['options']['response_format']['json_schema']['name'] == 'sciel_reflection_v4'
    card = f.c.state()['comparison_reviews'][f.group['id']]
    assert card['schema'] == reasoned_review.CARD_SCHEMA
    assert card['unverified_interpretation']['next_hypothesis_reason'] == (ABSTENTION if abstain else RATIONALE)
    assert (card['unverified_interpretation']['next_hypothesis'] is None) is abstain
    clone = restored(f)
    assert clone.state() == f.c.state()
    assert clone.context(f.db, f.parent, 0, 2) == f.c.context(f.db, f.parent, 0, 2)
    assert f.c.knowledge.state() == before and f.c.reflection_changes == 0
    assert f.c.attempts == f.c.outcomes == 1
    # The normal reflection boundary cannot invent a G or retry from a decision.
    # A subsequent search iteration is separate; this is not a semantic gate.
    assert len(f.calls) == 2 and f.c.aux_calls == 2  # inert C and R only


@pytest.mark.parametrize('damage', ['missing', 'blank', 'oversized'])
def test_invalid_reason_cannot_partially_change_lessons(damage):
    f = reasoned_fixture(abstain=True)
    old, = f.c.knowledge.add_lessons([lesson_proposal(f.record['id'])], allowed_evidence_ids=[f.record['id']])
    replacement = lesson_proposal(f.record['id'])
    replacement['rule'] = 'A replacement supported only by the saved ordinary observation.'
    f.reply.update(lessons=[replacement], retire_ids=[old])
    proposal = f.reply['comparison_reviews'][0]
    if damage == 'missing':
        proposal.pop('next_hypothesis_reason')
    else:
        proposal['next_hypothesis_reason'] = ' ' if damage == 'blank' else 'x' * 1025
    before = canonical_json(f.c.knowledge.state())
    f.c.complete_step()
    assert canonical_json(f.c.knowledge.state()) == before
    assert f.c.state()['comparison_reviews'] == {}
    assert f.c.reflections == 0 and len(f.calls) == 2


@pytest.mark.parametrize('abstain', [False, True])
def test_reason_and_target_are_removed_together_without_reallocating_context(abstain):
    arms = [reasoned_fixture(abstain=abstain, comparison_review_text=enabled) for enabled in (True, False)]
    contexts = []
    for f in arms:
        f.c.complete_step()
        contexts.append(f.c.context(f.db, f.parent, 0, 2))
    a, b = copy.deepcopy(contexts)
    on, off = a.pop('comparison_reviews'), b.pop('comparison_reviews')
    assert on['cards'][0]['unverified_interpretation']['next_hypothesis_reason']
    assert off['cards'][0]['unverified_interpretation'] is None
    assert common_review(on) == common_review(off)
    assert on['reserved_chars'] == off['reserved_chars']
    assert canonical_json(a) == canonical_json(b)
    for index, (f, context) in enumerate(zip(arms, contexts)):
        _, prompt = begin_fixture(f.c, context)
        assert decoded_context(prompt)['comparison_reviews'] == context['comparison_reviews']
        reason = f.reply['comparison_reviews'][0]['next_hypothesis_reason']
        assert (reason in prompt) is (index == 0)
        assert len(f.calls) == 2  # Request preparation is not dispatch.


def test_forged_reason_rejected_against_authoritative_card_before_attempt():
    f = reasoned_fixture(abstain=True)
    f.c.complete_step()
    context = f.c.context(f.db, f.parent, 0, 2)
    context['comparison_reviews']['cards'][0]['unverified_interpretation']['next_hypothesis_reason'] = 'A different valid-shaped reason.'
    before = f.c.state()
    with pytest.raises(ValueError, match='comparison review delivery'):
        begin_fixture(f.c, context)
    assert f.c.state() == before


@pytest.mark.parametrize('receiving', [1, 2])
def test_new_card_rejected_under_older_receiving_checkpoint_policy(receiving):
    f = reasoned_fixture(abstain=True)
    f.c.complete_step()
    state = f.c.state()
    # Deliberately forge the surrounding checkpoint config/identity to exercise
    # the card gate as well as the normal exact-config checkpoint boundary.
    old = fixture(comparison_review_version=receiving)
    state['config'] = copy.deepcopy(old.c.config)
    state['auxiliary_schema'] = old.c.state()['auxiliary_schema']
    with pytest.raises(ValueError, match='receiving version 3'):
        old.c.load_state(state)


def test_version_upgrade_is_not_a_silent_checkpoint_migration():
    old = fixture(comparison_review_version=2)
    new = reasoned_fixture(abstain=True)
    with pytest.raises(ValueError, match='configuration|config'):
        new.c.load_state(old.c.state())
