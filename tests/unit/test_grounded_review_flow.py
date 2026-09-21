"""Controller, checkpoint and prompt flow with inert model responses only."""
import copy
import json

import pytest

from sciel.hypothesis import grounded_review
from sciel.hypothesis.comparison_review import validate_cards
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from test_comparison_review_flow import fixture, restored, reflection_event
from test_comparison_review_text_flow import common_review
from test_research_measurement_state import begin_fixture
from test_research_paired_comparisons import decoded_context, proposal as lesson_proposal


def grounded_proposal(group, knowledge):
    facts = grounded_review.source_catalog(group, records=knowledge._evidence(), seen=knowledge._seen)
    return dict(comparison_id=group['id'], comparison_sha256=canonical_sha(group),
        claims=[dict(fact_id='quality:outcome', value=facts['facts']['quality:outcome'])],
        conjecture='A different reinsertion order may affect the resulting packing.',
        limitations='The causal hypothesis is untested; forecast agreement would not prove it.',
        next_hypothesis=dict(schema='sciel-comparison-target/1',
            base_source_sha256=group['challenger_source_sha256'],
            comparator_source_sha256=group['comparator_source_sha256'],
            change='Change only the reinsertion order.',
            quality_gate='paired_raw_exact_zero_loss_strict_v1', diagnostics=[]))


def grounded_fixture(**settings):
    f = fixture(comparison_review_version=2, **settings)
    f.reply['comparison_reviews'] = [grounded_proposal(f.group, f.c.knowledge)]
    return f


def test_grounded_reflection_checkpoint_and_next_prompt_preserve_host_facts():
    f = grounded_fixture()
    before = f.c.knowledge.state()
    f.c.complete_step()
    assert reflection_event(f)['error'] is None
    assert f.calls[-1]['options']['response_format']['json_schema']['name'] == 'sciel_reflection_v3'
    card = f.c.state()['comparison_reviews'][f.group['id']]
    assert card['schema'] == grounded_review.CARD_SCHEMA
    assert canonical_sha(card['origin_snapshot']) == f.c.knowledge._seen[f.record['id']]
    assert card['unverified_interpretation'] == {k: f.reply['comparison_reviews'][0][k]
                                               for k in grounded_review.MODEL_FIELDS}
    clone = restored(f)
    assert clone.state() == f.c.state()
    context = f.c.context(f.db, f.parent, 0, 2)
    assert context == clone.context(f.db, f.parent, 0, 2)
    presented, = context['comparison_reviews']['cards']
    assert 'origin_snapshot' not in presented
    assert presented['host_facts'] == card['host_facts']
    _, prompt = begin_fixture(f.c, context)
    assert decoded_context(prompt)['comparison_reviews'] == context['comparison_reviews']
    assert f.c.knowledge.state() == before
    assert len(f.calls) == 2  # Only inert C and R, no G dispatch.


@pytest.mark.parametrize('damage', ['fact', 'seed', 'version', 'target', 'origin'])
def test_bad_review_does_not_partially_retire_or_learn(damage):
    f = grounded_fixture()
    old, = f.c.knowledge.add_lessons([lesson_proposal(f.record['id'])], allowed_evidence_ids=[f.record['id']])
    replacement = lesson_proposal(f.record['id'])
    replacement['rule'] = 'A replacement supported only by the same saved observation.'
    f.reply.update(lessons=[replacement], retire_ids=[old])
    p = f.reply['comparison_reviews'][0]
    if damage == 'fact': p['claims'][0]['value'] = 'improved'
    elif damage == 'seed': p['claims'][0]['fact_id'] = 'pair:4294967295:raw_direction'
    elif damage == 'version': p['interpretation'] = p.pop('conjecture')
    elif damage == 'target': p['next_hypothesis']['quality_gate'] = 'exact_only_near_zero'
    else: p['next_hypothesis']['base_source_sha256'] = 'f' * 64
    before = canonical_json(f.c.knowledge.state())
    f.c.complete_step()
    assert canonical_json(f.c.knowledge.state()) == before
    assert f.c.state()['comparison_reviews'] == {}


def test_new_claims_and_targets_are_withheld_together_in_text_ablation():
    arms = [grounded_fixture(comparison_review_text=mode) for mode in (True, False)]
    contexts = []
    for f in arms:
        f.c.complete_step()
        contexts.append(f.c.context(f.db, f.parent, 0, 2))
    a, b = copy.deepcopy(contexts)
    on, off = a.pop('comparison_reviews'), b.pop('comparison_reviews')
    assert on['cards'][0]['unverified_interpretation']['claims']
    assert off['cards'][0]['unverified_interpretation'] is None
    assert common_review(on) == common_review(off)
    assert on['reserved_chars'] == off['reserved_chars']
    assert canonical_json(a) == canonical_json(b)
    for f, context in zip(arms, contexts):
        _, prompt = begin_fixture(f.c, context)
        assert decoded_context(prompt)['comparison_reviews'] == context['comparison_reviews']


def test_retained_origin_survives_aging_without_returning_to_evidence():
    f = grounded_fixture()
    f.c.complete_step()
    cards = f.c.state()['comparison_reviews']
    # Rebuild solely from the retained group and lifetime digest, with no live
    # ordinary record available. The snapshot is private to this card.
    assert validate_cards(cards, [f.group], seen=f.c.knowledge._seen) == cards
    changed = copy.deepcopy(cards)
    changed[f.group['id']]['origin_snapshot']['hypothesis']['text'] = 'A forged old hypothesis.'
    with pytest.raises(ValueError, match='lifetime observation'):
        validate_cards(changed, [f.group], seen=f.c.knowledge._seen)
    with pytest.raises(ValueError, match='lifetime observation'):
        validate_cards(cards, [f.group], seen={})


def test_foreign_presentation_fact_mutation_is_rejected_before_attempt():
    f = grounded_fixture(comparison_review_text=False)
    f.c.complete_step()
    context = f.c.context(f.db, f.parent, 0, 2)
    context['comparison_reviews']['cards'][0]['host_facts']['facts']['quality:outcome'] = 'improved'
    before = f.c.state()
    with pytest.raises(ValueError, match='comparison review delivery'):
        begin_fixture(f.c, context)
    assert f.c.state() == before
