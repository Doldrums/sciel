"""Legacy and grounded review coexistence, saved prompt pins and downgrade guards.

All auxiliary replies and histories are authored offline fixtures. No provider,
candidate, historical Python, external evaluator or solver runs.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from sciel.engine.reflection_context import prepare_reflection_context
from sciel.engine.research import ResearchController
from sciel.hypothesis import comparison_review, grounded_review
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from test_comparison_review import WRITER, group, proposal as legacy_proposal
from test_comparison_review_flow import fixture, restored
from test_external_history_v2_actual import actual_history
from test_grounded_review_flow import grounded_fixture


@pytest.fixture
def mixed():
    groups = [group(1), group(2)]
    old = legacy_proposal(groups[0])
    old['interpretation'] = 'An immutable historical interpretation, with no checked claims.'
    new = dict(comparison_id=groups[1]['id'], comparison_sha256=canonical_sha(groups[1]),
        claims=[dict(fact_id='quality:outcome', value='mixed')],
        conjecture='A grid change is an untested explanation.',
        limitations='The mechanism and its causal benefit remain unverified.', next_hypothesis=None)
    old_cards = comparison_review.build_cards([old], groups, WRITER)
    new_cards = comparison_review.build_cards([new], groups, WRITER, version=2, records={}, seen={})
    return groups, {**old_cards, **new_cards}, old, new


def test_mixed_retained_cards_rebuild_without_upgrading_old_prose(mixed):
    groups, cards, old, new = mixed
    before = canonical_json(cards)
    rebuilt = comparison_review.validate_cards(cards, groups, seen={})
    assert canonical_json(rebuilt) == before
    assert rebuilt[groups[0]['id']]['schema'] == comparison_review.CARD_SCHEMA
    assert rebuilt[groups[1]['id']]['schema'] == grounded_review.CARD_SCHEMA
    assert rebuilt[groups[0]['id']]['unverified_interpretation'] == {
        key: old[key] for key in ('interpretation', 'limitations', 'next_hypothesis')}
    assert rebuilt[groups[1]['id']]['unverified_interpretation'] == {
        key: new[key] for key in grounded_review.MODEL_FIELDS}
    assert 'claims' not in rebuilt[groups[0]['id']]['unverified_interpretation']
    assert 'interpretation' not in rebuilt[groups[1]['id']]['unverified_interpretation']
    rebuilt[groups[0]['id']]['unverified_interpretation']['interpretation'] = 'altered'
    assert canonical_json(cards) == before


@pytest.mark.parametrize('target_schema', ['sciel-comparison-review-card/1', 'sciel-comparison-review-card/2'])
def test_relabeling_one_card_does_not_convert_its_contract(mixed, target_schema):
    groups, cards, _, _ = mixed
    damaged = copy.deepcopy(cards)
    target = next(card for card in damaged.values() if card['schema'] != target_schema)
    target['schema'] = target_schema
    with pytest.raises(ValueError):
        comparison_review.validate_cards(damaged, groups, seen={})


def test_mixed_presentation_preserves_legacy_bytes_and_withholds_both_model_shapes(mixed):
    groups, cards, _, _ = mixed
    before = canonical_json(cards)
    legacy_view = comparison_review.context(cards, groups, 'p1', 10000, seen={})
    assert legacy_view['presented_count'] == 2
    by_id = {card['comparison_id']: card for card in legacy_view['cards']}
    assert canonical_json(by_id[groups[0]['id']]) == canonical_json(cards[groups[0]['id']])
    assert 'origin_snapshot' not in by_id[groups[1]['id']]
    on, off = [comparison_review.context(cards, groups, 'p1', 10000, seen={}, text_enabled=flag)
               for flag in (True, False)]
    assert on['presented_count'] == off['presented_count'] == 2
    assert on['reserved_chars'] == off['reserved_chars']
    for a, b in zip(on['cards'], off['cards']):
        assert a['comparison_id'] == b['comparison_id']
        assert a['host_facts'] == b['host_facts']
        assert a['writer'] == b['writer']
        assert a['unverified_interpretation'] is not None
        assert b['unverified_interpretation'] is None
    assert canonical_json(cards) == before


@pytest.mark.parametrize('settings', [
    {'comparison_reviews': False},
    {'comparison_reviews': True},
    {'comparison_reviews': True, 'comparison_review_version': 1},
])
def test_authored_legacy_prompt_and_context_preserve_version_one(actual_history, settings):
    _, final = actual_history
    before_final = canonical_json(final)
    config = {**final['config'], **settings}
    kb = KnowledgeBase(**{key: config[key] for key in ('max_lessons', 'max_experiments', 'max_fragments')})
    kb.load_state(final['knowledge'])
    before = kb.state()
    prepared = prepare_reflection_context(kb, final['conclusions'], config=config, interval=1, through=2)
    explicit = {**config, 'comparison_review_version': 1} if settings['comparison_reviews'] else config
    repeated = prepare_reflection_context(kb, final['conclusions'], config=explicit, interval=1, through=2)
    assert prepared == repeated
    assert 'next_hypothesis_reason' not in prepared['prompt']
    assert prepared['records']
    assert all(row['id'] in prepared['prompt'] for row in prepared['records'])
    assert prepared['context_event']['experiment_ids'] == [row['id'] for row in prepared['records']]
    assert prepared['response_format']['json_schema']['name'] == (
        'sciel_reflection_v2' if settings['comparison_reviews'] else 'sciel_reflection_v1')
    assert kb.state() == before and canonical_json(final) == before_final


def _empty_receiver(source, config):
    return ResearchController(config, source.c.llm,
        NS(research=lambda **_: pytest.fail('restore emitted research activity')),
        lambda _: pytest.fail('restore charged a model call'), metric=source.c.metric)


def test_normal_checkpoint_downgrade_rejected_atomically():
    f = grounded_fixture()
    f.c.complete_step()
    target = _empty_receiver(f, {**f.c.config, 'comparison_review_version': 1})
    before = target.state()
    with pytest.raises(ValueError, match='configuration mismatch'):
        target.load_state(f.c.state())
    assert target.state() == before


@pytest.mark.parametrize('explicit', [False, True])
def test_rehashed_legacy_envelope_cannot_hide_retained_v2_card(explicit):
    f = grounded_fixture()
    f.c.complete_step()
    config = copy.deepcopy(f.c.config)
    if explicit:
        config['comparison_review_version'] = 1
    else:
        config.pop('comparison_review_version')
    target = _empty_receiver(f, config)
    before = target.state()
    damaged = f.c.state()
    # Deliberately defeat superficial config/envelope equality checks while
    # leaving the authentic grounded card intact: receiving policy must reject.
    for key in ('config', 'auxiliary_schema', 'presentation'):
        damaged[key] = copy.deepcopy(before[key])
    with pytest.raises(ValueError, match='receiving version 2'):
        target.load_state(damaged)
    assert target.state() == before


def test_boolean_checkpoint_version_cannot_equal_explicit_legacy_one():
    f = fixture(comparison_review_version=1, reflect=True)
    good = f.c.state()
    bad = copy.deepcopy(good)
    bad['config']['comparison_review_version'] = True
    assert bad['config'] == good['config']  # Python's bool/int equality is the trap.
    target = _empty_receiver(f, f.c.config)
    before = target.state()
    with pytest.raises(ValueError, match='comparison_review_version'):
        target.load_state(bad)
    assert target.state() == before
    assert restored(f).state() == good
