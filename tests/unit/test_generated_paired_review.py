"""Exact saved-data reviews; no provider, candidate or evaluator execution."""
import copy
import json

import pytest

from sciel.hypothesis import generated_paired_review as review
from sciel.hypothesis import reasoned_review
from sciel.hypothesis.comparison_targets import QUALITY_GATE, SCHEMA as TARGET_SCHEMA
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha

import test_generated_paired_history as synthetic


@pytest.fixture
def component():
    return synthetic.capsule.__wrapped__()


@pytest.fixture
def writer():
    return dict(phase='reflection', reference=1, raw_response_sha256='a' * 64)


def proposal(component, index=0, **changes):
    group = component['comparisons'][index]
    value = dict(comparison_id=group['id'], comparison_sha256=canonical_sha(group),
        claims=[dict(fact_id='metric:raw_object', value='candidate_raw_radii')],
        conjecture='A prospective explanation, not an established mechanism.',
        limitations='Saved paired geometry does not establish causal explanation.',
        next_hypothesis=None,
        next_hypothesis_reason='No source edit is proposed by this inert fixture.')
    return {**value, **changes}


def target(group):
    return dict(schema=TARGET_SCHEMA, base_source_sha256=group['challenger_source_sha256'],
        comparator_source_sha256=group['comparator_source_sha256'], change='Bounded prospective change.',
        quality_gate=QUALITY_GATE, diagnostics=[])


def test_exact_card_has_separate_route_no_ordinary_origin_and_wholly_unverified_text(component, writer):
    original = copy.deepcopy(component)
    inputs = [proposal(component, i) for i in range(3)]
    cards = review.build_cards(inputs, component, writer)
    rebuilt = review.validate_cards(cards, component)
    assert rebuilt == cards and rebuilt is not cards
    for card, submitted, group in zip(cards.values(), inputs, component['comparisons'], strict=True):
        assert card['schema'] == review.CARD_SCHEMA
        assert card['branch'] == group['branch'] != 'p1'
        assert card['receiving_root'] == component['root_parent']
        assert card['origin_experiment_id'] is None
        assert card['source_component_sha256'] == canonical_sha(component)
        assert card['unverified_interpretation'] == {k: submitted[k] for k in reasoned_review.MODEL_FIELDS}
        assert set(card['host_facts']).isdisjoint(reasoned_review.MODEL_FIELDS)
        assert card['writer'] == writer
    rebuilt[inputs[0]['comparison_id']]['host_facts']['facts']['metric:raw_object'] = 'changed'
    assert cards[inputs[0]['comparison_id']]['host_facts']['facts']['metric:raw_object'] == 'candidate_raw_radii'
    assert component == original


def test_whole_study_coverage_deduplicates_reused_receipts_and_root_has_no_authored_forecast(component):
    catalogs = [review.source_catalog(component, g['id']) for g in component['comparisons']]
    assert all(c['draw_coverage']['planned_count'] == 9 for c in catalogs)
    assert all(c['draw_coverage']['status_counts'] == {'valid': 9} for c in catalogs)
    assert catalogs[0]['facts']['authorship:comparator:present'] == 'false'
    assert 'native:comparator:status' not in catalogs[0]['facts']
    both = catalogs[2]['authored_forecasts']
    assert [f['role'] for f in both] == ['comparator', 'challenger']
    assert len({f['proposal_id'] for f in both}) == 2
    for f in both:
        assert f['native_forecast_status'] == 'UNMEASURED'
        assert f['spec_sha256'] == canonical_sha(f['spec'])
        assert all(row['unavailable_reason'] for row in f['rows'])
    # No trace in this synthetic saved receipt means unknown, not a tie.
    assert catalogs[2]['facts']['restart:comparator:11:effective_locations_changed:observed'] == 'unknown'
    assert catalogs[2]['facts']['restart:comparator:11:effective_locations_changed:verdict'] == 'INCONCLUSIVE'


def test_incomplete_known_loss_keeps_both_stored_outcome_and_strict_verdict(component, writer):
    root = synthetic.history.builder.source_for()
    a, b = [p['source'] for p in component['proposals']]
    component['comparisons'] = [synthetic.group(component, 'a-root', root, a,
        right_radius=.09, incomplete=True), synthetic.group(component, 'b-root', root, b,
        right_radius=.11, incomplete=True)]
    catalog = review.source_catalog(component, component['comparisons'][0]['id'])
    assert catalog['stored_quality_outcome'] == 'incomplete'
    assert catalog['strict_quality']['verdict'] == 'REFUTED'
    assert catalog['facts']['quality:strict_reason'] == 'known_paired_loss'
    assert catalog['facts']['pair:11:raw_direction'] == 'negative'
    assert catalog['facts']['pair:12:raw_direction'] == 'unknown'
    assert catalog['facts']['pair:12:measured_no_loss'] == 'unknown'
    assert catalog['draw_coverage']['status_counts'] == {'valid': 3, 'not_dispatched': 6}
    assert catalog['facts']['restart:challenger:12:availability'] == 'receipt_not_dispatched'
    p = proposal(component, claims=[dict(fact_id='quality:strict_verdict', value='REFUTED')])
    assert review.build_cards([p], component, writer)
    p['claims'][0]['value'] = 'INCONCLUSIVE'
    with pytest.raises(ValueError, match='claim differs'):
        review.build_cards([p], component, writer)


@pytest.mark.parametrize('change', [
    {'comparison_id': 'comparison:' + '0' * 64},
    {'comparison_sha256': '0' * 64},
    {'claims': [{'fact_id': 'invented:claim', 'value': 'true'}]},
    {'claims': [{'fact_id': 'metric:raw_object', 'value': 'radius_decreased_certified_packing'}]},
    {'claims': []},
    {'next_hypothesis_reason': ''},
    {'next_hypothesis_reason': ' '},
    {'next_hypothesis_reason': 'x' * 1025},
    {'interpretation': 'Unsupported legacy model field.'},
])
def test_invalid_or_false_proposal_rejected_without_input_mutation(component, writer, change):
    p = proposal(component, **change)
    before = copy.deepcopy([p, component, writer])
    with pytest.raises(ValueError):
        review.build_cards([p], component, writer)
    assert [p, component, writer] == before


def test_next_target_uses_only_declared_group_sources_and_fixed_quality(component, writer):
    group = component['comparisons'][0]
    p = proposal(component, next_hypothesis=target(group))
    assert review.build_cards([p], component, writer)
    p['next_hypothesis']['base_source_sha256'] = component['comparisons'][1]['challenger_source_sha256']
    with pytest.raises(ValueError, match='unavailable source'):
        review.build_cards([p], component, writer)
    p['next_hypothesis'] = target(group)
    p['next_hypothesis']['quality_gate'] = 'exact_only'
    with pytest.raises(ValueError, match='quality gate'):
        review.build_cards([p], component, writer)


@pytest.mark.parametrize('mutate', [
    lambda c: c.__setitem__('schema', 'sciel-comparison-review-card/3'),
    lambda c: c.__setitem__('branch', 'p1'),
    lambda c: c.__setitem__('origin_experiment_id', 'r000003a1'),
    lambda c: c.__setitem__('source_component_sha256', '0' * 64),
    lambda c: c['receiving_root'].__setitem__('id', 'other'),
    lambda c: c['host_facts']['facts'].__setitem__('quality:strict_verdict', 'REFUTED'),
    lambda c: c['host_facts']['draw_coverage'].__setitem__('planned_count', 18),
    lambda c: c['host_facts']['authored_forecasts'][0].__setitem__('native_forecast_status', 'CONFIRMED'),
    lambda c: c['host_facts']['authored_forecasts'][0].__setitem__('source_sha256', '0' * 64),
    lambda c: c.__setitem__('extra_unverified_field', 'leak'),
    lambda c: c['unverified_interpretation'].__setitem__('unsupported', 'leak'),
    lambda c: c['writer'].__setitem__('reference', True),
])
def test_retained_card_facts_routes_and_schema_cannot_be_self_rehashed(component, writer, mutate):
    cards = review.build_cards([proposal(component)], component, writer)
    mutate(next(iter(cards.values())))
    canonical_sha(cards)  # A caller can recompute a checksum; this grants no authority.
    with pytest.raises(ValueError):
        review.validate_cards(cards, component)


def test_raw_writer_authentication_remains_a_sidecar_responsibility(component, writer):
    cards = review.build_cards([proposal(component)], component, writer)
    next(iter(cards.values()))['unverified_interpretation']['next_hypothesis_reason'] = 'Different valid prose.'
    # No raw R is present in this API. Structural validation alone must never
    # be described as authenticating model prose; the sidecar replays raw R.
    assert review.validate_cards(cards, component) == cards


def test_empty_cards_still_validate_component_and_invalid_writer_does_not_hide_behind_empty_list(component):
    assert review.validate_cards({}, component) == {}
    with pytest.raises(ValueError, match='writer'):
        review.build_cards([], component, {})
    component['proposals'][0]['response']['text'] += 'tampered'
    with pytest.raises(ValueError):
        review.validate_cards({}, component)


def test_source_block_contains_every_group_or_rejects_whole_oversized_block(component, monkeypatch):
    block = review.source_block(component)
    raw = block.split(review.PREFIX, 1)[1]
    catalogs = json.loads(raw)
    assert [c['comparison_id'] for c in catalogs] == [g['id'] for g in component['comparisons']]
    assert all(c['schema'] == review.CATALOG_SCHEMA for c in catalogs)
    monkeypatch.setattr(review, 'MAX_SOURCE_CHARS', len(block) - 1)
    with pytest.raises(ValueError, match='whole-study'):
        review.source_block(component)


def test_presentation_keeps_all_model_fields_in_one_maskable_object(component, writer):
    card = next(iter(review.build_cards([proposal(component)], component, writer).values()))
    presented = review.presentation_card(card)
    assert presented['schema'] == review.PRESENTATION_SCHEMA
    assert presented['unverified_interpretation'] == card['unverified_interpretation']
    assert presented['host_facts'] == card['host_facts']
    assert set(reasoned_review.MODEL_FIELDS).isdisjoint(presented)
    assert card['schema'] == review.CARD_SCHEMA


def test_authored_study_has_both_role_forecasts_and_exact_incomplete_quality():
    from test_generated_paired_integration import pilot_artifacts, component as actual_component
    component = actual_component.__wrapped__(pilot_artifacts.__wrapped__())
    before = canonical_json(component)
    catalogs = json.loads(review.source_block(component).split(review.PREFIX, 1)[1])
    assert [c['strict_quality']['verdict'] for c in catalogs] == ['INCONCLUSIVE', 'REFUTED', 'REFUTED']
    assert [c['stored_quality_outcome'] for c in catalogs] == ['incomplete'] * 3
    assert all(c['origin_experiment_id'] is None and c['receiving_root']['id'] == 'p1' for c in catalogs)
    assert all(c['draw_coverage']['status_counts'] == {'valid': 3, 'not_dispatched': 6} for c in catalogs)
    seed = 763606949
    first = catalogs[0]
    assert first['branch'] == 'on_off'
    off, on = first['authored_forecasts']
    assert (off['role'], on['role']) == ('comparator', 'challenger')
    assert off['proposal_id'] != on['proposal_id']
    assert list(off['rows'][0]['result']['verdicts'].values()) != ['INCONCLUSIVE'] * 3
    for name, expected in [('effective_locations_changed', 'CONFIRMED'),
                           ('result_changed', 'REFUTED'), ('accepted_interval_gain', 'CONFIRMED')]:
        assert first['facts'][f'restart:comparator:{seed}:{name}:verdict'] == expected
    for name, expected in [('effective_locations_changed', 'CONFIRMED'),
                           ('result_changed', 'INCONCLUSIVE'), ('accepted_interval_gain', 'REFUTED')]:
        assert first['facts'][f'restart:challenger:{seed}:{name}:verdict'] == expected
    assert all(c['facts']['pair:763606949:raw_direction'] == 'negative' and
               c['facts']['pair:763606949:certified_direction'] == 'positive' for c in catalogs[1:])
    assert all(c['facts']['pair:1167415771:raw_direction'] == 'unknown' for c in catalogs)
    for forecast in (off, on):
        result = forecast['rows'][0]['result']
        assert result['coverage']['retained'] == 16 and result['coverage']['accepted'] == 7
        assert result['coverage']['omitted'] > 0
        assert result['per_case_details_omitted_count'] == 7
        assert forecast['native_forecast_status'] == 'UNMEASURED'
    assert canonical_json(component) == before
