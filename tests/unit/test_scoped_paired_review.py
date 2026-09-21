"""Pure opt-in card checks over literal receipts; no candidate/provider execution."""
import copy
import json

import pytest

from sciel.hypothesis import component_targets, generated_paired_review as legacy
from sciel.hypothesis import scoped_paired_review as review
from sciel.hypothesis.comparison_targets import QUALITY_GATE
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha

import test_generated_paired_history as synthetic
from test_generated_paired_review import proposal as old_proposal


@pytest.fixture
def component():
    return synthetic.capsule.__wrapped__()


@pytest.fixture
def writer():
    return dict(phase='reflection', reference=1, raw_response_sha256='a' * 64)


def proposal(component, index=2, **changes):
    value = old_proposal(component, index)
    value['evidence_refs'] = {field: [dict(comparison_id=value['comparison_id'],
        fact_id='metric:raw_object')] for field in review.PROSE_FIELDS}
    return {**value, **changes}


def target(component, index=2):
    group = component['comparisons'][index]
    return dict(schema=component_targets.SCHEMA, component_sha256=canonical_sha(component),
        receiving_root=copy.deepcopy(component['root_parent']),
        policy_sha256=canonical_sha(component['policy']),
        motivating_evidence=[dict(comparison_id=group['id'], fact_id='quality:strict_verdict')],
        base_source_sha256=group['comparator_source_sha256'],
        comparator_source_sha256=component['root_parent']['source_sha256'],
        change='A bounded prospective edit, requiring separate generation and measurement.',
        quality_gate=QUALITY_GATE, diagnostics=[])


def test_new_contrast_is_separate_from_reviewed_pair_and_bound_to_component(component, writer):
    p = proposal(component, next_hypothesis=target(component))
    saved = copy.deepcopy([component, p, writer])
    cards = review.build_cards([p], component, writer)
    card = cards[p['comparison_id']]
    assert card['schema'] == review.CARD_SCHEMA
    assert card['origin_experiment_id'] is None
    assert card['branch'] == component['comparisons'][2]['branch']
    assert card['receiving_root'] == component['root_parent']
    assert card['source_component_sha256'] == canonical_sha(component)
    assert card['unverified_interpretation'] == {key: p[key] for key in review.MODEL_FIELDS}
    assert p['next_hypothesis']['comparator_source_sha256'] not in {
        card['comparator_source_sha256'], card['challenger_source_sha256']}
    assert card['unverified_interpretation']['next_hypothesis']['quality_gate'] == QUALITY_GATE
    assert card['evidence_bindings']['target_motivation'] == [dict(
        **p['next_hypothesis']['motivating_evidence'][0],
        comparison_sha256=p['comparison_sha256'], value='CONFIRMED')]
    assert review.validate_cards(cards, component) == cards
    assert [component, p, writer] == saved
    cards[p['comparison_id']]['evidence_bindings']['conjecture'][0]['value'] = 'changed'
    assert p == saved[1]


def test_each_prose_field_can_reference_other_group_with_own_group_anchor(component, writer):
    p = proposal(component)
    for field in review.PROSE_FIELDS:
        p['evidence_refs'][field].append(dict(comparison_id=component['comparisons'][0]['id'],
                                            fact_id='quality:strict_verdict'))
    card = review.build_cards([p], component, writer)[p['comparison_id']]
    for field in review.PROSE_FIELDS:
        resolved = card['evidence_bindings'][field]
        assert resolved == [dict(**p['evidence_refs'][field][0],
            comparison_sha256=p['comparison_sha256'], value='candidate_raw_radii'),
            dict(**p['evidence_refs'][field][1],
            comparison_sha256=canonical_sha(component['comparisons'][0]), value='CONFIRMED')]
    assert set(review.MODEL_FIELDS).isdisjoint(card)


def test_null_target_still_requires_bounded_anchored_reason_and_refs(component, writer):
    p = proposal(component)
    card = review.build_cards([p], component, writer)[p['comparison_id']]
    assert card['unverified_interpretation']['next_hypothesis'] is None
    assert card['evidence_bindings']['target_motivation'] is None
    assert card['evidence_bindings']['next_hypothesis_reason']
    p['evidence_refs']['next_hypothesis_reason'] = []
    with pytest.raises(ValueError, match='reference count'):
        review.build_cards([p], component, writer)


def test_target_motivation_resolves_distinct_foreign_fact_without_prose_replacement(component, writer):
    p = proposal(component, next_hypothesis=target(component))
    foreign = dict(comparison_id=component['comparisons'][0]['id'], fact_id='metric:certified_object')
    p['next_hypothesis']['motivating_evidence'].append(foreign)
    cards = review.build_cards([p], component, writer)
    card = cards[p['comparison_id']]
    assert card['evidence_bindings']['target_motivation'][1] == dict(**foreign,
        comparison_sha256=canonical_sha(component['comparisons'][0]),
        value='radius_decreased_certified_packing')
    assert foreign not in p['evidence_refs']['next_hypothesis_reason']
    assert card['unverified_interpretation']['next_hypothesis_reason'] == p['next_hypothesis_reason']
    card['evidence_bindings']['target_motivation'][1]['value'] = 'candidate_raw_radii'
    with pytest.raises(ValueError, match='retained cards differ'):
        review.validate_cards(cards, component)


@pytest.mark.parametrize('mutate', [
    lambda p: p.pop('evidence_refs'),
    lambda p: p['evidence_refs'].pop('limitations'),
    lambda p: p['evidence_refs'].__setitem__('invented_field', []),
    lambda p: p['evidence_refs'].__setitem__('conjecture', None),
    lambda p: p['evidence_refs'].__setitem__('conjecture', []),
    lambda p: p['evidence_refs'].__setitem__('conjecture', p['evidence_refs']['conjecture'] * 2),
    lambda p: p['evidence_refs']['conjecture'][0].__setitem__('value', 'candidate_raw_radii'),
    lambda p: p['evidence_refs']['conjecture'][0].__setitem__('fact_id', 'x' * 257),
    lambda p: p.__setitem__('next_hypothesis_reason', 'x' * 1025),
    lambda p: p.__setitem__('claims', []),
    lambda p: p.__setitem__('conjecture', ''),
])
def test_shape_rejects_invalid_bounded_fields_without_repair(component, mutate):
    p = proposal(component)
    mutate(p)
    before = copy.deepcopy(p)
    with pytest.raises(ValueError):
        review.validate_proposals([p])
    assert p == before


@pytest.mark.parametrize('bad', ['unknown_group', 'unknown_fact', 'wrong_group_fact', 'unanchored'])
def test_reference_uses_its_own_group_catalog_and_anchor(component, writer, bad):
    p = proposal(component)
    refs = p['evidence_refs']['limitations']
    if bad == 'unknown_group':
        refs.append(dict(comparison_id='missing', fact_id='metric:raw_object'))
    elif bad == 'unknown_fact':
        refs[0]['fact_id'] = 'comparison_sha256'  # Metadata remains outside named facts.
    elif bad == 'wrong_group_fact':
        # Only a generated source has this native forecast fact. Root does not.
        refs.append(dict(comparison_id=component['comparisons'][0]['id'],
                         fact_id='native:comparator:status'))
    else:
        refs[0]['comparison_id'] = component['comparisons'][0]['id']
    assert review.validate_proposals([p])  # Shape acceptance is not factual admission.
    with pytest.raises(ValueError):
        review.build_cards([p], component, writer)


@pytest.mark.parametrize('mutation', [
    lambda t: t.__setitem__('component_sha256', '0' * 64),
    lambda t: t.__setitem__('policy_sha256', '0' * 64),
    lambda t: t['receiving_root'].__setitem__('id', 'other'),
    lambda t: t.__setitem__('base_source_sha256', '0' * 64),
    lambda t: t.__setitem__('quality_gate', 'exact_only'),
    lambda t: t.__setitem__('motivating_evidence', []),
])
def test_target_cannot_escape_component_root_policy_sources_or_strict_gate(component, writer, mutation):
    p = proposal(component, next_hypothesis=target(component))
    mutation(p['next_hypothesis'])
    with pytest.raises(ValueError):
        review.build_cards([p], component, writer)


@pytest.mark.parametrize('mutate', [
    lambda c: c.__setitem__('schema', legacy.CARD_SCHEMA),
    lambda c: c.__setitem__('branch', 'p1'),
    lambda c: c.__setitem__('origin_experiment_id', 'r000003a1'),
    lambda c: c.__setitem__('source_component_sha256', '0' * 64),
    lambda c: c['evidence_bindings']['conjecture'][0].__setitem__('value', 'false'),
    lambda c: c['evidence_bindings']['limitations'][0].__setitem__('comparison_sha256', '0' * 64),
    lambda c: c['evidence_bindings'].__setitem__('extra', []),
    lambda c: c['host_facts']['facts'].__setitem__('metric:raw_object', 'certified'),
    lambda c: c['unverified_interpretation']['evidence_refs']['conjecture'][0].__setitem__(
        'fact_id', 'metric:certified_object'),
])
def test_retained_card_and_resolved_evidence_rebuild_exactly(component, writer, mutate):
    p = proposal(component)
    cards = review.build_cards([p], component, writer)
    mutate(cards[p['comparison_id']])
    canonical_sha(cards)  # A fresh outer digest does not authenticate changed facts.
    with pytest.raises(ValueError):
        review.validate_cards(cards, component)


def test_prose_and_ref_selection_are_unverified_even_when_shape_and_fact_links_match(component, writer):
    p = proposal(component)
    p['conjecture'] = 'An intentionally unsupported causal assertion remains unverified.'
    cards = review.build_cards([p], component, writer)
    card = cards[p['comparison_id']]
    assert card['unverified_interpretation']['conjecture'] == p['conjecture']
    assert 'conjecture' not in card['host_facts']
    card['unverified_interpretation']['conjecture'] = 'Another valid but unauthenticated sentence.'
    assert review.validate_cards(cards, component) == cards
    # Raw writer and semantic delivery checks are separate authoritative layers.
    assert card['writer'] == writer


def test_empty_cards_still_authenticate_whole_component_and_writer(component):
    assert review.validate_cards({}, component) == {}
    with pytest.raises(ValueError, match='writer'):
        review.build_cards([], component, {})
    component['proposals'][0]['response']['text'] += ' tampered'
    with pytest.raises(ValueError):
        review.validate_cards({}, component)


def test_whole_list_character_budget_includes_refs_and_component_targets(component):
    p = proposal(component, next_hypothesis=target(component))
    # Valid long prose/claims/refs produce an oversized three-card list.
    rows = []
    for index in range(3):
        q = proposal(component, index)
        q.update(conjecture='a' * 1024, limitations='b' * 2048,
                 next_hypothesis_reason='c' * 1024,
                 claims=[dict(fact_id=f'claim:{n}', value='x' * 128) for n in range(8)])
        rows.append(q)
    assert review.validate_proposals([p])
    assert len(canonical_json(rows)) > review.MAX_PROPOSAL_CHARS
    with pytest.raises(ValueError, match='character limit'):
        review.validate_proposals(rows)


def test_legacy_shape_and_group_local_targets_unchanged(component, writer):
    old = old_proposal(component, 2)
    before = canonical_json(legacy.build_cards([old], component, writer))
    with pytest.raises(ValueError, match='proposal fields'):
        review.validate_proposals([old])
    modern = proposal(component, next_hypothesis=target(component))
    with pytest.raises(ValueError, match='proposal fields'):
        legacy.build_cards([modern], component, writer)
    assert canonical_json(legacy.build_cards([old], component, writer)) == before
    assert json.loads(before)[old['comparison_id']]['schema'] == legacy.CARD_SCHEMA


def test_wire_and_scope_bind_full_study_without_changing_legacy_schema(component):
    old_schema = canonical_json(legacy.reasoned_review.json_schema())
    schema = review.json_schema(component)
    props = schema['items']['properties']
    assert set(schema['items']['required']) == {'comparison_id', 'comparison_sha256', *review.MODEL_FIELDS}
    target_schema = props['next_hypothesis']
    assert target_schema['type'] == ['object', 'null']
    assert target_schema['properties']['schema']['enum'] == [component_targets.SCHEMA]
    assert target_schema['properties']['component_sha256']['enum'] == [canonical_sha(component)]
    sources = {component['root_parent']['source_sha256'],
               *[synthetic.history._digest(p['source']) for p in component['proposals']]}
    assert set(target_schema['properties']['comparator_source_sha256']['enum']) == sources
    assert set(props['evidence_refs']['properties']) == set(review.PROSE_FIELDS)
    value = json.loads(review.source_block(component).split(review.PREFIX, 1)[1])
    assert value['component_sha256'] == canonical_sha(component)
    assert value['policy_sha256'] == canonical_sha(component['policy'])
    assert value['receiving_root'] == component['root_parent']
    assert set(value['available_source_sha256']) == sources
    assert canonical_json(legacy.reasoned_review.json_schema()) == old_schema
    assert review.identity()['card_schema'] == review.CARD_SCHEMA


def test_scope_includes_authenticated_parent_even_when_not_a_measured_role(component):
    component['comparisons'] = [component['comparisons'][2]]
    root_sha = component['root_parent']['source_sha256']
    assert root_sha not in {component['comparisons'][0][key] for key in
                           ('comparator_source_sha256', 'challenger_source_sha256')}
    scope = json.loads(review.source_block(component).split(review.PREFIX, 1)[1])
    assert root_sha in scope['available_source_sha256']
