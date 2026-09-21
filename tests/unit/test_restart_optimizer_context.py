"""Historical /3 header transport, with all original tamper/budget checks.

The old builder and _pack_common are selected explicitly so /1-/2 observations
are never fabricated by deleting motion from a newly stamped /3 observation.
Only inert stored arrays are used; no numerical execution.
"""
from copy import deepcopy

import pytest

from sciel.hypothesis import restart_observations as observations
from sciel.hypothesis import restart_optimizer_trace as codec
from sciel.knowledge_base import paired_comparisons
from sciel.knowledge_base import restart_context as context
from test_optimizer_restart_observations import receipt


@pytest.fixture(autouse=True)
def legacy_v3_builder(monkeypatch):
    monkeypatch.setattr(context.motion_observations, 'build_observation',
                        observations.build_observation)
    # The historical /3 packer must run for EVERY admission trial, preserving
    # the original exact whole-card budgeting tests rather than measuring /4.
    monkeypatch.setattr(context, '_pack', context._pack_common)


@pytest.fixture(scope='module')
def groups():
    pairs = [dict(seed=seed, comparator=receipt(source='a'*64, seed=seed),
                  challenger=receipt(source='c'*64, seed=seed, success=False, status=8))
             for seed in (11, 12, 13)]
    result = {}
    for kind in ('optimizer', 'legacy', 'mixed'):
        selected = deepcopy(pairs)
        for index, pair in enumerate(selected):
            for role in ('comparator', 'challenger'):
                if kind == 'legacy' or (kind == 'mixed' and (index, role) == (0, 'comparator')):
                    diagnostic = pair[role]['restart_trace']
                    diagnostic['trace'] = codec.legacy_projection(diagnostic['trace'])
                    diagnostic['sidecar_sha256'] = context._digest(diagnostic['trace'])
        first = selected[0]['comparator']
        result[kind] = paired_comparisons.build_comparison(namespace='inert_header_transport',
            batch_id='b1', branch='inert_fixture', origin_experiment_id=None,
            comparator_source_sha256='a'*64, challenger_source_sha256='c'*64,
            adapter_identity=first['adapter_identity'], limits=first['limits'], pairs=selected,
            provenance={'synthetic_only': True, 'numerical_or_provider_calls': 0})
    return result


@pytest.mark.parametrize('kind', ['optimizer', 'legacy', 'mixed'])
def test_whole_observations_and_original_view_hashes_roundtrip_without_receipt_changes(groups, kind):
    group = groups[kind]
    before = context._wire(group)
    wire = context.build_context([group], max_chars=64000)
    assert wire['schema'] == context.PREVIOUS_SCHEMA == 'sciel-restart-observation-context/3'
    assert len(wire['observation_common']) == (2 if kind == 'mixed' else 1)
    expanded = context.expand_context(wire)
    assert expanded['schema'] == context.EXPANDED_SCHEMA and expanded['presented_count'] == 6
    for card in expanded['cards']:
        saved = next(pair for pair in group['pairs'] if pair['seed'] == card['seed'])[card['role']]
        original = observations.observation_view(observations.build_observation(saved,
            source_sha256=card['source_sha256'], seed=card['seed'], adapter_identity=group['adapter_identity']))
        assert context._wire(card['observation_view']) == context._wire(original)
    assert context._pack_common(expanded) == wire
    assert context.validate_context(wire) == wire
    legacy_wire = context._pack_legacy(expanded)
    assert wire['rendered_chars'] < legacy_wire['rendered_chars']
    assert wire['rendered_chars'] == len(context._wire(wire))
    detached = context.expand_context(wire)
    detached['cards'][0]['observation_view']['observation']['columns']['cases'].append('changed')
    assert context.expand_context(wire) == expanded
    assert context._wire(group) == before


@pytest.mark.parametrize('kind', ['optimizer', 'legacy', 'mixed'])
def test_previous_wire_reader_preserves_exact_wire_and_full_objects(groups, kind):
    expanded = context.expand_context(context.build_context([groups[kind]], max_chars=64000))
    legacy = context._pack_legacy(expanded)
    before = context._wire(legacy)
    assert legacy['schema'] == context.LEGACY_SCHEMA == 'sciel-restart-observation-context/2'
    assert 'observation_common' not in legacy
    assert all('observation_common_ref' not in card for card in legacy['cards'])
    assert context._wire(context.validate_context(legacy)) == before
    assert context.expand_context(legacy) == expanded
    assert context._wire(legacy) == before


@pytest.mark.parametrize('damage', [
    'duplicate', 'unused', 'first_use', 'bool_ref', 'missing_ref', 'large_ref',
    'same_header_in_remainder', 'conflicting_header_in_remainder', 'missing_header_field',
    'extra_header_field', 'modified_header', 'modified_policy', 'wrong_common_ref',
    'unavailable_has_ref', 'changed_receipt', 'changed_view_hash',
])
def test_rehashed_transport_rejects_noncanonical_or_conflicting_metadata(groups, damage):
    wire = context.build_context([groups['mixed']], max_chars=64000)
    table, cards = wire['observation_common'], wire['cards']
    if damage == 'duplicate':
        table.append(deepcopy(table[0]))
    elif damage == 'unused':
        extra = deepcopy(table[0]); extra['scope'] += ' unused'
        table.append(extra)
    elif damage == 'first_use':
        table.reverse()
        for card in cards: card['observation_common_ref'] = 1-card['observation_common_ref']
    elif damage == 'bool_ref': cards[0]['observation_common_ref'] = False
    elif damage == 'missing_ref': cards[0].pop('observation_common_ref')
    elif damage == 'large_ref': cards[0]['observation_common_ref'] = len(table)
    elif damage == 'same_header_in_remainder': cards[0]['observation']['scope'] = table[0]['scope']
    elif damage == 'conflicting_header_in_remainder': cards[0]['observation']['scope'] = 'different'
    elif damage == 'missing_header_field': table[0].pop('scope')
    elif damage == 'extra_header_field': table[0]['other'] = {}
    elif damage == 'modified_header': table[0]['scope'] += ' changed'
    elif damage == 'modified_policy': table[1]['result_geometry_policy']['tolerance'] = 0.
    elif damage == 'wrong_common_ref': cards[1]['observation_common_ref'] = 0
    elif damage == 'unavailable_has_ref':
        cards[0].update(observation=None, view_sha256=None, unavailable_reason='invalid_diagnostic')
    elif damage == 'changed_receipt': wire['receipt_refs'][0][3] = 'f'*64
    else: cards[0]['view_sha256'] = 'f'*64
    with pytest.raises(ValueError):
        context.validate_context(context._seal(wire))


def test_catalog_empty_when_only_omissions_and_null_for_unavailable_cards(groups):
    group = groups['optimizer']
    wire = context.build_context([group], max_items=0, max_chars=64000)
    assert wire['observation_common'] == [] and wire['cards'] == []
    assert wire['omitted_count'] == wire['selected_count'] == 6
    assert context.validate_context(wire) == wire
    damaged = deepcopy(group)
    # This caller fixture preserves valid numeric receipts; only diagnostics are damaged.
    for pair in damaged['pairs']:
        for role in ('comparator', 'challenger'):
            pair[role]['restart_trace']['trace']['trace_sha256'] = 'f'*64
    damaged = paired_comparisons.build_comparison(namespace='inert_bad_optional_diagnostic',
        **{key: damaged[key] for key in ('batch_id', 'branch', 'origin_experiment_id',
            'comparator_source_sha256', 'challenger_source_sha256', 'adapter_identity', 'limits', 'pairs')},
        provenance={'synthetic_only': True, 'numerical_or_provider_calls': 0})
    missing = context.build_context([damaged], max_chars=64000)
    assert missing['observation_common'] == [] and missing['presented_count'] == 6
    assert all(card['observation_common_ref'] is None and card['observation'] is None
               and card['unavailable_reason'] == 'invalid_diagnostic' for card in missing['cards'])


def test_every_trial_counts_header_table_and_exact_boundary_keeps_whole_views(groups):
    group = groups['optimizer']
    all_cards = context.build_context([group], max_chars=64000)
    exact = context.build_context([group], max_chars=all_cards['rendered_chars'])
    for _ in range(3):
        if exact['max_chars'] != exact['rendered_chars']:
            exact = context.build_context([group], max_chars=exact['rendered_chars'])
    assert exact['presented_count'] == 6
    assert exact['rendered_chars'] == exact['max_chars'] == len(context._wire(exact))
    smaller = context.build_context([group], max_chars=exact['rendered_chars']-1)
    assert smaller is not None and smaller['presented_count'] < 6
    assert smaller['rendered_chars'] <= smaller['max_chars']
    expanded = context.expand_context(smaller)
    assert any(row['reason'] == 'memory_char_limit' for row in expanded['omissions'])
    assert [row['receipt_ref'] for row in smaller['cards']] == list(range(smaller['presented_count']))
    assert all(row['observation_view']['included'] for row in expanded['cards'])
    assert context.build_context([group], max_chars=1) is None
    assert context.build_context([group], max_chars=0) is None
