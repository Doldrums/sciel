"""Lossless diagnostic transport from inert receipts; no helper/solver/API runs.

Context validation checks presentation consistency, not external authenticity.
Durable receipt reconstruction remains the source of diagnostic facts.
"""
from copy import deepcopy

import pytest

from sciel.knowledge_base import paired_comparisons
from sciel.knowledge_base import restart_context as context
from test_paired_comparisons import inputs
from test_restart_context import digest, traced_group


@pytest.fixture
def full_wire():
    return context.build_context([traced_group(traces='all')], max_chars=64000)


def _field_reference(wire, card_index, field):
    return next(ref for ref in wire['cards'][card_index]['observation_refs']
                if wire['observation_catalog'][ref][0] == field)


def _receipt_for(group, card):
    return next(pair for pair in group['pairs'] if pair['seed'] == card['seed'])[card['role']]


def test_field_catalog_restores_every_whole_observation_and_is_detached():
    group = traced_group(traces='all')
    before = deepcopy(group)
    wire = context.build_context([group], max_chars=64000)
    assert wire['schema'] == 'sciel-restart-observation-context/5'
    assert wire['presented_count'] == 6 and wire['omitted_count'] == 0
    assert all(set(card) == {'receipt_ref', 'observation_refs', 'view_sha256',
                             'unavailable_reason'} for card in wire['cards'])
    expanded = context.expand_context(wire)
    for card in expanded['cards']:
        saved = _receipt_for(group, card)
        expected = context.motion_observations.build_observation(saved,
            source_sha256=card['source_sha256'], seed=card['seed'],
            adapter_identity=group['adapter_identity'])
        assert card['observation_view'] == context.observations.observation_view(expected)
        assert card['receipt_sha256'] == digest(saved)
    assert context._pack(expanded) == wire
    assert context.validate_context(wire) == wire
    assert len(wire['observation_catalog']) < sum(
        len(card['observation_refs']) for card in wire['cards'])
    detached = context.expand_context(wire)
    detached['cards'][0]['observation_view']['observation']['scope'] = 'changed copy'
    assert context._pack(context.expand_context(wire)) == wire
    assert group == before


@pytest.mark.parametrize('ref', [True, False, -1, 1000000, 0.0, '0', None])
def test_field_reference_requires_an_exact_bounded_integer(full_wire, ref):
    wire = deepcopy(full_wire)
    wire['cards'][0]['observation_refs'][0] = ref
    with pytest.raises(ValueError, match='observation field reference'):
        context.validate_context(context._seal(wire))


@pytest.mark.parametrize('entry', [None, {}, ['scope'], ['scope', None, None],
                                  [True, None], ['', None]])
def test_field_catalog_entry_has_exact_named_pair_shape(full_wire, entry):
    wire = deepcopy(full_wire)
    wire['observation_catalog'][0] = entry
    with pytest.raises(ValueError, match='observation field entry'):
        context.validate_context(context._seal(wire))


def test_duplicate_catalog_entry_rejected_even_after_outer_rehash(full_wire):
    wire = deepcopy(full_wire)
    wire['observation_catalog'].append(deepcopy(wire['observation_catalog'][0]))
    with pytest.raises(ValueError, match='duplicate observation field'):
        context.validate_context(context._seal(wire))


def test_unused_catalog_entry_is_not_hidden_payload(full_wire):
    wire = deepcopy(full_wire)
    wire['observation_catalog'].append(['unused_owned_field', None])
    with pytest.raises(ValueError, match='unused observation field'):
        context.validate_context(context._seal(wire))


def test_repeated_field_reference_cannot_overwrite_a_previous_field(full_wire):
    wire = deepcopy(full_wire)
    refs = wire['cards'][0]['observation_refs']
    refs.insert(1, refs[0])
    with pytest.raises(ValueError, match='duplicate observation field name'):
        context.validate_context(context._seal(wire))


def test_semantically_equivalent_reindexed_catalog_breaks_first_use_order(full_wire):
    wire = deepcopy(full_wire)
    catalog = wire['observation_catalog']
    catalog[0], catalog[1] = catalog[1], catalog[0]
    for card in wire['cards']:
        card['observation_refs'] = [1 if ref == 0 else 0 if ref == 1 else ref
                                    for ref in card['observation_refs']]
    with pytest.raises(ValueError, match='first-use order'):
        context.validate_context(context._seal(wire))


def test_field_order_is_canonical_even_when_references_are_already_seen(full_wire):
    wire = deepcopy(full_wire)
    seen = set(wire['cards'][0]['observation_refs'])
    refs = wire['cards'][1]['observation_refs']
    positions = [i for i, ref in enumerate(refs) if ref in seen]
    assert len(positions) >= 2
    a, b = positions[:2]
    refs[a], refs[b] = refs[b], refs[a]
    with pytest.raises(ValueError, match='observation field order'):
        context.validate_context(context._seal(wire))


def test_rehashed_outer_catalog_cannot_change_inner_receipt_or_observation(full_wire):
    wire = deepcopy(full_wire)
    ref = _field_reference(wire, 0, 'receipt_sha256')
    wire['observation_catalog'][ref][1] = 'f' * 64
    with pytest.raises(ValueError):
        context.validate_context(context._seal(wire))


@pytest.mark.parametrize('field', ['bindings', 'prefix', 'quality_join', 'run'])
@pytest.mark.parametrize('damage', ['dict_instead_of_row', 'missing_value'])
def test_encoded_field_values_require_the_declared_exact_row(full_wire, field, damage):
    wire = deepcopy(full_wire)
    ref = _field_reference(wire, 0, field)
    data = wire['observation_catalog'][ref][1]
    assert type(data) is list
    wire['observation_catalog'][ref][1] = {} if damage == 'dict_instead_of_row' else data[:-1]
    with pytest.raises(ValueError, match='encoded record width'):
        context.validate_context(context._seal(wire))


@pytest.mark.parametrize('damage', ['quality_width', 'origin_count', 'origin_counter_width'])
def test_nested_run_rows_cannot_lose_a_quality_value_or_origin(full_wire, damage):
    wire = deepcopy(full_wire)
    ref = _field_reference(wire, 0, 'run')
    run = wire['observation_catalog'][ref][1]
    columns = wire['columns']['field_values']['run']
    if damage == 'quality_width':
        run[columns.index('initial_quality')].pop()
    elif damage == 'origin_count':
        run[columns.index('origins')].pop()
    else:
        run[columns.index('origins')][0].pop()
    with pytest.raises(ValueError, match='encoded record width|origin row count'):
        context.validate_context(context._seal(wire))


def test_declared_value_column_order_is_fixed_even_after_rehash(full_wire):
    wire = deepcopy(full_wire)
    columns = wire['columns']['field_values']['bindings']
    columns[0], columns[1] = columns[1], columns[0]
    with pytest.raises(ValueError, match='wire schema fields'):
        context.validate_context(context._seal(wire))


def test_value_encoding_preserves_signed_zero_and_false_without_coercion():
    original = dict(raw_sum=-0.0, trace_sum=0.0, maximum_roundoff=0.0,
                    within_roundoff=False, terminal_status_compatible=True)
    encoded = context._encode_field('quality_join', original)
    assert context._wire(encoded) == '[-0.0,0.0,0.0,false,true]'
    restored = context._decode_field('quality_join', encoded)
    assert context._wire(restored) == context._wire(original)
    assert original['within_roundoff'] is False


@pytest.mark.parametrize('schema', [context.LEGACY_SCHEMA, context.PREVIOUS_SCHEMA])
@pytest.mark.parametrize('with_optimizer_outcome', [False, True])
def test_legacy_context_readers_restore_original_observation_bytes(
        monkeypatch, schema, with_optimizer_outcome):
    # Deliberately use the historical builder; legacy wire is never fabricated
    # by stripping motion fields from a new observation and restamping hashes.
    monkeypatch.setattr(context.motion_observations, 'build_observation',
                        context.observations.build_observation)
    if with_optimizer_outcome:
        from test_optimizer_restart_observations import receipt
        data = inputs()
        for pair in data['pairs']:
            for role in ('comparator', 'challenger'):
                pair[role] = receipt(source=data[role + '_source_sha256'], seed=pair['seed'])
        data['adapter_identity'] = deepcopy(data['pairs'][0]['comparator']['adapter_identity'])
        group = paired_comparisons.build_comparison(**data)
    else:
        group = traced_group(traces='all')
    current_wire = context.build_context([group], max_chars=64000)
    expanded = context.expand_context(current_wire)
    pack = context._pack_legacy if schema == context.LEGACY_SCHEMA else context._pack_common
    legacy_wire = pack(expanded)
    assert legacy_wire['schema'] == schema
    assert context.validate_context(legacy_wire) == legacy_wire
    assert context.expand_context(legacy_wire) == expanded
    for card in context.expand_context(legacy_wire)['cards']:
        observation = card['observation_view']['observation']
        expected_schema = 'sciel-restart-observation/2' if with_optimizer_outcome else 'sciel-restart-observation/1'
        assert observation['schema'] == expected_schema
        assert 'motion_policy' not in observation
    assert context._pack(context.expand_context(legacy_wire)) == current_wire


def test_internal_view_omission_remains_explicit_without_partial_factoring(monkeypatch):
    group = traced_group(traces='all')
    original_view = context.observations.observation_view
    budgets = []
    def omitted_view(observation, *, max_chars):
        budgets.append(max_chars)
        return original_view(observation, max_chars=700)
    monkeypatch.setattr(context.observations, 'observation_view', omitted_view)
    wire = context.build_context([group], max_chars=12000)
    assert budgets == [context.observations.MAX_OBSERVATION_BYTES * 2] * 6
    assert wire['cards'] == [] and wire['observation_catalog'] == []
    decoded = context.expand_context(wire)
    assert decoded['omitted_count'] == 6
    assert {row['reason'] for row in decoded['omissions']} == {'card_char_limit'}
    assert wire['rendered_chars'] == len(context._wire(wire)) <= 12000


def test_none_unknown_and_invalid_cards_need_no_catalog_fields():
    def alter(data):
        data['pairs'][0]['challenger'] = None
        data['pairs'][1]['comparator'] = deepcopy(paired_comparisons.UNKNOWN_RECEIPT)
        trace = data['pairs'][1]['challenger']['restart_trace']['trace']
        trace['identity']['source_sha256'] = 'f' * 64
        trace['trace_sha256'] = digest({k: v for k, v in trace.items() if k != 'trace_sha256'})
    group = traced_group(traces='all', mutate=alter)
    wire = context.build_context([group], max_chars=64000)
    cards = context.expand_context(wire)['cards']
    expected = ['receipt_not_dispatched', 'dispatch_unknown', 'invalid_diagnostic']
    assert [row['unavailable_reason'] for row in cards if row['observation_view'] is None] == expected
    for wire_card, card in zip(wire['cards'], cards):
        assert (wire_card['observation_refs'] is None) == (card['observation_view'] is None)
    assert paired_comparisons.validate_comparison(group) == group


def test_transport_hash_is_not_authentication_of_the_original_group(full_wire):
    # No original group is provided to validate_context: a consistent hash
    # claim is not a lookup or an independent proof of its historical contents.
    wire = deepcopy(full_wire)
    original = wire['groups'][0]['group_sha256']
    wire['groups'][0]['group_sha256'] = 'f' * 64
    assert original != wire['groups'][0]['group_sha256']
    wire = context._seal(wire)
    assert context.validate_context(wire) == wire
    assert context.expand_context(wire)['groups'][0]['group_sha256'] != original
