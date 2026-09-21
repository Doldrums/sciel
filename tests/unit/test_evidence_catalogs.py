"""Pure JSON/data fixtures: no providers, candidates, evaluators or saved runs."""
import copy
from decimal import Decimal
import hashlib
import json
import math

import pytest

from sciel.knowledge_base.catalogs import (
    SCHEMA, canonical_json, canonical_sha, expand_measurements, pack_measurements,
)


def measurement(**changes):
    return {'units': 'unit_square_side_length', 'validity': 0,
            'invalid_semantics': 'penalty sentinel, not a physical zero',
            'details': 'Measured data, not causal evidence. ' * 40,
            'float_zero': -0.0, 'integer_zero': 0, 'nullable': None,
            'opaque': {'catalog_ref': 'literal data', 'metric': {'nested': True}},
            **changes}


def example():
    metric = measurement()
    value = {'metric': metric, 'records': [{'parent_cases': copy.deepcopy(metric)},
             {'evaluation': {'cases': copy.deepcopy(metric)}}], 'unknown': None,
             'hypothesis': {'text': 'Unverified гипотеза', 'delta_low': -0.1, 'delta_high': 1}}
    paths = [('metric',), ('records', 0, 'parent_cases'), ('records', 1, 'evaluation', 'cases')]
    return value, paths


def test_realistic_repeated_measurements_restore_types_and_all_evidence():
    value, paths = example(); before = canonical_json(value)
    envelope = pack_measurements(value, paths)
    assert envelope['schema'] == SCHEMA
    assert len(envelope['catalog']) == 1 and len(envelope['bindings']) == 3
    restored = expand_measurements(envelope)
    assert canonical_json(restored) == before == canonical_json(value)
    assert type(restored['metric']['integer_zero']) is int
    assert type(restored['metric']['float_zero']) is float
    assert math.copysign(1, restored['metric']['float_zero']) == -1
    assert restored['metric']['validity'] == 0
    assert restored['metric']['invalid_semantics'] == value['metric']['invalid_semantics']
    assert restored['hypothesis'] == value['hypothesis']
    assert canonical_sha(restored) == envelope['original_sha256']


def test_canonical_json_matches_explicit_wire_and_distinguishes_number_types():
    value = {'z': 'ё\u2028', 'a': [None, True, False, -0.0, 0, 1.0, 1, 1e-100]}
    expected = json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False, separators=(',', ':'))
    assert canonical_json(value) == expected
    assert canonical_sha(value) == hashlib.sha256(expected.encode('ascii')).hexdigest()
    assert len({canonical_sha(v) for v in (0, 0.0, -0.0, False, None)}) == 5
    assert canonical_sha('é') != canonical_sha('é')


def test_only_explicit_paths_authorize_refs_even_for_identical_literal_marker():
    data = measurement(); digest = canonical_sha(data)
    literal = {'catalog_ref': digest}
    value = {'first': data, 'second': copy.deepcopy(data), 'literal': literal,
             'artifact': {'metric': measurement(units='opaque task value')}}
    envelope = pack_measurements(value, [('first',), ('second',)])
    assert envelope['payload']['literal'] == envelope['payload']['first'] == literal
    assert envelope['payload']['artifact'] == value['artifact']
    assert {tuple(b['path']) for b in envelope['bindings']} == {('first',), ('second',)}
    restored = expand_measurements(envelope)
    assert restored['literal'] == literal and restored['literal'] != restored['first']
    assert canonical_json(restored) == canonical_json(value)


def test_opaque_cases_with_reserved_looking_fields_are_not_recursively_interpreted():
    data = measurement(bindings=[{'path': ['fake'], 'sha256': 'b'*64}],
                       schema=SCHEMA, original_sha256='a'*64,
                       catalog={'a'*64: {'catalog_ref': 'b'*64}})
    value = [data, copy.deepcopy(data)]
    envelope = pack_measurements(value, [(0,), (1,)])
    assert canonical_json(expand_measurements(envelope)) == canonical_json(value)


def test_mutation_never_aliases_original_envelope_or_other_restored_positions():
    shared = measurement(); value = {'a': shared, 'b': shared}; paths = [['a'], ['b']]
    original = copy.deepcopy(value); original_paths = copy.deepcopy(paths)
    envelope = pack_measurements(value, paths); saved = copy.deepcopy(envelope)
    restored = expand_measurements(envelope)
    restored['a']['opaque']['metric']['nested'] = 'changed'
    assert restored['b']['opaque']['metric']['nested'] is True
    assert value == original and paths == original_paths and envelope == saved
    next(iter(envelope['catalog'].values()))['details'] = 'modified'
    assert value == original


def test_deterministic_greedy_order_counts_catalog_and_binding_overhead():
    value, paths = example()
    value['small_a'] = {'n': 1}; value['small_b'] = {'n': 1}
    value['unique'] = measurement(unique='one occurrence')
    paths += [('small_a',), ('small_b',), ('unique',)]
    packed = pack_measurements(value, paths)
    reordered = dict(reversed(list(value.items())))
    assert canonical_json(packed) == canonical_json(pack_measurements(reordered, list(reversed(paths))))
    assert len(canonical_json(packed)) < len(canonical_json(pack_measurements(value)))
    assert len(packed['catalog']) == 1
    assert packed['payload']['small_a'] == {'n': 1}
    assert packed['payload']['unique'] == value['unique']


def test_equal_python_numbers_with_different_json_representation_never_merge():
    values = [measurement(number=n) for n in (1, 1.0, -0.0, 0.0)]
    value = [v for v in values for _ in range(2)]
    envelope = pack_measurements(value, [(i,) for i in range(len(value))])
    assert len(envelope['catalog']) == 4
    assert canonical_json(expand_measurements(envelope)) == canonical_json(value)


def test_missing_and_null_paths_remain_distinct_without_creating_data():
    value = {'null': None, 'array': [None], 'object': {}, 'literal': {'metric': None}}
    envelope = pack_measurements(value, [('missing', 'metric'), ('null', 'cases'),
        ('array', 0, 'cases'), ('array', 8), ('object', 'absent'), ('literal', 'metric')])
    restored = expand_measurements(envelope)
    assert restored == value and 'missing' not in restored and len(restored['array']) == 1
    assert envelope['bindings'] == [] and envelope['catalog'] == {}


@pytest.mark.parametrize('paths', [None, 'metric', {'metric'}, [None], ['metric'],
    [('records', True)], [('records', -1)], [('records', 1.0)], [('records', None)],
    [('metric',), ('metric',)], [('metric',), ('metric', 'units')], [(), ('metric',)],
    [('absent',), ('absent', 'deeper')]])
def test_malformed_duplicate_or_overlapping_path_plan_rejected(paths):
    value, _ = example()
    with pytest.raises(ValueError): pack_measurements(value, paths)


@pytest.mark.parametrize('value,path', [({'a': 1}, ('a', 'b')), ({'a': False}, ('a', 0)),
    ({'a': 'x'}, ('a',)), ({'a': []}, ('a',)), ({'a': {}}, ('a', 0)),
    ({'a': []}, ('a', '0'))])
def test_known_non_dictionary_target_or_wrong_container_path_rejected(value, path):
    with pytest.raises(ValueError): pack_measurements(value, [path])


@pytest.mark.parametrize('value', [None, False, 0, -0.0, 'unknown', [], {}, [1, {'x': 2}]])
def test_arbitrary_json_roundtrip_without_marked_paths(value):
    assert canonical_json(expand_measurements(pack_measurements(value))) == canonical_json(value)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), (1, 2),
    {1: 'integer key'}, {False: 'bool key'}, {'nested': {1, 2}}, Decimal('1.0'), b'bytes', object()])
def test_non_json_or_nonfinite_input_rejected(value):
    with pytest.raises(ValueError): pack_measurements({'value': value})
    with pytest.raises(ValueError): canonical_json(value)


@pytest.mark.parametrize('container', ['dict', 'list'])
def test_cycles_rejected_in_packing_and_expansion(container):
    value = {} if container == 'dict' else []
    if container == 'dict': value['self'] = value
    else: value.append(value)
    with pytest.raises(ValueError, match='cyclic'): pack_measurements(value)
    envelope = pack_measurements(None); envelope['payload'] = value
    with pytest.raises(ValueError, match='cyclic'): expand_measurements(envelope)


@pytest.mark.parametrize('change', ['schema', 'extra_key', 'missing_key', 'original_hash',
    'catalog_hash', 'catalog_scalar', 'catalog_unused', 'catalog_dangling', 'binding_extra',
    'binding_duplicate', 'binding_overlap', 'binding_bool', 'binding_negative', 'binding_tuple',
    'binding_missing', 'binding_null', 'binding_scalar', 'placeholder_extra', 'placeholder_hash',
    'payload_unbound', 'nonfinite', 'bindings_object'])
def test_expansion_rejects_corrupt_envelope_without_mutation(change):
    value, paths = example(); envelope = pack_measurements(value, paths)
    first = envelope['bindings'][0]; digest = first['sha256']; first_path = first['path']
    def replace_first(v):
        parent = envelope['payload']
        for part in first_path[:-1]: parent = parent[part]
        parent[first_path[-1]] = v
    if change == 'schema': envelope['schema'] += '-unknown'
    elif change == 'extra_key': envelope['extra'] = None
    elif change == 'missing_key': del envelope['original_sha256']
    elif change == 'original_hash': envelope['original_sha256'] = '0'*64
    elif change == 'catalog_hash': envelope['catalog'][digest]['units'] = 'wrong'
    elif change == 'catalog_scalar': envelope['catalog'][digest] = 'wrong'
    elif change == 'catalog_unused':
        v = {'extra': 'unused'}; envelope['catalog'][canonical_sha(v)] = v
    elif change == 'catalog_dangling': del envelope['catalog'][digest]
    elif change == 'binding_extra': first['extra'] = None
    elif change == 'binding_duplicate': envelope['bindings'].append(copy.deepcopy(first))
    elif change == 'binding_overlap': envelope['bindings'].append({'path': [], 'sha256': digest})
    elif change == 'binding_bool': first['path'] = ['records', True, 'parent_cases']
    elif change == 'binding_negative': first['path'] = ['records', -1]
    elif change == 'binding_tuple': first['path'] = tuple(first['path'])
    elif change == 'binding_missing': first['path'] = ['not_present']
    elif change == 'binding_null': first['path'] = ['unknown', 'deeper']
    elif change == 'binding_scalar': first['path'] = ['hypothesis', 'delta_low', 'deeper']
    elif change == 'placeholder_extra': replace_first({'catalog_ref': digest, 'extra': True})
    elif change == 'placeholder_hash': replace_first({'catalog_ref': 'b'*64})
    elif change == 'payload_unbound': envelope['payload']['hypothesis']['delta_low'] = 0
    elif change == 'nonfinite': envelope['payload']['unknown'] = float('nan')
    else: envelope['bindings'] = {}
    before = copy.deepcopy(envelope)
    with pytest.raises(ValueError): expand_measurements(envelope)
    # NaN is deliberately not comparable to itself; identity still stays intact.
    if change != 'nonfinite': assert envelope == before
    else: assert math.isnan(envelope['payload']['unknown'])


def test_root_binding_and_literal_reference_inside_catalog_expand_once_only():
    original = {'catalog_ref': 'literal', 'metric': measurement()}
    digest = canonical_sha(original)
    envelope = {'schema': SCHEMA, 'original_sha256': digest,
                'payload': {'catalog_ref': digest}, 'catalog': {digest: original},
                'bindings': [{'path': [], 'sha256': digest}]}
    assert canonical_json(expand_measurements(envelope)) == canonical_json(original)


def test_empty_dictionary_and_unique_large_dictionary_are_not_interned():
    value = {'a': {}, 'b': {}, 'unique': measurement()}
    envelope = pack_measurements(value, [('a',), ('b',), ('unique',)])
    assert envelope['catalog'] == {} and envelope['bindings'] == []
    assert expand_measurements(envelope) == value
