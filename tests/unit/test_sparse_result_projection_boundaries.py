"""Independent sparse-projection integrity boundaries; all saved values are inert."""
from copy import deepcopy
import json

import pytest

from sciel.research_tools.result_projection import (
    MIN_CONTEXT_CHARS, SPARSE_PROJECTION_SCHEMA, _context,
    projection_fields, preserves_measurements, success_context, verify_projection,
)
from test_observation_row_projection import sample
from test_paired_result_projection import witness


def encoded(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)


def original():
    return {'output': {'observations': {
        'escaped/a~雪': {'': None, 'enabled': False, 'count': 0, 'number': 1.0,
            'empty_object': {}, 'empty_array': [], '0': 'numeric key', 'vec': [False, 0, 0.0]},
        'summary': {f'c{i}': {'mean': i * .01, 'median': i * .02} for i in range(12)},
        'source': 'x' * 10000}}}


def projected(item, budget):
    saved = encoded(item)
    text = success_context(item, budget)
    assert len(text) <= budget and text.isascii()
    value = json.loads(text)
    verify_projection(value, item)
    assert encoded(item) == saved
    return value


def test_real_sparse_result_preserves_escaped_empty_and_typed_values_without_complete_subtree_claim():
    item = original()
    value = projected(item, 900)
    assert value['schema'] == SPARSE_PROJECTION_SCHEMA
    assert value['fields']['output']['observations']['escaped/a~雪'] == item['output']['observations']['escaped/a~雪']
    fields = {row['path']: row['value'] for row in projection_fields(value)}
    prefix = '/output/observations/escaped~1a~0雪/'
    assert fields[prefix] is None
    assert fields[prefix + 'enabled'] is False
    assert type(fields[prefix + 'count']) is int
    assert type(fields[prefix + 'number']) is float
    assert fields[prefix + 'empty_object'] == {} and fields[prefix + 'empty_array'] == []
    assert fields[prefix + '0'] == 'numeric key'
    assert encoded(fields[prefix + 'vec']) == '[false,0,0.0]'
    assert '/output/observations/escaped~1a~0雪' not in fields
    assert '/output/observations/summary' not in fields
    assert value['observations_state'] == 'partial' and value['omitted_fields'] == 1
    assert value['omitted_paths'] == ['/output/observations/source']


@pytest.mark.parametrize('damage', [
    'bool_to_int', 'int_to_bool', 'float_to_int', 'missing_leaf', 'extra_leaf',
    'array_to_object', 'partial_array', 'empty_to_missing', 'empty_type',
    'escaped_key', 'invented_subtree', 'stale_hash', 'false_complete', 'wrong_count',
    'wrong_omitted_path', 'empty_root', 'nonobject_root', 'undeclared_supplement',
])
def test_sparse_mutations_cannot_change_value_path_type_or_coverage(damage):
    item = original()
    value = projected(item, 900)
    tree = value['fields']['output']['observations']
    special = tree['escaped/a~雪']
    if damage == 'bool_to_int': special['enabled'] = 0
    elif damage == 'int_to_bool': special['count'] = False
    elif damage == 'float_to_int': special['number'] = 1
    elif damage == 'missing_leaf': del special['count']
    elif damage == 'extra_leaf': special['invented'] = 0
    elif damage == 'array_to_object': special['vec'] = {'0': False, '1': 0, '2': 0.0}
    elif damage == 'partial_array': special['vec'] = special['vec'][:2]
    elif damage == 'empty_to_missing': del special['empty_object']
    elif damage == 'empty_type': special['empty_object'] = []
    elif damage == 'escaped_key': tree['escaped~1a~0雪'] = tree.pop('escaped/a~雪')
    elif damage == 'invented_subtree': tree['summary'] = []
    elif damage == 'stale_hash': value['original_sha256'] = '0' * 64
    elif damage == 'false_complete': value['observations_state'] = 'complete'
    elif damage == 'wrong_count': value['omitted_fields'] = False
    elif damage == 'wrong_omitted_path': value['omitted_paths'] = ['/output/observations/summary']
    elif damage == 'empty_root': value['fields'] = {}
    elif damage == 'nonobject_root': value['fields'] = []
    else: value['comparison_projection'] = None
    with pytest.raises(ValueError):
        verify_projection(value, item)


@pytest.mark.parametrize('budget', [MIN_CONTEXT_CHARS, 100, 299, 450, 600, 899, 900, 1600, 2600, 50000])
def test_exact_budget_selection_never_regresses_historical_atomic_coverage(budget):
    item = original()
    old = json.loads(_context(item, budget))
    new = projected(item, budget)
    verify_projection(old, item)
    assert preserves_measurements(old, new)
    if new.get('schema') == SPARSE_PROJECTION_SCHEMA:
        assert not preserves_measurements(new, old)  # /4 must earn its format change.
    else:
        assert encoded(new) == encoded(old)
    reversed_item = json.loads(encoded(item), object_pairs_hook=lambda pairs: dict(reversed(pairs)))
    assert success_context(reversed_item, budget) == encoded(new)


@pytest.mark.parametrize(('builder', 'budget', 'supplement'), [
    (sample, 5000, 'observation_rows'), (witness, 8000, 'comparison_projection'),
])
def test_sparse_extra_summary_retains_whole_rows_and_every_paired_arm(builder, budget, supplement):
    item = builder()
    item['output']['observations']['summary'] = {
        f'cut{i}': {'mean': i * .01, 'median': i * .02} for i in range(18)}
    old = json.loads(_context(item, budget))
    new = projected(item, budget)
    assert new['schema'] == SPARSE_PROJECTION_SCHEMA
    assert preserves_measurements(old, new) and not preserves_measurements(new, old)
    if supplement == 'comparison_projection':
        assert new[supplement] == old[supplement]
        assert new[supplement]['shown_rows'] == 6
        for group in new[supplement]['groups']:
            assert len(group['rows']) == 3
        broken = deepcopy(new)
        broken[supplement]['groups'][0]['rows'].pop()
    else:
        assert new[supplement]['shown_rows'] >= old[supplement]['shown_rows'] > 0
        for array in new[supplement]['arrays']:
            for row in array['rows']:
                assert row['value'] == item['output']['observations']['arbitrary/a~雪'][row['index']]
        broken = deepcopy(new)
        del broken[supplement]['arrays'][0]['rows'][0]['value']['finite_steps']
    with pytest.raises(ValueError):
        verify_projection(broken, item)


def test_typed_coverage_does_not_confuse_bool_int_float_or_partial_container():
    old = {'output': {'observations': {'vec': [False, 0, 0.0], 'stat': 1.0}}}
    for path, replacement in [('stat', 1), ('vec', [0, 0, 0.0]), ('vec', [False, 0])]:
        changed = deepcopy(old)
        changed['output']['observations'][path] = replacement
        assert not preserves_measurements(old, changed)
    assert preserves_measurements(old, deepcopy(old))
