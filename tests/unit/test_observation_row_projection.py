"""Whole observation records at arbitrary pointers; no generated code runs."""
from copy import deepcopy
import json

import pytest

from sciel.methods.models import content_hash
from sciel.research_tools.result_projection import (
    MIN_CONTEXT_CHARS, ROW_PROJECTION_SCHEMA, _legacy_context, _paired_context,
    success_context, verify_projection,
)


def sample():
    return {'output': {'observations': {
        'arbitrary/a~雪': [{'tolerance': 10. ** -n, 'active_count': 15,
            'max_gain': -1.3299720696675932e-8, 'finite_steps': [
                {'step': 10. ** -k, 'value': 0.027765494023017284, 'valid': True}
                for k in range(3, 9)], 'labels': ['雪', 'a\\b', '"']}
            for n in (10, 8, 6) * 8],
        'baseline': 0.027765494023017503,
        'continuation_state': {'points': [[0.1, 0.2]] * 300},
        'source_tail': 'large echoed source' * 1000,
    }}, 'diagnostic': None}


def project(item, budget):
    before = deepcopy(item)
    text = success_context(item, budget)
    assert text.isascii() and len(text) <= budget and item == before
    value = json.loads(text)
    verify_projection(value, item)
    return value


def test_unknown_key_keeps_complete_exact_rows_before_state_and_source():
    item = sample()
    value = project(item, 3000)
    assert value['schema'] == ROW_PROJECTION_SCHEMA
    rows = value['observation_rows']; array = rows['arrays'][0]
    path = '/output/observations/arbitrary~1a~0雪'
    assert array['path'] == path
    assert 0 < rows['shown_rows'] < 24
    assert rows['shown_rows'] + rows['omitted_rows'] == rows['total_rows'] == 24
    assert array['original_sha256'] == content_hash(item['output']['observations']['arbitrary/a~雪'])
    for row in array['rows']:
        assert row['path'] == f"{path}/{row['index']}"
        original = item['output']['observations']['arbitrary/a~雪'][row['index']]
        assert row['value'] == original and row['original_sha256'] == content_hash(original)
    assert value['observations_state'] == 'partial'
    assert path in value['omitted_paths']
    assert 'not verified scientific claims' in rows['scope']


def test_order_is_source_index_not_outcome_and_oversize_rows_are_omitted_whole():
    item = sample()
    raw = item['output']['observations']['arbitrary/a~雪']
    raw[0]['large'] = 'x' * 100000
    raw[1]['max_gain'] = -100
    raw[2]['max_gain'] = 100000
    value = project(item, 2600)
    rows = value['observation_rows']['arrays'][0]['rows']
    assert rows[0]['index'] == 1 and rows[0]['value'] == raw[1]
    assert all(row['index'] != 0 for row in rows)
    # Hash authenticates the complete original row, never a selection of fields.
    assert rows[0]['original_sha256'] == content_hash(raw[1])


def test_multiple_arrays_and_nested_pointer_coverage():
    item = sample()
    item['output']['observations']['nested'] = {'other': [
        {'index': index, 'z': 'y' * 500} for index in range(24)]}
    value = project(item, 8000)
    rows = value['observation_rows']
    assert rows['total_arrays'] == rows['shown_arrays'] == 2
    assert rows['total_rows'] == 48
    assert rows['shown_rows'] + rows['omitted_rows'] == 48
    assert rows['omitted_arrays'] == 0
    assert [a['path'] for a in rows['arrays']] == [
        '/output/observations/arbitrary~1a~0雪', '/output/observations/nested/other']


def test_numerical_vectors_and_mixed_arrays_remain_atomic():
    item = {'output': {'observations': {'numbers': [1.2] * 1000,
        'geometry': [[1, 2]] * 1000, 'mixed': [{'v': 1}, 2] * 1000}}, 'diagnostic': 'x' * 5000}
    value = project(item, 1200)
    assert value['schema'] != ROW_PROJECTION_SCHEMA
    assert not value['fields']
    assert value['omitted_observation_fields'] == 3


def test_full_fitting_output_and_historical_views_stay_exact():
    item = sample()
    assert project(item, 100000) == item
    verify_projection(json.loads(_legacy_context(item, 3000)), item)
    verify_projection(json.loads(_paired_context(item, 3000)), item)
    complete = {'output': {'observations': {'records': [{'x': 1}, {'x': 2}],
        'source_tail': 'x' * 20000}}}
    assert project(complete, 2000)['schema'] == 'sciel-tool-result-projection/1'


@pytest.mark.parametrize('damage', ['value', 'partial_row', 'index', 'bool_index', 'path',
    'row_hash', 'array_hash', 'source_hash', 'duplicate', 'order', 'count', 'array_count',
    'scope', 'array_path', 'complete_claim', 'fields_duplicate', 'comparison_lie'])
def test_forged_whole_rows_and_coverage_rejected(damage):
    item = sample(); value = project(item, 8000)
    block = value['observation_rows']; array = block['arrays'][0]; row = array['rows'][0]
    if damage == 'value': row['value']['max_gain'] = 42
    elif damage == 'partial_row': row['value'].pop('finite_steps')
    elif damage == 'index': row['index'] = 9
    elif damage == 'bool_index': row['index'] = False
    elif damage == 'path': row['path'] += '/0'
    elif damage == 'row_hash': row['original_sha256'] = '0'*64
    elif damage == 'array_hash': array['original_sha256'] = '0'*64
    elif damage == 'source_hash': value['original_sha256'] = '0'*64
    elif damage == 'duplicate': array['rows'].append(deepcopy(row))
    elif damage == 'order': array['rows'].reverse()
    elif damage == 'count': block['omitted_rows'] += 1
    elif damage == 'array_count': block['total_arrays'] = True
    elif damage == 'scope': block['scope'] = 'Canonical benefit established'
    elif damage == 'array_path': array['path'] = '/output/observations/other'
    elif damage == 'complete_claim': value['observations_state'] = 'complete'
    elif damage == 'fields_duplicate': value['fields'].append({'path': array['path'],
        'value': item['output']['observations']['arbitrary/a~雪']})
    else: value['comparison_projection'] = {}
    with pytest.raises(ValueError):
        verify_projection(value, item)


def test_comparison_seed_groups_preserved_alongside_generic_rows():
    item = sample(); obs = item['output']['observations']
    obs['comparison_results'] = [dict(arm_id=arm, seed=seed, cpu_seconds=.1,
        best_result=-100 if arm == 'a' else 100, conditions={}, saved_state='s' * 6000)
        for arm in ('a', 'b') for seed in (0, 1)]
    value = project(item, 8500)
    assert value['schema'] == ROW_PROJECTION_SCHEMA
    comp = value['comparison_projection']
    assert comp['shown_rows'] == 4
    for group in comp['groups']:
        assert [r['values']['arm_id'] for r in group['rows']] == ['a', 'b']
    assert all(a['path'] != '/output/observations/comparison_results'
               for a in value['observation_rows']['arrays'])
    broken = deepcopy(value); broken['comparison_projection']['groups'][0]['rows'].pop()
    with pytest.raises(ValueError): verify_projection(broken, item)


def test_unicode_and_budget_boundaries_never_return_truncated_json():
    item = sample()
    for budget in (MIN_CONTEXT_CHARS, 100, 300, 600, 1000, 1500, 1999, 2000, 2500, 5000, 8000, 10000):
        project(item, budget)


@pytest.mark.parametrize('change', ['oversized', 'malformed'])
def test_invalid_comparison_group_never_falls_back_to_generic_single_arm(change):
    item = sample()
    comparisons = [dict(arm_id=arm, seed=0, cpu_seconds=1, best_result=1,
                        conditions={}) for arm in ('a', 'b')]
    if change == 'oversized':
        comparisons[1]['best_result'] = 'x' * 100000
    else:
        del comparisons[1]['best_result']
        comparisons[1]['extra'] = 'x' * 100000
    item['output']['observations']['comparison_results'] = comparisons
    value = project(item, 5000)
    assert value['schema'] == ROW_PROJECTION_SCHEMA
    assert value['comparison_projection'] is None
    assert all(a['path'] != '/output/observations/comparison_results'
               for a in value['observation_rows']['arrays'])
    assert all(f['path'] != '/output/observations/comparison_results' for f in value['fields'])


def test_existing_paired_group_cannot_be_lost_to_new_schema_overhead():
    item = {'output': {'observations': {
        'comparison_results': [dict(arm_id=arm, seed=0, cpu_seconds=1,
            best_result=index, conditions={'x': 1}, saved_state='s' * 6000)
            for index, arm in enumerate(('a', 'b'), 1)],
        'checks': [{'measurement': -.1, 'valid': True}] * 30,
        'source_tail': 'x' * 5000}}}
    previous = json.loads(_paired_context(item, 1500))
    assert previous['comparison_projection']['shown_rows'] == 2
    value = project(item, 1500)
    assert value['comparison_projection'] == previous['comparison_projection']


def test_complete_prior_array_is_not_traded_for_partial_rows_of_another():
    item = sample()
    original = [{'reading': i, 'measurement': -1.23} for i in range(2)]
    item['output']['observations']['small_records'] = original
    prior = json.loads(_paired_context(item, 5000))
    path = '/output/observations/small_records'
    assert next(f['value'] for f in prior['fields'] if f['path'] == path) == original
    value = project(item, 5000)
    assert next(f['value'] for f in value['fields'] if f['path'] == path) == original
    assert value['observation_rows']['total_arrays'] == 1
    assert value['observation_rows']['shown_rows'] > 0


def test_declared_continuation_state_records_cannot_evict_measurement_rows():
    item = sample()
    item['output']['observations']['continuation_state'] = {'history': [
        {'iteration': i, 'points': [[0.1, 0.2]] * 30} for i in range(20)]}
    value = project(item, 3000)
    arrays = value['observation_rows']['arrays']
    assert arrays and all('/continuation_state/' not in array['path'] for array in arrays)
    assert arrays[0]['rows'][0]['value']['max_gain'] == -1.3299720696675932e-8
