"""Pure saved-JSON projection checks; no authored code or providers execute."""
from copy import deepcopy
import hashlib
import json

import pytest

from sciel.research_tools.result_projection import (
    MIN_CONTEXT_CHARS, PROJECTION_SCHEMAS, success_context, verify_projection, projection_fields,
)


def wrapper(observations, diagnostics=None):
    return {'output': {'schema': 'sciel-tool-result/1', 'status': 'success',
            'observations': observations, 'diagnostics': diagnostics or {}, 'error': None},
        'result': {'status': 'success', 'host_status': 'ok'},
        'replay_policy': 'independent_execution', 'reason': None, 'diagnostic': None}


def pointer(item, path):
    for part in path.split('/')[1:]:
        item = item[part.replace('~1', '/').replace('~0', '~')]
    return item


def projected(item, limit):
    text = success_context(item, limit)
    assert text.isascii() and len(text) <= limit
    result = json.loads(text)
    assert result['schema'] in PROJECTION_SCHEMAS
    verify_projection(result, item)
    for field in projection_fields(result):
        assert pointer(item, field['path']) == field['value']
    return result


def test_full_result_semantically_unchanged_and_input_never_mutated():
    item = wrapper({'score': 0.0, 'message': '雪 "quoted"\nline', 'array': [1, None, False]})
    before = deepcopy(item)
    text = success_context(item, 5000)
    assert json.loads(text) == item and text.isascii()
    assert item == before
    assert success_context(item, len(text)) == text


def test_large_diagnostics_cannot_displace_actual_observations():
    item = wrapper({'measured_delta': 0.125, 'samples': 7}, {'trace': 'x' * 50000})
    assert len(json.dumps(item)) < 65536
    before = deepcopy(item)
    view = projected(item, 1500)
    values = {row['path']: row['value'] for row in projection_fields(view)}
    assert values['/output/observations/measured_delta'] == 0.125
    assert values['/output/observations/samples'] == 7
    assert '/output/diagnostics/trace' not in values
    assert view['observations_state'] == 'complete'
    assert view['omitted_observation_fields'] == 0
    assert '/output/diagnostics/trace' in view['omitted_paths']
    assert item == before


def test_opus_shaped_nested_summary_survives_large_per_layout_array():
    # Shape of the actual numerical experiment; small synthetic data only.
    summary = {'n_layouts': 7, 'p25_shrink_loss_score_units': {
        'max': 9.586342261853659e-10, 'mean': 2.9544068164772997e-10},
        'all_repairs_strictly_feasible': True}
    observations = {'inputs': {'selected_source_sha256': 'a' * 64},
        'numeric_experiment': {'available': True,
            'per_layout': [{'layout': str(i), 'detail': 'x' * 200} for i in range(100)],
            'summary': summary, 'target_sum': 2.635}, 'z_after_large_array': 42}
    item = wrapper(observations, {'stages': ['static_done', 'numeric_done']})
    view = projected(item, 2500)
    values = {row['path']: row['value'] for row in projection_fields(view)}
    assert values['/output/observations/numeric_experiment/summary/n_layouts'] == 7
    assert values['/output/observations/numeric_experiment/summary/p25_shrink_loss_score_units/max'] == summary['p25_shrink_loss_score_units']['max']
    assert values['/output/observations/z_after_large_array'] == 42
    assert '/output/observations/numeric_experiment/per_layout' not in values
    assert view['observations_state'] == 'partial'
    assert view['omitted_observation_fields'] == 1
    assert '/output/observations/numeric_experiment/per_layout' in view['omitted_paths']


@pytest.mark.parametrize('limit', [MIN_CONTEXT_CHARS, 64, 100, 300, 500, 1000])
def test_tiny_budgets_are_explicit_valid_json_not_fake_empty_observations(limit):
    item = wrapper({'unrenderable': 'x' * 30000})
    view = projected(item, limit)
    if 'fields' in view:
        assert view['observations_state'] == 'all_omitted'
        assert view['omitted_observation_fields'] == 1
    else:
        assert view['omitted'] is True
    assert 'observations' not in view


def test_original_empty_observations_differ_from_budget_omission():
    view = projected(wrapper({}, {'large': 'x' * 40000}), 1200)
    assert view['observations_state'] == 'originally_empty'
    assert {'path': '/output/observations', 'value': {}} in projection_fields(view)
    assert view['omitted_observation_fields'] == 0


def test_escaped_unicode_keys_pointers_values_and_whole_arrays():
    item = wrapper({'雪/a~b': {'"\n': 'λ\n"'}, 'whole': [1, 2, {'z': 0}],
                    'oversized': ['x' * 9000]}, {'trace': 'y' * 30000})
    view = projected(item, 1600)
    fields = {row['path']: row['value'] for row in projection_fields(view)}
    assert fields['/output/observations/雪~1a~0b/"\n'] == 'λ\n"'
    assert fields['/output/observations/whole'] == [1, 2, {'z': 0}]
    assert '/output/observations/oversized' not in fields
    assert not any(path.startswith('/output/observations/oversized/') for path in fields)


def test_omission_paths_bounded_without_eviction_or_order_dependence():
    obs = {f'large/{i}': 'x' * 2000 for i in range(30)}
    obs['tiny'] = 3
    item = wrapper(obs)
    view = projected(item, 1400)
    assert {'path': '/output/observations/tiny', 'value': 3} in projection_fields(view)
    assert len(view['omitted_paths']) <= 16
    assert len(view['omitted_paths']) + view['omitted_paths_omitted'] == view['omitted_fields']
    reordered = wrapper(dict(reversed(list(obs.items()))))
    assert success_context(item, 1400) == success_context(reordered, 1400)
    canonical = json.dumps(item, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert view['original_sha256'] == hashlib.sha256(canonical.encode('ascii')).hexdigest()


@pytest.mark.parametrize('limit', [MIN_CONTEXT_CHARS - 1, 0, -1, True, 100.0])
def test_unsupported_budget_rejected_instead_of_slicing_json(limit):
    with pytest.raises(ValueError, match='limit'):
        success_context(wrapper({'value': 1}), limit)


def test_nonfinite_json_rejected_without_inventing_a_value():
    with pytest.raises(ValueError):
        success_context(wrapper({'value': float('nan')}), 1000)
