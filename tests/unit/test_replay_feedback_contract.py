"""Bounded replay reasons over detached receipts; no tool code executes."""
from copy import deepcopy
import json

import pytest

from sciel.research_tools.replay_feedback import (
    build_feedback, is_projection, shorten_feedback, validate_feedback)
from test_research_tool_result_contract import receipt, result


def feedback(left, right, *, policy='exact_observations', changes=None):
    a = {'receipt': receipt(left), 'replay_policy': policy}
    b = {'receipt': receipt(right, **(changes or {})), 'replay_policy': policy}
    frozen = deepcopy((a, b))
    projected = build_feedback(a, b, tool_id='a'*64,
        reference_receipt_id='e'*64, replay_receipt_id='f'*64)
    assert (a, b) == frozen
    return projected


def test_exact_elapsed_observation_diff_is_actionable_but_diagnostics_are_excluded():
    value = feedback(result({'elapsed_time': 1.}, {'elapsed_time': 99.}),
                     result({'elapsed_time': 2.}, {'elapsed_time': 100.}))
    assert value['reason'] == 'observations_differ' and not value['matched']
    assert value['changed_paths'] == [{'path': '/elapsed_time', 'reference': '1.0', 'replay': '2.0'}]
    same = feedback(result({'sample': 1}, {'elapsed_time': 1}), result({'sample': 1}, {'elapsed_time': 2}))
    assert same['reason'] == 'observations_match' and same['measurement_reproduced']
    assert same['changed_paths'] == []


def test_independent_execution_reports_actual_difference_without_claiming_reproduction():
    value = feedback(result({'sample': 1}), result({'sample': 2}), policy='independent_execution')
    assert value['reason'] == 'independent_execution_passed'
    assert value['matched'] and not value['measurement_reproduced']
    assert value['changed_paths'][0]['path'] == '/sample'


@pytest.mark.parametrize('changes,reason', [
    ({'runtime_sha256': '9'*64}, 'binding_mismatch'),
    ({'input_sha256': '9'*64}, 'binding_mismatch'),
    ({'status': 'failed'}, 'replay_not_successful'),
])
def test_binding_and_execution_failure_never_disguised_as_measurement_change(changes, reason):
    value = feedback(result({'a': 1}), result({'a': 2}), changes=changes)
    assert value['reason'] == reason and not value['matched']
    if reason == 'binding_mismatch':
        assert value['binding_mismatches'] == list(changes)


def test_large_arrays_and_many_changes_remain_bounded_with_honest_omissions():
    value = feedback(result({str(i): [0]*10000 for i in range(20)}),
                     result({str(i): [1]*10000 for i in range(20)}))
    assert len(value['changed_paths']) == 8 and value['changed_paths_omitted'] == 12
    assert len(json.dumps(value)) < 3000
    short = shorten_feedback(value, 1)
    assert short['changed_paths_omitted'] == 19 and is_projection(short, value)
    short['changed_paths'][0]['path'] = '/fabricated'
    assert not is_projection(short, value)


def test_exact_json_pointer_and_added_removed_fields_are_not_unescaped_names():
    value = feedback(result({'a/b~': {'gone': 1}}), result({'a/b~': {'added': None}}))
    assert value['changed_paths'] == [
        {'path': '/a~1b~0/added', 'reference': '<absent>', 'replay': 'null'},
        {'path': '/a~1b~0/gone', 'reference': '1', 'replay': '<absent>'}]


def test_very_long_key_is_summarized_at_exact_parent_pointer():
    value = feedback(result({'long'*100: 1}), result({'long'*100: 2}))
    assert value['changed_paths'][0]['path'] == ''
    assert len(json.dumps(value)) < 1400


@pytest.mark.parametrize('patch', [
    {'reference_observations_sha256': None},
    {'replay_observations_sha256': None},
    {'policy': 'independent_execution'},
    {'binding_mismatches': ['runtime_sha256']},
    {'reason': 'binding_mismatch'},
    {'matched': True}, {'measurement_reproduced': True},
    {'changed_paths_omitted': True}, {'unexpected': True},
])
def test_inconsistent_typed_feedback_is_rejected(patch):
    value = feedback(result({'sample': 1}), result({'sample': 2}))
    assert not validate_feedback({**value, **patch})


def test_equal_hashes_cannot_be_described_as_observations_differ():
    value = feedback(result({'sample': 1}), result({'sample': 2}))
    value['replay_observations_sha256'] = value['reference_observations_sha256']
    assert not validate_feedback(value)


def test_projection_cannot_change_policy_and_policy_mismatch_raises():
    value = feedback(result({'sample': 1}), result({'sample': 2}))
    altered = deepcopy(value)
    altered.update(policy='independent_execution', matched=True,
                   status='independently_executed', reason='independent_execution_passed')
    assert validate_feedback(altered) and not is_projection(altered, value)
    with pytest.raises(ValueError, match='immutable policy'):
        build_feedback({'receipt': receipt(result()), 'replay_policy': 'exact_observations'},
            {'receipt': receipt(result()), 'replay_policy': 'independent_execution'},
            tool_id='a'*64, reference_receipt_id='e'*64, replay_receipt_id='f'*64)
