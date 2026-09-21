"""Scientific-result contracts over saved receipts; no generated tool executes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from sciel.methods.models import content_hash
from sciel.research_tools.result_contract import parse_result, assess_receipt, replay_evidence

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/research_tool_result_v4_failures.json'


def result(observations=None, diagnostics=None, *, status='success', error=None):
    return {'schema': 'sciel-tool-result/1', 'status': status,
            'observations': {} if observations is None else deepcopy(observations),
            'diagnostics': {} if diagnostics is None else deepcopy(diagnostics), 'error': error}


def receipt(output, **changes):
    value = dict(schema='sciel-method-execution/1', method_id='a'*64,
        input_sha256='b'*64, source_sha256='c'*64, runtime_sha256='d'*64,
        invocation_id='first', plan_id='plan', reservation_id='first',
        limits={'cpu_seconds': 10., 'wall_seconds': 15.}, dependency_ids=[],
        status='ok', dispatch_status='invoked', output=deepcopy(output),
        output_sha256=content_hash(output), reason=None, stderr='',
        resources={'cpu_seconds': .1, 'wall_seconds': .2})
    value.update(changes)
    value['receipt_sha256'] = content_hash(value)
    return value


def saved_rows():
    return json.loads(FIXTURE.read_text())['invocations']


def test_negative_finding_is_a_successful_check_not_a_tool_error():
    value = result({'hypothesis_supported': False, 'violations': 4, 'valid': False})
    parsed = parse_result(value)
    assert parsed == value and parsed is not value
    parsed['observations']['violations'] = 999
    assert value['observations']['violations'] == 4
    assert assess_receipt(receipt(value))['status'] == 'success'


@pytest.mark.parametrize('patch', [
    {'schema': 'sciel-tool-result/0'}, {'status': 'ok'}, {'status': True},
    {'observations': []}, {'diagnostics': []}, {'error': 'error on success'},
    {'status': 'tool_error', 'error': None}, {'status': 'tool_error', 'error': ''},
    {'status': 'tool_error', 'error': '  '}, {'status': 'tool_error', 'error': 42},
    {'extra': 'silently ignore me'},
])
def test_malformed_envelopes_are_not_inferred_or_coerced(patch):
    value = {**result(), **patch}
    with pytest.raises(ValueError):
        parse_result(value)
    assert assess_receipt(receipt(value))['status'] == 'invalid_output'


@pytest.mark.parametrize('field,value', [('observations', float('nan')), ('diagnostics', float('inf'))])
def test_nonfinite_result_data_is_rejected(field, value):
    malformed = result()
    malformed[field]['number'] = value
    with pytest.raises(ValueError):
        parse_result(malformed)


def test_actual_namespace_error_remains_raw_and_is_never_promoted_to_success():
    old = next(x['receipt'] for x in saved_rows() if x['tool_id'].startswith('ab57c'))
    before = deepcopy(old)
    assert old['status'] == 'ok'
    assert 'compute_max_radii' in old['output']['error']
    assert assess_receipt(old)['status'] == 'invalid_output'  # Missing explicit envelope.
    assert old == before
    future = receipt(result(status='tool_error', error=old['output']['error']))
    frozen = deepcopy(future)
    assessed = assess_receipt(future)
    assert assessed['host_status'] == 'ok' and assessed['status'] == 'tool_error'
    assert assessed['error'] == old['output']['error']
    assert future == frozen


def test_actual_elapsed_field_is_ignored_only_after_explicit_diagnostics_assignment():
    raw = next(x['receipt'] for x in saved_rows() if x['tool_id'].startswith('800ee'))
    before = deepcopy(raw)
    observations = {k: v for k, v in raw['output'].items() if k != 'elapsed_time'}
    first = receipt(result(observations, {'elapsed_time': raw['output']['elapsed_time']}))
    later = receipt(result(observations, {'elapsed_time': 1.2345}), invocation_id='later')
    proof = replay_evidence(first, later, 'exact_observations')
    assert proof['matched'] is True and proof['measurement_reproduced'] is True
    assert proof['reference_observations_sha256'] == proof['replay_observations_sha256']
    # No name-based exclusion: time kept in scientific observations still differs.
    first = receipt(result(raw['output']))
    changed = deepcopy(raw['output']); changed['elapsed_time'] += .1
    assert replay_evidence(first, receipt(result(changed)), 'exact_observations')['matched'] is False
    assert raw == before


def test_exact_observation_change_fails_without_automatic_policy_fallback():
    first, later = receipt(result({'radius': .25})), receipt(result({'radius': .250001}))
    proof = replay_evidence(first, later, 'exact_observations')
    assert proof['policy'] == 'exact_observations'
    assert proof['matched'] is False and proof['measurement_reproduced'] is False
    assert proof['reference_observations_sha256'] != proof['replay_observations_sha256']


def test_independent_execution_is_bound_without_claiming_outcome_identity():
    first, later = receipt(result({'sample': 1})), receipt(result({'sample': 2}))
    proof = replay_evidence(first, later, 'independent_execution')
    assert proof['policy'] == 'independent_execution' and proof['matched'] is True
    assert proof['measurement_reproduced'] is False
    assert proof['reference_observations_sha256'] != proof['replay_observations_sha256']


@pytest.mark.parametrize('field', ['method_id', 'input_sha256', 'source_sha256', 'runtime_sha256'])
@pytest.mark.parametrize('policy', ['exact_observations', 'independent_execution'])
def test_both_replay_policies_require_same_bound_method_input_source_runtime(field, policy):
    first = receipt(result({'probe': 1}))
    later = receipt(result({'probe': 1}), **{field: 'f'*64})
    assert replay_evidence(first, later, policy)['matched'] is False


@pytest.mark.parametrize('status,expected', [('failed', 'host_failed'), ('unknown', 'unknown')])
def test_host_failure_or_unknown_is_not_overridden_by_successful_looking_output(status, expected):
    value = receipt(result({'all_good': True}), status=status,
                    dispatch_status='unknown' if status == 'unknown' else 'invoked')
    assert assess_receipt(value)['status'] == expected
    assert replay_evidence(receipt(result({'all_good': True})), value, 'independent_execution')['matched'] is False


def test_tool_error_never_satisfies_either_replay_policy():
    good = receipt(result({'probe': 1}))
    broken = receipt(result(status='tool_error', error='caught diagnostic failure'))
    for policy in ('exact_observations', 'independent_execution'):
        assert replay_evidence(good, broken, policy)['matched'] is False
        assert replay_evidence(broken, good, policy)['matched'] is False
