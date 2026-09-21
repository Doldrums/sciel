"""Synthetic collector/codec boundaries only; no helper, solver or process."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.hypothesis import restart_trace as old
from sciel.hypothesis import restart_optimizer_trace as codec

IDENTITY = dict(source_sha256='a'*64, scaffold_sha256='b'*64,
                runtime_sha256='c'*64, seed=17, invocation_id='d'*32)


def seal(value):
    value = deepcopy(value)
    value.pop('trace_sha256', None)
    value['trace_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return value


def add_start(collector, index=0, *, result=None, success=False, returned=True,
              outcome=True, unique=False, complete=True):
    collector.begin_start(index, incumbent_sum=1., improvements=0)
    if index % 4 == 0:
        offset = index/100 if unique else 0
        args = dict(best=np.arange(78, dtype=float)/100+offset,
                    initial=np.full(78, offset, dtype=float), noise=np.full(78, .01+offset),
                    ordinary=np.arange(78, dtype=float)/100+.01+offset)
        collector.begin_helper(args)
        collector.helper_end(raw_kind='none', post_inputs=args, effective=args['ordinary'].copy(),
            origin='saved_ordinary_fallback', incumbent_sum=1., improvements=0)
    collector.optimizer_enter(incumbent_sum=1., improvements=0)
    if returned:
        collector.optimizer_exit(incumbent_sum=1., improvements=0)
        if outcome:
            collector.optimizer_outcome(success, result if result is not None else SimpleNamespace(status=8))
        collector.optimizer_result(np.arange(78, dtype=float)/100 + (index*.03125+.004 if unique else 0))
    if complete:
        collector.end_start('completed', incumbent_sum=1., improvements=0)


def capture(*, success=False, result=None, returned=True, outcome=True):
    collector = codec.RestartTrace(IDENTITY)
    add_start(collector, success=success, result=result, returned=returned, outcome=outcome,
              complete=returned and outcome)
    return collector.finish('restart_limit' if returned and outcome else 'error', incumbent_sum=1., improvements=0)


def test_v3_roundtrip_preserves_false_success_nonzero_status_and_result_snapshot():
    value = capture()
    call = value['calls'][0]
    assert value['schema'] == codec.TRACE_SCHEMA
    assert call['optimizer_outcome'] == dict(success=dict(value=False, reason=None), status=dict(value=8, reason=None))
    assert call['optimizer_result']['ref'] in value['vectors']
    assert codec.parse_trace(codec.encode_trace(value), **IDENTITY) == value
    detached = codec.validate_trace(value)
    detached['calls'][0]['optimizer_outcome']['status']['value'] = 0
    assert call['optimizer_outcome']['status']['value'] == 8


@pytest.mark.parametrize('status, expected', [(0, 0), (-7, -7), (np.int64(9), 9),
    (-(2**63), -(2**63)), (2**63-1, 2**63-1)])
def test_bounded_integer_status_is_not_interpreted_as_success(status, expected):
    value = capture(success=True, result=SimpleNamespace(status=status))
    assert value['calls'][0]['optimizer_outcome'] == {
        'success': {'value':True,'reason':None}, 'status':{'value':expected,'reason':None}}


@pytest.mark.parametrize('status', [True, np.bool_(False), 0., '0', 2**63, -(2**63)-1, None])
def test_unsupported_status_is_unavailable_not_coerced(status):
    outcome = capture(result=SimpleNamespace(status=status))['calls'][0]['optimizer_outcome']
    assert outcome['status'] == dict(value=None, reason='unsupported_type')
    assert outcome['success'] == dict(value=False, reason=None)


def test_status_missing_and_read_error_are_distinct_and_do_not_invent_code():
    class Broken:
        @property
        def status(self):
            raise RuntimeError('must not be echoed')
    missing = capture(result=SimpleNamespace())['calls'][0]['optimizer_outcome']['status']
    error = capture(result=Broken())['calls'][0]['optimizer_outcome']['status']
    assert missing == dict(value=None, reason='missing')
    assert error == dict(value=None, reason='read_error')


@pytest.mark.parametrize('returned,reason', [(False,'not_returned'),(True,'not_observed')])
def test_absent_success_read_is_explicit_not_false(returned, reason):
    value = capture(returned=returned, outcome=False)
    assert value['calls'][0]['optimizer_outcome'] == {
        field:dict(value=None,reason=reason) for field in ('success','status')}


def test_status_access_is_once_per_retained_call_not_ordinary_or_omitted():
    collector = codec.RestartTrace(IDENTITY)
    class Observed:
        reads = 0
        @property
        def status(self):
            self.reads += 1
            return 0
    result = Observed()
    for index in range(81):
        add_start(collector, index, result=result)
    value = collector.finish('restart_limit', incumbent_sum=1., improvements=0)
    assert result.reads == len(value['calls']) == 16
    assert value['total_starts'] == 81 and value['prefix']['omitted_count'] == 5
    assert value['aggregates']['ordinary']['optimizer_returned'] == 60


def test_byte_cap_still_drops_whole_prefix_records_with_metadata_counted():
    collector = codec.RestartTrace(IDENTITY)
    for index in range(81):
        add_start(collector, index, unique=True)
    value = collector.finish('restart_limit', incumbent_sum=1., improvements=0)
    assert 0 < len(value['calls']) < 16
    assert value['prefix']['omission_reason'] == 'byte_limit'
    assert all('optimizer_outcome' in call for call in value['calls'])
    assert len(codec.encode_trace(value).encode()) <= old.MAX_TRACE_BYTES == codec.MAX_TRACE_BYTES


@pytest.mark.parametrize('damage', ['bool_status','int_success','not_returned','missing_value','extra','removed'])
def test_rehashed_outcome_contradictions_rejected(damage):
    value = capture()
    call = value['calls'][0]; outcome = call['optimizer_outcome']
    if damage == 'bool_status': outcome['status']['value'] = True
    elif damage == 'int_success': outcome['success']['value'] = 0
    elif damage == 'not_returned': outcome.update({f:dict(value=None,reason='not_returned') for f in outcome})
    elif damage == 'missing_value': outcome['status'] = dict(value=8, reason='missing')
    elif damage == 'extra': outcome['message'] = 'invented'
    else: call.pop('optimizer_outcome')
    with pytest.raises(ValueError): codec.validate_trace(seal(value))


def test_legacy_projection_preserves_all_original_arrays_phases_and_counts():
    value = capture(success=True)
    projected = codec.legacy_projection(value)
    assert old.validate_trace(projected) == projected
    assert projected['schema'] == codec.LEGACY_RESULT_SCHEMA
    assert projected['trace_sha256'] != value['trace_sha256']
    assert 'optimizer_outcome' not in projected['calls'][0]
    expected = deepcopy(value); expected['schema'] = old.TRACE_SCHEMA
    for call in expected['calls']: call.pop('optimizer_outcome')
    assert projected == seal(expected)
    legacy1 = deepcopy(projected); legacy1['schema'] = old.LEGACY_TRACE_SCHEMA
    for call in legacy1['calls']: call.pop('optimizer_result')
    used = {ref for call in legacy1['calls'] for ref in old._rt_record_refs(call)}
    legacy1['vectors'] = {key:item for key,item in legacy1['vectors'].items() if key in used}
    for legacy in (projected, seal(legacy1)):
        assert codec.parse_trace(codec.encode_trace(legacy)) == legacy
        assert codec.legacy_projection(legacy) == legacy
        assert 'optimizer_outcome' not in codec.validate_trace(legacy)['calls'][0]


def test_duplicate_keys_and_external_binding_mismatch_rejected():
    value = capture()
    with pytest.raises(ValueError): codec.validate_trace(value, seed=18)
    raw = codec.encode_trace(value).replace('"seed":17', '"seed":17,"seed":17', 1)
    with pytest.raises(ValueError, match='duplicate'): codec.parse_trace(raw)
