"""Pure receipt/array fixtures; never execute a candidate, helper or optimizer."""
from copy import deepcopy
import hashlib
import json
import math

import numpy as np
import pytest

from sciel.eval import packing_geometry as geometry
from sciel.hypothesis import restart_observations as observations
from sciel.hypothesis import restart_trace as codec


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def synthetic_adapter():
    value = {'schema': 'fake-repeat-adapter/2', 'runtime': {'kind': 'inert-fixture'}}
    return {**value, 'identity_sha256': sha(value)}


def seal_trace(value):
    value = deepcopy(value)
    value['trace_sha256'] = sha({k: v for k, v in value.items() if k != 'trace_sha256'})
    return value


def synthetic_receipt(source_sha256='a'*64, seed=17, *, adapter_identity=None, mode='fallback',
                      result='same', scheduled_calls=1, legacy=False, raw_sum=.25, gain=0.):
    """Full fake numerical receipt with real pure trace validation and geometry.

    No function named by source_sha256 is imported or invoked. Shared by the
    Engine/KB fake tests to exercise the actual projection without a mock codec.
    """
    adapter = deepcopy(adapter_identity) if adapter_identity is not None else synthetic_adapter()
    centers = [[.5, .5]] + [[0., 0.]]*25
    radii = [raw_sum] + [0.]*25
    capture = geometry.validate_packing({'centers': centers, 'radii': radii})
    certificate = geometry.certify_packing(capture)
    assert capture['valid'] and certificate['status'] == 'certified'
    identity = dict(source_sha256=source_sha256, seed=seed, runtime_sha256=sha(adapter['runtime']),
                    scaffold_sha256='b'*64, invocation_id=sha([source_sha256, seed])[:32])
    collector = codec.RestartTrace(identity)
    initial_sum = capture['sum_radii'] - gain
    best = np.array([x for pair in centers for x in pair] + radii, dtype=float)
    initial = best.copy(); noise = np.full(78, .0001); ordinary = best + noise
    count = 4*(scheduled_calls-1)+1 if scheduled_calls else 0
    for index in range(count):
        collector.begin_start(index, incumbent_sum=initial_sum, improvements=0)
        effective = ordinary.copy()
        if index % 4 == 0:
            inputs = dict(best=best.copy(), initial=initial.copy(), noise=noise.copy(), ordinary=ordinary.copy())
            collector.begin_helper(inputs)
            if mode == 'unresolved':
                break
            if mode == 'helper_error':
                collector.helper_end(raw_kind='exception', post_inputs=inputs, effective=None, origin=None,
                    error='ValueError', error_stage='helper', incumbent_sum=initial_sum, improvements=0)
                break
            accepted = mode in ('accepted', 'best')
            if mode == 'best': effective = best.copy()
            collector.helper_end(raw_kind='value' if accepted else 'none', raw_value=effective if accepted else None,
                raw_type='ndarray' if accepted else None, post_inputs=inputs, effective=effective,
                origin='accepted_return' if accepted else 'saved_ordinary_fallback',
                incumbent_sum=initial_sum, improvements=0)
        collector.optimizer_enter(incumbent_sum=initial_sum, improvements=0)
        x = (None if result == 'missing' else np.ones(77) if result == 'badshape' else
             np.full(78, np.nan) if result == 'nonfinite' else effective.copy())
        if result == 'changed': x[0] += .01
        collector.optimizer_exit(incumbent_sum=initial_sum, improvements=0, result=x)
        final = index == count-1
        collector.end_start('completed', incumbent_sum=capture['sum_radii'] if final else initial_sum,
                            improvements=int(final and gain > 0))
    status = 'error' if mode == 'helper_error' else 'deadline' if mode == 'unresolved' else 'restart_limit'
    trace = collector.finish(status, incumbent_sum=capture['sum_radii'], improvements=int(gain > 0))
    if legacy:
        trace['schema'] = codec.LEGACY_TRACE_SCHEMA
        for call in trace['calls']: call.pop('optimizer_result')
        used = {d['ref'] for call in trace['calls'] for d in
            [*call['inputs'].values(), call['raw_return']['vector'], call['effective']['vector']] if d['ref'] is not None}
        trace['vectors'] = {key: vector for key, vector in trace['vectors'].items() if key in used}
        trace = codec.validate_trace(seal_trace(trace))
    valid = mode != 'helper_error'
    return dict(schema='sciel-repeat-draw/1', source_sha256=source_sha256, seed=seed, adapter_identity=adapter,
        seed_applied=dict(python_random=seed, numpy_legacy=seed, python_hash_seed=str(seed),
            before_candidate_import=True, determinism_proven=False),
        limits=dict(candidate_wall_seconds=30., certification_wall_seconds=10., candidate_cpu_seconds=35,
            certification_cpu_seconds=35, candidate_mem_bytes=2**31, certification_mem_bytes=2**31),
        status='valid' if valid else 'candidate_failure', reason=None if valid else 'inert fixture failure',
        capture=capture, certificate=certificate, raw_sum=capture['sum_radii'] if valid else None,
        certified_sum=certificate['sum_exact'] if valid else None,
        started_at='2026-09-17T00:00:00+00:00', completed_at='2026-09-17T00:00:01+00:00', elapsed_seconds=1.,
        candidate_elapsed_seconds=.8, certification_elapsed_seconds=.1, identity_elapsed_seconds=0.,
        restart_trace=dict(status='captured', reason=None, trace=trace, sidecar_sha256=sha(trace)))


def build(receipt=None, **kwargs):
    receipt = synthetic_receipt() if receipt is None else receipt
    return observations.build_observation(receipt, source_sha256=receipt['source_sha256'], seed=receipt['seed'],
        adapter_identity=receipt['adapter_identity'], **kwargs)


def row(value, index=0):
    return observations.observation_cases(value)[index]


def descriptor(value, name, index=0):
    item = row(value, index)
    ref = item['snapshot_refs'][value['columns']['snapshots'].index(name)]
    return value['descriptor_catalog'][ref]


def test_fallback_and_equal_explicit_return_have_different_observed_origins():
    fallback, accepted, best = [build(synthetic_receipt(mode=mode)) for mode in ('fallback', 'accepted', 'best')]
    assert row(fallback)['helper_accepted'] is False
    assert row(fallback)['effective_origin'] == 'saved_ordinary_fallback'
    assert row(accepted)['helper_accepted'] is True and row(accepted)['raw_return_kind'] == 'value'
    assert row(best)['helper_accepted'] is True
    assert row(best)['result_values_equal_best'] is True
    assert row(accepted)['optimizer_returned'] is True and row(accepted)['phase_improvements'] == [0]*5
    assert accepted['run']['origins']['accepted_return']['gain'] == 0.
    assert 'data_base64' not in canonical(accepted) and 'polarity' not in accepted


def test_optimizer_return_changed_result_and_incumbent_gain_are_separate():
    value = build(synthetic_receipt(mode='accepted', result='changed'))
    assert row(value)['optimizer_returned'] is True
    assert row(value)['result_values_equal_effective'] is False
    assert row(value)['phase_sums'] == [.25]*5
    assert value['run']['origins']['accepted_return']['gain'] == 0.
    gained = build(synthetic_receipt(mode='fallback', gain=.01))
    assert row(gained)['helper_accepted'] is False and row(gained)['phase_improvements'] == [0,0,0,0,1]
    assert gained['run']['origins']['saved_ordinary_fallback']['gain'] > 0


@pytest.mark.parametrize('result,finite,shape', [('missing',None,None), ('badshape',True,[77]), ('nonfinite',False,[78])])
def test_result_availability_never_implies_valid_geometry_or_equality(result, finite, shape):
    value = build(synthetic_receipt(result=result))
    item = descriptor(value, 'optimizer_result')
    assert item['finite'] is finite and item['shape'] == shape
    assert row(value)['optimizer_returned'] is True
    assert row(value)['result_values_equal_effective'] is None


def test_legacy_result_snapshot_is_unknown_not_effective_or_incumbent():
    receipt = synthetic_receipt(legacy=True)
    original = deepcopy(receipt)
    value = build(receipt)
    assert value['trace_schema'] == codec.LEGACY_TRACE_SCHEMA
    assert descriptor(value, 'optimizer_result')['reason'] == 'legacy_not_recorded'
    assert row(value)['optimizer_returned'] is True and row(value)['result_values_equal_best'] is None
    assert receipt == original


@pytest.mark.parametrize('mode', ['helper_error', 'unresolved'])
def test_failed_and_unfinished_handoffs_do_not_invent_acceptance_or_returns(mode):
    value = build(synthetic_receipt(mode=mode))
    item = row(value)
    assert item['optimizer_returned'] is False and item['optimizer_entered'] is False
    assert item['helper_accepted'] is (False if mode == 'helper_error' else None)
    assert item['effective_origin'] is None
    assert descriptor(value, 'optimizer_result')['state'] == 'not_applicable'


def test_prefix_counts_do_not_mislabel_whole_run_or_select_later_calls():
    value = build(synthetic_receipt(mode='accepted', scheduled_calls=21))
    assert len(value['cases']) == 16
    assert [row(value,i)['restart_index'] for i in range(16)] == list(range(0,64,4))
    assert value['run']['total_scheduled_calls'] == 21
    assert value['prefix']['omitted_count'] == 5 and value['prefix']['omission_reason'] == 'prefix_limit'
    view = observations.observation_view(value)
    assert view['included'] and len(canonical(view)) <= 8192


@pytest.mark.parametrize('status', ['absent', 'missing', 'invalid', 'error'])
def test_unavailable_trace_preserves_measured_receipt_without_fabricated_counts(status):
    receipt = synthetic_receipt()
    if status == 'absent': receipt.pop('restart_trace')
    else: receipt['restart_trace'] = dict(status=status, reason='fixture_unavailable', sidecar_sha256=None, trace=None)
    original = deepcopy(receipt)
    value = build(receipt)
    assert value['status'] == 'unavailable' and value['receipt_status'] == 'valid'
    assert value['run'] is value['prefix'] is None and value['cases'] == []
    assert receipt == original


@pytest.mark.parametrize('key', ['source_sha256', 'seed', 'adapter_identity', 'scaffold_sha256'])
def test_external_binding_mismatch_rejected(key):
    receipt = synthetic_receipt()
    kwargs = dict(source_sha256=receipt['source_sha256'], seed=receipt['seed'], adapter_identity=receipt['adapter_identity'], scaffold_sha256='b'*64)
    kwargs[key] = 18 if key == 'seed' else synthetic_adapter() if key == 'adapter_identity' else 'f'*64
    if key == 'adapter_identity':
        kwargs[key]['runtime']['kind'] = 'other'
        kwargs[key]['identity_sha256'] = sha({k:v for k,v in kwargs[key].items() if k != 'identity_sha256'})
    with pytest.raises(ValueError): observations.build_observation(receipt, **kwargs)


@pytest.mark.parametrize('field', ['seed', 'source_sha256', 'runtime_sha256'])
def test_rehashed_trace_identity_cannot_bypass_receipt_join(field):
    receipt = synthetic_receipt(); trace = receipt['restart_trace']['trace']
    trace['identity'][field] = 18 if field == 'seed' else 'f'*64
    receipt['restart_trace']['trace'] = seal_trace(trace)
    with pytest.raises(ValueError): build(receipt)


def test_supplied_quality_join_booleans_cannot_rescue_wrong_result_sum():
    receipt = synthetic_receipt()
    trace = receipt['restart_trace']['trace']
    for phase in trace['calls'][0]['phases'].values(): phase['sum'] += .01
    trace['initial_quality']['sum'] += .01; trace['final_quality']['sum'] += .01
    receipt['restart_trace']['trace'] = seal_trace(trace)
    receipt['restart_trace']['quality_join'] = {'within_roundoff': True, 'terminal_status_compatible': True}
    value = build(receipt)
    assert value['status'] == 'unavailable' and value['reason'] == 'trace_result_mismatch'
    assert value['quality_join']['within_roundoff'] is False
    assert receipt['raw_sum'] == .25


def test_rehashed_summary_is_not_evidence_without_original_receipt():
    receipt = synthetic_receipt(); value = build(receipt)
    value['case_state_catalog'][0][value['columns']['case_state'].index('helper_accepted')] = True
    value['observation_sha256'] = sha({k:v for k,v in value.items() if k != 'observation_sha256'})
    with pytest.raises(ValueError, match='differs from durable receipt'):
        observations.validate_observation(value, receipt=receipt, source_sha256=receipt['source_sha256'],
            seed=receipt['seed'], adapter_identity=receipt['adapter_identity'])


def test_whole_card_omission_is_deterministic_and_preserves_binding():
    value = build(synthetic_receipt(scheduled_calls=21))
    view = observations.observation_view(value, max_chars=1200)
    assert view['included'] is False and view['observation'] is None
    assert view['observation_sha256'] == value['observation_sha256'] and view['bindings'] == value['bindings']
    assert observations.validate_observation_view(view) == view
    assert observations.observation_view(value, max_chars=1200) == view
    with pytest.raises(ValueError, match='omission metadata'): observations.observation_view(value, max_chars=10)


def test_view_and_validated_observation_are_detached():
    receipt = synthetic_receipt(); value = build(receipt)
    checked = observations.validate_observation(value, receipt=receipt, source_sha256=receipt['source_sha256'],
        seed=receipt['seed'], adapter_identity=receipt['adapter_identity'])
    view = observations.observation_view(value)
    restored = observations.validate_observation_view(view)
    restored['observation']['cases'][0][0] = 999
    checked['bindings']['seed'] = 999
    assert value['bindings']['seed'] == 17 and view['observation']['cases'][0][0] == 0


def test_phase_catalog_preserves_signed_zero_without_merging_recorded_values():
    receipt = synthetic_receipt(raw_sum=0.)
    trace = receipt['restart_trace']['trace']
    trace['calls'][0]['phases']['start']['sum'] = -0.0
    receipt['restart_trace']['trace'] = seal_trace(trace)
    value = build(receipt)
    phases = row(value)['phase_sums']
    assert math.copysign(1., phases[0]) == -1.
    assert all(math.copysign(1., x) == 1. for x in phases[1:])
    refs = row(value)['phase_quality_refs']
    assert refs[0] != refs[1]
    assert len(value['phase_catalog']) == 2


def test_rehashed_transport_cannot_claim_equality_with_unknown_other_snapshot():
    view = observations.observation_view(build(synthetic_receipt(mode='best')))
    value = view['observation']
    assert row(value)['result_values_equal_best'] is True
    value['descriptor_catalog'].append(dict(state='unavailable', reason='array_unavailable',
                                          dtype=None, shape=None, finite=None))
    refs = value['cases'][0][value['columns']['cases'].index('snapshot_refs')]
    refs[value['columns']['snapshots'].index('best')] = len(value['descriptor_catalog']) - 1
    value['observation_sha256'] = sha({k:v for k,v in value.items() if k != 'observation_sha256'})
    view['observation_sha256'] = value['observation_sha256']
    view['view_sha256'] = sha({k:v for k,v in view.items() if k != 'view_sha256'})
    with pytest.raises(ValueError, match='unknown result equality'):
        observations.validate_observation_view(view)


def test_all_sixteen_state_rows_expand_losslessly_to_original_call_facts():
    receipt = synthetic_receipt(mode='accepted', scheduled_calls=21)
    value = build(receipt)
    cases = observations.observation_cases(value)
    assert len(cases) == 16 and len(value['case_state_catalog']) == 1
    for actual, original in zip(cases, receipt['restart_trace']['trace']['calls'], strict=True):
        assert actual['ordinal'] == original['ordinal']
        assert actual['restart_index'] == original['restart_index']
        assert actual['raw_return_kind'] == original['raw_return']['kind']
        assert actual['effective_origin'] == original['effective']['origin']
        assert actual['helper_finished'] == original['helper_finished']
        assert actual['error'] == original['error'] and actual['terminal'] == original['terminal']
        assert actual['helper_accepted'] is True
        assert actual['optimizer_entered'] == (original['phases']['optimizer_entered'] is not None)
        assert actual['optimizer_returned'] == (original['phases']['optimizer_returned'] is not None)
        phases = [original['phases'][name] for name in value['columns']['phases']]
        assert canonical(actual['phase_sums']) == canonical([None if p is None else p['sum'] for p in phases])
        assert actual['phase_improvements'] == [None if p is None else p['improvements'] for p in phases]
        assert actual['input_unchanged'] == [original['inputs'][k]['ref'] == original['post_inputs'][k]['ref']
                                              for k in value['columns']['input_unchanged']]
        assert actual['result_values_equal_effective'] is True
        assert actual['result_values_equal_best'] is False
    cases[0]['input_unchanged'][0] = False
    assert observations.observation_cases(value)[0]['input_unchanged'][0] is True
