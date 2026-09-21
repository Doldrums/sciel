"""Forecast arithmetic over inert collectors and immutable saved receipts only."""
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.hypothesis import restart_predictions as forecasts
from sciel.hypothesis import restart_motion_observations as motion
from sciel.hypothesis import restart_optimizer_trace as codec
from test_restart_motion_observations import older


def spec(**expected):
    return {'schema': forecasts.SCHEMA,
            'expected': dict.fromkeys(forecasts.NAMES, True) | expected}


def receipt(modes=('restored',), *, gain=False, ambiguous=False):
    """Fixed arrays and status objects; no helper or optimization is executed."""
    adapter = {'runtime': {'fixture': 'inert prediction vectors'}}
    adapter['identity_sha256'] = forecasts._sha(adapter)
    identity = dict(source_sha256='a'*64, seed=17, runtime_sha256=forecasts._sha(adapter['runtime']),
                    scaffold_sha256='b'*64, invocation_id='c'*32)
    collector = codec.RestartTrace(identity)
    best = np.array([coordinate for i in range(26)
                     for coordinate in (.125+(i%5)*.15625, .125+(i//5)*.15625)] + [.015625]*26)
    best[52] = 0.  # Exact 1e-6 radius-residual boundary is representable.
    if ambiguous:
        best[48:52] = [.95, .95, .95, .95]
        best[76:] = 0.
    total, improvements, finished = float(math.fsum(best[52:])), 0, 'restart_limit'
    for index in range(4*(len(modes)-1)+1 if modes else 0):
        quality = dict(incumbent_sum=total, improvements=improvements)
        collector.begin_start(index, **quality)
        mode = modes[index//4] if index%4 == 0 else 'fallback'
        effective = best.copy()
        if index%4 == 0:
            inputs = dict(best=best, initial=best.copy(), noise=np.zeros(78), ordinary=best.copy())
            collector.begin_helper(inputs)
            if mode == 'unfinished':
                collector.end_start('deadline', **quality); finished = 'deadline'; break
            if mode in ('center', 'restored_center', 'invalid', 'missing'):
                effective[0] += .0078125
            collector.helper_end(raw_kind='none' if mode == 'fallback' else 'value',
                raw_type='NoneType' if mode == 'fallback' else 'ndarray',
                raw_value=None if mode == 'fallback' else effective, post_inputs=inputs,
                effective=effective, origin='saved_ordinary_fallback' if mode == 'fallback' else 'accepted_return',
                **quality)
        if mode == 'unentered':
            collector.end_start('deadline', **quality); finished = 'deadline'; break
        collector.optimizer_enter(**quality)
        collector.optimizer_exit(**quality)
        # Success is deliberately false even when geometry and change are valid.
        collector.optimizer_outcome(False, SimpleNamespace(status=8))
        result = effective.copy()
        if mode == 'restored_center': result = best.copy()
        elif mode == 'radius': result[52] = .0078125
        elif mode == 'threshold': result[52] = forecasts.EPSILON
        elif mode == 'above_threshold': result[52] = math.nextafter(forecasts.EPSILON, math.inf)
        elif mode == 'invalid': result[0] = -1.
        elif mode == 'missing': result = None
        collector.optimizer_result(result)
        if gain and index%4 == 0 and mode != 'fallback':
            total = math.nextafter(total, math.inf); improvements += 1
        collector.end_start('completed', incumbent_sum=total, improvements=improvements)
    trace = collector.finish(finished, incumbent_sum=total, improvements=improvements)
    return dict(schema='sciel-repeat-draw/1', status='valid', source_sha256='a'*64, seed=17,
        adapter_identity=adapter, raw_sum=total, restart_trace=dict(status='captured', reason=None,
            sidecar_sha256='d'*64, trace=trace))


def build(saved):
    return motion.build_observation(saved, source_sha256=saved['source_sha256'],
                                    seed=saved['seed'], adapter_identity=saved['adapter_identity'])


def evaluate(modes=('restored',), **options):
    return forecasts.evaluate(spec(), build(receipt(modes, **options)))


@pytest.mark.parametrize('bad', [None, [], {}, {'schema': 'wrong', 'expected': {}},
    spec(extra=True), {'schema': forecasts.SCHEMA, 'expected': {'result_changed': True}},
    spec(result_changed=1), spec(result_changed=None), spec(result_changed='true'),
    spec() | {'epsilon': 0.}])
def test_schema_rejects_missing_unknown_or_coerced_claims(bad):
    with pytest.raises(ValueError, match='invalid specification'):
        forecasts.validate_spec(bad)


def test_schema_detached_and_provider_matches_fixed_registry():
    original = spec(result_changed=False)
    checked = forecasts.validate_spec(original)
    checked['expected']['result_changed'] = True
    assert original['expected']['result_changed'] is False
    schema = forecasts.json_schema()
    assert schema['additionalProperties'] is False
    assert schema['properties']['expected']['required'] == list(forecasts.NAMES)
    assert set(schema['properties']['expected']['properties']) == set(forecasts.NAMES)
    schema['properties']['expected']['properties'].clear()
    assert len(forecasts.json_schema()['properties']['expected']['properties']) == 3
    assert all(name in forecasts.instruction() for name in forecasts.NAMES)
    assert 'not causal credit' in forecasts.instruction()
    assert forecasts.instruction().startswith('Include restart_predictions ')
    assert 'EACH planned paired challenger draw' in forecasts.instruction()
    assert 'first 16 scheduled calls' in forecasts.instruction()


def test_center_radius_and_interval_are_separate_not_optimizer_success():
    center = evaluate(('center',), gain=True)
    assert set(center['observed'].values()) == {True}
    assert center['cases'] == [[0, 0, True, True, True, True]]
    radius = evaluate(('radius',))
    assert radius['observed'] == dict(effective_locations_changed=False,
                                     result_changed=True, accepted_interval_gain=False)
    assert center['coverage'] == dict(retained=1, accepted=1, unknown_acceptance=0, omitted=0)


def test_persistence_has_strict_fixed_boundary_and_gain_no_epsilon():
    assert evaluate(('threshold',))['observed']['result_changed'] is False
    result = evaluate(('above_threshold',), gain=True)
    assert result['observed']['result_changed'] is True
    assert result['observed']['accepted_interval_gain'] is True
    phases = receipt(('restored',), gain=True)['restart_trace']['trace']['calls'][0]['phases']
    assert 0 < phases['terminal']['sum']-phases['optimizer_entered']['sum'] < 1e-12


@pytest.mark.parametrize('modes,expected', [
    (('center', 'missing'), True), (('restored', 'missing'), None),
    (('restored', 'restored'), False), (('center', 'unfinished'), True),
    (('restored', 'unfinished'), None), ((), None), (('fallback',), None)])
def test_existential_three_values_include_unknown_acceptance(modes, expected):
    result = evaluate(modes)
    assert result['observed']['result_changed'] is expected
    if 'unfinished' in modes:
        assert result['coverage']['unknown_acceptance'] == 1
        assert result['cases'][-1][2:] == [None, None, None, None]


def test_known_invalid_geometry_is_false_while_valid_ambiguous_is_unknown():
    assert evaluate(('invalid',), ambiguous=True)['observed']['result_changed'] is False
    assert evaluate(('center',), ambiguous=True)['observed']['result_changed'] is None
    assert evaluate(('unentered',))['observed']['accepted_interval_gain'] is None


def test_whole_run_accepted_call_outside_prefix_supplies_no_witness():
    saved = receipt(('fallback',)*16 + ('center',), gain=True)
    result = forecasts.evaluate(spec(), build(saved))
    assert saved['restart_trace']['trace']['aggregates']['accepted_return']['helper_calls'] == 1
    assert result['coverage'] == dict(retained=16, accepted=0, unknown_acceptance=0, omitted=1)
    assert result['cases'] == [] and set(result['observed'].values()) == {None}


def test_missing_trace_and_quality_mismatch_remain_unknown_not_zero():
    saved = receipt()
    saved['restart_trace'] = dict(status='missing', reason='sidecar_missing', sidecar_sha256=None, trace=None)
    result = forecasts.evaluate(spec(), build(saved))
    assert result['status'] == 'unavailable'
    assert set(result['coverage'].values()) == {None}
    assert result['cases'] == [] and set(result['observed'].values()) == {None}
    saved = receipt(); saved['raw_sum'] += .1
    mismatch = forecasts.evaluate(spec(), build(saved))
    assert mismatch['reason'] == 'trace_result_mismatch'
    assert set(mismatch['observed'].values()) == {None}


@pytest.mark.parametrize('version', [1, 2])
def test_legacy_result_without_known_validity_is_not_fabricated(version):
    result = forecasts.evaluate(spec(), build(older(receipt(('center',), gain=True), version)))
    assert result['observed'] == dict(effective_locations_changed=True,
                                     result_changed=None, accepted_interval_gain=True)


def test_expected_false_agreement_is_distinct_from_quality_and_source_binding():
    saved = receipt()
    observation = build(saved)
    original = deepcopy(observation)
    declared = spec(**dict.fromkeys(forecasts.NAMES, False))
    result = forecasts.evaluate(declared, observation)
    assert set(result['verdicts'].values()) == {'CONFIRMED'}
    assert observation == original and result['expected'] == declared['expected']
    assert result['spec_sha256'] == forecasts._sha(declared)
    assert result['receipt_sha256'] == observation['receipt_sha256']
    assert result['observation_sha256'] == observation['observation_sha256']
    assert result['result_sha256'] == forecasts._sha({k:v for k,v in result.items() if k != 'result_sha256'})
    assert not {'score', 'actual_delta', 'evidence_ids', 'lesson_ids'} & set(result)
    result['cases'][0][3] = True
    assert forecasts.evaluate(declared, observation)['cases'][0][3] is False


def test_motion_projection_tampering_is_rejected_before_forecast_use():
    observation = build(receipt())
    observation['cases'][0][-2] = .5
    with pytest.raises(ValueError): forecasts.evaluate(spec(), observation)


def test_sparse_accepted_prefix_with_large_omission_does_not_infer_persistence():
    modes = ['fallback'] * 476
    for ordinal in (4, 8, 12):
        modes[ordinal] = 'restored_center'
    saved = receipt(modes)
    result = forecasts.evaluate(spec(), build(saved))
    assert result['observed'] == dict(effective_locations_changed=True,
                                     result_changed=False, accepted_interval_gain=False)
    assert [row[1] for row in result['cases']] == [16, 32, 48]
    assert result['coverage'] == dict(retained=16, accepted=3, unknown_acceptance=0, omitted=460)
    assert len(forecasts._canonical(result)) < 1500
