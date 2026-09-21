"""Fixed arrays only; no helper, optimizer or provider call."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.hypothesis import restart_motion as motion
from sciel.hypothesis import restart_optimizer_trace as current
from sciel.hypothesis import restart_trace as legacy


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def seal(value, name):
    value = deepcopy(value)
    value[name] = hashlib.sha256(canonical({k: v for k, v in value.items() if k != name}).encode()).hexdigest()
    return value


def packing():
    centers = [[.125 + (i % 5) * .15625, .125 + (i // 5) * .15625] for i in range(26)]
    return np.array([v for pair in centers for v in pair] + [.03125 + i * 2.**-15 for i in range(26)])


def trace(*, best=None, effective=None, result='same', calls=1, schema=2):
    best = packing() if best is None else best
    effective = best.copy() if effective is None else effective
    result = effective.copy() if type(result) is str and result == 'same' else result
    identity = dict(source_sha256='a'*64, scaffold_sha256='b'*64, runtime_sha256='c'*64,
                    seed=17, invocation_id='d'*32)
    collector = (current if schema == 3 else legacy).RestartTrace(identity)
    quality = float(sum(best[52:]))
    for index in range(4*(calls-1)+1):
        collector.begin_start(index, incumbent_sum=quality, improvements=0)
        if index % 4 == 0:
            inputs = dict(best=best.copy(), initial=best.copy(), noise=np.zeros(78), ordinary=best.copy())
            collector.begin_helper(inputs)
            collector.helper_end(raw_kind='value', raw_type='ndarray', raw_value=effective,
                post_inputs=inputs, effective=effective, origin='accepted_return',
                incumbent_sum=quality, improvements=0)
        collector.optimizer_enter(incumbent_sum=quality, improvements=0)
        collector.optimizer_exit(incumbent_sum=quality, improvements=0, result=result)
        if schema == 3: collector.optimizer_outcome(False, SimpleNamespace(status=8))
        collector.end_start('completed', incumbent_sum=quality, improvements=0)
    value = collector.finish('restart_limit', incumbent_sum=quality, improvements=0)
    if schema == 1:
        value['schema'] = legacy.LEGACY_TRACE_SCHEMA
        for call in value['calls']: call.pop('optimizer_result')
        used = {ref for call in value['calls'] for ref in legacy._rt_record_refs(call)}
        value['vectors'] = {key: data for key, data in value['vectors'].items() if key in used}
        value = legacy.validate_trace(seal(value, 'trace_sha256'))
    return value


def row(value):
    return motion.motion_cases(motion.build_motion(value))[0]


def test_center_permutation_and_radius_restoration_are_distinct_from_indexwise_equality():
    best = packing(); effective = best.copy()
    effective[:4] = best[[2, 3, 0, 1]]
    returned = effective.copy(); returned[52:54] = best[[53, 52]]
    saved = trace(best=best, effective=effective, result=returned)
    before = canonical(saved); facts = row(saved)
    assert facts['effective_locations_changed'] is False
    assert facts['result_center_max'] == 0.
    assert facts['match_status'] == 'matched'
    assert facts['match_center_max'] == facts['match_radius_max'] == 0.
    assert not np.array_equal(returned, best) and not np.array_equal(returned, effective)
    assert canonical(saved) == before


def test_persistent_location_change_records_magnitudes_without_success_threshold():
    best = packing(); effective = best.copy(); effective[0] += .001
    returned = effective.copy(); returned[0] += .002
    facts = row(trace(best=best, effective=effective, result=returned))
    assert facts['effective_locations_changed'] is True
    assert facts['result_center_max'] == abs(returned[0] - effective[0])
    assert facts['match_center_max'] == abs(returned[0] - best[0])
    assert facts['match_radius_max'] == 0.
    assert 'feasible' not in facts and 'success' not in facts


def test_numeric_multiset_preserves_multiplicity_and_treats_signed_zero_equally():
    best = packing(); best[0] = 0.
    effective = best.copy(); effective[0] = -0.
    assert row(trace(best=best, effective=effective))['effective_locations_changed'] is False
    effective[:2] = best[2:4]
    facts = row(trace(best=best, effective=effective))
    assert facts['effective_locations_changed'] is True
    assert facts['match_status'] == 'ambiguous' and facts['match_reason'] == 'non_bijective_nearest'


@pytest.mark.parametrize('kind,reason', [('tie', 'nearest_tie'), ('reuse', 'non_bijective_nearest')])
def test_matching_ambiguity_never_invents_errors(kind, reason):
    best = packing(); returned = best.copy()
    returned[:2] = ((best[:2] + best[2:4]) / 2 if kind == 'tie' else best[2:4])
    facts = row(trace(best=best, result=returned))
    assert facts['match_status'] == 'ambiguous' and facts['match_reason'] == reason
    assert facts['match_center_max'] is facts['match_radius_max'] is None
    assert facts['result_center_max'] > 0.


@pytest.mark.parametrize('schema,result,reason', [
    (1, None, 'result:legacy_not_recorded'), (2, None, 'result:array_unavailable'),
    (2, np.zeros(77), 'result:shape_not_78'), (2, np.zeros(78, dtype=bool), 'result:non_real_dtype'),
    (3, np.full(78, np.nan), 'result:nonfinite_vector'),
])
def test_missing_and_unsupported_snapshots_remain_unknown(schema, result, reason):
    facts = row(trace(result=result, schema=schema))
    assert facts['effective_locations_changed'] is False
    assert facts['result_center_max'] is None and facts['result_center_reason'] == reason
    assert facts['match_status'] == 'unavailable' and facts['match_reason'] == reason
    assert facts['match_center_max'] is facts['match_radius_max'] is None


def test_finite_invalid_geometry_still_has_descriptive_motion():
    best = packing(); returned = best.copy(); returned[52:] *= -1
    facts = row(trace(best=best, result=returned, schema=3))
    assert facts['match_status'] == 'matched' and facts['result_center_max'] == 0.
    assert facts['match_radius_max'] == max(2*best[52:])
    assert facts['match_center_max'] == 0.


def test_lossy_integer_result_is_unknown_instead_of_false_zero_motion():
    best = np.array([float(i) for i in range(52)] + [0.]*26)
    best[0] = float(2**53)
    exact = best.astype(np.int64)
    unchanged = row(trace(best=best, result=exact))
    assert unchanged['result_center_max'] == unchanged['match_center_max'] == 0.
    changed = exact.copy(); changed[0] += 1
    assert int(changed[0]) == 2**53 + 1 and int(exact[0]) == 2**53
    facts = row(trace(best=best, result=changed))
    assert facts['result_center_max'] is None
    assert facts['result_center_reason'] == facts['match_reason'] == 'result:integer_not_exact_binary64'
    assert facts['match_status'] == 'unavailable'


def test_finite_vectors_with_nonfinite_distance_arithmetic_are_explicitly_unavailable():
    returned = packing(); returned[:2] = 1.7e308
    facts = row(trace(result=returned))
    assert facts['result_center_max'] is None and facts['result_center_reason'] == 'nonfinite_distance'
    assert facts['match_status'] == 'unavailable' and facts['match_reason'] == 'nonfinite_distance'


def test_rehashed_false_motion_rejected_against_original_trace_but_not_authentication_by_hash():
    saved = trace(); value = motion.build_motion(saved)
    value['cases'][0][-1] = .5
    value = seal(value, 'motion_sha256')
    assert motion.motion_cases(value)[0]['match_radius_max'] == .5  # Transport only.
    with pytest.raises(ValueError, match='projection differs'):
        motion.validate_motion(value, trace=saved, **saved['identity'])


@pytest.mark.parametrize('key', ['source_sha256', 'runtime_sha256', 'seed'])
def test_external_source_runtime_seed_bindings_are_required_when_supplied(key):
    saved = trace(); bindings = dict(saved['identity'])
    bindings[key] = 999 if key == 'seed' else 'f'*64
    with pytest.raises(ValueError): motion.build_motion(saved, **bindings)


def test_whole_retained_prefix_omissions_detachment_and_payload_bound():
    saved = trace(calls=21, schema=3)
    value = motion.build_motion(saved, **saved['identity'])
    assert len(value['cases']) == 16 and value['prefix'] == saved['prefix']
    assert value['prefix']['omitted_count'] == 5 and len(canonical(value).encode()) <= motion.MAX_BYTES
    assert motion.validate_motion(value, trace=saved, **saved['identity']) == value
    expanded = motion.motion_cases(value); expanded[0]['effective_locations_changed'] = True
    assert motion.motion_cases(value)[0]['effective_locations_changed'] is False
    damaged = deepcopy(value); damaged['cases'][0][0] = False
    with pytest.raises(ValueError): motion.motion_cases(seal(damaged, 'motion_sha256'))


def test_restart32_quantifies_near_restoration_without_execution():
    best = packing()
    returned = best.copy()
    center_delta, radius_delta = 2.**-36, 2.**-37
    returned[0] += center_delta
    returned[52] += radius_delta
    saved = trace(best=best, result=returned, calls=9)
    original = canonical(saved)
    value = motion.build_motion(saved, **saved['identity'])
    facts = next(case for case in motion.motion_cases(value) if case['restart_index'] == 32)
    assert facts['effective_locations_changed'] is False and facts['match_status'] == 'matched'
    assert facts['result_center_max'] == facts['match_center_max'] == center_delta
    assert facts['match_radius_max'] == radius_delta
    assert 0 < center_delta < 1e-10 and 0 < radius_delta < 1e-10
    assert value['trace_schema'] == legacy.TRACE_SCHEMA and canonical(saved) == original
