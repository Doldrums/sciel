"""Synthetic /3 observations and stable /1-/2 presentation.

Collector methods only receive fixed arrays and inert status objects. No helper,
optimizer, candidate or provider source is executed.
"""
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.eval import packing_geometry as geometry
from sciel.hypothesis import restart_observations as observations
from sciel.hypothesis import restart_optimizer_observations as optimizer
from sciel.hypothesis import restart_optimizer_trace as codec
from sciel.knowledge_base import paired_comparisons, restart_context
from test_restart_observations import canonical, sha, synthetic_adapter


def receipt(*, source='a' * 64, seed=11, success=True, status=0, result='valid',
            calls=1, returned=True, observed=True, realistic=False):
    """A declared inert numerical fixture, not an executed packing receipt."""
    adapter = synthetic_adapter()
    identity = dict(source_sha256=source, seed=seed, runtime_sha256=sha(adapter['runtime']),
        scaffold_sha256='b' * 64, invocation_id=sha([source, seed, 'synthetic-status'])[:32])
    collector = codec.RestartTrace(identity)
    centers = [[.125 + (i % 5) * .15625, .125 + (i // 5) * .15625] for i in range(26)]
    radii = [.03125] * 26
    initial = np.array([x for pair in centers for x in pair] + radii)
    best = initial.copy()
    quality, improvements = math.fsum(best[52:]), 0
    for index in range(4 * (calls - 1) + 1):
        collector.begin_start(index, incumbent_sum=quality, improvements=improvements)
        scheduled = index % 4 == 0
        effective = best.copy()
        if scheduled:
            ordinal = index // 4
            noise = np.array([((j + ordinal) % 7 - 3) * 2.**-16 for j in range(78)])
            ordinary = best + noise
            inputs = dict(best=best.copy(), initial=initial.copy(), noise=noise, ordinary=ordinary)
            collector.begin_helper(inputs)
            collector.helper_end(raw_kind='value', raw_type='ndarray', raw_value=effective,
                post_inputs=inputs, effective=effective, origin='accepted_return',
                incumbent_sum=quality, improvements=improvements)
        collector.optimizer_enter(incumbent_sum=quality, improvements=improvements)
        if not returned:
            break
        # Two explicitly synthetic callback improvements make the full prefix
        # nonconstant, while no result.x is credited with incumbent admission.
        if realistic and scheduled and index < 8:
            best = best.copy(); best[52:] += 2.**-20
            quality, improvements = math.fsum(best[52:]), improvements + 1
        collector.optimizer_exit(incumbent_sum=quality, improvements=improvements)
        if not observed:
            break
        outcome = SimpleNamespace() if status == 'missing' else SimpleNamespace(status=status)
        collector.optimizer_outcome(success, outcome)
        x = effective.copy()
        if realistic:
            x[:52] += ((index // 4) % 3) * 2.**-24
        if result == 'invalid': x[0] = -1.
        elif result == 'wrongshape': x = np.zeros(77)
        elif result == 'nonfinite': x[0] = np.nan
        elif result == 'missing': x = None
        collector.optimizer_result(x)
        collector.end_start('completed', incumbent_sum=quality, improvements=improvements)
    trace = collector.finish('deadline' if not returned else 'error' if not observed else 'restart_limit',
        incumbent_sum=quality, improvements=improvements)
    capture = geometry.validate_packing(dict(centers=best[:52].reshape(26, 2).tolist(), radii=best[52:].tolist()))
    certificate = geometry.certify_packing(capture)
    assert capture['valid'] and certificate['status'] == 'certified'
    valid = returned and observed
    return dict(schema='sciel-repeat-draw/1', source_sha256=source, seed=seed,
        adapter_identity=adapter, status='valid' if valid else 'candidate_failure',
        reason=None if valid else 'synthetic stopped fixture', capture=capture, certificate=certificate,
        raw_sum=capture['sum_radii'] if valid else None,
        certified_sum=certificate['sum_exact'] if valid else None,
        seed_applied=dict(python_random=seed, numpy_legacy=seed, python_hash_seed=str(seed),
            before_candidate_import=True, determinism_proven=False),
        limits=dict(candidate_wall_seconds=30., certification_wall_seconds=10., candidate_cpu_seconds=35,
            certification_cpu_seconds=35, candidate_mem_bytes=2**31, certification_mem_bytes=2**31),
        started_at='2026-09-17T00:00:00+00:00', completed_at='2026-09-17T00:00:01+00:00',
        elapsed_seconds=1., candidate_elapsed_seconds=.8, certification_elapsed_seconds=.1,
        identity_elapsed_seconds=0., restart_trace=dict(status='captured', reason=None,
            trace=trace, sidecar_sha256=sha(trace)))


def build(value, **overrides):
    bindings = dict(source_sha256=value['source_sha256'], seed=value['seed'],
        adapter_identity=value['adapter_identity'])
    return observations.build_observation(value, **dict(bindings, **overrides))


def row(value, index=0):
    return observations.observation_cases(value)[index]


def reseal(value):
    value['observation_sha256'] = sha({k: v for k, v in value.items() if k != 'observation_sha256'})
    return value


@pytest.mark.parametrize('success,status,result,valid', [
    (True, 0, 'invalid', False), (False, 8, 'valid', True), (True, 0, 'valid', True),
])
def test_reported_success_geometry_and_incumbent_admission_are_separate(success, status, result, valid):
    saved = receipt(success=success, status=status, result=result)
    original = deepcopy(saved)
    value = build(saved); case = row(value)
    assert value['schema'] == optimizer.SCHEMA and value['trace_schema'] == codec.TRACE_SCHEMA
    assert case['success'] is success and case['status'] == status
    assert case['result_tolerance_valid'] is valid
    assert case['helper_accepted'] is True and case['optimizer_returned'] is True
    assert len(set(case['phase_sums'])) == 1 and case['phase_improvements'] == [0] * 5
    assert value['run']['origins']['accepted_return']['gain'] == 0.
    assert value['receipt_sha256'] == sha(saved)
    assert value['trace_sha256'] == saved['restart_trace']['trace']['trace_sha256']
    assert saved == original


@pytest.mark.parametrize('result,expected', [('missing', None), ('wrongshape', False), ('nonfinite', False)])
def test_bad_or_missing_snapshot_is_not_inferred_from_optimizer_success(result, expected):
    value = build(receipt(result=result, success=True))
    case = row(value)
    assert case['success'] is True and case['optimizer_returned'] is True
    assert case['result_tolerance_valid'] is expected
    assert case['result_geometry_reason']
    assert case['result_values_equal_best'] is None
    if result == 'missing':
        assert case['result_geometry_reason'] == 'array_unavailable'


@pytest.mark.parametrize('returned,reason', [(False, 'not_returned'), (True, 'not_observed')])
def test_unobserved_status_and_unreturned_optimizer_stay_unknown(returned, reason):
    value = build(receipt(returned=returned, observed=False))
    case = row(value)
    assert case['success'] is None and case['success_reason'] == reason
    assert case['status'] is None and case['status_reason'] == reason
    assert case['optimizer_returned'] is returned and case['result_tolerance_valid'] is None
    assert case['result_geometry_reason'] == ('array_unavailable' if returned else 'not_applicable')


def test_missing_status_is_not_zero_and_success_does_not_fill_it():
    case = row(build(receipt(status='missing')))
    assert case['success'] is True and case['status'] is None
    assert case['status_reason'] == 'missing' and case['result_tolerance_valid'] is True


@pytest.mark.parametrize('mode', ['completed_unknown', 'unreturned_observed', 'unobserved_status'])
def test_rehashed_presentation_cannot_contradict_optimizer_phase(mode):
    value = build(receipt(returned=mode != 'unreturned_observed',
                          observed=mode != 'unobserved_status'))
    state = value['optimizer_state_catalog'][0]
    if mode == 'completed_unknown':
        state[:4] = [None, 'not_observed', None, 'not_observed']
    else:
        state[:4] = ([True, None, 0, None] if mode == 'unreturned_observed'
                     else [None, 'not_observed', 0, None])
    with pytest.raises(ValueError):
        observations.observation_view(reseal(value))


@pytest.mark.parametrize('adapter', [None, {}, {'runtime': None}])
def test_malformed_adapter_is_value_error_not_attribute_or_key_error(adapter):
    saved = receipt()
    with pytest.raises(ValueError):
        build(saved, adapter_identity=adapter)


def test_rehashed_false_geometry_cannot_import_against_original_receipt():
    saved = receipt(result='invalid')
    value = build(saved)
    state = value['optimizer_state_catalog'][value['cases'][0][-1]]
    state[optimizer.STATE_COLUMNS.index('result_tolerance_valid')] = True
    state[optimizer.STATE_COLUMNS.index('result_geometry_reason')] = None
    value['observation_sha256'] = sha({k: v for k, v in value.items() if k != 'observation_sha256'})
    # Presentation validation is not a geometry proof; receipt reconstruction is.
    with pytest.raises(ValueError, match='projection differs'):
        observations.validate_observation(value, receipt=saved, source_sha256=saved['source_sha256'],
            seed=saved['seed'], adapter_identity=saved['adapter_identity'])
    assert row(build(saved))['result_tolerance_valid'] is False


@pytest.mark.parametrize('field', ['source_sha256', 'runtime_sha256', 'schema'])
def test_trace_identity_or_schema_mismatch_is_not_downgraded_to_legacy(field):
    saved = receipt(); trace = saved['restart_trace']['trace']
    if field == 'schema': trace['schema'] = 'sciel-restart-trace/999'
    else: trace['identity'][field] = 'f' * 64
    trace['trace_sha256'] = sha({k: v for k, v in trace.items() if k != 'trace_sha256'})
    with pytest.raises(ValueError):
        build(saved)


def test_state_catalog_dedup_whole_view_and_detached_expansion():
    value = build(receipt(calls=16))
    assert len(value['cases']) == 16 and len(value['optimizer_state_catalog']) == 1
    assert all(case[-1] == 0 for case in value['cases'])
    view = observations.observation_view(value)
    assert view['included'] and len(canonical(view)) <= 8192
    assert observations.validate_observation_view(view) == view
    tiny = observations.observation_view(value, max_chars=1000)
    assert tiny['included'] is False and tiny['observation'] is None
    expanded = observations.observation_cases(value)
    expanded[0]['success'] = False
    assert row(value)['success'] is True
    damaged = deepcopy(value); damaged['cases'][0][-1] = True
    damaged['observation_sha256'] = sha({k: v for k, v in damaged.items() if k != 'observation_sha256'})
    with pytest.raises(ValueError): observations.observation_view(damaged)


@pytest.mark.parametrize('mode', ['duplicate', 'unused', 'first_use'])
def test_rehashed_state_catalog_requires_unique_used_canonical_order(mode):
    value = build(receipt(calls=2))
    other = deepcopy(value['optimizer_state_catalog'][0])
    if mode != 'duplicate': other[:4] = [False, None, 8, None]
    value['optimizer_state_catalog'].append(other)
    if mode == 'first_use':
        value['cases'][0][-1], value['cases'][1][-1] = 1, 0
    with pytest.raises(ValueError):
        observations.observation_view(reseal(value))


@pytest.mark.parametrize('version,expected_sha', [
    (1, 'b72b4c127b66886d21f0743beeb0302ad346c1a91cb81bae1a02d4a5d1035dde'),
    (2, 'f4c8ef5a4175dffdc06337605e2b92e3ded0673306fa909b147c8c23aa9c672e'),
])
def test_legacy_projections_remain_byte_exact_without_new_outcomes(version, expected_sha):
    from test_restart_motion_observations import older
    saved = older(receipt(calls=2, realistic=True), version)
    before = canonical(saved)
    value = build(saved)
    # A compact golden hash pins the entire legacy wire view, including omissions.
    assert sha(observations.observation_view(value)) == expected_sha
    assert value['schema'] == observations.SCHEMA
    assert 'optimizer_state_catalog' not in value and 'result_geometry_policy' not in value
    assert canonical(saved) == before


def test_two_realistic_synthetic_full_prefixes_fit_current_pair_context_cap():
    from sciel.hypothesis import restart_motion_observations as motion_observations
    pairs = [dict(seed=seed,
        comparator=receipt(source='a' * 64, seed=seed, success=False, status=8, calls=16, realistic=True),
        challenger=receipt(source='c' * 64, seed=seed, success=True, status=0, calls=16, realistic=True))
        for seed in (11, 12, 13)]
    first = pairs[0]['comparator']
    group = paired_comparisons.build_comparison(namespace='synthetic_optimizer_capacity',
        batch_id='b1', branch='inert_fixture', origin_experiment_id=None,
        comparator_source_sha256='a' * 64, challenger_source_sha256='c' * 64,
        adapter_identity=first['adapter_identity'], limits=first['limits'], pairs=pairs,
        provenance={'synthetic_only': True, 'candidate_optimizer_provider_calls': 0})
    wire = restart_context.build_context([group], max_chars=12000)
    assert wire is not None and len(canonical(wire)) <= 12000
    expanded = restart_context.expand_context(wire)
    assert expanded['selected_count'] == 6
    for role in ('comparator', 'challenger'):
        cards = [c for c in expanded['cards'] if c['seed'] == 11 and c['role'] == role]
        assert len(cards) == 1, (role, expanded['omissions'], len(canonical(wire)))
        card = cards[0]
        value = card['observation_view']['observation']
        assert len(value['cases']) == value['prefix']['captured_count'] == 16
        assert value['schema'] == motion_observations.SCHEMA
        assert canonical(motion_observations.base_observation(value)) == canonical(build(pairs[0][role]))
        assert len(motion_observations.observation_cases(value)) == 16
    assert len(canonical(wire)) + len(restart_context.RULES) + len(restart_context.PREFIX) + 1 <= 12802
