"""Host arithmetic and inert collector fixtures; no helper/optimizer/provider."""
from copy import deepcopy

import pytest

from sciel.hypothesis import restart_observations as base
from sciel.hypothesis import restart_motion_observations as enriched
from sciel.hypothesis import restart_motion as motion
from sciel.hypothesis import restart_optimizer_trace as codec
from sciel.hypothesis import restart_trace as legacy
from test_optimizer_restart_observations import receipt


def bindings(saved):
    return dict(source_sha256=saved['source_sha256'], seed=saved['seed'],
                adapter_identity=saved['adapter_identity'])


def build(saved):
    return enriched.build_observation(saved, **bindings(saved))


def seal(value):
    value['observation_sha256'] = base._hash({k: v for k, v in value.items() if k != 'observation_sha256'})
    return value


def older(saved, version):
    """Convert authored traces into explicitly older wire versions."""
    saved = deepcopy(saved)
    trace = codec.legacy_projection(saved['restart_trace']['trace'])
    if version == 1:
        trace['schema'] = legacy.LEGACY_TRACE_SCHEMA
        for case in trace['calls']:
            case.pop('optimizer_result')
        used = {ref for case in trace['calls'] for ref in legacy._rt_record_refs(case)}
        trace['vectors'] = {ref: value for ref, value in trace['vectors'].items() if ref in used}
        trace['trace_sha256'] = base._hash({k: v for k, v in trace.items() if k != 'trace_sha256'})
        legacy.validate_trace(trace)
    saved['restart_trace']['trace'] = trace
    saved['restart_trace']['sidecar_sha256'] = base._hash(trace)
    return saved


@pytest.mark.parametrize('version', [1, 2, 3])
def test_exact_legacy_base_default_and_motion_roundtrip_detached(version):
    saved = receipt(calls=4, realistic=True)
    if version < 3:
        saved = older(saved, version)
    before = base._canonical(saved)
    original = base.build_observation(saved, **bindings(saved))
    value = build(saved)
    assert value['schema'] == enriched.SCHEMA
    assert value['base_schema'] == original['schema']
    assert base._canonical(enriched.base_observation(value)) == base._canonical(original)
    assert base._canonical(base.build_observation(saved, **bindings(saved))) == base._canonical(original)
    assert base._canonical(saved) == before
    trace = saved['restart_trace']['trace']
    expected = motion.build_motion(trace, **trace['identity'])
    assert value['motion_sha256'] == expected['motion_sha256']
    assert value['motion_policy_sha256'] == base._hash(motion.POLICY)
    old_cases = base.observation_cases(original)
    cases = base.observation_cases(value)
    for old, new, moved in zip(old_cases, cases, motion.motion_cases(expected), strict=True):
        assert {key: new[key] for key in old} == old
        assert {key: new[key] for key in moved} == moved
    restored = enriched.base_observation(value); restored['bindings']['seed'] += 1
    cases[0]['match_status'] = 'modified copy'
    assert base.observation_cases(value)[0]['match_status'] != 'modified copy'
    assert base.validate_observation(value, receipt=saved, **bindings(saved)) == value
    assert base.validate_observation_view(base.observation_view(value))['observation'] == value


@pytest.mark.parametrize('result,valid', [('invalid', False), ('wrongshape', False), ('nonfinite', False), ('missing', None)])
def test_motion_does_not_reinterpret_result_validity_or_unknown(result, valid):
    saved = receipt(result=result, success=False, status=8)
    facts = base.observation_cases(build(saved))[0]
    assert facts['success'] is False and facts['status'] == 8
    assert facts['result_tolerance_valid'] is valid
    if result == 'invalid':
        assert facts['result_center_max'] > 0.  # Finite invalid geometry still has descriptive movement.
    else:
        assert facts['result_center_max'] is None and facts['match_status'] == 'unavailable'
        assert facts['match_center_max'] is facts['match_radius_max'] is None


def test_unreturned_and_legacy_result_are_explicitly_unavailable():
    for saved, reason in ((receipt(returned=False), 'result:not_applicable'),
                          (older(receipt(), 1), 'result:legacy_not_recorded')):
        value = build(saved); facts = enriched.observation_cases(value)[0]
        assert facts['result_center_max'] is None and facts['result_center_reason'] == reason
        assert facts['match_status'] == 'unavailable' and facts['match_reason'] == reason


@pytest.mark.parametrize('diagnostic', [None, 'missing'])
def test_unavailable_receipt_has_no_fabricated_motion(diagnostic):
    saved = receipt()
    if diagnostic is None:
        saved.pop('restart_trace')
    else:
        saved['restart_trace'] = dict(status='missing', reason='no_saved_sidecar', trace=None, sidecar_sha256=None)
    old = base.build_observation(saved, **bindings(saved)); value = build(saved)
    assert value['status'] == 'unavailable' and value['motion_sha256'] is None
    assert value['cases'] == value['motion_state_catalog'] == value['phase_refs_catalog'] == []
    assert enriched.base_observation(value) == old
    assert base.observation_cases(value) == []


@pytest.mark.parametrize('field', ['phase_refs_ref', 'snapshot_refs_ref', 'motion_state_ref'])
def test_bool_or_missing_catalog_reference_rejected_even_rehashed(field):
    value = build(receipt())
    position = value['columns']['cases'].index(field)
    for bad in (False, -1, 99, None):
        altered = deepcopy(value); altered['cases'][0][position] = bad
        with pytest.raises(ValueError): enriched.validate_shape(seal(altered))


@pytest.mark.parametrize('name', ['phase_refs_catalog', 'snapshot_refs_catalog', 'motion_state_catalog'])
def test_duplicate_unused_and_reordered_catalog_entries_rejected(name):
    value = build(receipt(calls=4, realistic=True))
    altered = deepcopy(value); altered[name].append(deepcopy(altered[name][0]))
    with pytest.raises(ValueError, match='duplicate'): enriched.validate_shape(seal(altered))
    altered = deepcopy(value); extra = deepcopy(altered[name][0]); extra[0] = 'invented'; altered[name].append(extra)
    with pytest.raises(ValueError): enriched.validate_shape(seal(altered))
    if len(value[name]) > 1:
        altered = deepcopy(value); altered[name].reverse()
        with pytest.raises(ValueError): enriched.validate_shape(seal(altered))


def test_rehashed_false_motion_is_not_receipt_evidence():
    saved = receipt(calls=2, realistic=True); value = build(saved)
    index = value['columns']['cases'].index('match_center_max')
    value['cases'][1][index] = .5
    restored_motion = enriched._restore_motion(value, enriched._rows(value))
    value['motion_sha256'] = motion._hash({k: v for k, v in restored_motion.items() if k != 'motion_sha256'})
    value = seal(value)
    assert enriched.validate_shape(value) == value  # A consistent claim, not authenticated truth.
    with pytest.raises(ValueError, match='differs from durable receipt'):
        enriched.validate_observation(value, receipt=saved, **bindings(saved))


def test_original_base_hash_and_motion_hash_are_independently_bound():
    value = build(receipt())
    for key in ('base_observation_sha256', 'motion_sha256', 'motion_policy_sha256'):
        altered = deepcopy(value); altered[key] = 'f' * 64
        with pytest.raises(ValueError): base._compact(seal(altered))


@pytest.mark.parametrize('key', ['source_sha256', 'seed', 'scaffold_sha256', 'adapter_identity'])
def test_external_identity_mismatch_rejected(key):
    saved = receipt(); args = bindings(saved)
    if key == 'adapter_identity':
        args[key] = deepcopy(args[key]); args[key]['runtime']['different'] = True
    else:
        args[key] = 99 if key == 'seed' else 'f' * 64
    with pytest.raises(ValueError): enriched.build_observation(saved, **args)


@pytest.mark.parametrize('source,seed', [('a' * 64, 17), ('b' * 64, 31)])
def test_full16_view_fits_without_loss_or_status_fabrication(source, seed):
    saved = receipt(source=source, seed=seed, calls=16, realistic=True)
    original = base.build_observation(saved, **bindings(saved))
    value = build(saved)
    view = base.observation_view(value)
    assert base._canonical(enriched.base_observation(value)) == base._canonical(original)
    assert view['included'] and len(value['cases']) == 16
    assert len(base._canonical(view)) <= 8192
    assert len(value['snapshot_refs_catalog']) < 16
    assert base.observation_view(value, max_chars=1000)['included'] is False
    assert value['prefix'] == saved['restart_trace']['trace']['prefix']
    assert all(case['success'] is True and case['status'] == 0 for case in base.observation_cases(value))
