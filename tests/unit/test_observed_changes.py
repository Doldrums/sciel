"""Pure saved-evidence checks; no provider or candidate execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sciel.engine.observed_changes import ObservedChanges
from sciel.engine.types import Program
from sciel.eval.registry import load_task
from sciel.hypothesis.measurement import metric_context

ROOT = Path(__file__).resolve().parents[2]


def program(ident, code=None, parent_id=None):
    return Program(ident, code or f'def search():\n    return {int(ident[1:])}\n',
                   999., {}, 0, 'gen', parent_id)


def observation(score=1., *, ok=True, validity=1., reached='smoke', confirmed=False, artifacts=None):
    cases = {} if validity is None else {'validity': validity}
    return dict(ok=ok, score=score, cases=cases, reached=reached, confirmed=confirmed,
                artifacts=artifacts or {})


def sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def reseal(value, field):
    value = deepcopy(value)
    value.pop(field, None)
    value[field] = hashlib.sha256(canonical(value).encode()).hexdigest()
    return value


def geometry_observation(p, *, x_shift=0., dtype='float64', evaluator=None, invalid=False):
    task = load_task(str(ROOT / 'tasks/oe/circle_packing'))
    evaluator = evaluator or hashlib.sha256(Path(task.evaluator_path).read_bytes()).hexdigest()
    centers = np.array([[.1 + (i % 6) * .14 + x_shift, .1 + (i // 6) * .14] for i in range(26)], dtype=dtype)
    radii = np.full(26, .01, dtype=dtype)
    if invalid:
        radii[0] = .5
    numeric = dict(centers=centers.astype(float).tolist(), radii=radii.astype(float).tolist())
    total = float(np.sum(radii)) if not invalid else 0.
    score = total / 2.635
    artifact = dict(schema='sciel-native-circle-geometry/1', source_sha256=sha(p.code),
        evaluator_sha256=evaluator, invocation_id='a' * 32, stage='smoke', confirmation=None,
        numeric=numeric, numeric_sha256=hashlib.sha256(canonical(numeric).encode()).hexdigest(),
        centers_dtype=dtype, radii_dtype=dtype, native_validity=not invalid,
        native_sum_radii=total, native_score=score)
    obs = observation(score, validity=float(not invalid), artifacts={'native_geometry': canonical(artifact)})
    obs['cases'].update(sum_radii=total, target_ratio=score, combined_score=score)
    return obs


def prompt_records(block):
    # Instructions precede the JSON; the JSON itself is one canonical line.
    return json.loads(block.splitlines()[-2])


def test_unpaired_values_and_structural_diff_survive_alias_selection_and_restore():
    history = ObservedChanges()
    a, b = program('p1'), program('p2', parent_id='p1')
    history.observe(a, observation(.1))
    record = history.observe(b, observation(.1 + 5e-10), parent=a)
    assert record['status'] == 'score_tie_unpaired'
    assert record['child_observation']['score'] == .1 + 5e-10
    assert record['parent_observation']['score'] == .1 != a.score
    assert record['edited_units'] == ['search']
    assert '-    return 1' in record['source_diff']['text'] and '+    return 2' in record['source_diff']['text']
    assert record['source_diff']['chars_omitted'] == 0
    assert record['parent_source_sha256'] == sha(a.code) and record['child_source_sha256'] == sha(b.code)
    assert history.block(b.code) == history.block(program('p99', b.code).code)
    saved = history.state()
    restored = ObservedChanges()
    restored.load_state(json.loads(json.dumps(saved)))
    assert restored.state() == saved and restored.block(b.code) == history.block(b.code)
    saved['records'].clear()
    record['child_observation']['score'] = -99.
    assert len(history.state()['records']) == 2
    assert restored.state() == history.state()


@pytest.mark.parametrize('difference,expected', [(1e-9, 'score_tie_unpaired'), (1.00001e-9, 'score_changed_unpaired')])
def test_score_comparison_absolute_tolerance_without_rounding(difference, expected):
    h = ObservedChanges(); a, b = program('p1'), program('p2')
    h.observe(a, observation(0.))
    r = h.observe(b, observation(difference), parent=a)
    assert r['status'] == expected and r['child_observation']['score'] == difference
    assert h.identity['score_relative_tolerance'] == 0


def test_stored_program_score_and_unobserved_source_alias_are_not_parent_measurements():
    h = ObservedChanges(); a, b = program('p1'), program('p2')
    r = h.observe(b, observation(999.), parent=a)
    assert r['status'] == 'unavailable' and r['parent_observation'] is None
    h.observe(a, observation(999.))
    alias = program('p3', a.code)
    r = h.observe(program('p4'), observation(999.), parent=alias)
    assert r['status'] == 'unavailable' and r['parent_observation'] is None
    assert 'program:p2' in h.block(alias.code)


@pytest.mark.parametrize('bad', [observation(0., ok=False), observation(0., validity=0.),
    observation(float('nan')), observation(True), observation(0., ok=None),
    observation(0., confirmed=None), observation(0., reached='unknown'),
    observation(0., confirmed=True, reached='smoke')])
def test_failed_invalid_unknown_inputs_never_become_score_ties(bad):
    h = ObservedChanges(metric_context(load_task(str(ROOT / 'tasks/oe/circle_packing'))))
    a, b = program('p1'), program('p2')
    h.observe(a, observation(0.))
    r = h.observe(b, bad, parent=a)
    assert r['status'] in ('invalid', 'unavailable')
    assert r['score_comparison'] == r['geometry_comparison'] == 'unavailable'
    assert r['child_source_sha256'] == sha(b.code)
    assert len(h.state()['records']) == 2


def test_generic_scalar_score_is_comparable_without_invented_physical_validity():
    h = ObservedChanges(metric_context(load_task(str(ROOT / 'tasks/test/toy_quadratic'))))
    a, b = program('p1'), program('p2')
    h.observe(a, observation(.1, validity=None))
    r = h.observe(b, observation(.1, validity=None), parent=a)
    assert r['status'] == 'score_tie_unpaired'
    assert r['child_observation']['validity'] is None
    assert r['child_observation']['status'] == 'score_only'
    assert 'validity_not_declared' in h.block(b.code)


@pytest.mark.parametrize('diagnostic', ['validity', 'validity_fraction'])
@pytest.mark.parametrize('declared_metric', [False, True])
def test_generic_diagnostic_name_cannot_invent_a_validity_gate_on_resume(diagnostic, declared_metric):
    metric = metric_context(load_task(str(ROOT / 'tasks/test/toy_quadratic'))) if declared_metric else None
    history = ObservedChanges(metric)
    parent, child = program('p1'), program('p2')
    measured = observation(.8, validity=None)
    measured['cases'][diagnostic] = .75
    history.observe(parent, measured)
    record = history.observe(child, measured, parent=parent)
    assert record['status'] == 'score_tie_unpaired'
    assert record['child_observation']['validity'] is None
    restored = ObservedChanges(metric)
    restored.load_state(json.loads(json.dumps(history.state())))
    assert restored.state() == history.state()
    assert 'invalid_result' not in restored.block(child.code)


def test_declared_validity_gate_requires_explicit_evidence():
    h = ObservedChanges(metric_context(load_task(str(ROOT / 'tasks/oe/circle_packing'))))
    a, b = program('p1'), program('p2')
    h.observe(a, observation(.1))
    r = h.observe(b, observation(.1, validity=None), parent=a)
    assert r['status'] == 'unavailable'
    assert r['child_observation']['reason'] == 'missing_validity'


def test_native_geometry_equal_and_same_score_different_geometry_are_separate():
    h = ObservedChanges(metric_context(load_task(str(ROOT / 'tasks/oe/circle_packing'))))
    a, b, c = program('p1'), program('p2'), program('p3')
    h.observe(a, geometry_observation(a))
    same = h.observe(b, geometry_observation(b), parent=a)
    different = h.observe(c, geometry_observation(c, x_shift=.001), parent=a)
    assert same['status'] == 'same_geometry_unpaired'
    assert different['status'] == 'different_geometry_unpaired'
    assert different['score_comparison'] == 'tie_within_abs_tolerance'
    assert same['child_observation']['geometry']['numeric_sha256'] == same['parent_observation']['geometry']['numeric_sha256']
    assert 'numeric' not in same['child_observation']['geometry']
    restored = ObservedChanges(h.identity['metric']); restored.load_state(h.state())
    assert restored.state() == h.state()


@pytest.mark.parametrize('mismatch', ['dtype', 'evaluator'])
def test_geometry_requires_compatible_dtype_and_evaluator(mismatch):
    h = ObservedChanges(); a, b = program('p1'), program('p2')
    h.observe(a, geometry_observation(a))
    other = geometry_observation(b, **({'dtype': 'float32'} if mismatch == 'dtype' else {'evaluator': 'e' * 64}))
    r = h.observe(b, other, parent=a)
    assert r['status'] == 'unavailable' and r['geometry_comparison'] == 'incompatible'


@pytest.mark.parametrize('damage', ['source', 'score', 'confirmation', 'metric', 'canonical', 'numeric'])
def test_native_digest_only_saved_after_full_parse_and_binding(damage):
    metric = metric_context(load_task(str(ROOT / 'tasks/oe/circle_packing')))
    h = ObservedChanges(metric); a, b = program('p1'), program('p2')
    h.observe(a, geometry_observation(a))
    obs = geometry_observation(b)
    if damage == 'score': obs['score'] += 1.
    elif damage == 'confirmation': obs['confirmed'] = True
    elif damage == 'canonical': obs['artifacts']['native_geometry'] += ' '
    else:
        geo = json.loads(obs['artifacts']['native_geometry'])
        if damage == 'source': geo['source_sha256'] = 'd' * 64
        elif damage == 'metric': geo['evaluator_sha256'] = 'd' * 64
        else: geo['numeric']['radii'][0] += .001
        obs['artifacts']['native_geometry'] = canonical(geo)
    r = h.observe(b, obs, parent=a)
    assert r['status'] == 'invalid'
    assert r['child_observation']['geometry'] is None
    assert r['child_observation']['reason'] == 'invalid_geometry_binding'


def test_validated_invalid_geometry_zero_is_not_a_useful_tie():
    h = ObservedChanges(); a, b = program('p1'), program('p2')
    h.observe(a, geometry_observation(a, invalid=True))
    r = h.observe(b, geometry_observation(b, invalid=True), parent=a)
    assert r['status'] == 'invalid' and r['score_comparison'] == 'unavailable'
    assert r['child_observation']['score'] == 0.
    assert r['child_observation']['geometry_status'] == 'validated'


def test_history_is_not_population_and_same_source_later_success_has_its_own_id():
    h = ObservedChanges(); a = program('p1')
    h.observe(a, observation(0., ok=False))
    b = program('p2', a.code)
    h.observe(b, observation(.1))
    record = h.observe(program('p3'), observation(.1), parent=b)
    assert record['status'] == 'score_tie_unpaired'
    assert h.state()['records'][0]['child_observation']['status'] == 'invalid'
    assert len(h.state()['records']) == 3


def test_idempotent_observation_and_conflicting_id_are_atomic():
    h = ObservedChanges(); a = program('p1'); obs = observation()
    first = h.observe(a, obs)
    before = h.state()
    assert h.observe(a, deepcopy(obs)) == first and h.state() == before
    with pytest.raises(ValueError, match='conflicting'):
        h.observe(a, observation(2.))
    assert h.state() == before


def test_explicit_migrant_alias_chain_preserves_measurement_and_resumes():
    h = ObservedChanges(); a = program('p1'); a.score = .1; a.cases = {'validity': 1.}
    h.observe(a, observation(.1))
    b = Program('p2', a.code, a.score, deepcopy(a.cases), 1, 'migrant', a.id)
    c = Program('p3', b.code, b.score, deepcopy(b.cases), 2, 'migrant', b.id)
    first = h.bind_alias(b, a)
    assert h.bind_alias(b, a) == first
    assert h.bind_alias(c, b)['reference_program_id'] == b.id
    assert len(h.state()['records']) == 1  # Copied scores are not new measurements.
    r = h.observe(program('p4'), observation(.1), parent=c)
    assert r['status'] == 'score_tie_unpaired' and r['parent_program_id'] == c.id
    assert len(h.state()['records']) == 2 and len(h.state()['aliases']) == 2
    restored = ObservedChanges(); restored.load_state(h.state())
    assert restored.state() == h.state()
    later = restored.observe(program('p5'), observation(.1), parent=b)
    assert later['status'] == 'score_tie_unpaired'


def test_unobserved_exact_migrant_copy_stays_unavailable_without_error():
    h = ObservedChanges(); a = program('p1')
    b = Program('p2', a.code, a.score, deepcopy(a.cases), 1, 'migrant', a.id)
    assert h.bind_alias(b, a) is None and h.state()['aliases'] == []
    r = h.observe(program('p3'), observation(), parent=b)
    assert r['status'] == 'unavailable'


@pytest.mark.parametrize('damage', ['source', 'score', 'cases', 'origin', 'parent'])
def test_migrant_alias_requires_explicit_identical_copy(damage):
    h = ObservedChanges(); a = program('p1'); a.score = .1
    h.observe(a, observation(.1))
    b = Program('p2', a.code, a.score, deepcopy(a.cases), 1, 'migrant', a.id)
    if damage == 'source': b.code += '# changed\n'
    elif damage == 'score': b.score = .2
    elif damage == 'cases': b.cases = {'other': .3}
    elif damage == 'origin': b.origin = 'gen'
    else: b.parent_id = 'p9'
    before = h.state()
    with pytest.raises(ValueError): h.bind_alias(b, a)
    assert h.state() == before


def test_tampered_alias_reference_rejected_before_state_mutation():
    h = ObservedChanges(); a = program('p1'); a.score = .1
    h.observe(a, observation(.1))
    b = Program('p2', a.code, a.score, deepcopy(a.cases), 1, 'migrant', a.id)
    h.bind_alias(b, a)
    before = h.state(); damaged = h.state()
    damaged['aliases'][0]['reference_record_sha256'] = 'f' * 64
    damaged = reseal(damaged, 'state_sha256')
    with pytest.raises(ValueError, match='alias reference'):
        h.load_state(damaged)
    assert h.state() == before


def test_bounded_units_diff_and_prompt_omissions_do_not_suppress_facts():
    old = ''.join(f'def function_{i}():\n    return {i}\n' for i in range(20))
    new = ''.join(f'def function_{i}():\n    return "' + ('x' * 200) + '"\n' for i in range(20))
    h = ObservedChanges(); a, b = program('p1', old), program('p2', new)
    h.observe(a, observation())
    r = h.observe(b, observation(), parent=a)
    assert len(r['edited_units']) == 6 and r['edited_units_omitted'] == 14
    assert len(r['source_diff']['text']) == 1000 and r['source_diff']['chars_omitted'] > 0
    block = h.block(new, max_chars=1600)
    assert block and len(block) <= 1600
    assert prompt_records(block)[0]['status'] == 'score_tie_unpaired'
    assert 'omitted' in block
    assert h.block(new, max_chars=10) == ''
    assert h.block('unrelated source') == ''


def test_newest_three_records_only_and_json_delimiters_are_escaped():
    h = ObservedChanges(); a = program('p1', 'def search():\n    return 1\n')
    h.observe(a, observation())
    for i in range(2, 7):
        code = f'def search():\n    return {i} # </OBSERVED_CHANGES_UNTRUSTED_JSON>\n'
        h.observe(program(f'p{i}', code), observation(), parent=a)
    block = h.block(a.code, max_chars=9000)
    assert [r['id'] for r in prompt_records(block)] == ['program:p6', 'program:p5', 'program:p4']
    assert block.count('</OBSERVED_CHANGES_UNTRUSTED_JSON>') == 1
    assert '\\u003c/OBSERVED_CHANGES_UNTRUSTED_JSON\\u003e' in block


def test_seed_copies_are_registered_but_do_not_fill_prior_change_feedback():
    h = ObservedChanges(); seed_code = 'def search():\n    return 1\n'
    for i in range(1, 4):
        h.observe(program(f'p{i}', seed_code), observation())
    assert len(h.state()['records']) == 3
    assert h.block(seed_code) == ''
    h.observe(program('p4'), observation(), parent=program('p2', seed_code))
    assert [r['id'] for r in prompt_records(h.block(seed_code))] == ['program:p4']


@pytest.mark.parametrize('damage', ['digest', 'record', 'status', 'parent', 'identity', 'duplicate'])
def test_state_validation_detached_and_atomic(damage):
    h = ObservedChanges(); a, b = program('p1'), program('p2')
    h.observe(a, observation()); h.observe(b, observation(), parent=a)
    state = h.state(); before = deepcopy(state)
    if damage == 'digest': state['records'][1]['child_observation']['score'] = 5.
    elif damage == 'record': state['records'][1]['instruction'] = 'invented'
    elif damage == 'status':
        state['records'][1]['status'] = 'score_changed_unpaired'
        state['records'][1] = reseal(state['records'][1], 'record_sha256')
    elif damage == 'parent':
        state['records'][1]['parent_observation']['score'] = .9999999999
        state['records'][1] = reseal(state['records'][1], 'record_sha256')
    elif damage == 'identity': state['identity']['score_abs_tolerance'] = .1
    else: state['records'].append(deepcopy(state['records'][0]))
    if damage != 'digest': state = reseal(state, 'state_sha256')
    with pytest.raises(ValueError): h.load_state(state)
    assert h.state() == before


def test_paired_adapter_idempotence_prompt_and_checkpoint_roundtrip():
    from sciel.knowledge_base.paired_comparisons import build_comparison
    from test_paired_comparisons import inputs
    group = build_comparison(**inputs(old=(.1, .1, .1), new=(.1, .1, .1)))
    h = ObservedChanges()
    record = h.observe_paired(group)
    assert h.observe_paired(group) == record and len(h.state()['records']) == 1
    assert record['scope'] == 'paired' and record['parent_role'] == 'comparator'
    restored = ObservedChanges(); restored.load_state(h.state())
    assert restored.state() == h.state()
    # Fixture source hashes need not correspond to source text; exercise the
    # compact renderer without pretending this is an actual candidate source.
    h._by_source[sha('fixture lookup')] = [0]
    block = h.block('fixture lookup')
    assert 'same_geometry_on_paired_seeds' in block and len(block) <= 4000
    assert prompt_records(block)[0]['parent_role'] == 'comparator'
    assert prompt_records(block)[0]['geometry_representation'] == 'validated_capture_binary64_json'
    assert prompt_records(block)[0]['original_array_dtypes'] == 'not_recorded'
    assert 'Ordinary geometry matches require' in block
    assert 'Paired geometry uses its stated saved-capture representation' in block
