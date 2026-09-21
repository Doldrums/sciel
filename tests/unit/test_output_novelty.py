"""Registry joins use inert, fully captured evaluator outputs, not scores."""
from copy import deepcopy
import json

import pytest

from sciel.engine.output_novelty import OutputNoveltyRegistry, is_repeat, validate_verdict
from sciel.eval.output_identity import adapt_output, digest, sealed
from test_output_identity import fixture


def observed(registry, tmp_path, evaluation='e1', source='1' * 64, island=0, **kwargs):
    task, _, outcome, binding = fixture(tmp_path, source=source, **kwargs)
    binding['island_id'] = island
    binding['receipt_sha256'] = digest(evaluation)
    identity = adapt_output(task, source, outcome, binding)
    assert identity is not None
    return registry.observe(evaluation, identity, binding), identity, binding


def registry():
    return OutputNoveltyRegistry({'task_hash': 'a' * 64, 'engine_contract': 'test-runtime/1'})


def reseal(value):
    return sealed({k: v for k, v in value.items() if k != 'sha256'})


def test_cross_island_eviction_independent_resume_and_aliases(tmp_path):
    r = registry()
    first, _, _ = observed(r, tmp_path)
    r.bind_program('e1', 'p1')
    r.bind_program('e1', 'p2')  # Seed clones/aliases are not new measurements.
    r.bind_program('e1', 'p1')
    assert first['status'] == 'new_output' and first['accept']
    # No population dependency: p1 may be evicted and still authenticates history.
    restored = registry()
    restored.load_state_dict(json.loads(json.dumps(r.state_dict())))
    second, _, _ = observed(restored, tmp_path, 'e2', '2' * 64, 17)
    assert is_repeat(second) and second['accept'] is False
    assert second['match'] == dict(evaluation_id='e1', source_sha256='1' * 64,
                                   island_id=0, program_id='p1')
    assert restored.summary() == dict(evaluations=2, distinct_outputs=1, new_output=1,
                                      exact_recorded_repeat=1, unsupported=0)


def test_repeat_measurements_registered_once_and_idempotent_exact_eval_id(tmp_path):
    r = registry()
    first, identity, binding = observed(r, tmp_path)
    assert r.observe('e1', identity, binding) == first
    assert r.summary()['evaluations'] == 1
    changed = dict(binding, receipt_sha256='d' * 64)
    with pytest.raises(ValueError, match='conflicts'): r.observe('e1', identity, changed)
    repeated, identity2, binding2 = observed(r, tmp_path, 'e2', '2' * 64)
    assert r.observe('e2', identity2, binding2) == repeated
    assert r.summary()['evaluations'] == 2


@pytest.mark.parametrize('kwargs', [dict(output={'different': [0]}), dict(reached='smoke'),
    dict(input_scope='artifact_sha256', input_sha='3' * 64),
    dict(seed_scope='input_seed', seed_sha='4' * 64),
    dict(reached='confirm', confirmation=dict(trials=2, selected_trial=0, selection='lowest'), confirmed=True)])
def test_output_input_seed_and_stage_scope_do_not_cross_match(tmp_path, kwargs):
    r = registry()
    observed(r, tmp_path)
    second, _, _ = observed(r, tmp_path, 'e2', '2' * 64, **kwargs)
    assert second['status'] == 'new_output' and second['accept']


@pytest.mark.parametrize('damage', ['seal', 'source', 'task', 'binding', 'scope', 'weak'])
def test_invalid_identity_can_never_reject(tmp_path, damage):
    r = registry()
    _, identity, binding = observed(r, tmp_path)
    if damage == 'seal': identity['sha256'] = 'f' * 64
    elif damage == 'source': identity['evidence']['source_sha256'] = '2' * 64
    elif damage == 'task': identity['scope']['task_hash'] = '2' * 64
    elif damage == 'binding': identity['evidence']['binding_sha256'] = '2' * 64
    elif damage == 'scope': identity['scope']['seeds']['kind'] = 'unknown'
    elif damage == 'weak': identity['scope']['adapter'] = 'case_vector_and_score'
    if damage != 'seal': identity = reseal(identity)
    result = r.observe('e2', identity, binding)
    assert result['status'] == 'unsupported' and result['accept'] and not is_repeat(result)
    r.bind_program('e2', 'accepted-without-output-adapter')


def test_frozen_verdict_stays_original_when_alias_added_later(tmp_path):
    r = registry()
    observed(r, tmp_path)
    repeat, identity, binding = observed(r, tmp_path, 'e2', '2' * 64)
    assert repeat['match']['program_id'] is None
    r.bind_program('e1', 'admitted-later')
    restored = registry()
    restored.load_state_dict(r.state_dict())
    assert restored.observe('e2', identity, binding) == repeat
    assert r.state_dict() == restored.state_dict()


@pytest.mark.parametrize('damage', ['digest', 'runtime', 'duplicate', 'false_repeat', 'missing_match',
    'false_new', 'source', 'wrong_alias', 'duplicate_alias'])
def test_restore_is_atomic_and_checks_resealed_history_joins(tmp_path, damage):
    r = registry()
    observed(r, tmp_path)
    r.bind_program('e1', 'p1')
    observed(r, tmp_path, 'e2', '2' * 64)
    saved = r.state_dict()
    state = deepcopy(saved)
    row = state['records'][1]
    if damage == 'digest': state['sha256'] = 'f' * 64
    elif damage == 'runtime': state['identity']['engine_contract'] = 'other/1'
    elif damage == 'duplicate': state['records'].append(deepcopy(row))
    elif damage == 'false_repeat':
        row['identity']['output_sha256'] = '9' * 64
        row['identity'] = reseal(row['identity'])
        row['verdict']['identity'] = deepcopy(row['identity'])
    elif damage == 'missing_match': row['verdict']['match']['evaluation_id'] = 'absent'
    elif damage == 'false_new': row['verdict'].update(status='new_output', accept=True, match=None)
    elif damage == 'source': row['binding']['source_sha256'] = '9' * 64
    elif damage == 'wrong_alias': row['verdict']['match']['program_id'] = 'unknown-program'
    elif damage == 'duplicate_alias': state['records'][0]['program_ids'].append('p1')
    if damage != 'digest':
        row['verdict'] = reseal(row['verdict'])
        state = reseal(state)
    with pytest.raises(ValueError): r.load_state_dict(state)
    assert r.state_dict() == saved


def test_validation_rejects_wrong_admission_and_research_source(tmp_path):
    r = registry()
    first, _, _ = observed(r, tmp_path)
    with pytest.raises(ValueError, match='source differs'):
        validate_verdict(first, source_sha256='9' * 64)
    first['accept'] = False
    with pytest.raises(ValueError, match='admission'):
        validate_verdict(reseal(first))


def test_detached_mutation_deepcopy_and_unsupported_roundtrip(tmp_path):
    r = registry()
    identity = r.identity
    identity['engine_contract'] = 'mutated externally'
    assert r.identity['engine_contract'] == 'test-runtime/1'
    _, _, outcome, binding = fixture(tmp_path)
    value = r.observe('e1', None, binding)
    r.bind_program('e1', 'p1')
    copy = deepcopy(r)
    copy.bind_program('e1', 'p2')
    assert r.state_dict()['records'][0]['program_ids'] == ['p1']
    value['binding']['source_sha256'] = 'f' * 64
    restored = registry()
    restored.load_state_dict(r.state_dict())
    assert restored.state_dict() == r.state_dict()
