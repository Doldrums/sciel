"""Pure saved-output fixtures; no candidate or evaluator code is executed."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.output_identity import (CONFIG_SCHEMA, WITNESS_SCHEMA, adapt_output, canonical,
    digest, sealed, validate_identity, validate_output_identity_config)


def fixture(tmp_path, *, source='1' * 64, output=None, reached='full', confirmation=None,
            confirmed=False, input_scope='task_fixed', input_sha=None, seed_scope='unpaired',
            seed_sha=None, invocation='saved-evaluation-1'):
    evaluator = tmp_path / 'evaluator.py'
    evaluator.write_text('# inert trusted evaluator fixture; never executed\n')
    task = SimpleNamespace(task_hash='a' * 64, evaluator_path=str(evaluator), confirm_n=2, output_identity={
        'schema': CONFIG_SCHEMA, 'adapter': 'canonical-json/1', 'artifact': 'output_identity',
        'input_scope': input_scope, 'seed_scope': seed_scope})
    witness = dict(schema=WITNESS_SCHEMA, source_sha256=source,
        evaluator_sha256=hashlib.sha256(evaluator.read_bytes()).hexdigest(), invocation_id=invocation,
        stage='full' if reached == 'confirm' else reached, confirmation=confirmation,
        input_sha256=input_sha, seed_sha256=seed_sha, output=output if output is not None else {
            'sequence': [1, 2, 3], 'label': 'exact π output', 'negative_zero': -0.0})
    cases = {'metric': .8}
    if confirmation is not None:
        cases['n_trials'] = float(confirmation['trials'])
    outcome = CascadeOutcome(EvalResult(ok=True, score=.8, cases=cases, confirmed_sampled=confirmed,
        artifacts={'output_identity': canonical(witness)}), raw_first_score=.5,
        reached=reached, confirmed=confirmed)
    binding = dict(task_hash=task.task_hash, source_sha256=source, island_id=0,
                   receipt_sha256='b' * 64, request_sha256='c' * 64)
    return task, witness, outcome, binding


def identity(tmp_path, **kwargs):
    task, witness, outcome, binding = fixture(tmp_path, **kwargs)
    value = adapt_output(task, binding['source_sha256'], outcome, binding)
    assert value is not None
    return value, binding


def test_generic_opt_in_full_witness_and_inputs_remain_unchanged(tmp_path):
    task, witness, outcome, binding = fixture(tmp_path)
    original = deepcopy((witness, outcome, binding, task.output_identity))
    value = adapt_output(task, binding['source_sha256'], outcome, binding)
    assert validate_identity(value) == value
    assert value['output_sha256'] == digest(witness['output'])
    assert value['scope']['seeds'] == {'kind': 'unpaired', 'sha256': None}
    assert (witness, outcome, binding, task.output_identity) == original
    assert 'output' not in value  # Immutable receipt retains full witness.


def test_never_fallback_to_score_cases_digest_or_unknown_task(tmp_path):
    task, witness, outcome, binding = fixture(tmp_path)
    for artifacts in ({}, {'output_identity': 'digest-only:' + 'f' * 64},
                      {'native_geometry': canonical({'numeric_sha256': 'a' * 64})}):
        outcome.result.artifacts = artifacts
        assert adapt_output(task, binding['source_sha256'], outcome, binding) is None
    outcome.result.artifacts = {'output_identity': canonical(witness)}
    task.output_identity = None
    assert adapt_output(task, binding['source_sha256'], outcome, binding) is None


@pytest.mark.parametrize('damage', ['source', 'task', 'evaluator', 'stage', 'failed', 'partial',
    'noncanonical', 'duplicate_json', 'nan', 'input', 'seed', 'bool_confirmation', 'view', 'big'])
def test_invalid_or_unscoped_artifact_is_permissive(tmp_path, damage):
    task, witness, outcome, binding = fixture(tmp_path)
    if damage == 'source': witness['source_sha256'] = '2' * 64
    elif damage == 'task': binding['task_hash'] = '3' * 64
    elif damage == 'evaluator': witness['evaluator_sha256'] = '3' * 64
    elif damage == 'stage': witness['stage'] = 'smoke'
    elif damage == 'failed': outcome.result.ok = False
    elif damage == 'partial': del witness['output']
    elif damage == 'input': witness['input_sha256'] = '3' * 64
    elif damage == 'seed': witness['seed_sha256'] = '3' * 64
    elif damage == 'bool_confirmation': outcome.confirmed = 0
    elif damage == 'view': witness['schema'] = 'sciel-full-output-view/1'
    elif damage == 'big': witness['output'] = 'x' * (1024 * 1024)
    text = canonical(witness)
    if damage == 'noncanonical': text = json.dumps(witness, indent=2)
    elif damage == 'duplicate_json': text = text.replace('"schema":', '"schema":"duplicate","schema":')
    elif damage == 'nan': text = text.replace('"output":', '"output":NaN,"unused":')
    outcome.result.artifacts = {'output_identity': text}
    assert adapt_output(task, binding['source_sha256'], outcome, binding) is None


def test_same_score_different_full_output_and_representation_distinguish(tmp_path):
    first, _ = identity(tmp_path, output={'x': [0.0, 1.0]})
    second, _ = identity(tmp_path, output={'x': [0.0, 2.0]})
    signed, _ = identity(tmp_path, output={'x': [-0.0, 1.0]})
    integer, _ = identity(tmp_path, output={'x': [0, 1]})
    assert len({v['output_sha256'] for v in (first, second, signed, integer)}) == 4


def test_selected_confirmation_only_not_all_trials_or_paired_seed_claim(tmp_path):
    confirmation = dict(trials=3, selected_trial=0, selection='lowest_score')
    a, _ = identity(tmp_path, reached='confirm', confirmation=confirmation, confirmed=True)
    b, _ = identity(tmp_path, reached='confirm', confirmation=dict(confirmation, selected_trial=2), confirmed=True)
    assert a['scope'] == b['scope'] and a['output_sha256'] == b['output_sha256']
    assert a['evidence'] != b['evidence']
    assert a['scope']['stage']['selected_output_only'] is True
    c, _ = identity(tmp_path, reached='confirm', confirmation=dict(confirmation, trials=4), confirmed=True)
    assert a['scope'] != c['scope']


def test_variable_inputs_and_environment_seeds_require_explicit_hashes(tmp_path):
    for kwargs in ({'input_scope': 'artifact_sha256'}, {'seed_scope': 'input_seed'}):
        task, _, outcome, binding = fixture(tmp_path, **kwargs)
        assert adapt_output(task, binding['source_sha256'], outcome, binding) is None
    a, _ = identity(tmp_path, input_scope='artifact_sha256', input_sha='2' * 64,
                    seed_scope='input_seed', seed_sha='3' * 64)
    b, _ = identity(tmp_path, input_scope='artifact_sha256', input_sha='4' * 64,
                    seed_scope='input_seed', seed_sha='3' * 64)
    assert a['scope'] != b['scope']


@pytest.mark.parametrize('damage', ['recorded_count', 'requested_count', 'not_sampled', 'not_confirmed', 'bool_count'])
def test_confirmation_identity_bound_to_actual_cascade_protocol(tmp_path, damage):
    task, _, outcome, binding = fixture(tmp_path, reached='confirm', confirmed=True,
        confirmation=dict(trials=3, selected_trial=1, selection='lowest_score'))
    if damage == 'recorded_count': outcome.result.cases['n_trials'] = 4.0
    elif damage == 'requested_count': task.confirm_n = 4
    elif damage == 'not_sampled': outcome.result.confirmed_sampled = False
    elif damage == 'not_confirmed': outcome.confirmed = False
    elif damage == 'bool_count': outcome.result.cases['n_trials'] = True
    assert adapt_output(task, binding['source_sha256'], outcome, binding) is None


def test_native_complete_geometry_adapter_and_penalty_not_identity():
    from sciel.eval.registry import load_task
    from test_native_geometry_contract import fixture as geometry, observation
    task = load_task(str(Path(__file__).resolve().parents[2] / 'tasks/oe/circle_packing'))
    witness = geometry()
    raw = observation(witness)
    binding = dict(task_hash=task.task_hash, source_sha256=witness['source_sha256'], island_id=2)
    outcome = CascadeOutcome(EvalResult(ok=True, score=raw['score'], cases=raw['cases'],
        artifacts={'native_geometry': canonical(witness)}), raw_first_score=.2, reached='full')
    value = adapt_output(task, witness['source_sha256'], outcome, binding)
    assert value is not None and value['output_sha256'] == witness['numeric_sha256']
    assert value['scope']['adapter'] == 'native-circle-geometry/1'
    # Full raw invalid evidence is preserved by the evaluator but never gates.
    witness = geometry(numeric={'centers': [[-.1, .5]] * 26, 'radii': [.01] * 26}, validity=False)
    raw = observation(witness)
    outcome.result = EvalResult(ok=True, score=raw['score'], cases=raw['cases'],
                                artifacts={'native_geometry': canonical(witness)})
    assert adapt_output(task, witness['source_sha256'], outcome, binding) is None


@pytest.mark.parametrize('config', [{}, {'schema': CONFIG_SCHEMA}, {'adapter': 'score'},
    {'schema': CONFIG_SCHEMA, 'adapter': 'canonical-json/1', 'artifact': '../escape',
     'input_scope': 'task_fixed', 'seed_scope': 'unpaired'}])
def test_descriptor_rejects_untrusted_weak_or_ambiguous_configuration(config):
    with pytest.raises(ValueError): validate_output_identity_config(config)
