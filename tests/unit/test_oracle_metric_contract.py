"""Oracle aggregate metrics and public artifacts, with inert saved trial fixtures.

No Docker, candidate source, provider or numerical optimizer is executed here.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.eval._runner import _normalize
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.output_identity import adapt_output
from sciel.eval.registry import load_task
from sciel.hypothesis.measurement import metric_context
from test_oracle_protocol import _evaluator

TASK = Path(__file__).resolve().parents[2]/'tasks/oe/function_minimization_oracle'


def trial(value, status='ok'):
    return {'status':status, 'score':1/(1+value) if status == 'ok' else 0.,
        'queries':1, 'query_budget':50, 'best_point':[0.,0.], 'best_value':value,
        'observations':[{'query':1,'point':[0.,0.],'value':value}],
        'source_sha256':'a'*64, 'image_id':'sha256:'+'b'*64,
        'container_name':'fixture', 'elapsed_seconds':.1, 'startup_seconds':.01,
        'error':None if status == 'ok' else 'fixture failure', 'stderr':''}


def evaluate(monkeypatch, tmp_path, trials, *, stage='full'):
    evaluator = _evaluator()
    source = tmp_path/'unused_candidate.py'
    source.write_text('# inert bytes only; never imported\n')
    actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    trials = deepcopy(trials)
    for value in trials:
        value['source_sha256'] = actual_hash
    commitment = {'schema':evaluator.runtime.PROTOCOL, 'batch':'public-batch',
        'role':'development','instance_count':len(trials),'commitment_sha256':'c'*64,
        'created_at_unix':123.,'commitment_path':'/public/commitment.json'}
    calls = []
    monkeypatch.setattr(evaluator.runtime,'verify_runtime_image',lambda image: {
        'image_id':'sha256:'+'b'*64,'assets_sha256':'d'*64,'base_image':'pinned-fixture'})
    def instances(n, **kwargs):
        assert n == len(trials)
        return [{'secret_seed':'PRIVATE_SEED_DO_NOT_EXPOSE','centre':[.12,.34],
                 'index':i} for i in range(n)], deepcopy(commitment)
    monkeypatch.setattr(evaluator.runtime,'create_instances',instances)
    def saved_trial(candidate, instance, image):
        assert candidate == source.read_bytes()
        calls.append(instance['index'])
        return deepcopy(trials[instance['index']])
    monkeypatch.setattr(evaluator.runtime,'run_container_trial',saved_trial)
    result = (evaluator.confirm(source, len(trials)) if stage == 'confirm'
              else getattr(evaluator,stage)(source))
    assert calls == list(range(len(trials)))
    return result, trials, commitment


@pytest.mark.parametrize('stage,n', [('smoke',1),('full',5),('confirm',20)])
def test_exact_native_mean_preserves_existing_score_and_public_trials(monkeypatch,tmp_path,capsys,stage,n):
    result, trials, commitment = evaluate(monkeypatch,tmp_path,[trial(i+.25) for i in range(n)],stage=stage)
    mean = sum(t['score'] for t in trials)/n
    assert result['ok'] is True
    assert result['score'] == result['cases']['combined_score'] == mean
    assert result['cases']['mean_best_query_score'] == mean
    assert result['cases']['validity'] == result['cases']['reliability_score'] == 1.
    assert json.loads(result['artifacts']['trials']) == trials
    assert json.loads(result['artifacts']['instance_commitment']) == commitment
    public = json.dumps(result) + capsys.readouterr().out
    assert 'PRIVATE_SEED_DO_NOT_EXPOSE' not in public and 'centre' not in public
    assert 'full_output' not in result['artifacts'] and 'output_identity' not in result['artifacts']
    # Exercise the existing serialized evaluator contract, including numeric artifacts.
    returned = EvalResult.from_json(_normalize(result))
    assert returned.ok is True and returned.score == mean
    assert returned.cases['mean_best_query_score'] == mean


def test_full_stage_partial_failures_contribute_zero_without_discarding_success(monkeypatch,tmp_path):
    result,trials,_ = evaluate(monkeypatch,tmp_path,[trial(1),trial(3),trial(0,'timeout'),
        trial(0,'protocol_error'),trial(0,'candidate_error')])
    mean = (.5+.25)/5
    assert result['ok'] is True and result['score'] == mean
    assert result['cases']['validity'] == 1. and result['cases']['reliability_score'] == .4
    assert result['cases']['mean_best_query_score'] == mean
    assert json.loads(result['artifacts']['trials']) == trials


@pytest.mark.parametrize('stage,status', [('full','infrastructure_error'),('full','startup_timeout'),
    ('confirm','protocol_error'),('confirm','infrastructure_error')])
def test_aggregate_failures_zero_all_metric_cases_and_keep_raw_mean(monkeypatch,tmp_path,stage,status):
    trials = [trial(1),trial(3),trial(0,status),trial(1),trial(3)]
    result,saved,_ = evaluate(monkeypatch,tmp_path,trials,stage=stage)
    mean = sum(t['score'] for t in saved)/len(saved)
    assert result['ok'] is False and result['score'] == 0.
    assert result['cases']['combined_score'] == result['cases']['mean_best_query_score'] == 0.
    assert result['cases']['validity'] == 0. and result['cases']['reliability_score'] == .8
    assert result['artifacts']['raw_trial_mean'] == mean
    assert json.loads(result['artifacts']['trials']) == saved


def test_all_failed_trials_are_not_a_measured_zero(monkeypatch,tmp_path):
    result,_,_ = evaluate(monkeypatch,tmp_path,[trial(0,'timeout')]*5)
    assert result['ok'] is False and result['cases']['reliability_score'] == 0.
    assert result['score'] == result['cases']['mean_best_query_score'] == result['cases']['validity'] == 0.


@pytest.mark.parametrize('stage,confirmed', [('full',False),('confirm',True)])
def test_equal_means_never_manufacture_output_identity_for_private_batches(stage,confirmed):
    task = load_task(str(TASK))
    assert task.output_identity is None
    metric = metric_context(task)
    assert metric['semantics']['denominator'] == 1.
    assert task.primary_metric == 'combined_score' and task.known_best == 1.
    for source,commitment in [('a'*64,'c'*64),('b'*64,'d'*64)]:
        result = EvalResult(ok=True,score=.5,cases={'combined_score':.5,'mean_best_query_score':.5,
            'validity':1.,'n_trials':20. if confirmed else 5.},confirmed_sampled=confirmed,
            artifacts={'instance_commitment':json.dumps({'commitment_sha256':commitment}),
                       'trials':json.dumps([trial(1)]*(20 if confirmed else 5))})
        outcome=CascadeOutcome(result,reached=stage,confirmed=confirmed,raw_first_score=.5)
        assert adapt_output(task,source,outcome,dict(task_hash=task.task_hash,
            source_sha256=source,island_id=0)) is None
