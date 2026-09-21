"""Actual Engine, saved evaluator artifacts and checkpoint; all scientific leaves fake."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
from unittest.mock import Mock

import pytest

from sciel.engine import checkpoint
from sciel.engine.research_artifacts import evaluation_id_for_program,saved_evaluation_for_program


def fixture(tmp_path,monkeypatch,*,warm=True,novelty=True,research=False):
    import test_output_novelty_engine as base
    from test_parallel_engine import source
    initial=source('run_search',7 if warm else 0)
    warm_info={'schema':'sciel-warm-start/1','source':initial,
        'source_sha256':hashlib.sha256(initial.encode()).hexdigest(),
        'parent':{'run_id':'inert-prior-run','checkpoint_sha256':'a'*64,'journal_sha256':'b'*64,
                  'task':'fixture/run_search','task_hash':'c'*64,'program_id':'p99',
                  'previous_score':.99,'transfer':'source_only_fresh_evaluation'}}
    config=base.config
    def settings(*args,**kwargs):
        kwargs.update(dynamic_islands=True,max_islands=3,spawn_stall=1,pop_cap=1,grid_bins=1)
        kwargs['candidate_novelty']['enabled']=novelty
        kwargs['candidate_novelty']['output_repeats']=novelty
        if warm:kwargs['warm_start']=deepcopy(warm_info)
        return config(*args,**kwargs)
    monkeypatch.setattr(base,'config',settings)
    build,model,evaluated,directory=base.setup(tmp_path,monkeypatch,[1,2,7,7,7,7,7],{},
        parallel=False,iterations=10,scores={0:.4,7:.4,1:.8,2:.7},
        research={'enabled':research,'max_aux_calls':0,'reflection_interval':0})
    generate=model.generate
    def hypothesis(*args,**kwargs):
        reply=generate(*args,**kwargs)
        if research:reply.text='HYPOTHESIS_JSON: '+json.dumps({'hypothesis':'Inert source change.',
            'predicted_low':-1,'predicted_high':1,'reasoning':'Not measured yet.','confidence':.5})+'\n'+reply.text
        return reply
    model.generate=hypothesis
    engine=build()
    engine.log.run_start(seed=23,budget=None,counters=checkpoint.counters(engine),task=engine.task.qualified_name,
        task_hash=engine.task.task_hash,config=engine.cfg)
    engine.seed(portfolio=False)
    baseline=engine.db.get('p1');baseline.card='Retained baseline card.'
    engine._step(0);engine._step(0)
    assert baseline.id not in engine.db.islands[0].programs
    assert baseline.code==initial and baseline.score==.4
    assert engine.B_evals==2 and engine._last_improve_eval==1
    return engine,baseline,build,model,evaluated,directory


def trigger(engine,monkeypatch,*,real_nochild=False):
    if not real_nochild:monkeypatch.setattr(engine,'_paradigm_child',Mock(return_value=False))
    engine.reward.improvement_event=Mock(wraps=engine.reward.improvement_event)
    before=dict(B_evals=engine.B_evals,measurements=engine.n_evaluation_results,
        observations=len(engine.observed_changes.state()['records']),
        lessons=deepcopy(engine.research.knowledge.lessons) if engine.research else [],
        queued=deepcopy(engine.research.queue) if engine.research else [],
        output_records=len(engine.output_novelty.state_dict()['records']) if engine.output_novelty else None)
    engine._maybe_spawn_island()
    copied=next(iter(engine.db.islands[-1].programs.values()))
    assert engine.B_evals==before['B_evals'] and engine.n_evaluation_results==before['measurements']
    assert len(engine.observed_changes.state()['records'])==before['observations']
    if engine.output_novelty:assert len(engine.output_novelty.state_dict()['records'])==before['output_records']
    if engine.research:
        assert engine.research.knowledge.lessons==before['lessons'] and engine.research.queue==before['queued']
    engine.reward.improvement_event.assert_not_called()
    return copied


def same_evidence(engine,baseline,copied):
    assert copied.code==baseline.code and copied.score==baseline.score and copied.cases==baseline.cases
    assert copied.cases is not baseline.cases
    assert copied.card==baseline.card and copied.parent_id==baseline.id and copied.origin=='migrant'
    original_id=evaluation_id_for_program(engine,baseline)
    assert original_id is not None and evaluation_id_for_program(engine,copied)==original_id
    assert engine.research_evaluation_bindings[copied.id]==engine.research_evaluation_bindings[baseline.id]
    assert engine.evaluation_timings[copied.id]==engine.evaluation_timings[baseline.id]
    assert engine.eval_times[copied.id]==engine.eval_times[baseline.id]
    assert engine.observed_changes.record_for_program(copied.id)==engine.observed_changes.record_for_program(baseline.id)
    assert saved_evaluation_for_program(engine,copied)==saved_evaluation_for_program(engine,baseline)
    if engine.candidate_novelty:assert engine.candidate_novelty._ids[copied.id]==engine.candidate_novelty._ids[baseline.id]


@pytest.mark.parametrize('warm',[False,True])
@pytest.mark.parametrize('novelty',[False,True])
@pytest.mark.parametrize('cold',[False,True])
def test_fallback_copies_original_evicted_measured_baseline_and_cold_provenance(tmp_path,monkeypatch,warm,novelty,cold):
    engine,baseline,build,model,evaluated,directory=fixture(tmp_path,monkeypatch,warm=warm,novelty=novelty)
    if cold:
        engine._finish('fixture_before_spawn');engine.close()
        engine=build(resume=True);baseline=engine.db.get(baseline.id)
        engine.log.run_start(seed=23,budget=None,counters=checkpoint.counters(engine),
            restored_checkpoint=engine._restored_checkpoint,task=engine.task.qualified_name,task_hash=engine.task.task_hash)
    previous_calls=list(evaluated);previous_provider=list(model.calls);warm_info=deepcopy(engine.cfg.get('warm_start'))
    try:
        copied=trigger(engine,monkeypatch)
        same_evidence(engine,baseline,copied)
        assert evaluated==previous_calls and model.calls==previous_provider
        assert engine.cfg.get('warm_start')==warm_info
        engine._finish('fixture_after_spawn')
    finally:engine.close()
    restored=build(resume=True)
    try:
        same_evidence(restored,restored.db.get(baseline.id),restored.db.get(copied.id))
        assert restored.cfg.get('warm_start')==warm_info
        assert evaluated==previous_calls and model.calls==previous_provider
    finally:restored.close()


@pytest.mark.parametrize('research',[False,True])
def test_actual_nochild_spawn_uses_baseline_without_extra_evaluation_or_C(tmp_path,monkeypatch,research):
    engine,baseline,_,model,evaluated,_=fixture(tmp_path,monkeypatch,research=research)
    prior=list(evaluated);n=len(model.calls);cost=engine.cost_total
    try:
        copied=trigger(engine,monkeypatch,real_nochild=True)
        same_evidence(engine,baseline,copied)
        assert evaluated==prior
        assert len(model.calls)==n+3 and engine.cost_total==pytest.approx(cost+.09)
    finally:engine.close()


def test_later_same_source_measurement_cannot_replace_original_baseline_receipt(tmp_path,monkeypatch):
    engine,baseline,_,_,evaluated,_=fixture(tmp_path,monkeypatch)
    original=evaluation_id_for_program(engine,baseline)
    try:
        later=engine._evaluate(baseline.code,remeasurement=True)
        assert later.evaluation_id!=original
        calls=list(evaluated)
        copied=trigger(engine,monkeypatch)
        same_evidence(engine,baseline,copied)
        assert evaluation_id_for_program(engine,copied)==original and evaluated==calls
    finally:engine.close()


def test_legacy_missing_bindings_stay_unavailable_without_latest_source_guess(tmp_path,monkeypatch):
    engine,baseline,_,_,evaluated,_=fixture(tmp_path,monkeypatch,novelty=False)
    engine.research_evaluation_bindings.clear();engine.evaluation_timings.clear();engine.eval_times.clear()
    try:
        previous=list(evaluated);copied=trigger(engine,monkeypatch)
        assert copied.code==baseline.code and copied.score==baseline.score and copied.parent_id==baseline.id
        assert evaluation_id_for_program(engine,copied) is None
        assert copied.id not in engine.evaluation_timings and copied.id not in engine.eval_times
        assert saved_evaluation_for_program(engine,copied)['status']=='unavailable'
        assert evaluated==previous
    finally:engine.close()


@pytest.mark.parametrize('missing',['source','root_role'])
def test_missing_retained_baseline_does_not_invent_a_fallback_score(tmp_path,monkeypatch,missing):
    engine,baseline,_,_,evaluated,_=fixture(tmp_path,monkeypatch)
    if missing=='source':baseline.code+='\n# changed missing source\n'
    else:baseline.origin='gen'
    before=len(engine.db._all);calls=list(evaluated)
    monkeypatch.setattr(engine,'_paradigm_child',Mock(return_value=False))
    try:
        with pytest.raises(ValueError,match='retained initial baseline'):engine._maybe_spawn_island()
        assert len(engine.db._all)==before and evaluated==calls
    finally:engine.close()
