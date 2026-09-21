"""Real Engine/WAL admission with inert outputs and fake transport/evaluator leaves."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint
from sciel.engine.llm.client import COST_PROVENANCE_SCHEMA, SUBSCRIPTION_ENDPOINT_SHA256
from sciel.engine.loop import Engine
from sciel.eval.output_identity import sealed
from test_output_novelty_engine import setup
from test_parallel_engine import events, number


def _unpriced(provider):
    generate=provider.generate
    def returned(*args,**kwargs):
        response=generate(*args,**kwargs)
        return replace(response,cost=None,finish_reason='stop',cost_provenance={
            'schema':COST_PROVENANCE_SCHEMA,'source':'subscription_unpriced',
            'provider':'codex_responses','response_status':'completed',
            'endpoint_sha256':SUBSCRIPTION_ENDPOINT_SHA256})
    provider.generate=returned


def _history(engine):
    return engine.output_novelty.state_dict()['records']


def _receipts(directory):
    return {str(path.relative_to(directory)):hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in (directory/'.parallel',directory/'research_evidence')
            for path in folder.rglob('*.json') if path.name=='receipt.json'
            or path.parent.name=='evaluations'}


@pytest.mark.parametrize('parallel',[False,True])
def test_source_and_near_output_rejections_share_generation_cap(tmp_path,monkeypatch,parallel):
    build,provider,evaluated,directory=setup(tmp_path,monkeypatch,[0,1,2],
        {0:[0.0],1:[0.1],2:[2.0]},resolution=1.0,parallel=parallel)
    engine=build();result=engine.run()
    assert evaluated==[0,1,2] and len(provider.calls)==3
    assert result['B_evals']==1 and result['n_output_repeats']==1
    assert result['cost_total']==pytest.approx(.09)
    assert [number(p.code) for p in engine.db._all.values()]==[0,2]
    rejected=next(row for row in _history(engine)
                  if row['verdict']['status']=='non_improving_near_repeat')
    assert rejected['verdict']['match']['program_id']=='p1'
    assert rejected['evaluation_id'] in provider.calls[-1]
    assert 'non_improving_near_repeat' in provider.calls[-1]
    assert rejected['verdict']['near_output']['match']['score']==.5


def test_rejected_history_blocks_later_near_copy_outside_original_radius(tmp_path,monkeypatch):
    build,provider,evaluated,directory=setup(tmp_path,monkeypatch,[1,2,3],
        {0:[0.0],1:[.75],2:[1.5],3:[2.25]},resolution=1.0,scores={1:.4,2:.4,3:.4})
    engine=build();result=engine.run()
    assert evaluated==[0,1,2,3] and len(provider.calls)==3
    assert result['B_evals']==0 and result['n_output_repeats']==3
    assert result['cost_total']==pytest.approx(.09)
    assert [number(p.code) for p in engine.db._all.values()]==[0]
    history=_history(engine)
    for index,row in enumerate(history[1:],1):
        assert row['verdict']['status']=='non_improving_near_repeat'
        assert row['verdict']['match']['evaluation_id']==history[index-1]['evaluation_id']
        assert row['program_ids']==[]
    iteration=next(row for row in events(directory) if row['event']=='iteration')
    assert iteration['reason']=='output_repeat_cap'
    assert not iteration.get('is_pop_event') and not iteration.get('is_island_event')
    restored=build(resume=True)
    try:assert restored.output_novelty.state_dict()==engine.output_novelty.state_dict()
    finally:restored.close()


def test_source_exact_output_and_near_output_use_one_shared_two_resample_budget(tmp_path,monkeypatch):
    build,provider,evaluated,directory=setup(tmp_path,monkeypatch,[0,1,2],
        {0:[0.0],1:[0.0],2:[.1]},resolution=1.0)
    engine=build();result=engine.run()
    assert len(provider.calls)==3 and evaluated==[0,1,2]
    assert result['B_evals']==0 and result['n_output_repeats']==2
    assert [row['verdict']['status'] for row in _history(engine)]==[
        'new_output','exact_recorded_repeat','non_improving_near_repeat']
    assert len(engine.db._all)==1 and result['cost_total']==pytest.approx(.09)


def test_one_ulp_measured_improvement_is_admitted_inside_resolution(tmp_path,monkeypatch):
    improved=math.nextafter(.5,math.inf)
    build,provider,evaluated,directory=setup(tmp_path,monkeypatch,[1],
        {0:[0.0],1:[.1]},resolution=1.0,scores={1:improved})
    engine=build();result=engine.run()
    assert evaluated==[0,1] and len(provider.calls)==1
    assert result['B_evals']==1 and result['n_output_repeats']==0
    assert max(p.score for p in engine.db._all.values())==improved
    row=_history(engine)[-1]
    assert row['verdict']['status']=='new_output' and row['verdict']['accept'] is True
    assert row['near_measurement']['score']==improved


def test_parallel_siblings_compare_near_outputs_across_islands(tmp_path,monkeypatch):
    build,provider,evaluated,directory=setup(tmp_path,monkeypatch,[1,2,3],
        {0:[0.0],1:[10.0],2:[10.1],3:[20.0]},resolution=1.0,
        workers=2,iterations=2,islands=2)
    engine=build();result=engine.run()
    assert len(provider.calls)==3 and len(evaluated)==4
    assert result['B_evals']==2 and result['n_output_repeats']==1
    rejected=next(row for row in _history(engine)
                  if row['verdict']['status']=='non_improving_near_repeat')
    assert rejected['verdict']['match']['island_id']!=rejected['binding']['island_id']
    assert rejected['verdict']['match']['program_id'] is None


@pytest.mark.parametrize('pause',['stop_after_near_receipt','crash_before_commit'])
def test_paid_near_rejection_cold_resume_never_repeats_call_or_unpriced_charge(tmp_path,monkeypatch,pause):
    build,provider,evaluated,directory=setup(tmp_path,monkeypatch,[1,2],
        {0:[0.0],1:[.1],2:[2.0]},resolution=1.0)
    _unpriced(provider)
    first=True
    if pause=='stop_after_near_receipt':
        original=Engine._evaluate
        def paused(self,*args,**kwargs):
            nonlocal first
            outcome=original(self,*args,**kwargs)
            if first and outcome.output_novelty['status']=='non_improving_near_repeat':
                first=False;(directory/'STOP').touch()
            return outcome
        monkeypatch.setattr(Engine,'_evaluate',paused)
        result=build().run()
        assert result['reason']=='paused'
    else:
        original=Engine._commit_step
        def crash(self,*args,**kwargs):
            nonlocal first
            if first:first=False;raise RuntimeError('inert near admission crash')
            return original(self,*args,**kwargs)
        monkeypatch.setattr(Engine,'_commit_step',crash)
        with pytest.raises(CampaignStopped,match='admission crash'):build().run()
    frozen=_receipts(directory)
    engine=build(resume=True);result=engine.run()
    assert len(provider.calls)==2 and evaluated==[0,1,2]
    assert result['cost_total']==0 and result['llm_unpriced_calls']==2
    assert result['B_evals']==1 and result['n_output_repeats']==1
    assert all(_receipts(directory).get(path)==digest for path,digest in frozen.items())
    physical=[json.loads(path.read_text())['payload']['value']
              for path in (directory/'.parallel').rglob('receipt.json')]
    llm=[value for value in physical if type(value) is dict and 'text' in value]
    assert len(llm)==2 and all(value['cost'] is None for value in llm)
    restored=build(resume=True)
    try:assert restored.output_novelty.state_dict()==engine.output_novelty.state_dict()
    finally:restored.close()


@pytest.mark.parametrize('damage',['score','erase_measurement','missing_receipt'])
def test_cold_resume_reauthenticates_receipt_not_resealed_checkpoint(tmp_path,monkeypatch,damage):
    build,provider,evaluated,directory=setup(tmp_path,monkeypatch,[1],
        {0:[0.0],1:[2.0]},resolution=1.0)
    build().run()
    path=directory/'checkpoint.json';value=json.loads(path.read_text())
    state=value['output_novelty'];row=state['records'][-1]
    if damage=='score':
        row['near_measurement']['score']=.7
        row['near_measurement']=sealed({key:item for key,item in row['near_measurement'].items() if key!='sha256'})
        row['verdict']['near_output'].update(score=.7,measurement_sha256=row['near_measurement']['sha256'])
        row['verdict']=sealed({key:item for key,item in row['verdict'].items() if key!='sha256'})
    elif damage=='erase_measurement':
        row['near_measurement']=None;row['verdict']['near_output']=None
        row['verdict']=sealed({key:item for key,item in row['verdict'].items() if key!='sha256'})
    else:
        (directory/'research_evidence/evaluations'/(row['binding']['receipt_sha256']+'.json')).unlink()
    value['output_novelty']=sealed({key:item for key,item in state.items() if key!='sha256'})
    corrupt=directory/'tampered-checkpoint.json';corrupt.write_text(json.dumps(value))
    engine=build();before=deepcopy(engine.output_novelty.state_dict())
    try:
        with pytest.raises(checkpoint.IdentityMismatch,match='output novelty'):
            checkpoint.load(engine,str(corrupt),allow_identity_mismatch=True)
        assert engine.output_novelty.state_dict()==before
        assert len(provider.calls)==1 and evaluated==[0,1]
    finally:engine.close()


def test_measured_near_rejections_keep_records_without_c_lessons_fragments_or_credit(tmp_path,monkeypatch):
    build,provider,evaluated,directory=setup(tmp_path,monkeypatch,[1,2,3],
        {0:[0.0],1:[.1],2:[.2],3:[.3]},resolution=1.0,
        research={'enabled':True,'max_aux_calls':None,'reflection_interval':0})
    original=provider.generate
    def with_hypothesis(*args,**kwargs):
        response=original(*args,**kwargs)
        return replace(response,text='HYPOTHESIS_JSON: '+json.dumps({
            'hypothesis':'Test a small output change.','predicted_low':0.,'predicted_high':.1,
            'reasoning':'An inert test forecast.','confidence':.5})+'\n'+response.text)
    provider.generate=with_hypothesis
    engine=build();result=engine.run()
    assert evaluated==[0,1,2,3] and len(provider.calls)==3
    assert result['B_evals']==0 and result['n_output_repeats']==3
    research=engine.research
    assert research.queue==[] and research.knowledge.lessons==[] and research.knowledge.fragments==[]
    assert len(research.knowledge.experiments)==3
    for row in research.knowledge.experiments:
        assert row['status']=='evaluated' and row['evaluation']['ok'] is True
        assert row['output_novelty']['status']=='non_improving_near_repeat'
        assert row['code_fragment']==''
        feedback=research.knowledge.failed_hypotheses(row['branch'])
        linked=next(text for text in feedback if 'id='+row['id']+';' in text)
        assert 'Measured non-improving near-output repeat' in linked
        assert 'Measured exact recorded-output repeat' not in linked
        assert 'not admitted and not efficacy evidence' in linked
        assert '"status":"non_improving_near_repeat"' in linked
    assert result['cost_total']==pytest.approx(.09)
    iteration=next(row for row in events(directory) if row['event']=='iteration')
    assert not iteration.get('is_pop_event') and not iteration.get('is_island_event')
    restored=build(resume=True)
    try:assert restored.research.state()==research.state()
    finally:restored.close()


def test_near_copy_positive_vs_weak_parent_still_has_no_efficacy_credit(tmp_path,monkeypatch):
    build,provider,evaluated,directory=setup(tmp_path,monkeypatch,[1,2,3],
        {0:[0.0],1:[10.0],2:[10.1],3:[20.0]},resolution=1.0,
        scores={1:.8,2:.7,3:.6},workers=2,iterations=2,islands=2,
        research={'enabled':True,'max_aux_calls':0,'reflection_interval':0})
    original=provider.generate
    def with_hypothesis(*args,**kwargs):
        response=original(*args,**kwargs)
        return replace(response,text='HYPOTHESIS_JSON: '+json.dumps({
            'hypothesis':'Test a changed output.','predicted_low':0.,'predicted_high':.3,
            'reasoning':'An inert forecast.','confidence':.5})+'\n'+response.text)
    provider.generate=with_hypothesis
    engine=build();result=engine.run()
    assert len(provider.calls)==3 and len(evaluated)==4
    assert result['B_evals']==2 and result['n_output_repeats']==1
    rejected=[row for row in engine.research.knowledge.experiments
              if row.get('output_novelty',{}).get('status')=='non_improving_near_repeat']
    assert len(rejected)==1
    row=rejected[0]
    assert row['score']==.7 and row['actual_delta']>0 and row['parent_score']==.5
    assert row['code_fragment']==''
    assert row['id'] not in [item['id'] for item in engine.research.queue]
    assert all(fragment['experiment_id']!=row['id'] for fragment in engine.research.knowledge.fragments)
    assert 2 not in [number(program.code) for program in engine.db._all.values()]


def test_research_submission_reports_bound_near_refusal_without_generator(tmp_path):
    from sciel.engine.output_novelty import OutputNoveltyRegistry
    from sciel.engine.research_candidate import evaluate_submission
    from sciel.engine.types import NoChild
    from sciel.engine.validation import source_sha256
    from sciel.eval.output_identity import adapt_output,adapt_near_output,digest
    from test_output_identity import fixture
    from test_near_output import policy
    from test_research_candidate_submission import SOURCE,provenance,snapshot
    task,_,prior,binding=fixture(tmp_path,source='e'*64,output=[0.0])
    resolution=policy(task,1.0)
    registry=OutputNoveltyRegistry({'task_hash':task.task_hash},near_policy=resolution)
    def observe(ident,outcome,bind):
        bind['receipt_sha256']=digest({'evaluation_id':ident,'result':outcome.result.to_json()})
        identity=adapt_output(task,bind['source_sha256'],outcome,bind)
        measurement=adapt_near_output(task,bind['source_sha256'],outcome,bind,
                                      identity=identity,policy=resolution)
        return registry.observe(ident,identity,bind,measurement=measurement)
    observe('prior',prior,binding);registry.bind_program('prior','p-archived')
    calls=[]
    def evaluate(code):
        calls.append(code)
        _,_,outcome,bind=fixture(tmp_path,source=source_sha256(code),output=[.1])
        outcome.evaluation_id='tool-canonical'
        outcome.output_novelty=observe(outcome.evaluation_id,outcome,bind)
        return outcome
    result,feedback=evaluate_submission(SOURCE,provenance(),snapshot(),evaluate)
    assert calls==[SOURCE] and isinstance(result,NoChild) and result.cost==0
    assert feedback['status']=='output_repeat_rejected' and feedback['efficacy']=='unknown'
    assert result.meta['evaluation_count']==result.meta['output_repeat_rejections']==1
    verdict=result.meta['evaluations'][0]['output_novelty']
    assert verdict['status']=='non_improving_near_repeat'
    assert verdict['match']['program_id']=='p-archived'
    assert verdict['binding']['source_sha256']==source_sha256(SOURCE)
