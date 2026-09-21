"""Actual orchestration with fake returned observations; no candidate/evaluator/API execution."""
import copy
import json
from pathlib import Path
import sys

import pytest

from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base.presentation import expand_reflection_payload
import yaml

from sciel.campaign.budget import CampaignStopped
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.engine.research import digest
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from sciel.hypothesis import CONTRACT
from sciel.hypothesis.measurement import evaluation_observation, metric_context


ROOT=Path(__file__).resolve().parents[2]
SEED='def run_packing():\n    return 0.1\n'
BASE_CASES={'sum_radii':.527, 'validity':1., 'combined_score':.2, 'signed_zero':-0.0, 'n_trials':1}


def rows(folder):
    path=folder/'sciel_events.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def decode(prompt,marker):
    value = decode_evidence(json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())[0])
    return expand_reflection_payload(value) if value.get("schema") == "sciel-reflection-presentation/2" else value


def observation(kind):
    if kind=='raised': return None
    if kind=='failure':
        result=EvalResult(False,-.25,{'error_code':-9,'combined_score':-777.},
                          artifacts={'error_type':'TimeoutError','error':'synthetic timeout'},timing=.125)
        return CascadeOutcome(result,-.375,'smoke',False)
    score=0. if kind=='invalid' else .887793997246478
    cases={'sum_radii':0. if kind=='invalid' else 2.3393371827444693,
           'validity':0. if kind=='invalid' else 1., 'combined_score':123.25,
           'n_trials':3, 'signed_zero':-0.0}
    # Contradictory diagnostic combined_score is intentional: explicit score wins.
    result=EvalResult(True,score,cases,timing=.125,artifacts={'fixture':'not physical geometry'})
    confirmed=kind=='confirm'
    return CascadeOutcome(result,.9012345678912345 if confirmed else score,
                          'confirm' if confirmed else 'full',confirmed)


@pytest.fixture
def task(tmp_path):
    folder=tmp_path/'task'; folder.mkdir()
    definition=yaml.safe_load((ROOT/'tasks/oe/circle_packing/task.yaml').read_text())
    definition.update(suite='test',name='measurement_fixture')
    (folder/'task.yaml').write_text(yaml.safe_dump(definition))
    (folder/'initial_program.py').write_text(SEED)
    (folder/'evaluator.py').write_text('raise AssertionError("evaluator execution forbidden")\n')
    return load_task(str(folder))


class FakeModel:
    models=['fake']; weights=[1.0]; max_retries=1
    def __init__(self,task,folder,*,duplicate=False):
        self.descriptor=metric_context(task); self.folder=folder; self.duplicate=duplicate
        self.calls=[]; self.contexts=[]; self.card_contexts=[]; self.conclusions=[]; self.engine=None; self.counter=0
    def choose_model(self,rng): return 'fake'
    def generate(self,prompt,model,temperature=None):
        if prompt.startswith('IDEA STAGE 1:'):
            context=decode(prompt,'RESEARCH_CONTEXT_JSON:')
            event=rows(self.folder)[-1]
            assert event['phase']=='idea' and event['action']=='request_start' and event['stage']=='card'
            assert event['prompt_sha256']==digest(prompt)
            parent=self.engine.db.get(event['parent_id'])
            assert context=={'metric':self.descriptor,'parent_score':parent.score,'parent_cases':parent.cases}
            assert 'No code fences or forecast are requested here.' in prompt
            self.card_contexts.append(copy.deepcopy(context))
            text='IDEA_CARD_JSON: '+json.dumps({'family':'fixture','strategy':'synthetic geometry',
                                              'mechanism':'return inert fixture code'})
            role='card'
        elif 'Review this one experiment.' in prompt:
            record=decode(prompt,'EVIDENCE_JSON:'); self.conclusions.append(record)
            text='CONCLUSION_JSON: '+json.dumps({'verdict':record['forecast_verdict'],
                'explanation':'Recorded fixture observation only.','lessons':[]})
            role='conclusion'
        elif 'Review existing experimental evidence and lessons.' in prompt:
            text='REFLECTION_JSON: {"lessons":[],"retire_ids":[]}'
            role='reflection'
        else:
            context=decode(prompt,'RESEARCH_CONTEXT_JSON:')
            assert context['metric']==self.descriptor
            # This assertion runs before the fake response/charge, against durable begin().
            attempt=next(r for r in reversed(rows(self.folder)) if r.get('phase')=='attempt')
            assert attempt['contract']==CONTRACT
            assert attempt['evidence']['metric']==context['metric']
            assert attempt['evidence']['evaluation'] is None
            assert attempt['evidence']['parent_cases']==context['parent_cases']
            parent=self.engine.db.get(attempt['evidence']['parent_id'])
            assert context['parent_cases']==parent.cases
            assert context['parent_score']==parent.score
            assert context['parent_source_sha256']==digest(parent.code)
            self.contexts.append(copy.deepcopy(context)); self.counter+=1
            code=SEED if self.duplicate else f'def run_packing():\n    return {self.counter+1}.0\n'
            text='HYPOTHESIS_JSON: '+json.dumps({'hypothesis':'Synthetic score change.',
                'predicted_low':-1.,'predicted_high':1.,'reasoning':'No geometry is executed.',
                'confidence':.5})+'\n```python\n'+code+'```'
            role='code'
        self.calls.append({'role':role,'prompt':prompt,'text':text,'cost':.01})
        return LLMResponse(text,model,.01,25,15)


def build(task,folder,monkeypatch,kind,*,two_stage=False,portfolio=False,budget=1):
    import sciel.engine.loop as loop_module
    import sciel.eval.sandbox as sandbox_module
    cfg={'task':task.qualified_name,'budget':budget,'max_iterations':budget,'dup_cap':1,
        'num_islands':1,'random_seed':51,'schedule_policy':'round_robin','milestone_fracs':[],
        'op_fixed':'idea' if two_stage else 'full_rewrite','policy':{'enabled':False},
        'portfolio_seed':portfolio,'dynamic_islands':False,'idea_island':two_stage,
        'idea_two_stage':{'enabled':two_stage},'idea_card_selection':{'enabled':False},
        'embed_grid':False,'embed_niche':False,'three_head':False,'crossover':False,
        'confirm_records':False,'llm':{'mock':True},
        'research':{'enabled':True,'reflection_interval':2,'max_aux_calls':10}}
    model=FakeModel(task,folder,duplicate=kind=='duplicate'); returned=observation(kind)
    calls=[]
    def no_subprocess(*args,**kwargs): pytest.fail('real evaluator/candidate execution forbidden')
    monkeypatch.setattr(sandbox_module,'run_stage',no_subprocess)
    def fake_cascade(actual_task,path,**kwargs):
        assert actual_task is task
        code=Path(path).read_text(); calls.append(code)
        if code==SEED:
            assert len(calls)==1
            return CascadeOutcome(EvalResult(True,.2,copy.deepcopy(BASE_CASES),timing=.01),.2,'full',False)
        proposals=[r for r in rows(folder) if r.get('phase')=='hypothesis']
        assert proposals and proposals[-1]['evidence']['source_sha256']==digest(code)
        assert proposals[-1]['evidence']['evaluation'] is None
        if kind=='raised': raise RuntimeError('synthetic evaluator failure')
        return copy.deepcopy(returned)
    monkeypatch.setattr(loop_module,'cascade_evaluate',fake_cascade)
    engine=Engine(cfg,task,str(folder),model,python=sys.executable); model.engine=engine
    return engine,model,returned,calls


@pytest.mark.parametrize('two_stage',[False,True])
@pytest.mark.parametrize('kind',['valid','confirm','invalid','failure','raised','duplicate'])
def test_worker_and_two_stage_preserve_exact_metric_observation(task,tmp_path,monkeypatch,kind,two_stage):
    folder=tmp_path/'run'; engine,model,returned,eval_calls=build(task,folder,monkeypatch,kind,two_stage=two_stage)
    if kind=='raised':
        with pytest.raises(CampaignStopped): engine.run()
    else: engine.run()
    record,=engine.research.knowledge.experiments
    assert record['metric']==metric_context(task)
    assert record['parent_cases']==BASE_CASES
    assert json.dumps(record['parent_cases']['signed_zero'])=='-0.0'
    assert len(model.contexts)==1 and model.contexts[0]['parent_cases']==BASE_CASES
    assert len(model.card_contexts)==int(two_stage)
    if two_stage:
        assert model.card_contexts[0]=={key:model.contexts[0][key] for key in ('metric','parent_score','parent_cases')}
    expected=None if kind in {'raised','duplicate'} else evaluation_observation(returned)
    assert record['evaluation']==expected
    if kind in {'raised','failure','duplicate'}:
        assert record['score'] is None and record['actual_delta'] is None
        assert record['status']==('no_child' if kind=='duplicate' else 'eval_failed')
    else:
        assert record['status']=='evaluated'
        assert record['score']==returned.result.score
        assert record['actual_delta']==returned.result.score-.2
        assert record['score']!=record['evaluation']['cases']['combined_score']
        if kind=='invalid':
            assert record['score']==record['evaluation']['cases']['sum_radii']==0.
            assert record['metric']['semantics']['invalid_numerator_semantics']=='penalty_sentinel_not_physical_measurement'
    journal=rows(folder)
    outcome,=[r for r in journal if r.get('phase')=='outcome']
    assert outcome['evidence']['evaluation']==expected
    evaluations=[r for r in journal if r.get('policy')=='evaluation']
    assert len(evaluations)==(1 if kind in {'raised','duplicate'} else 2)
    if expected:
        assert evaluations[-1]['source_sha256']==record['source_sha256']
        assert {k:evaluations[-1][k] for k in expected}==expected
        assert model.conclusions[0]['evaluation']==expected
        assert model.conclusions[0]['metric']==record['metric']
        assert model.conclusions[0]['parent_cases']==BASE_CASES
    else: assert model.conclusions==[]
    assert len(eval_calls)==(1 if kind=='duplicate' else 2)
    saved=json.loads((folder/'checkpoint.json').read_text())
    assert saved['research']['metric']==record['metric']
    assert saved['research']['knowledge']['experiments'][0]['evaluation']==expected
    assert engine.cost_total==pytest.approx(sum(c['cost'] for c in model.calls))


def test_portfolio_uses_same_metric_path_and_new_parent_cases_are_measured(task,tmp_path,monkeypatch):
    folder=tmp_path/'portfolio'
    engine,model,returned,eval_calls=build(task,folder,monkeypatch,'valid',portfolio=True)
    engine.run()
    records=engine.research.knowledge.experiments
    assert len(records)==2 and len(eval_calls)==3  # seed, portfolio, ordinary child
    expected=evaluation_observation(returned)
    assert all(r['evaluation']==expected and r['metric']==metric_context(task) for r in records)
    assert records[0]['parent_cases']==BASE_CASES
    assert len(model.conclusions)==2 and engine.research.reflections==1
    assert len({r['id'] for r in records})==2 and engine.B_evals==1
    for context,record in zip(model.contexts,records):
        parent=engine.db.get(record['parent_id'])
        assert context['parent_cases']==record['parent_cases']==parent.cases
    assert engine.cost_total==pytest.approx(sum(c['cost'] for c in model.calls))
