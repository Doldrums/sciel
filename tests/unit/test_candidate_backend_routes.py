"""Real canonical backend/WAL; physical Docker and model/tool leaves inert."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
from unittest.mock import Mock

import pytest
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.research_artifacts import saved_evaluation_for_program
from test_candidate_backend import setup_backend, ROOT
from test_parallel_engine import Provider,events,source


def poison(monkeypatch):
    from sciel.eval import cascade,sandbox,registry
    from sciel.engine import loop,factory
    sentinel=Mock(side_effect=AssertionError('forbidden native candidate path'))
    for module,name in [(loop,'cascade_evaluate'),(cascade,'run_stage'),(sandbox,'run_stage'),
                        (registry,'determinism_check'),(factory,'build_llm')]:
        monkeypatch.setattr(module,name,sentinel)
    return sentinel


def evidence(run):
    rows=[]
    for p in (run/'.parallel').rglob('intent.json'):
        v=json.loads(p.read_text())
        if v.get('kind')=='evaluation':rows.append((p,v,json.loads((p.parent/'receipt.json').read_text())))
    return rows


@pytest.mark.parametrize('portfolio',[False,True])
@pytest.mark.parametrize('warm',[False,True])
def test_seed_portfolio_ordinary_and_cold_replay_use_candidate_backend_only(tmp_path,monkeypatch,portfolio,warm):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    cfg.update(portfolio_seed=portfolio,max_iterations=2)
    initial=source('run_packing',99)
    if warm:
        cfg['warm_start']={'schema':'sciel-warm-start/1','source':initial,
            'source_sha256':hashlib.sha256(initial.encode()).hexdigest(),
            'parent':{'run_id':'inert-prior','checkpoint_sha256':'a'*64,'journal_sha256':'b'*64,
                      'task':task.qualified_name,'task_hash':task.task_hash,'program_id':'p99',
                      'previous_score':999,'transfer':'source_only_fresh_evaluation'}}
    sentinel=poison(monkeypatch);provider=Provider('run_packing');run=tmp_path/'run'
    engine=build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider)
    engine.run()
    expected=3+2*int(portfolio)
    assert len(executor.calls)==4*expected
    assert engine.n_evaluation_results==expected
    assert engine.db.get('p1').code==(initial if warm else Path(task.initial_program_path).read_text())
    assert engine.db.get('p1').score!=999
    rows=evidence(run)
    assert len(rows)==expected
    assert all(v['request']['evaluation_backend']==engine.evaluation_backend.identity for _,v,_ in rows)
    before=deepcopy(executor.calls),deepcopy(provider.calls)
    restored=build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider)
    checkpoint.load(restored,str(run/'checkpoint.json'))
    assert restored.run()['iteration_no']==2
    assert (executor.calls,provider.calls)==before
    sentinel.assert_not_called()


@pytest.mark.parametrize('independent',[False,True])
def test_real_tool_submission_uses_same_canonical_backend_and_saved_execution(tmp_path,monkeypatch,independent):
    from test_independent_research_engine import designed
    from test_research_tool_result_flow import SavedExecutor,result
    from test_research_yield_engine_integration import YieldProvider
    from test_research_tool_controller import IMAGE
    from test_sequential_research_episode import decision,finish,current_receipt
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    candidate=Path(task.initial_program_path).read_text()+'\nFIXTURE_ROUTE = 19\n'
    def submit(prompt):
        return designed(decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt)),
                        kind='closure',budget_action='none')
    actions=[designed(decision()),submit,
             designed(finish(conclusion='inconclusive'),kind='closure',budget_action='none')]
    class Planner(YieldProvider):
        def generate(self,*args,**kwargs):
            reply=super().generate(*args,**kwargs)
            # Only inert generated fixture source changes entrypoint.
            if '```python' in reply.text:reply.text=reply.text.replace('def run_search():','def run_packing():')
            return reply
    provider=Planner(actions);tools=SavedExecutor([result({'candidate_source':candidate,'gain':0})])
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor',lambda image:tools)
    cfg.update(max_iterations=2 if independent else 1,research={'enabled':False},
        research_tools={'enabled':True,'image_id':IMAGE,'interval':1000,'proactive_every_attempts':1000,
            'max_scientific_steps':3,'yield_after_success':independent,'independent_research':independent,
            'research_jobs_per_wave':1,'shared_observations':True,'experimental_protocol':True,
            'max_requests':3,'share_between_islands':False})
    sentinel=poison(monkeypatch);run=tmp_path/'run'
    engine=build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider);engine.run()
    children=[p for p in engine.db._all.values() if p.origin=='research_tool']
    assert len(children)==1 and children[0].code==candidate
    rows=evidence(run)
    submitted=[(p,v,r) for p,v,r in rows if v['request']['source']==candidate]
    assert len(submitted)==1
    p,v,r=submitted[0]
    assert v['request']['evaluation_backend']==engine.evaluation_backend.identity
    assert ('research:' in v['key']) is independent
    saved=saved_evaluation_for_program(engine,children[0])
    assert saved['status']=='available'
    assert engine.n_evaluation_results==len(rows) and len(executor.calls)==4*len(rows)
    assert len(tools.calls)==1
    prior=deepcopy(executor.calls),deepcopy(provider.calls),deepcopy(provider.planner_prompts),deepcopy(tools.calls)
    restored=build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider)
    checkpoint.load(restored,str(run/'checkpoint.json'));restored.run()
    assert (executor.calls,provider.calls,provider.planner_prompts,tools.calls)==prior
    sentinel.assert_not_called()


def test_warm_dynamic_nochild_fallback_retains_safe_measurement_without_native_bypass(tmp_path,monkeypatch):
    from sciel.engine.llm.client import LLMResponse
    from sciel.engine.research_artifacts import evaluation_id_for_program
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    initial=Path(task.initial_program_path).read_text()+'\nWARM_FIXTURE = 77\n'
    cfg.update(dynamic_islands=True,max_islands=3,spawn_stall=1,max_iterations=2,
        warm_start={'schema':'sciel-warm-start/1','source':initial,
            'source_sha256':hashlib.sha256(initial.encode()).hexdigest(),
            'parent':{'run_id':'inert-prior','checkpoint_sha256':'a'*64,'journal_sha256':'b'*64,
                      'task':task.qualified_name,'task_hash':task.task_hash,'program_id':'p99',
                      'previous_score':999,'transfer':'source_only_fresh_evaluation'}})
    class RepeatsForSpawn(Provider):
        def generate(self,prompt,model,temperature=None,**options):
            if 'Bootstrap spawn:' in prompt:
                self.calls.append(prompt)
                return LLMResponse('```python\n'+initial+'```',model,.03,20,20)
            return super().generate(prompt,model,temperature,**options)
    sentinel=poison(monkeypatch);provider=RepeatsForSpawn('run_packing');run=tmp_path/'run'
    engine=build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider);engine.run()
    assert len(engine.db.islands)==3
    baseline=engine.db.get('p1')
    copies=[p for p in engine.db._all.values() if p.island==2 and p.origin=='migrant']
    assert len(copies)==1
    copied=copies[0]
    assert copied.code==baseline.code==initial and copied.parent_id==baseline.id
    assert copied.score==baseline.score and copied.cases==baseline.cases
    assert evaluation_id_for_program(engine,copied)==evaluation_id_for_program(engine,baseline)
    assert saved_evaluation_for_program(engine,copied)==saved_evaluation_for_program(engine,baseline)
    assert len(evidence(run))==engine.n_evaluation_results==3 and len(executor.calls)==12
    assert len(provider.calls)==5 and engine.cost_total==pytest.approx(.15)
    assert engine.B_evals==2
    before=deepcopy(executor.calls),deepcopy(provider.calls)
    restored=build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider)
    checkpoint.load(restored,str(run/'checkpoint.json'));restored.run()
    assert restored.db.get(copied.id).code==initial
    assert saved_evaluation_for_program(restored,restored.db.get(copied.id))==saved_evaluation_for_program(engine,copied)
    assert (executor.calls,provider.calls)==before
    sentinel.assert_not_called()
