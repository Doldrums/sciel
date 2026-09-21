"""Real Engine + ResearchTools + cooperative WAL; only leaf transports are fake.

No generated candidate or tool source executes. The provider returns fixed JSON,
the evaluator reads source as text, and the isolated backend returns saved data.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from sciel.engine import checkpoint
from sciel.campaign.budget import CampaignStopped
from sciel.engine.factory import build_engine
from sciel.engine.loop import Engine
from sciel.engine.llm.client import LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.research_tools.controller import ResearchTools
from test_parallel_engine import config, source, task_fixture
from test_research_tool_controller import IMAGE
from test_research_tool_result_flow import SavedExecutor, result
from test_sequential_research_episode import decision, finish, current_receipt


@pytest.mark.parametrize('crash_before_admission',[False,True])
def test_saved_evaluation_to_two_stage_tool_to_canonical_population_and_resume(tmp_path, monkeypatch, crash_before_admission):
    tasks = tmp_path/'tasks'
    task = task_fixture(tasks)
    seed_source = source('run_search',0)
    candidate_source = source('run_search',2)
    original_output = {'incumbent':[0.0,1.23456789012345,-2], 'seed':123,
                       'conditions':{'precision':'binary64','evaluation_budget':1}}
    continuation = {'incumbent':[0.0,1.23456789012345,3], 'seed':123,
                    'rng_state':[7,0,19], 'conditions':deepcopy(original_output['conditions'])}
    evaluations = []

    def evaluate(current_task, path, **options):
        code = Path(path).read_text()
        assert code in (seed_source,candidate_source)
        evaluations.append(code)
        score = .1 if code==seed_source else .2
        saved_output = original_output if code==seed_source else continuation
        returned = EvalResult(True,score,{current_task.primary_metric:score},timing=.012,
                              artifacts={'saved_solution':deepcopy(saved_output)})
        return CascadeOutcome(returned,score,'full',False)

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate',evaluate)
    backend = SavedExecutor([result({'continuation_state':continuation,'gain':.1}),
                             result({'candidate_source':candidate_source,'continuation_state':continuation})])
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor',lambda image:backend)

    class Provider(MockLLM):
        def __init__(self):
            super().__init__(); self.prompts=[]
        def generate(self,prompt,model,temperature=None,**options):
            self.prompts.append(prompt)
            index=len(self.prompts)
            assert 'SCIEL_RESEARCH_TOOL_EPISODE' in prompt
            if index==1:
                action=decision()
            elif index==2:
                assert 'continuation_state' in prompt
                action=decision(code=decision()['code']+'# second scientific stage\n',
                                input_receipt_id=current_receipt(prompt))
            elif index==3:
                action=decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt))
            elif index==4:
                assert 'canonical_evaluation' in prompt and '0.2' in prompt
                action=finish(conclusion='inconclusive',next_decision='Test the method on another branch.')
            else:
                pytest.fail('No ordinary generation or repeat physical proposal is expected')
            return LLMResponse(json.dumps(action),model,.03,20,20)

    cfg=config(task,tmp_path/'capacity',workers=1,num_islands=1,max_iterations=1,
        research={'enabled':False},research_tools={'enabled':True,'image_id':IMAGE,
            'interval':1,'proactive_every_attempts':1,'max_scientific_steps':3,
            'max_requests':2,'share_between_islands':False})
    provider=Provider(); directory=tmp_path/'run'
    engine=build_engine(cfg,str(tasks),str(directory),python=sys.executable,llm=provider)
    assert isinstance(engine.research_tools,ResearchTools)
    if crash_before_admission:
        def stop_before_admission(self,plan,result,**kwargs):
            assert result.origin=='research_tool' and result.child_code==candidate_source
            assert evaluations==[seed_source,candidate_source]
            raise RuntimeError('simulated crash after tool candidate evaluation before admission')
        with monkeypatch.context() as stopped:
            stopped.setattr(Engine,'_commit_step',stop_before_admission)
            with pytest.raises(CampaignStopped,match='simulated crash'):
                engine.run()
        assert len(provider.prompts)==4 and len(backend.calls)==2
        # Receipted provider, tool and canonical-evaluator calls must all replay
        # through the real cooperative WAL without repeating physical work.
        engine=build_engine(cfg,str(tasks),str(directory),python=sys.executable,llm=provider)
        checkpoint.load(engine,str(directory/'checkpoint.json'))
    completed=engine.run()
    assert evaluations==[seed_source,candidate_source]
    assert completed['iteration_no']==completed['B_evals']==1
    assert completed['cost_total']==pytest.approx(.12)
    assert len(provider.prompts)==4 and len(backend.calls)==2
    first=backend.calls[0]['inputs']['research_targets']['selected']['saved_evaluation']
    assert first['status']=='available'
    assert first['record']['result']['artifacts']['saved_solution']==original_output
    assert backend.calls[1]['inputs']['continuation_state']==continuation
    assert backend.calls[1]['inputs']['research_stage']['observations']['continuation_state']==continuation
    children=[p for p in engine.db._all.values() if p.origin!='seed']
    assert len(children)==1 and children[0].origin=='research_tool'
    assert children[0].code==candidate_source and children[0].score==.2
    saved_state=deepcopy(engine.research_tools.state())
    research=next(iter(saved_state['lifecycle']['research_state'].values()))
    assert research['scientific_steps']==2 and research['known_llm_cost']==pytest.approx(.12)
    assert research['resources']['cpu_seconds']==.02
    assert research['status']=='closed' and research['pending_method_trial'] is not None
    calls=[json.loads(path.read_text()) for path in (directory/'.parallel').rglob('intent.json')]
    assert sum(row['kind']=='tool_execution' for row in calls)==2
    assert sum(row['kind']=='evaluation' for row in calls)==2
    for path in (directory/'.parallel').rglob('intent.json'):
        assert path.with_name('receipt.json').is_file()
    # Real checkpoint loading authenticates the cell revisions, stage receipts,
    # saved evaluator binding, accounting and canonical child without any replay.
    resumed=build_engine(cfg,str(tasks),str(directory),python=sys.executable,llm=provider)
    checkpoint.load(resumed,str(directory/'checkpoint.json'))
    assert resumed.research_tools.state()==saved_state
    assert resumed.db.get(children[0].id).code==candidate_source
    resumed.run()
    assert evaluations==[seed_source,candidate_source]
    assert len(backend.calls)==2 and len(provider.prompts)==4
