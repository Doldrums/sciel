"""Real Engine/CooperativeIO commits ordinary waves around detached method stages.

Provider/evaluator/executor are deterministic fixture leaves. No model-authored
candidate or tool source is executed; no worker subprocess or network starts.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from sciel.research_tools.jobs import BackgroundJobs
from sciel.research_tools.job_worker import run_job
from test_independent_research_engine import setup
from test_parallel_engine import source
from test_research_tool_result_flow import result
from test_sequential_research_episode import current_receipt, decision, finish


def research_state(engine):
    states=engine.research_tools.state()['lifecycle']['research_state']
    assert len(states)==1
    return next(iter(states.values()))


def test_accepted_transfer_submit_stop_resume_and_ordinary_progress_between_stages(tmp_path,monkeypatch):
    monkeypatch.setattr(BackgroundJobs,'_spawn',lambda self:None)
    first_source=source('run_search',19); next_source=source('run_search',21)
    initial_state={'seed':11,'rng':[4,5,6],'array':[.25]*800}
    recipient_state={'seed':11,'rng':[7,8,9],'array':[.5]*800}
    actions=[decision(),
        lambda prompt: decision(code=decision()['code']+'# second segment\n',input_receipt_id=current_receipt(prompt)),
        lambda prompt: decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt)),
        finish(conclusion='inconclusive')]
    build,provider,backend,evaluations,directory=setup(tmp_path,monkeypatch,actions,
        [result({'continuation_state':initial_state}),result({'candidate_source':first_source}),
         result({'continuation_state':recipient_state}),result({'candidate_source':next_source})],
        iterations=8,interleaved=True,tools_overrides={'yield_after_success':False,
            'jobs':{'enabled':True},'experimental_protocol':False})
    submits=[]; stop_once=True
    original=BackgroundJobs.submit
    def submit(self,*args,**kwargs):
        nonlocal stop_once
        ack=original(self,*args,**kwargs)
        submits.append(deepcopy(ack))
        if stop_once:
            stop_once=False
            (directory/'STOP').touch()
        return ack
    monkeypatch.setattr(BackgroundJobs,'submit',submit)
    engine=build(); paused=engine.run()
    assert paused['reason']=='paused'
    assert len(submits)==1 and len(backend.calls)==2
    assert not engine.research_tools.state()['jobs']  # Accepted job is still a WAL-owned side effect.
    first_job=submits[0]['job_id']
    spec=engine.research_tools.jobs.spec(first_job)
    assert spec['request']['owner']['method_transfer']['stage_index']==0
    original_question=next(iter(engine.research_tools.state()['continuations']['items'].values()))['snapshot_artifact_id']
    ack_receipts={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (directory/'.parallel').rglob('receipt.json')}
    assert ack_receipts
    assert len(provider.planner_prompts)==4 and evaluations.count(first_source)==1
    resumed=build(resume=True); resumed._stop_at=paused['B_evals']+2
    complete=resumed.run()
    assert complete['reason']=='paused' and complete['iteration_no']>paused['iteration_no']
    assert len(submits)==1 and len(backend.calls)==2 and len(provider.planner_prompts)==4
    state=research_state(resumed); progress=state['background_transfer']
    assert progress['job_ids']==[first_job] and progress['stage_results']==[]
    assert not state['transfer_review_pending']
    assert all(hashlib.sha256(__import__('pathlib').Path(path).read_bytes()).hexdigest()==digest
               for path,digest in ack_receipts.items())
    question_after=next(iter(resumed.research_tools.state()['continuations']['items'].values()))['snapshot_artifact_id']
    assert question_after==original_question
    # Finish only the fake executor leaf after multiple real ordinary waves committed.
    run_job(directory,first_job,resumed.research_tools.library)
    first_receipt=deepcopy(resumed.research_tools.jobs.result(first_job))
    progressed=build(resume=True); progressed._stop_at=complete['B_evals']+2
    middle=progressed.run()
    assert middle['reason']=='paused' and middle['iteration_no']>complete['iteration_no']
    assert len(submits)==2 and len(backend.calls)==3 and len(provider.planner_prompts)==4
    next_job=submits[-1]['job_id']
    next_spec=progressed.research_tools.jobs.spec(next_job)['request']
    assert next_spec['parent_job_id']==first_job
    assert next_spec['inputs']['continuation_state']==recipient_state
    assert next_spec['inputs']['research_targets']==spec['request']['inputs']['research_targets']
    assert next_spec['inputs']['search_evidence']==spec['request']['inputs']['search_evidence']
    assert progressed.research_tools.jobs.result(first_job)==first_receipt
    run_job(directory,next_job,progressed.research_tools.library)
    final=build(resume=True); final._stop_at=middle['B_evals']+3
    done=final.run()
    assert done['iteration_no']>middle['iteration_no']
    assert len(submits)==2 and len(backend.calls)==4 and len(provider.planner_prompts)==4
    assert evaluations.count(first_source)==evaluations.count(next_source)==1
    state=research_state(final)
    assert state['transfer_review_pending'] and 'background_transfer' not in state
    assert state['pending_method_trial'] is None
    assert state['resources']['cpu_seconds']==pytest.approx(.04)
    assert state['known_llm_cost']==pytest.approx(.12)
    children=[p for p in final.db._all.values() if p.origin=='research_tool']
    assert {p.code for p in children}=={first_source,next_source}
    assert done['cost_total']==pytest.approx(.03*(len(provider.calls)+len(provider.planner_prompts)))
    restored=build(resume=True)
    assert restored.research_tools.state()==final.research_tools.state()
