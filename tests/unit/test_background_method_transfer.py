"""A saved multi-stage method progresses through durable jobs, not a wave wait.

The real controller, job ledger, snapshot/receipt authentication and continuation
queue are exercised; fake executor outputs are data, no authored code is run.
"""
from copy import deepcopy
import json
import hashlib
from pathlib import Path

import pytest

from research_target_fixtures import bind_snapshot
from sciel.methods.models import content_hash
from sciel.research_tools.job_worker import run_job
from test_research_continuation_queue import queued, lease
from test_research_tool_controller import IO, LLM, snapshot
from test_research_tool_result_flow import collect, result
from test_sequential_research_episode import build, decision, finish, current_receipt, lineage


def prepared(tmp_path, *, failed=False):
    old_state = {'seed':7,'rng':[1,2,3],'array':[.25]*3000}
    new_state = {'seed':7,'rng':[9,8,7],'array':[.5]*3000}
    tools, backend, events, costs = build(tmp_path, [result({'continuation_state':old_state}),
        result({'candidate_source':'def run_search(): return 10\n'}),
        result(status='tool_error', error='fixture failed') if failed else result({'continuation_state':new_state}),
        result({'candidate_source':'def run_search(): return 11\n'})],
        background=True, independent_research=True, share_between_islands=False)
    first = lineage(snapshot(), ancestors=[('root','a'*64)])
    cell = queued(tools, first)
    _, current, plan = lease(tools)
    evaluations = []
    def evaluate(source, provenance):
        evaluations.append((source,deepcopy(provenance)))
        return {'status':'evaluated','score':.1*len(evaluations),'timing_seconds':.01}
    original = tools.execute(plan, current, LLM([decision(cpu_seconds=120,wall_seconds=240),
        lambda prompt: decision(cpu_seconds=360,wall_seconds=480,code=decision()['code']+'# next\n',input_receipt_id=current_receipt(prompt)),
        lambda prompt: decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt)),
        finish(conclusion='inconclusive')]), io=IO(),candidate_evaluator=evaluate)
    tools.commit(original)
    recipient = snapshot(iteration=1)
    recipient.parent_id='sibling'; recipient.parent_code='def run_search(): return 7\n'
    bind_snapshot(recipient); lineage(recipient, ancestors=[('root','a'*64)])
    def refresh(saved, state, **work):
        if work['work_kind']=='method_transfer':
            return deepcopy(recipient)
        saved.iteration_no += 1
        return saved
    return tools, backend, events, costs, cell, evaluate, evaluations, refresh, new_state


def stage(tools, refresh, evaluate, *, io=None):
    _, current = tools.reserve_continuations(refresh)[0]
    episode = tools.execute(current.tool_episode_plan,current,LLM([]),io=io or IO(),candidate_evaluator=evaluate)
    tools.commit(episode)
    return episode


def roundtrip(tools):
    saved = json.loads(json.dumps(tools.state()))
    tools.load_state(saved)
    assert tools.state()==saved
    return saved


def test_two_stage_jobs_pin_recipient_chain_and_canonical_once(tmp_path):
    tools, backend, events, costs, cell, evaluate, evaluations, refresh, new_state = prepared(tmp_path)
    first = stage(tools,refresh,evaluate)
    job = first['submitted_jobs'][0]['job_id']
    assert len(backend.calls)==2 and len(evaluations)==1 and costs==[.03]*4
    state = roundtrip(tools)
    helper=Path(__import__('sciel.research_tools.transfer_jobs',fromlist=['']).__file__)
    assert tools.identity['implementation']['transfer_jobs.py']==hashlib.sha256(helper.read_bytes()).hexdigest()
    bad_identity=deepcopy(state);bad_identity['identity_sha256']='0'*64
    with pytest.raises(ValueError,match='identity'):
        tools.load_state(bad_identity)
    progress = state['lifecycle']['research_state'][cell]['background_transfer']
    frozen_id = progress['snapshot_artifact_id']
    assert progress['job_ids']==[job] and progress['stage_results']==[]
    assert tools.reserve_continuations(refresh)==[]
    # Another cell is schedulable while this numerical stage has not run.
    other = queued(tools,snapshot(1,2))
    _, current, plan = lease(tools)
    assert plan['cell_id']==other
    tools.commit(tools.execute(plan,current,LLM([finish(conclusion='inconclusive')]),io=IO()))
    collect(tools,job)
    state = roundtrip(tools)
    old_result = state['collected_jobs'][job]['invocation_artifact_id']
    resources = deepcopy(state['lifecycle']['research_state'][cell]['resources'])
    tools.collect_background(IO())
    assert tools.state()['lifecycle']['research_state'][cell]['resources']==resources
    def no_reselection(saved,state,**work):
        assert work['work_kind']=='method_transfer_continuation'
        assert saved.parent_id=='sibling'
        saved.iteration_no=9
        saved.research_search_evidence['later_history']='not part of pinned inputs'
        return saved
    second = stage(tools,no_reselection,evaluate)
    next_job = second['submitted_jobs'][0]['job_id']
    request = tools.jobs.spec(next_job)['request']
    first_request = tools.jobs.spec(job)['request']
    assert first_request['limits']['cpu_seconds']==120
    assert request['limits']['cpu_seconds']==360
    assert request['tool_id']!=first_request['tool_id']
    assert request['parent_job_id']==job
    assert request['inputs']['continuation_state']==new_state
    assert request['inputs']['research_stage']['prior_receipt_id']==old_result
    assert request['inputs']['research_targets']==first_request['inputs']['research_targets']
    assert request['inputs']['search_evidence']==first_request['inputs']['search_evidence']
    assert second['research_state']['background_transfer']['snapshot_artifact_id']==frozen_id
    roundtrip(tools)
    collect(tools,next_job); roundtrip(tools)
    final = stage(tools,no_reselection,evaluate)
    assert len(backend.calls)==4 and len(evaluations)==2
    assert evaluations[-1][0]=='def run_search(): return 11\n'
    assert evaluations[-1][1]['discovery_resources']['cpu_seconds']==pytest.approx(.04)
    assert evaluations[-1][1]['independent_branch_trial'] is True
    assert 'background_transfer' not in final['research_state']
    assert final['research_state']['transfer_review_pending']
    assert final['research_state']['pending_method_trial'] is None
    roundtrip(tools)
    _, original_question = tools.reserve_continuations(refresh)[0]
    assert original_question.parent_id=='p0'  # Question never inherits temporary recipient.
    assert original_question.tool_episode_plan['work_kind']=='question'
    assert all(e.get('benefit')!='established' for e in events)


@pytest.mark.parametrize('failure', ['tool_error','cancel','unknown'])
def test_terminal_stage_failure_retires_chain_without_retry_or_candidate(tmp_path,failure):
    tools, backend, _, _, cell, evaluate, evaluations, refresh, _ = prepared(tmp_path,failed=True)
    episode=stage(tools,refresh,evaluate); job=episode['submitted_jobs'][0]['job_id']
    if failure=='cancel':
        tools.jobs.cancel(job)
    if failure=='unknown':
        # The real job service records a previously dispatched, unreceipted leaf
        # as unknown on reconciliation; no fixture output or retry is invented.
        from sciel.research_tools.jobs import write_json
        write_json(tools.jobs.root/job/'dispatch.json',
            {'job_id':job,'request_hash':tools.jobs.spec(job)['request_hash']},immutable=True)
        run_job(tools.lifecycle.run_dir,job,tools.library)
        tools.collect_background(IO())
    else:
        collect(tools,job)
    saved=roundtrip(tools)
    before=saved['lifecycle']['research_state'][cell]
    if failure=='unknown':
        assert before['resources']['unknown_cpu_invocations']==1
        assert saved['collected_jobs'][job]['resources']['cpu_seconds'] is None
        tools.collect_background(IO())
        assert tools.state()['lifecycle']['research_state'][cell]['resources']==before['resources']
    calls=len(backend.calls)
    final=stage(tools,refresh,evaluate)
    assert len(backend.calls)==calls and len(evaluations)==1
    assert final['research_state']['pending_method_trial'] is None
    assert final['research_state']['transfer_review_pending']
    assert 'background_transfer' not in final['research_state']
    roundtrip(tools)


@pytest.mark.parametrize('change', ['snapshot','references','job','cursor','result','missing'])
def test_checkpoint_rejects_resealed_wrong_stage_or_recipient(tmp_path,change):
    tools, _, _, _, cell, evaluate, _, refresh, _ = prepared(tmp_path)
    episode=stage(tools,refresh,evaluate); job=episode['submitted_jobs'][0]['job_id']
    collect(tools,job)
    saved=roundtrip(tools); bad=deepcopy(saved)
    progress=bad['lifecycle']['research_state'][cell]['background_transfer']
    if change=='snapshot':
        progress['snapshot_artifact_id']=saved['continuations']['items'][cell]['snapshot_artifact_id']
    elif change=='references': progress['reference_receipt_ids'].reverse()
    elif change=='job': progress['job_ids'][0]='0'*64
    elif change=='cursor': progress['stage_results']=[]
    elif change=='missing': bad['lifecycle']['research_state'][cell].pop('background_transfer')
    else: progress['stage_results'][0]['receipt_id']='0'*64
    with pytest.raises((ValueError,KeyError)):
        tools.load_state(bad)
    assert tools.state()==saved


def test_four_stage_saved_method_exceeds_local_opportunity_cap_but_yields_between_jobs(tmp_path):
    stages=4
    tools, backend, _, costs=build(tmp_path,
        [result({'continuation_state':{'segment':i,'array':[i]*150}}) for i in range(3)] +
        [result({'candidate_source':'def run_search(): return 10\n'})] +
        [result({'continuation_state':{'segment':i,'array':[9+i]*150}}) for i in range(3)] +
        [result({'candidate_source':'def run_search(): return 11\n'})],
        independent_research=True,background=True,yield_after_success=True,share_between_islands=False)
    assert tools.config['max_scientific_steps']==3
    queued(tools,lineage(snapshot(),ancestors=[('root','a'*64)]))
    previous=None
    for index in range(stages):
        _,current,plan=lease(tools)
        proposal=decision(code=decision()['code']+f'# saved segment {index}\n',
                          input_receipt_id=previous or '')
        episode=tools.execute(plan,current,LLM([proposal]),io=IO())
        previous=episode['receipt_ids'][-1]
        tools.commit(episode)
    evaluations=[]
    def evaluate(source,provenance):
        evaluations.append(source)
        return {'status':'evaluated','score':.1,'timing_seconds':.02}
    _,current,plan=lease(tools)
    episode=tools.execute(plan,current,LLM([
        decision(action='submit_candidate',candidate_receipt_id=previous),
        finish(conclusion='inconclusive')]),io=IO(),candidate_evaluator=evaluate)
    tools.commit(episode)
    sibling=snapshot(iteration=5);sibling.parent_id='sibling';sibling.parent_code='def run_search(): return 7\n'
    bind_snapshot(sibling);lineage(sibling,ancestors=[('root','a'*64)])
    def refresh(saved,state,**work):
        return deepcopy(sibling) if work['work_kind']=='method_transfer' else saved
    jobs=[]
    for index in range(stages):
        submitted=stage(tools,refresh,evaluate)
        job=submitted['submitted_jobs'][0]['job_id'];jobs.append(job)
        assert len(backend.calls)==stages+index
        assert submitted['research_state']['background_transfer']['job_ids']==jobs
        assert not submitted['research_state']['transfer_review_pending']
        roundtrip(tools)
        collect(tools,job);roundtrip(tools)
    final=stage(tools,refresh,evaluate)
    assert len(backend.calls)==8 and len(evaluations)==2
    assert final['research_state']['pending_method_trial'] is None
    assert final['research_state']['transfer_review_pending']
    assert costs==[.03]*6
