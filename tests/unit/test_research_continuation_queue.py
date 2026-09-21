"""Durable queue state and lifecycle safety using in-memory fake tool leaves."""
from copy import deepcopy

import pytest
from sciel.engine.research_cells_context import compact_context

from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import collect, result
from test_sequential_research_episode import build, decision, finish, current_receipt


def queued(tools, current):
    assert prepare(tools, current) is None
    return next(reversed(tools.state()['continuations']['items']))


def lease(tools):
    def refresh(current, state, **work):
        current.research_cells_context = compact_context(tools.context(current))
        return current
    key, current = tools.reserve_continuations(refresh)[0]
    return key, current, current.tool_episode_plan


def test_background_waits_without_planner_and_completed_receipt_resumes_same_cell(tmp_path):
    tools, backend, events, costs = build(tmp_path, [result({'gain': 0, 'continuation_state': {'seed': 11}})],
        independent_research=True, background=True, yield_after_success=True, share_between_islands=False)
    cell_id = queued(tools, snapshot())
    _, current, plan = lease(tools)
    episode = tools.execute(plan, current, LLM([decision(execution='background', cpu_seconds=2, wall_seconds=5)]), io=IO())
    tools.commit(episode)
    assert tools.state()['continuations']['items'][cell_id]['status'] == 'waiting'
    assert tools.reserve_continuations(lambda current, state, **work: current) == []
    assert not backend.calls and costs == [.03]
    other = queued(tools, snapshot(1))
    _, other_snapshot, other_plan = lease(tools)
    assert other_plan['cell_id'] == other
    tools.commit(tools.execute(other_plan, other_snapshot, LLM([finish(conclusion='inconclusive')]), io=IO()))
    collect(tools, episode['submitted_jobs'][0]['job_id'])
    _, resumed, resumed_plan = lease(tools)
    assert resumed_plan['cell_id'] == cell_id
    assert resumed_plan['research_state']['status'] == 'open'
    received = []
    def close(prompt):
        received.append(current_receipt(prompt))
        assert 'gain' in prompt and 'continuation_state' in prompt
        return finish()
    tools.commit(tools.execute(resumed_plan, resumed, LLM([close]), io=IO()))
    state = tools.state()
    assert state['continuations']['items'][cell_id]['status'] == 'closed'
    assert received == [state['collected_jobs'][episode['submitted_jobs'][0]['job_id']]['invocation_artifact_id']]
    assert len(backend.calls) == 1 and costs == [.03] * 3
    tools.load_state(deepcopy(state))
    assert tools.state() == state


def test_waiting_branch_does_not_block_another_ready_cell(tmp_path):
    tools, _, _, _ = build(tmp_path, [result({'candidate_source':'def run_search(): return 8\n'})],
                          independent_research=True, share_between_islands=False)
    first = queued(tools, snapshot())
    _, current, plan = lease(tools)
    episode = tools.execute(plan,current,LLM([decision(),
        lambda prompt: decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt)),
        finish()]),io=IO(),candidate_evaluator=lambda *_: {'status':'evaluated','score':.3})
    tools.commit(episode)
    second = queued(tools, snapshot(1))
    leased = tools.reserve_continuations(lambda current, state, **work: None if current.island == 0 else current)
    assert len(leased) == 1 and leased[0][1].tool_episode_plan['cell_id'] == second
    assert tools.state()['continuations']['items'][first]['reason'] == 'no_independent_branch'


def test_terminal_question_is_not_automatically_reopened_by_next_trigger(tmp_path):
    tools, _, _, _ = build(tmp_path, [result({'gain': 0})], independent_research=True)
    old = queued(tools, snapshot())
    _, current, plan = lease(tools)
    tools.commit(tools.execute(plan, current, LLM([decision(), finish()]), io=IO()))
    new = queued(tools, snapshot(iteration=1))
    assert new != old
    _, current, plan = lease(tools)
    assert 'research_state' not in plan
    assert 'different uncertainty' in plan['question']
    assert tools.state()['continuations']['items'][old]['status'] == 'closed'
    assert tools.state()['lifecycle']['research_state'][old]['conclusion'] == 'refuted'


def test_resume_authenticates_queue_source_and_owner(tmp_path):
    tools, _, _, _ = build(tmp_path, [], independent_research=True)
    cell = queued(tools, snapshot())
    saved = tools.state()
    tools.load_state(deepcopy(saved))
    artifact_id = saved['continuations']['items'][cell]['snapshot_artifact_id']
    record = tools._read_artifact(artifact_id, 'continuation_snapshot')
    record['snapshot']['parent_code'] += '# unbound change\n'
    tampered = tools._artifact('continuation_snapshot', record)
    bad = deepcopy(saved)
    bad['continuations']['items'][cell]['snapshot_artifact_id'] = tampered
    with pytest.raises(ValueError):
        tools.load_state(bad)
    assert tools.state() == saved


def test_committed_checkpoint_cannot_strand_a_scheduled_lease(tmp_path):
    tools, _, _, _ = build(tmp_path, [], independent_research=True)
    cell = queued(tools, snapshot())
    lease(tools)
    assert tools.state()['continuations']['items'][cell]['status'] == 'scheduled'
    with pytest.raises(ValueError, match='scheduled research lease'):
        tools.load_state(tools.state())


def test_fifo_completion_requeues_after_already_ready_questions(tmp_path):
    tools, _, _, _ = build(tmp_path, [result({'gain': 0})], independent_research=True, yield_after_success=True)
    first, second = queued(tools, snapshot()), queued(tools, snapshot(1))
    _, current, plan = lease(tools)
    assert plan['cell_id'] == first
    tools.commit(tools.execute(plan, current, LLM([decision()]), io=IO()))
    _, _, following = lease(tools)
    assert following['cell_id'] == second


def test_eight_k_independent_prompt_keeps_protocol_state_and_repair_cause(tmp_path):
    from test_research_experimental_protocol import protocol
    tools, _, _, _ = build(tmp_path, [result({'gain': 0, 'seed': 11})],
        independent_research=True, shared_observations=True, experimental_protocol=True,
        max_prompt_chars=8000, yield_after_success=True)
    queued(tools, snapshot())
    _, current, plan = lease(tools)
    episode = tools.execute(plan, current, LLM([decision(experimental_protocol=protocol())]), io=IO())
    tools.commit(episode)
    _, current, plan = lease(tools)
    prompt = tools._prompt(plan, current, plan['visible_tool_ids'],
        [{'status': 'proposal_invalid', 'reason': 'EXACT_REPAIR_CAUSE'}], 1,
        research_state=plan['research_state'])
    assert len(prompt) <= 8000
    assert 'EXACT_REPAIR_CAUSE' in prompt and 'EXPERIMENT PROTOCOL' in prompt
    assert 'QUEUE:' in prompt and 'PERSISTENT RESEARCH DECISION JSON:' in prompt
    assert episode['receipt_ids'][0] in prompt


def test_background_pipeline_authenticates_candidate_checkpoint_jobs_on_resume(tmp_path, monkeypatch):
    from research_target_fixtures import bind_snapshot
    from test_sequential_research_episode import lineage
    outputs = [result({'continuation_state': {'stage': n}}) for n in (1, 2, 3)]
    outputs.append(result({'candidate_source': 'def run_search(): return 19\n'}))
    tools, backend, _, costs = build(tmp_path, outputs, independent_research=True,
        background=True, yield_after_success=True)
    original = lineage(snapshot(), ancestors=[('root', 'a' * 64)])
    cell = queued(tools, original)
    prior_job = None
    for stage in range(4):
        _, current, plan = lease(tools)
        action = decision(execution='background', cpu_seconds=2, wall_seconds=5)
        if prior_job:
            action.update(action='continue', job_id=prior_job)
        episode = tools.execute(plan, current, LLM([action]), io=IO())
        tools.commit(episode)
        prior_job = episode['submitted_jobs'][0]['job_id']
        collect(tools, prior_job)
    _, current, plan = lease(tools)
    episode = tools.execute(plan, current, LLM([
        lambda prompt: decision(action='submit_candidate', candidate_receipt_id=current_receipt(prompt)),
        finish(conclusion='inconclusive')]), io=IO(),
        candidate_evaluator=lambda *_: {'status': 'evaluated', 'score': .3, 'efficacy': 'unknown'})
    tools.commit(episode)
    sibling = snapshot(iteration=1)
    sibling.parent_id = 'sibling'
    sibling.parent_code = 'def run_search(): return 7\n'
    bind_snapshot(sibling)
    lineage(sibling, ancestors=[('root', 'a' * 64)])
    sibling.research_cells_context = compact_context(tools.context(sibling))
    _, current = tools.reserve_continuations(lambda current, state, **work: sibling)[0]
    transfer = tools.execute(current.tool_episode_plan,current,LLM([]),io=IO())
    assert not transfer['research_state']['transfer_review_pending']
    assert len(transfer['submitted_jobs']) == 1
    tools.commit(transfer)
    saved = tools.state()
    state = saved['lifecycle']['research_state'][cell]
    assert 'blocked_reason' not in state['pending_method_trial']
    assert len(state['background_transfer']['reference_receipt_ids']) == 4
    assert len(saved['jobs']) == 5 and len(saved['collected_jobs']) == 4
    traversed = []
    traverse = tools._method_pipeline
    def traced(pending, owner, *, job_state=None):
        traversed.append(job_state)
        return traverse(pending, owner, job_state=job_state)
    monkeypatch.setattr(tools, '_method_pipeline', traced)
    tools.load_state(deepcopy(saved))
    assert traversed and traversed[0] == saved
    assert tools.state() == saved and len(backend.calls) == 4 and costs == [.03] * 6
    assert tools.reserve_continuations(lambda current, state, **work: current) == []


def test_background_parent_job_fallback_uses_supplied_state_not_controller_baseline(tmp_path):
    tools, _, _, _ = build(tmp_path, [], independent_research=True, background=True)
    # Fixed minimal traversal records exercise the legacy parent-job fallback.
    # The full resume test above separately authenticates real fixture receipts.
    records = {}
    candidate_state = {'jobs': {}, 'collected_jobs': {}}
    previous_job = None
    for index in range(4):
        job, receipt = str(index + 1) * 64, str(index + 5) * 64
        records[receipt] = {'cell_id': 'cell', 'result': {'status': 'success'},
                            'inputs': {}, 'receipt': {'invocation_id': job}}
        candidate_state['jobs'][job] = {'parent_job_id': previous_job}
        candidate_state['collected_jobs'][job] = {'invocation_artifact_id': receipt}
        previous_job = job
    original_read = tools._read_artifact
    tools._read_artifact = lambda ident, kind: deepcopy(records[ident])
    try:
        pending = {'submission_receipt_id': receipt}
        assert len(tools._method_pipeline(pending, 'cell', job_state=candidate_state)) == 4
        assert len(tools._method_pipeline(pending, 'cell')) == 1
    finally:
        tools._read_artifact = original_read
