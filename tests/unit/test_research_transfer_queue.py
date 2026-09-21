"""Transfer fairness, identity and durable queue tests; fake leaves only."""
from copy import deepcopy

import pytest

from sciel.methods.models import content_hash
from sciel.research_tools.research_steps import (
    new_state, validate_state, enqueue_method_trial, complete_method_trial, pending_method_trials)
from test_research_continuation_queue import queued, lease
from test_research_tool_controller import IO, LLM, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish, current_receipt, lineage
from research_target_fixtures import bind_snapshot


def intent(char):
    return {'tool_id':char*64, 'source_program_id':'p'+char,
            'source_sha256':'f'*64,'submission_receipt_id':char*64}


def test_transfer_fifo_deduplicates_without_overwriting_and_removes_selected_only():
    state = new_state()
    first, second, third = [intent(c) for c in 'abc']
    for item in (first, second, first, third):
        enqueue_method_trial(state,item)
    assert pending_method_trials(state) == [first,second,third]
    first['tool_id'] = 'e'*64
    assert state['pending_method_trial']['tool_id'] == 'a'*64
    complete_method_trial(state,'b'*64)
    assert [r['submission_receipt_id'] for r in pending_method_trials(state)] == ['a'*64,'c'*64]
    complete_method_trial(state,'a'*64)
    assert state['pending_method_trial'] == third and state['pending_method_trial_backlog'] == []
    validate_state(state)
    with pytest.raises(ValueError,match='unique pending'):
        complete_method_trial(state,'a'*64)
    complete_method_trial(state,'c'*64)
    assert not pending_method_trials(state)


@pytest.mark.parametrize('mutation', ['duplicate','orphan_tail','invalid_tail','empty_head','review_type'])
def test_malformed_transfer_state_is_rejected(mutation):
    state = new_state()
    enqueue_method_trial(state,intent('a'))
    if mutation == 'duplicate':
        state['pending_method_trial_backlog'] = [deepcopy(state['pending_method_trial'])]
    elif mutation == 'orphan_tail':
        state['pending_method_trial'] = None
        state['pending_method_trial_backlog'] = [intent('a')]
    elif mutation == 'invalid_tail':
        state['pending_method_trial_backlog'] = [dict(intent('b'), submission_receipt_id='wrong')]
    elif mutation == 'empty_head':
        state['pending_method_trial'] = {}
    else:
        state['transfer_review_pending'] = 1
    with pytest.raises(ValueError):
        validate_state(state)


def test_eligible_transfer_gets_turn_while_question_stays_open_then_reviews_original_source(tmp_path):
    source = 'def run_search(): return 19\n'
    tools,backend,_,costs = build(tmp_path,
        [result({'candidate_source':source}), result({'continuation_state':{'stage':2}}),
         result({'candidate_source':'def run_search(): return 21\n'})],
        independent_research=True,yield_after_success=True)
    original = lineage(snapshot(), ancestors=[('root','a'*64)])
    cell = queued(tools,original)
    _, current, plan = lease(tools)
    tools.commit(tools.execute(plan,current,LLM([decision()]),io=IO()))
    _, current, plan = lease(tools)
    def submit(prompt):return decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt))
    evaluate=lambda *_: {'status':'evaluated','score':.4}
    second = tools.execute(plan,current,LLM([submit,decision(code=decision()['code']+'# new check\n')]),
                           io=IO(),candidate_evaluator=evaluate)
    tools.commit(second)
    before = tools.state()
    assert before['lifecycle']['research_state'][cell]['status'] == 'open'
    assert before['continuations']['items'][cell]['last_work_kind'] == 'question'
    question_snapshot = before['continuations']['items'][cell]['snapshot_artifact_id']
    other = snapshot(iteration=2)
    other.parent_id,other.parent_code = 'independent', 'def run_search(): return 7\n'
    bind_snapshot(other)
    lineage(other,ancestors=[('root','a'*64)])
    def choose(saved,state,*,work_kind,pending_trial):
        return other if work_kind == 'method_transfer' else saved
    _, transferred = tools.reserve_continuations(choose)[0]
    plan = transferred.tool_episode_plan
    assert plan['work_kind'] == 'method_transfer' and transferred.parent_id == other.parent_id
    no_llm = LLM([])
    outcome = tools.execute(plan,transferred,no_llm,io=IO(),candidate_evaluator=evaluate)
    tools.commit(outcome)
    saved = tools.state()
    assert not no_llm.prompts
    assert saved['continuations']['items'][cell]['snapshot_artifact_id'] == question_snapshot
    assert saved['lifecycle']['research_state'][cell]['transfer_review_pending']
    assert not pending_method_trials(saved['lifecycle']['research_state'][cell])
    tools.load_state(deepcopy(saved))
    _, review = tools.reserve_continuations(choose)[0]
    assert review.parent_id == original.parent_id and review.parent_code == original.parent_code
    assert review.tool_episode_plan['work_kind'] == 'question'
    llm = LLM([finish(conclusion='inconclusive')])
    tools.commit(tools.execute(review.tool_episode_plan,review,llm,io=IO()))
    assert source in llm.prompts[0] or outcome['receipt_ids'][0] in llm.prompts[0]
    assert not tools.state()['lifecycle']['research_state'][cell]['transfer_review_pending']
    assert len(backend.calls) == 3 and costs == [.03]*4


def test_question_plan_cannot_smuggle_a_transfer_reference(tmp_path):
    tools,backend,_,costs = build(tmp_path,[],independent_research=True)
    queued(tools,snapshot())
    _,current,plan=lease(tools)
    plan['method_trial_receipt_id']='a'*64
    plan['action_id']=content_hash({k:v for k,v in plan.items() if k!='action_id'})
    with pytest.raises(ValueError,match='transfer differs'):
        tools.execute(plan,current,LLM([]),io=IO())
    assert not backend.calls and not costs
