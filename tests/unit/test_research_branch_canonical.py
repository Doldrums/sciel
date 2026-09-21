"""Independent method trials must measure their saved program, not just run it.

Generated Python is inert text. These exercise the actual controller, receipts,
cell commits and resume authentication with fixture transport leaves.
"""
from copy import deepcopy
import json

import pytest

from research_target_fixtures import bind_snapshot
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import action, result
from test_sequential_research_episode import build, decision, finish, current_receipt, lineage


def test_saved_two_stage_method_is_canonically_evaluated_on_independent_branch(tmp_path):
    original_source = 'def run_search(): return 3\n'
    branch_source = 'def run_search(): return 15\n'
    original_state = {'seed': 17, 'best': [1, 2], 'rng': [8, 3]}
    branch_state = {'seed': 17, 'best': [7, 8], 'rng': [9, 4]}
    tools, backend, events, costs = build(tmp_path, [
        result({'continuation_state': original_state}), result({'candidate_source': original_source}),
        result({'continuation_state': branch_state}), result({'candidate_source': branch_source})])
    first_snapshot = lineage(snapshot(), ancestors=[('root', 'a' * 64)])
    evaluations = []

    def evaluator(source, provenance):
        evaluations.append((source, deepcopy(provenance)))
        return {'status': 'evaluated', 'score': len(evaluations) / 10,
                'timing_seconds': .025, 'efficacy': 'unknown'}

    first = tools.execute(prepare(tools, first_snapshot), first_snapshot, LLM([
        decision(), lambda prompt: decision(code=action()['code'] + '# next stage\n',
                                            input_receipt_id=current_receipt(prompt)),
        lambda prompt: decision(action='submit_candidate', candidate_receipt_id=current_receipt(prompt)),
        finish(conclusion='inconclusive')]), io=IO(), candidate_evaluator=evaluator)
    tools.commit(first)
    tools.load_state(json.loads(json.dumps(tools.state())))
    sibling = snapshot(iteration=1)
    sibling.parent_id = 'independent_sibling'
    sibling.parent_code = 'def run_search(): return 7\n'
    bind_snapshot(sibling)
    lineage(sibling, ancestors=[('root', 'a' * 64)])

    def interpret(prompt):
        assert 'canonical_evaluation' in prompt and '0.2' in prompt
        return finish(conclusion='inconclusive', next_decision='Use matched controls to estimate benefit.')

    no_planner = LLM([])
    second = tools.execute(prepare(tools, sibling), sibling, no_planner,
                           io=IO(), candidate_evaluator=evaluator)
    assert [source for source, _ in evaluations] == [original_source, branch_source]
    assert len(backend.calls) == 4 and costs == [.03] * 4
    assert not no_planner.prompts and second['research_state']['transfer_review_pending']
    provenance = evaluations[-1][1]
    assert provenance['independent_branch_trial'] is True
    assert provenance['reference_submission_receipt_id'] == first['candidate_submission']['provenance']['receipt_id']
    assert provenance['target_binding']['candidate_id'] == 'independent_sibling'
    assert provenance['discovery_resources']['cpu_seconds'] == pytest.approx(.04)
    assert provenance['known_llm_cost'] == pytest.approx(.12)
    assert 'continuation_state' not in backend.calls[2]['inputs']
    assert backend.calls[3]['inputs']['continuation_state'] == branch_state
    assert second['candidate_submission']['source'] == branch_source
    assert second['research_state']['pending_method_trial'] is None
    measured = [step for step in second['research_state']['steps'] if step.get('independent_branch_trial')]
    assert len(measured) == 1 and measured[0]['benefit'] == 'not_established'
    assert measured[0]['canonical_evaluation_id']
    tools.commit(second)
    state = tools.state()
    tools.load_state(json.loads(json.dumps(state)))
    assert tools.state() == state
    assert len(evaluations) == 2 and len(backend.calls) == 4
    review = deepcopy(first_snapshot)
    review.iteration_no = 2
    interpretation = tools.execute(prepare(tools, review), review, LLM([interpret]),
                                   io=IO(), candidate_evaluator=evaluator)
    tools.commit(interpretation)
    assert costs == [.03] * 5
    assert interpretation['research_state']['status'] == 'closed'
    assert not interpretation['research_state']['transfer_review_pending']
    assert len(evaluations) == 2 and len(backend.calls) == 4


@pytest.mark.parametrize('status', ['eval_failed', 'output_repeat_rejected', 'rejected'])
def test_failed_or_duplicate_candidate_does_not_schedule_transfer_as_new_method(tmp_path, status):
    tools, _, _, _ = build(tmp_path, [result({'candidate_source': 'def run_search(): return 2\n'})])
    current = lineage(snapshot())
    episode = tools.execute(prepare(tools, current), current, LLM([
        decision(), lambda prompt: decision(action='submit_candidate', candidate_receipt_id=current_receipt(prompt)),
        finish(conclusion='inconclusive')]), io=IO(),
        candidate_evaluator=lambda *_: {'status': status, 'score': None, 'efficacy': 'unknown'})
    assert episode['research_state']['pending_method_trial'] is None
    tools.commit(episode)
    tools.load_state(json.loads(json.dumps(tools.state())))
