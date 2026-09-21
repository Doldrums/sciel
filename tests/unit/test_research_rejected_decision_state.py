"""Rejected proposals retain their ledger evidence without replacing decisions.

All model responses and execution receipts are fixtures. No generated code runs.
"""
from copy import deepcopy
import json

import pytest

from sciel.research_tools.research_steps import TEXT_FIELDS, new_state, record_step
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish


def planner_state(prompt):
    marker = 'PERSISTENT RESEARCH DECISION JSON:\n'
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]


@pytest.mark.parametrize('kind', ['invisible_tool', 'unknown_stage', 'unknown_candidate'])
def test_postparse_rejection_preserves_decision_through_resume_and_next_prompt(tmp_path, kind):
    tools, backend, events, costs = build(tmp_path, [result({'gain': 0})], max_requests=1)
    current = snapshot()
    accepted = decision()
    rejected_fields = {key: 'Rejected proposal ' + key for key in TEXT_FIELDS}
    rejected = decision(**rejected_fields, conclusion='supported')
    error = 'receipt is not a successful visible invocation'
    if kind == 'invisible_tool':
        rejected.update(action='use', tool_id='f' * 64)
        error = 'tool_id is not visible'
    elif kind == 'unknown_stage':
        rejected['input_receipt_id'] = 'f' * 64
    else:
        rejected.update(action='submit_candidate', candidate_receipt_id='f' * 64)

    def never_evaluate(*args):
        pytest.fail('A rejected receipt must not reach candidate evaluation')

    episode = tools.execute(prepare(tools, current), current, LLM([accepted, rejected]),
                            io=IO(), candidate_evaluator=never_evaluate)
    state = episode['research_state']
    assert state['status'] == 'repair_exhausted'
    assert {key: state[key] for key in TEXT_FIELDS} == {key: accepted[key] for key in TEXT_FIELDS}
    assert state['current_experiment'] == accepted['experiment']
    assert state['conclusion'] is None
    assert state['scientific_steps'] == 1
    assert state['resources'] == {'cpu_seconds': .01, 'wall_seconds': .02, 'unknown_cpu_invocations': 0}
    assert len(backend.calls) == 1 and costs == [.03, .03]
    assert state['known_llm_cost'] == episode['known_llm_cost'] == pytest.approx(.06)
    invalid = state['steps'][-1]
    assert invalid['action'] == 'invalid' and invalid['status'] == 'proposal_invalid'
    assert {key: invalid[key] for key in TEXT_FIELDS} == rejected_fields
    assert invalid['conclusion'] == 'supported' and error in invalid['error']
    assert json.loads(tools._read_artifact(invalid['proposal_id'], 'proposal')['response']['text']) == rejected
    assert 'terminal_decision_proposal_id' not in episode

    tools.commit(episode)
    saved = deepcopy(tools.state())
    tools.load_state(saved)
    assert tools.state() == saved
    committed = [event for event in events if event['phase'] == 'research_state'][-1]['research_state']
    assert committed['question'] == accepted['question'] and committed['conclusion'] is None
    next_snapshot = snapshot(iteration=1)
    next_plan = prepare(tools, next_snapshot)
    assert next_plan['cell_id'] == episode['plan']['cell_id']
    assert next_plan['question'] == accepted['question']

    def inspect_next_prompt(prompt):
        view = planner_state(prompt)
        assert view['question'] == accepted['question'] and view['conclusion'] is None
        assert view['experiment'] == accepted['experiment']
        return finish(conclusion='inconclusive')

    tools.execute(next_plan, next_snapshot, LLM([inspect_next_prompt]), io=IO())
    assert len(backend.calls) == 1  # Closing the question does not rerun the tool.
    assert costs == [.03, .03, .03]


def test_rejected_supported_finish_without_evidence_does_not_promote_a_conclusion(tmp_path):
    tools, backend, _, costs = build(tmp_path, [], max_requests=1)
    current = snapshot()
    rejected = finish(conclusion='supported', question='A claim with no experiment')
    episode = tools.execute(prepare(tools, current), current, LLM([rejected]), io=IO())
    state = episode['research_state']
    assert state['status'] == 'repair_exhausted'
    assert all(state[key] == '' for key in (*TEXT_FIELDS, 'current_experiment'))
    assert state['conclusion'] is None and state['scientific_steps'] == 0
    assert state['known_llm_cost'] == pytest.approx(.03) and costs == [.03]
    assert not backend.calls and not episode['cards']
    assert 'terminal_decision_proposal_id' not in episode
    invalid = state['steps'][0]
    assert invalid['status'] == 'proposal_invalid' and invalid['conclusion'] == 'supported'
    assert invalid['question'] == rejected['question']
    assert 'requires recorded experiment evidence' in invalid['error']
    tools.commit(episode)
    saved = deepcopy(tools.state())
    tools.load_state(saved)
    assert tools.state() == saved


@pytest.mark.parametrize('rejected_conclusion', [None, 'supported'])
def test_rejected_step_cannot_clear_or_replace_prior_conclusion_but_valid_step_can(rejected_conclusion):
    state = new_state()
    record_step(state, {'action': 'finish', 'status': 'closed', 'question': 'Existing question',
                       'experiment': 'Existing experiment', 'conclusion': 'refuted'}, cost=.03)
    before = deepcopy(state)
    rejected = {'action': 'invalid', 'status': 'proposal_invalid',
                **{key: 'Rejected ' + key for key in TEXT_FIELDS}, 'conclusion': rejected_conclusion}
    record_step(state, rejected, cost=.04)
    assert all(state[key] == before[key] for key in (*TEXT_FIELDS, 'current_experiment', 'conclusion'))
    assert state['steps'][-1] == rejected and state['known_llm_cost'] == pytest.approx(.07)
    record_step(state, {'action': 'create', 'status': 'success', 'question': 'Next valid question',
                       'experiment': 'Next valid experiment', 'conclusion': None}, cost=.02, scientific=True)
    assert state['question'] == 'Next valid question'
    assert state['current_experiment'] == state['experiment'] == 'Next valid experiment'
    assert state['conclusion'] is None and state['scientific_steps'] == 1
    assert state['known_llm_cost'] == pytest.approx(.09)
