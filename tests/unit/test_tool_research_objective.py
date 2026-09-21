"""Task-focused tool planning with real frozen plans and deterministic receipts.

The fake transport/executor never invokes a provider or evaluates submitted code.
These tests establish delivery/compatibility, not model compliance or usefulness.
"""
from copy import deepcopy
import json

import pytest

from sciel.engine.research_targets import tool_inputs
from sciel.research_tools.controller import RESULTS_VIEW_MARKER, response_format
from sciel.research_tools.lifecycle import CellLifecycle
from test_research_cell_lifecycle import lab, prepare as prepare_cell, episode
from test_research_tool_controller import IO, LLM, prepare
from test_research_tool_result_contract import result
from test_research_tool_result_flow import action, controller
from test_tool_authoring_references import RecordingLLM
from test_tool_input_projection import target_fixture, rendered_view, assert_view_matches_inputs
from test_tool_prior_results_prompt import view


def test_question_preserves_durable_island_anchor_identity_and_restore(lab):
    plan = prepare_cell(lab, island=3, anchor=17)
    question = plan['question']
    assert "uncertainty about the supplied candidate's behavior under this task" in question
    assert 'next search decision' in question and 'reusable check' in question
    assert 'island 3, record evaluation 17; this explains scheduling only.' in question
    assert len(question) < 600 and 'What limits island' not in question
    lab.lifecycle.commit(episode(lab, plan))
    state = lab.lifecycle.state()
    restored = CellLifecycle(lab.run, lab.config, lambda **_: None)
    restored.load_state(state)
    assert restored.state() == state
    lab.lifecycle = restored
    same = prepare_cell(lab, island=3, anchor=17, parent='p_later_generation')
    assert same['cell_id'] == plan['cell_id'] and same['question'] == question
    lab.lifecycle.commit(episode(lab, same))
    after_record = prepare_cell(lab, island=3, anchor=18)
    assert after_record['cell_id'] != plan['cell_id']
    assert 'record evaluation 18' in after_record['question']
    assert restored.cells.get(plan['cell_id'])['question'] == question


@pytest.mark.parametrize('task', [
    'Investigate parser behavior on valid and malformed strings.',
    'Investigate numerical residuals of a bounded solver.',
])
@pytest.mark.parametrize('budget', [8000, 64000])
@pytest.mark.parametrize('background', [False, True])
def test_actual_initial_repair_and_cached_requests_keep_objective_and_contracts(
        tmp_path, task, budget, background):
    current = target_fixture('zero', huge=True)
    current.task_prompt = task + '\n' + 'Long quoted study data. ' * 5000
    current.kb_lessons = ['Prior candidate lesson. ' * 4000]
    before_targets = deepcopy(current.research_targets)
    cause = "Required input stagnation_evidence is absent."
    prior_observations = {'measured_probe': 2.58, 'samples': [0.125, 0.25, 0.5] * 30}
    tools, backend, _, costs = controller(tmp_path, [
        result(prior_observations),
        result(status='tool_error', error=cause),
        result({'hypothesis_supported': False, 'source_check': 'counterexample'})],
        background=background, max_prompt_chars=budget)
    # Produce prior memory through the real immutable invocation and committed
    # cell lifecycle. An invented prior_card hash has no delivery authority.
    prior_snapshot = target_fixture('zero')
    prior_plan = tools.prepare(prior_snapshot, signals={'schema':'sciel-stagnation-signals/1',
        'population_stall':{'threshold_reached':True, 'last_record_B_evals':1}})
    prior_action = action()
    prior_action['code'] += '# Separate prior measurement fixture.\n'
    previous = tools.execute(prior_plan, prior_snapshot, LLM([prior_action]), io=IO())
    tools.commit(previous)
    saved_state = deepcopy(tools.state())
    tools.load_state(saved_state)
    assert tools.state() == saved_state
    current.research_cells_context = tools.context_view(current)
    original, = current.research_cells_context['cards']
    receipt = tools._read_artifact(original['source']['receipt_id'], 'invocation')
    assert receipt['receipt']['output']['observations'] == prior_observations
    assert json.loads(original['observations']['output_excerpt'])['output']['observations'] == prior_observations
    assert original['source']['receipt_id'] == previous['receipt_ids'][0]
    before_context = deepcopy(current.research_cells_context)
    plan = prepare(tools, current)
    proposals = [action(), action(revision=True)]
    llm = RecordingLLM(proposals)
    io = IO()
    completed = tools.execute(plan, current, llm, io=io)
    assert len(llm.prompts) == 2 and len(backend.calls) == 3
    assert costs == [.03, .03, .03] and completed['known_llm_cost'] == pytest.approx(.06)
    for prompt, call, options in zip(llm.prompts, backend.calls[1:], llm.options):
        assert len(prompt) <= budget
        objective = prompt.split('RESEARCH OBJECTIVE:', 1)[1].split('Choose create', 1)[0]
        assert 'task/candidate uncertainty, not host trigger counters' in objective
        assert 'uncertainty, supplied input/source, check and next candidate/experiment decision' in objective
        assert 'source/results, failures and prior checks' in objective
        assert 'another supported check or skip' in objective
        assert prompt.index('RESEARCH OBJECTIVE:') < prompt.index(RESULTS_VIEW_MARKER)
        assert 'Static checks, CPU search' in prompt
        assert 'Negative findings may succeed; improvement is not required' in prompt
        assert "or payload['stagnation_evidence']" in prompt
        assert 'module-level synchronous def run(payload)' in prompt
        assert 'rules apply to candidates, not the tool API' in prompt
        assert 'Never submit a standalone candidate as the tool.' in prompt
        assert 'exact_observations needs identical observations' in prompt
        assert 'independent_execution needs success, NOT reproducibility' in prompt
        assert_view_matches_inputs(rendered_view(prompt), call['inputs'])
        expected_inputs = tool_inputs(current, tools.task_identity, {})
        if background:
            expected_inputs['execution_budget'] = call['kwargs']['limits']
        assert call['inputs'] == expected_inputs
        assert call['inputs']['parent_score'] == 0.0
        assert not {'stagnation', 'stagnation_evidence', 'scheduling_context'} & call['inputs'].keys()
        prior = view(prompt)
        assert len(prior['cards']) + prior['omitted_count'] == 1
        if budget == 64000:
            assert plan['question'] in prompt
            assert 'SCHEDULING CONTEXT (planning only; not the research objective or runtime input):' in prompt
            assert 'STAGNATION EVIDENCE:' not in prompt
            assert prior['omitted_count'] == 0
            assert prior['cards'][0]['observations'] == before_context['cards'][0]['observations']
        reason = options['response_format']['json_schema']['schema']['properties']['reason']
        assert reason['type'] == 'string'
        assert 'task/candidate uncertainty' in reason['description']
        assert 'for skip, explain the limitation' in reason['description']
    assert cause in llm.prompts[1]
    assert current.research_targets == before_targets
    assert current.research_cells_context == before_context
    # Legitimate negative observations remain successful; no new benefit gate.
    assert completed['invocations'][-1]['result']['status'] == 'success'
    tools.commit(completed)
    assert tools.state()['published'] == []  # No independent replay yet.
    cached = tools.execute(plan, current, RecordingLLM(proposals), io=io)
    assert cached['receipt_ids'] == completed['receipt_ids']
    assert len(backend.calls) == 3 and costs == [.03, .03, .03]


def test_reason_guidance_adds_no_new_required_fields_or_acceptance_rule():
    schema = response_format()['json_schema']['schema']
    assert set(schema['required']) == {
        'action', 'reason', 'name', 'description', 'code', 'tool_id', 'arguments_json',
        'replay_policy', 'parents', 'dependencies'}
    assert schema['properties']['reason']['type'] == 'string'
    assert set(schema['properties']['reason']) == {'type', 'description'}
