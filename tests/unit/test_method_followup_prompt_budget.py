"""Chronological method history must not erase an actual tool repair cause.

Prompt assembly over an inert committed receipt; no model/tool/evaluator is called.
"""
from copy import deepcopy
import json

import pytest

from sciel.engine.research_cells_context import compact_context
from sciel.research_tools.followups import build_followup
from test_method_followups import ledger, record
from test_replay_feedback_contract import feedback
from test_research_tool_controller import prepare
from test_research_tool_engine import FakeTools
from test_research_tool_result_contract import result
from test_research_tool_result_flow import controller
from test_tool_prior_results_prompt import known_jobs, target_fixture


@pytest.mark.parametrize('budget', [8000, 64000])
@pytest.mark.parametrize('followup_count', [0, 1, 8])
def test_combined_context_keeps_repair_cause_receipt_and_terminal_exception(
        tmp_path, budget, followup_count):
    current = target_fixture('failed', huge=True)
    current.kb_lessons = ['Long retained lesson 雪🙂 ' * 5000]
    tools, backend, _, costs = controller(
        tmp_path, [], background=True, max_prompt_chars=budget)
    # The imported old prior_card() had an invented receipt and no committed
    # owner. Keep the production evidence boundary: save an inert invocation
    # through the existing fixture helper and commit its real lifecycle joins.
    prior_episode = FakeTools.saved_observation(tools, current,
        label='followup-prompt-prior', commit=True,
        observations={'measured_probe': 2.58, 'samples': [0.125, 0.25, 0.5] * 30})
    prior_card = prior_episode['cards'][0]
    current.research_cells_context = compact_context([prior_card])
    plan = prepare(tools, current)
    plan['visible_jobs'] = known_jobs()
    comparison_count = max(1, followup_count)
    plan['author_replay_feedback'] = [
        feedback(result({'probe': 1}), result({'probe': 2}))] * comparison_count
    if followup_count:
        history = ledger()
        for number in range(1, followup_count + 1):
            history.add(build_followup(record(number)))
        plan['method_followups'] = history.planner_view(['c' * 64], max_chars=8000)
    before = deepcopy(plan)
    item = {
        'tool_id': 'a' * 64,
        'receipt_id': 'b' * 64,
        'result': {
            'status': 'tool_error', 'host_status': 'ok',
            'error': 'terminal repair cause' + ' long diagnostic' * 1000,
        },
        'stderr': ('Traceback\n' + 'middle stack detail\n' * 1000
                   + "KeyError: 'measurement_status'"),
    }
    prompt = tools._prompt(plan, current, [], [item], 1)

    assert len(prompt) <= budget
    assert 'terminal repair cause' in prompt
    assert 'Receipt: ' + item['receipt_id'] in prompt
    assert "KeyError: 'measurement_status'" in prompt

    def block(marker):
        return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]

    targets = block('FROZEN_INPUT_VIEW_JSON:\n')
    assert targets['access_paths']['fields']['score'] == '/measurement/record/child_observation/score'
    assert targets['access_paths']['fields']['measurement_status'] == '/measurement/status'
    prior = block('RESEARCH_CHECK_RESULTS_VIEW_JSON:\n')
    assert len(prior['cards']) + prior['omitted_count'] == 1
    for shown in prior['cards']:
        assert shown['source'] == prior_card['source']
    if budget == 64000:
        assert len(prior['cards']) == 1
        assert prior['cards'][0]['source']['receipt_id'] == prior_episode['receipt_ids'][0]
        assert 'measured_probe' in prior['cards'][0]['observations']['output_excerpt']
    replay = block('REPLAY_FEEDBACK_JSON:\n')
    assert len(replay['comparisons']) + replay['comparisons_omitted'] == comparison_count
    jobs = block('KNOWN JOBS (compact last committed observations):\n')
    assert len(jobs['jobs']) + jobs['jobs_omitted'] == 1
    if followup_count:
        followup = block('METHOD_FOLLOWUPS_VIEW_JSON:\n')
        assert len(followup['records']) + followup['omitted_count'] == followup_count
        assert 'causal benefit are unknown' in followup['scope']
        if budget == 64000:
            assert len(followup['records']) == min(2, followup_count)
    else:
        assert 'METHOD_FOLLOWUPS_VIEW_JSON:\n' not in prompt
    assert plan == before
    assert backend.calls == [] and costs == []
