"""Maintenance observations survive lesson growth without changing tool inputs.

All executions below use saved JSON receipts; no proposed source is executed.
"""
from copy import deepcopy
import json

import pytest

from sciel.engine.research_cells_context import compact_context
from sciel.methods.models import content_hash
from sciel.research_tools.controller import RESULTS_VIEW_MARKER, _research_results_prompt
from test_research_cell_context import card, replay_card
from test_research_tool_controller import IO, LLM, prepare
from test_research_tool_result_flow import action, controller
from test_research_tool_result_contract import result
from test_tool_input_projection import target_fixture, rendered_view, assert_view_matches_inputs


def prior_card():
    value = replay_card()
    value['observations']['output_excerpt'] = json.dumps({'output': result({
        'measured_probe': 2.58, 'samples': [0.125, 0.25, 0.5] * 30})})
    return value


@pytest.fixture(autouse=True)
def synthetic_presentation_receipt(monkeypatch):
    """These invented prior cards test rendering, not artifact authentication.

    The real Engine/WAL and forgery suites cover the host evidence boundary.
    Stub only this file's exact synthetic receipt; real new invocations still
    traverse production authentication. Prompt/hash/budget assertions remain.
    """
    from sciel.research_tools.lifecycle import CellLifecycle
    original = CellLifecycle._delivery_measurement
    synthetic_id = prior_card()['source']['receipt_id']
    def render_fixture(self, card, episode=None):
        if card['schema'] == 'sciel-research-check/3' and card['source']['receipt_id'] == synthetic_id:
            return card
        return original(self, card, episode)
    monkeypatch.setattr(CellLifecycle, '_delivery_measurement', render_fixture)


def view(prompt):
    return json.JSONDecoder().raw_decode(prompt.split(RESULTS_VIEW_MARKER, 1)[1])[0]


def known_jobs():
    return [{'job_id': '7' * 64, 'tool_id': 'c' * 64, 'status': 'completed',
             'collected': True, 'reviewed': False, 'continuation_available': False,
             'tool_result_status': 'success'}]


@pytest.mark.parametrize('background', [False, True])
def test_actual_controller_initial_and_repair_keep_prior_observations_despite_large_lessons(
        tmp_path, background):
    current = target_fixture('zero', huge=True)
    current.kb_lessons = ['Large retained lesson and evidence ' * 4000]
    current.artifacts = {'diagnostic': 'old unrelated details ' * 4000}
    current.research_cells_context = compact_context([prior_card()])
    before = deepcopy(current.research_cells_context)
    error = 'Measured repair cause: invalid response shape.'
    tools, backend, _, costs = controller(tmp_path,
        [result(status='tool_error', error=error), result({'new_check': True})],
        background=background, max_prompt_chars=64000)
    plan = prepare(tools, current)
    if background:
        plan['visible_jobs'] = known_jobs()
        plan['action_id'] = content_hash({k: v for k, v in plan.items() if k != 'action_id'})
    llm = LLM([action(), action(revision=True)])
    episode = tools.execute(plan, current, llm, io=IO())
    assert len(llm.prompts) == len(backend.calls) == 2 and costs == [.03, .03]
    assert episode['known_llm_cost'] == .06
    original = before['cards'][0]
    for prompt, call in zip(llm.prompts, backend.calls):
        assert len(prompt) <= 64000
        projected = view(prompt)
        assert projected['schema'] == 'sciel-tool-research-results-view/1'
        assert projected['omitted_count'] == 0 and projected['omission_reasons'] == {}
        row, = projected['cards']
        assert row['card_sha256'] == content_hash(original)
        for key in ('source', 'cell_id', 'revision_id', 'observations', 'replay_comparison'):
            assert row[key] == original[key]
        assert row['omitted_fields'] == [] and row['output_excerpt_omitted_chars'] == 0
        assert prompt.index(RESULTS_VIEW_MARKER) < prompt.index('FROZEN_INPUT_VIEW_JSON:')
        assert prompt.index(RESULTS_VIEW_MARKER) < prompt.index('RECENT RESEARCH CONTEXT')
        assert_view_matches_inputs(rendered_view(prompt), call['inputs'])
        assert 'research_check_results' not in call['inputs']
        assert 'sciel-tool-research-results-view/1' not in json.dumps(call['inputs'])
        if background:
            jobs, _ = json.JSONDecoder().raw_decode(prompt.split(
                'KNOWN JOBS (compact last committed observations):\n', 1)[1])
            assert jobs['jobs'][0]['job_id'] == '7' * 64 and jobs['jobs_omitted'] == 0
    assert error in llm.prompts[1]
    assert current.research_cells_context == before


@pytest.mark.parametrize('background', [False, True])
def test_minimum_budget_reports_omission_without_losing_input_contract_or_repair(
        tmp_path, background):
    current = target_fixture('failed', huge=True)
    current.kb_lessons = ['Large retained lesson ' * 5000]
    current.research_cells_context = compact_context([prior_card()])
    tools, backend, _, _ = controller(tmp_path,
        [result(status='tool_error', error='terminal repair cause'), result({'fixed': True})],
        background=background, max_prompt_chars=8000)
    plan = prepare(tools, current)
    if background:
        plan['visible_jobs'] = known_jobs()
        plan['action_id'] = content_hash({k: v for k, v in plan.items() if k != 'action_id'})
    llm = LLM([action(), action(revision=True)])
    tools.execute(plan, current, llm, io=IO())
    for prompt, call in zip(llm.prompts, backend.calls):
        assert len(prompt) <= 8000
        block = view(prompt)
        assert len(block['cards']) + block['omitted_count'] == 1
        if not block['cards']:
            assert block['omission_reasons']['planner_prompt_budget'] == 1
        else:
            assert block['cards'][0]['source'] == current.research_cells_context['cards'][0]['source']
        assert_view_matches_inputs(rendered_view(prompt), call['inputs'])
        assert 'module-level synchronous def run(payload)' in prompt
        if background:
            assert 'COMPUTE JOB CONTROLS:' in prompt and "payload['execution_budget']" in prompt
            jobs, _ = json.JSONDecoder().raw_decode(prompt.split(
                'KNOWN JOBS (compact last committed observations):\n', 1)[1])
            assert len(jobs['jobs']) + jobs['jobs_omitted'] == 1
    assert 'terminal repair cause' in llm.prompts[1]


@pytest.mark.parametrize('budget', [1800, 2200, 3000])
def test_excerpt_projection_keeps_whole_identities_and_counts_escaped_unicode(budget):
    original = card(text='snow 雪🙂 "quoted"\n' * 100)
    before = deepcopy(original)
    context = compact_context([original])
    assert len(context['cards']) == 1
    prompt = _research_results_prompt(context, budget)
    assert len(prompt) <= budget
    block = view(prompt)
    row, = block['cards']
    assert row['source'] == original['source']
    assert row['cell_id'] == original['cell_id'] and row['revision_id'] == original['revision_id']
    assert row['card_sha256'] == content_hash(original)
    retained = row['observations']['output_excerpt']
    assert original['observations']['output_excerpt'].startswith(retained)
    assert len(retained) + row['output_excerpt_omitted_chars'] == len(original['observations']['output_excerpt'])
    assert row['output_excerpt_omitted_chars'] > 0
    assert row['omitted_fields']
    assert original == before


def test_upstream_and_planner_omissions_remain_distinct_and_whole():
    context = compact_context([card('a'), card('f')])
    context['omitted_count'] = 3
    context['omission_reasons'] = {'character_limit': 3}
    projected = view(_research_results_prompt(context, 1900))
    assert len(projected['cards']) == 1
    assert projected['omitted_count'] == 4
    assert projected['omission_reasons'] == {'character_limit': 3, 'planner_prompt_budget': 1}
    assert context['omitted_count'] == 3 and len(context['cards']) == 2


def test_empty_context_needs_no_extra_block_and_invalid_card_is_explicit():
    assert _research_results_prompt(None, 0) == ''
    bad = card(); bad['source']['receipt_id'] = {'untrusted': 'shape'}
    projected = view(_research_results_prompt([bad], 400))
    assert projected['cards'] == [] and projected['omission_reasons'] == {'invalid_card': 1}


@pytest.mark.parametrize('budget', [8000, 64000])
def test_author_comparison_and_prior_observation_share_budget_with_explicit_omissions(tmp_path, budget):
    from test_replay_feedback_contract import feedback
    from sciel.research_tools.replay_feedback import is_projection
    current = target_fixture('zero', huge=True)
    current.kb_lessons = ['many lessons ' * 5000]
    current.research_cells_context = compact_context([prior_card()])
    tools, backend, _, costs = controller(tmp_path, [], background=True, max_prompt_chars=budget)
    plan = prepare(tools, current)
    comparison = feedback(result({'probe': 1}), result({'probe': 2}))
    plan['author_replay_feedback'] = [deepcopy(comparison) for _ in range(8)]
    prompt = tools._prompt(plan, current, [], [{
        'result': {'status': 'tool_error', 'error': 'terminal repair cause ' + 'detail ' * 2000}}], 1)
    assert len(prompt) <= budget
    comparisons, _ = json.JSONDecoder().raw_decode(prompt.split('REPLAY_FEEDBACK_JSON:\n', 1)[1])
    assert len(comparisons['comparisons']) + comparisons['comparisons_omitted'] == 8
    assert all(is_projection(row, comparison) for row in comparisons['comparisons'])
    prior = view(prompt)
    assert len(prior['cards']) + prior['omitted_count'] == 1
    if budget == 64000:
        assert comparisons['comparisons'] and prior['cards']
        assert prior['cards'][0]['observations'] == current.research_cells_context['cards'][0]['observations']
    assert 'terminal repair cause' in prompt
    assert backend.calls == [] and costs == []


@pytest.mark.parametrize('long_failure', [False, True])
def test_minimum_combined_failed_target_jobs_prior_result_and_author_replay_keeps_repair(
        tmp_path, long_failure):
    from test_replay_feedback_contract import feedback
    current = target_fixture('failed', huge=True)
    current.research_cells_context = compact_context([prior_card()])
    tools, backend, _, costs = controller(tmp_path, [], background=True, max_prompt_chars=8000)
    plan = prepare(tools, current)
    plan['visible_jobs'] = known_jobs()
    plan['author_replay_feedback'] = [feedback(result({'probe': 1}), result({'probe': 2}))]
    error = 'terminal repair cause'
    item = {'tool_id': 'a' * 64, 'receipt_id': 'b' * 64,
            'result': {'status': 'tool_error', 'host_status': 'ok', 'error': error}}
    if long_failure:
        item['result']['error'] += ' long diagnostic' * 1000
        item['stderr'] = 'Traceback\n' + 'middle stack detail\n' * 1000 + "KeyError: 'measurement_status'"
    prompt = tools._prompt(plan, current, [], [item], 1)
    assert len(prompt) <= 8000 and error in prompt
    assert 'Receipt: ' + item['receipt_id'] in prompt
    projected = view(prompt)
    assert len(projected['cards']) + projected['omitted_count'] == 1
    comparisons, _ = json.JSONDecoder().raw_decode(prompt.split('REPLAY_FEEDBACK_JSON:\n', 1)[1])
    assert len(comparisons['comparisons']) + comparisons['comparisons_omitted'] == 1
    assert rendered_view(prompt)['access_paths']['fields']['measurement_status'] == '/measurement/status'
    if long_failure:
        assert "KeyError: 'measurement_status'" in prompt
    assert backend.calls == [] and costs == []
