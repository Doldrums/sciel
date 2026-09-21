"""Yield successful evidence, never reset an unfinished repair budget."""
import json

import pytest

from sciel.research_tools.controller import response_format
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish


@pytest.mark.parametrize('budget', [8000, 64000])
def test_repair_success_yields_open_and_keeps_actual_cost_and_observation(tmp_path, budget):
    tools, backend, events, costs = build(tmp_path,
        [result(status='tool_error', error='missing optional dataset'), result({'measured_gain':0})],
        max_prompt_chars=budget, yield_after_success=True)
    current = snapshot()
    llm = LLM([decision(), decision(code=decision()['code']+'# repaired source\n')])
    episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    assert len(llm.prompts) == len(backend.calls) == 2
    assert 'missing optional dataset' in llm.prompts[1]
    assert 'YIELD:' in llm.prompts[0]
    assert 'finish closes without executing next_decision advice' in llm.prompts[0]
    assert episode['research_state']['status'] == 'open'
    assert episode['research_state']['conclusion'] is None
    assert episode['research_state']['scientific_steps'] == 1
    assert episode['research_state']['resources']['cpu_seconds'] == .02
    assert costs == [.03, .03] and episode['known_llm_cost'] == pytest.approx(.06)
    card = episode['cards'][0]
    assert card['source']['receipt_id'] == episode['receipt_ids'][-1]
    assert 'measured_gain' in episode['context'] and 'decision' not in card
    yielded = [event for event in events if event['phase'] == 'research_yielded']
    assert len(yielded) == 1 and yielded[0]['receipt_id'] == episode['receipt_ids'][-1]
    tools.commit(episode)
    saved = json.loads(json.dumps(tools.state()))
    tools.load_state(saved)
    assert tools.state() == saved and tools.state()['published'] == []


def test_failed_trials_never_yield_or_reset_repair_allowance(tmp_path):
    tools, backend, events, costs = build(tmp_path,
        [result(status='tool_error', error='missing dataset')]*2,
        max_requests=2, yield_after_success=True)
    current = snapshot(); llm = LLM([decision(), decision()])
    episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    assert len(llm.prompts) == len(backend.calls) == 2
    assert episode['research_state']['status'] == 'repair_exhausted'
    assert episode['research_state']['scientific_steps'] == 0
    assert not any(event['phase'] == 'research_yielded' for event in events)
    assert costs == [.03, .03]


@pytest.mark.parametrize('reason', [
    'The source statically proves this branch is unreachable; no empirical trial is warranted.',
    'The supplied measurement already answers this narrow question; no additional trial is needed.',
    'No valid observation is available and the required external input is missing.'])
def test_no_need_static_and_blocked_closures_do_not_force_tool_calls(tmp_path, reason):
    tools, backend, events, _ = build(tmp_path, [], yield_after_success=True)
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current,
        LLM([finish(conclusion='inconclusive', reason=reason)]), io=IO())
    assert backend.calls == []
    assert episode['research_state']['status'] == 'closed'
    assert episode['cards'][0]['decision']['interpretation'] == reason
    assert not any(event['phase'] == 'research_yielded' for event in events)


def test_finish_guidance_changes_descriptions_without_new_required_fields():
    schema = response_format(sequential=True)['json_schema']['schema']
    assert set(schema['required']) == set(schema['properties'])
    assert set(schema['required']) == {'reason','name','description','code','tool_id','arguments_json',
        'action','question','decision_if_supported','decision_if_refuted','experiment','next_decision',
        'input_receipt_id','candidate_receipt_id','conclusion','replay_policy','parents','dependencies'}
    assert 'why further trials' in schema['properties']['reason']['description']
    assert 'not automatically scheduled' in schema['properties']['next_decision']['description']
    assert 'Static/no-need' in schema['properties']['action']['description']
