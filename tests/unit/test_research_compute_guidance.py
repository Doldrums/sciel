"""Actual planner compute guidance and staged receipts, using only fake leaves."""
from copy import deepcopy
import json

import pytest

from sciel.research_tools.controller import parse_action
from sciel.engine.research_cells_context import compact_context
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_research_experimental_protocol import comparison, protocol
from test_sequential_research_episode import build, current_receipt, decision, finish


def assert_guidance(prompt, budget):
    assert len(prompt) <= budget
    for phrase in ('null=defaults, not targets', 'throughput/progress-vs-CPU',
                   '120-180 CPU-s TOTAL', 'incumbent/RNG/counters/conditions/trace',
                   "payload['execution_budget']", 'respect explicit task bounds'):
        assert phrase in prompt
    if 'CPU/WALL seconds:' in prompt:
        assert 'both modes share maxima' in prompt and 'Long stages: background' in prompt
        assert 'A pilot only bounds tested effort' in prompt and 'not minimum; no zero-gain escalation' in prompt
        assert 'kill loses unreturned state' in prompt
        assert 'default=20/30; maximum=3600/7200' in prompt
    else:
        assert 'Both modes share maxima' in prompt and 'execution=background; foreground waits' in prompt
        assert "Tiny pilots don't refute methods" in prompt
        assert "not a minimum. Don't escalate zero-gain polishing" in prompt
        limits = json.JSONDecoder().raw_decode(prompt.split('DEFAULT LIMITS: ', 1)[1])[0]
        maxima = json.JSONDecoder().raw_decode(prompt.split('MAXIMUM PER INVOCATION: ', 1)[1])[0]
        assert limits['cpu_seconds'] == 20 and limits['wall_seconds'] == 30
        assert maxima['max_cpu_seconds'] == 3600 and maxima['max_wall_seconds'] == 7200



@pytest.mark.parametrize('budget', [8000, 64000])
def test_actual_planners_keep_compute_guidance_and_exact_larger_segment_state(tmp_path, budget):
    saved = {'incumbent': [1.23456789012345, 0.0], 'RNG': [7, 11, 13],
             'counters': {'iterations': 17}, 'conditions': {'seed': 43},
             'progress_trace': [{'cpu_seconds': 1.2, 'native_metric': 5.25}]}
    conditions = {'inputs': 'same'}
    def output(cpu, score):
        return result({'native_metric': score, 'continuation_state': saved,
            'comparison_results': [dict(arm_id=arm, seed=seed, cpu_seconds=cpu,
                best_result={'native_metric': score}, saved_state=saved, conditions=conditions)
                for arm in ('a', 'b') for seed in (11, 17)]})
    tools, backend, _, costs = build(tmp_path, [output(2, 5.25), output(40, 5.25)],
        background=True, experimental_protocol=True, shared_observations=True,
        independent_research=True, yield_after_success=True, share_between_islands=False, max_prompt_chars=budget)
    current = snapshot(); current.kb_lessons = ['Unrelated retained lesson 雪🙂 ' * 5000]
    first = decision(execution='foreground', cpu_seconds=20, wall_seconds=30,
        experimental_protocol=comparison())
    def later(prompt):
        assert_guidance(prompt, budget)
        if budget == 64000:
            assert '5.25' in prompt  # Actual observation; small contexts declare omissions.
        else:
            assert 'omitted' in prompt
            assert current_receipt(prompt)
        receipt = current_receipt(prompt)
        spec = comparison(); spec.update(cpu_seconds_per_arm=40, budget_action='escalate',
            budget_reason='Measure whether a longer segment improves the retained incumbent.',
            new_uncertainty='Does more search effort change this negative pilot?', prior_receipt_ids=[receipt])
        return decision(action='use', tool_id=backend.calls[0]['bundle']['id'],
            execution='foreground', cpu_seconds=180, wall_seconds=360,
            input_receipt_id=receipt, experimental_protocol=spec)
    def close(prompt):
        assert_guidance(prompt, budget)
        if budget == 64000:
            assert '5.25' in prompt
        else:
            assert 'omitted' in prompt
            assert current_receipt(prompt)
        return finish(conclusion='inconclusive', reason='The two measured budgets have the same result; no further polishing is warranted.',
            experimental_protocol={**protocol(), 'kind':'closure', 'budget_action':'none'})
    prepare(tools, current)  # Detach the first independent research opportunity.
    def refresh(saved_snapshot, state, **work):
        saved_snapshot.research_cells_context = compact_context(tools.context(saved_snapshot))
        return saved_snapshot
    prompts = []
    for action in (first, later, close):
        _, current = tools.reserve_continuations(refresh)[0]
        plan = current.tool_episode_plan
        llm = LLM([action])
        episode = tools.execute(plan, current, llm, io=IO())
        prompts.extend(llm.prompts)
        tools.commit(episode)
        checkpoint = deepcopy(tools.state()); tools.load_state(checkpoint)
        assert tools.state() == checkpoint
    assert len(backend.calls) == 2 and costs == [.03] * 3
    for prompt in prompts:
        assert_guidance(prompt, budget)
    assert backend.calls[1]['inputs']['continuation_state'] == saved
    assert backend.calls[1]['inputs']['research_stage']['observations'] == output(2, 5.25)['observations']
    assert backend.calls[1]['kwargs']['limits']['cpu_seconds'] == 180
    assert backend.calls[1]['inputs']['execution_budget']['wall_seconds'] == 360
    assert episode['research_state']['status'] == 'closed'



@pytest.mark.parametrize('budget', [8000, 64000])
def test_guidance_preserves_static_no_need_closure_without_a_tool_quota(tmp_path, budget):
    tools, backend, _, costs = build(tmp_path, [], background=True,
        experimental_protocol=True, shared_observations=True, max_prompt_chars=budget)
    def no_need(prompt):
        assert_guidance(prompt, budget)
        return finish(conclusion='inconclusive', reason='The required input is absent; numerical execution cannot answer this question.',
            experimental_protocol={**protocol(), 'kind':'closure', 'budget_action':'none'})
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current, LLM([no_need]), io=IO())
    assert episode['research_state']['status'] == 'closed'
    assert not backend.calls and costs == [.03]


def test_foreground_override_uses_real_envelope_without_raising_defaults(tmp_path):
    tools, _, _, _ = build(tmp_path, [], background=True)
    request = decision(execution='foreground', cpu_seconds=180, wall_seconds=360)
    parsed = parse_action(json.dumps(request), tools.config)
    assert parsed['limits']['cpu_seconds'] == 180 and tools.config['limits']['cpu_seconds'] == 20
    with pytest.raises(ValueError, match='resource envelope'):
        parse_action(json.dumps({**request, 'cpu_seconds': 3601}), tools.config)
