"""Tool-reference semantics at the actual planner and ordinary G boundaries.

Only deterministic transports/executor receipts are used; no authored tool or
candidate code is executed. Frozen tool visibility remains the admission gate.
"""
from copy import deepcopy
import hashlib
import json
import sys

import pytest

from sciel.engine.factory import build_engine
from sciel.research_tools.controller import PROMPT_MARKER, parse_action, response_format
from test_parallel_engine import ResearchProvider, config, evaluator, task_fixture
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_contract import result
from test_research_tool_result_flow import action, controller
from test_tool_input_projection import target_fixture, rendered_view, assert_view_matches_inputs


class RecordingLLM(LLM):
    def __init__(self, actions):
        super().__init__(actions)
        self.options = []

    def generate(self, prompt, model, temperature=None, **options):
        self.options.append(deepcopy(options))
        return super().generate(prompt, model, temperature, **options)


@pytest.mark.parametrize('field,bad', [('parents', 'p10'), ('parents', 'p33'),
    ('parents', 'a' * 64), ('dependencies', 'p10'), ('dependencies', 'b' * 64)])
def test_first_reference_rejection_explains_semantics_and_empty_repair_succeeds(tmp_path, field, bad):
    tools, backend, events, costs = controller(tmp_path, [result({'checked': True})])
    invalid = action(); invalid[field] = [bad]
    def repair(prompt):
        feedback = prompt.split('PREVIOUS ATTEMPT FEEDBACK:\n', 1)[1]
        assert field + ': invalid or unavailable tool reference' in feedback
        assert bad in feedback
        assert 'TOOL-version IDs' in feedback and 'source/input SHA256' in feedback
        assert 'Use [] for a new independent tool' in feedback
        assert 'Visible TOOL-version IDs: []' in feedback
        return action()
    current = snapshot(); plan = prepare(tools, current); io = IO()
    llm = RecordingLLM([invalid, repair])
    episode = tools.execute(plan, current, llm, io=io)
    tools.commit(episode)
    assert len(llm.prompts) == 2 and len(backend.calls) == 1
    assert costs == [.03, .03] and episode['known_llm_cost'] == pytest.approx(.06)
    assert len(episode['registered_ids']) == 1
    manifest = backend.calls[0]['bundle']['manifest']
    assert manifest['parents'] == manifest['dependencies'] == []
    assert [e['phase'] for e in events].count('rejected') == 1
    for options in llm.options:
        properties = options['response_format']['json_schema']['schema']['properties']
        for name in ('parents', 'dependencies'):
            assert 'TOOL-version IDs' in properties[name]['description']
            assert 'this array must be []' in properties[name]['description']
            assert 'enum' not in properties[name]['items']  # No invalid empty enum.
    # Replaying the same episode cannot double-charge its existing charge IDs.
    again = tools.execute(plan, current, RecordingLLM([invalid, action()]), io=io)
    assert costs == [.03, .03] and len(backend.calls) == 1
    assert again['receipt_ids'] == episode['receipt_ids']


def test_same_episode_created_tool_can_be_valid_parent_and_dependency(tmp_path):
    tools, backend, _, costs = controller(tmp_path, [
        result(status='tool_error', error='Repair this method'), result({'checked': True})])
    def repair(prompt):
        ident = next(tools.library.objects.glob('*.json')).stem
        value = action(revision=True, parents=[ident]); value['dependencies'] = [ident]
        return value
    current = snapshot(); llm = RecordingLLM([action(), repair])
    episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    first, second = episode['registered_ids']
    assert tools.library.get(second)['manifest']['parents'] == [first]
    assert tools.library.get(second)['manifest']['dependencies'] == [first]
    assert backend.calls[1]['kwargs']['dependency_bundles'][0]['id'] == first
    properties = llm.options[1]['response_format']['json_schema']['schema']['properties']
    assert all(properties[k]['items']['enum'] == [first] for k in ('parents', 'dependencies'))
    assert costs == [.03, .03]


def test_library_version_outside_frozen_episode_is_not_admitted(tmp_path):
    tools, backend, _, costs = controller(tmp_path, [])
    current = snapshot(); plan = prepare(tools, current)
    value = action()
    other = tools.library.register(value['name'], value['description'], value['code'], 1)
    value['parents'] = [other['id']]
    episode = tools.execute(plan, current, RecordingLLM([value, value]), io=IO())
    assert plan['visible_tool_ids'] == [] and episode['registered_ids'] == []
    assert backend.calls == [] and costs == [.03, .03]


@pytest.mark.parametrize('field,value', [('parents', 'p10'), ('dependencies', [None]),
    ('parents', ['BAD']), ('dependencies', ['c' * 64])])
def test_invalid_formats_and_unknown_hashes_share_actionable_first_diagnostic(tmp_path, field, value):
    tools, _, _, _ = controller(tmp_path, [])
    proposal = action(); proposal[field] = value
    with pytest.raises(ValueError) as caught:
        parse_action(json.dumps(proposal), tools.config, visible_tool_ids=[])
    message = str(caught.value)
    assert message.startswith(field + ':')
    assert 'Use [] for a new independent tool' in message
    assert 'Visible TOOL-version IDs: []' in message
    # Existing standalone parser API remains available without membership data.
    valid = action(parents=['a' * 64])
    assert parse_action(json.dumps(valid), tools.config)['parents'] == ['a' * 64]


@pytest.mark.parametrize('background', [False, True])
def test_minimum_prompt_keeps_reference_repair_and_planning_runtime_distinction(tmp_path, background):
    tools, backend, _, costs = controller(tmp_path, [result({'checked': True})],
        background=background, max_prompt_chars=8000)
    current = target_fixture('zero', huge=True)
    current.task_prompt = 'Long task detail ' * 5000
    current.kb_lessons = ['Long old lesson ' * 5000]
    bad = action(parents=['p10'])
    llm = RecordingLLM([bad, action()]); episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    inputs = backend.calls[0]['inputs']
    assert len(llm.prompts) == 2 and costs == [.03, .03]
    for prompt in llm.prompts:
        assert len(prompt) <= 8000
        assert 'SCHEDULING CONTEXT is planning-only, not a runtime payload field' in prompt
        assert "or payload['stagnation_evidence']" in prompt
        assert 'Use [] for a new independent' in prompt
        assert_view_matches_inputs(rendered_view(prompt), inputs)
        if background:
            assert 'COMPUTE JOB CONTROLS:' in prompt and "payload['execution_budget']" in prompt
    feedback = llm.prompts[1].split('PREVIOUS ATTEMPT FEEDBACK:\n', 1)[1]
    assert 'parents: invalid or unavailable tool reference' in feedback and 'p10' in feedback
    assert 'Visible TOOL-version IDs: []' in feedback
    assert not {'stagnation', 'stagnation_evidence', 'stagnation_signals'} & inputs.keys()
    assert inputs['parent_score'] == 0.0
    assert len(episode['registered_ids']) == 1


def test_two_bad_references_exhaust_planner_but_real_ordinary_generation_continues(tmp_path, monkeypatch):
    tasks = tmp_path/'tasks'; task = task_fixture(tasks)
    provider = ResearchProvider(concurrent=1)
    evaluations, _ = evaluator(monkeypatch)
    engine = build_engine(config(task, tmp_path/'capacity', workers=1, num_islands=1),
        str(tasks), str(tmp_path/'search'), python=sys.executable, llm=provider)
    tools, backend, events, costs = controller(tmp_path/'tool_fixture', [])
    try:
        engine.log.run_start(); engine.seed(portfolio=False)
        engine.research_tools = tools; engine.cfg['research_tools'] = tools.config
        engine._parallel_io = IO()
        engine._parallel_io.current = 'G:fixture'
        engine._parallel_io.sequences = {}
        engine._parallel_wave = 1
        step = engine._prepare_step(0, spawn=False)
        step['snap'].tool_episode_plan = prepare(tools, step['snap'])
        candidate_sha = hashlib.sha256(step['snap'].parent_code.encode()).hexdigest()
        proposals = RecordingLLM([action(parents=['p10']), action(parents=[candidate_sha])])
        class Transport:
            def generate(self, prompt, model, temperature=None, **options):
                return (proposals if prompt.startswith(PROMPT_MARKER) else provider).generate(
                    prompt, model, temperature, **options)
        outcome = engine._execute_step(step, llm=Transport())
        assert outcome.ok and provider.roles == ['G'] and len(evaluations) == 2
        assert len(proposals.prompts) == 2 and costs == [.03, .03]
        assert backend.calls == [] and outcome.meta['research_tools']['registered_ids'] == []
        assert len([e for e in events if e['phase'] == 'rejected']) == 2
    finally:
        engine.close()
