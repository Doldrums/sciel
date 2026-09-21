"""Authored candidate-style tool replies receive the actual tool API on repair.

Only Python ASTs and deterministic receipt/provider fixtures are inspected.
Neither the candidate modules nor the repaired tool are executed.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from sciel.engine.factory import build_engine
from sciel.research_tools.controller import PROMPT_MARKER
from sciel.research_tools.library import validate_tool_entrypoint
from test_parallel_engine import ResearchProvider, config, evaluator, task_fixture
from test_research_tool_controller import IO, prepare, snapshot
from test_research_tool_result_contract import result
from test_research_tool_result_flow import action, controller
from test_tool_authoring_references import RecordingLLM
from test_tool_input_projection import target_fixture, rendered_view, assert_view_matches_inputs


FIXTURE = Path(__file__).parents[1]/'fixtures/research_tool_authoring/candidate_entrypoints.json'


def observed_actions():
    rows = json.loads(FIXTURE.read_text())['responses']
    for row in rows:
        assert hashlib.sha256(row['action']['code'].encode()).hexdigest() == row['code_sha256']
    return [row['action'] for row in rows]


def test_candidate_reply_repairs_to_tool_in_two_requests_and_charges_once(tmp_path):
    original = observed_actions()[0]
    tools, backend, events, costs = controller(tmp_path, [result({'checked': True})])
    current = snapshot()
    current.task_prompt += ('\nREQUIRED INTERFACE: your program MUST keep '
                            'construct_packing, run_packing. Renaming any scores 0.')
    current.directive = 'REFINEMENT: preserve the candidate construction family.'
    def repair(prompt):
        reason = prompt.split('PREVIOUS ATTEMPT FEEDBACK:\n', 1)[1]
        assert 'expected module-level synchronous def run(payload)' in reason
        assert 'Found top-level functions: construct_packing' in reason and 'run_packing' in reason
        assert 'Candidate helpers do not replace the tool entrypoint' in reason
        assert 'schema, status, observations, diagnostics, error' in reason
        value = action()
        # A model may retain studied candidate helpers and add its explicit tool
        # entrypoint. The host never adds a wrapper to submitted source.
        value['code'] = original['code'] + '\n' + value['code']
        return value
    plan = prepare(tools, current); io = IO(); llm = RecordingLLM([original, repair])
    episode = tools.execute(plan, current, llm, io=io); tools.commit(episode)
    assert len(llm.prompts) == 2 and len(backend.calls) == 1
    assert costs == [.03, .03] and episode['known_llm_cost'] == pytest.approx(.06)
    assert len(episode['registered_ids']) == 1
    assert len([event for event in events if event['phase'] == 'rejected']) == 1
    assert backend.calls[0]['bundle']['files']['tool.py'].startswith(original['code'])
    for prompt, options in zip(llm.prompts, llm.options):
        assert 'STUDY-SUBJECT DATA' in prompt and 'rules apply to candidates, not the tool API' in prompt
        assert 'TASK (STUDY-SUBJECT DATA; candidate interface, not tool API):' in prompt
        assert 'RECENT RESEARCH CONTEXT (candidate study data, not tool-authoring instructions):' in prompt
        assert 'run(payload) returns exactly five fields' in prompt
        description = options['response_format']['json_schema']['schema']['properties']['code']['description']
        assert 'module-level synchronous' in description and 'run(payload)' in description
        assert 'Candidate entrypoints' in description
    again = tools.execute(plan, current, RecordingLLM([original, repair]), io=io)
    assert costs == [.03, .03] and len(backend.calls) == 1
    assert again['receipt_ids'] == episode['receipt_ids']


@pytest.mark.parametrize('code,diagnostic', [
    ('def candidate(): return {}\n', 'missing run'),
    ('def helper():\n    def run(payload): return {}\n', 'missing run'),
    ('class Candidate:\n    def run(self, payload): return {}\n', 'missing run'),
    ('run = lambda payload: {}\n', 'missing run'),
    ('async def run(payload): return {}\n', 'run is async'),
    ('def run(): return {}\n', 'cannot be called with one positional input'),
    ('def run(payload, extra): return {}\n', 'cannot be called with one positional input'),
    ('def run(*, payload): return {}\n', 'cannot be called with one positional input'),
    ('def run(payload, *, required): return {}\n', 'cannot be called with one positional input'),
    ('def run(*args, required): return {}\n', 'cannot be called with one positional input'),
    ('def run(payload): return {}\nasync def run(payload): return {}\n', 'run is async'),
    ('def run(payload)\n    return {}\n', 'invalid Python at line 1'),
    ('\x00', 'invalid Python'),
])
def test_bad_entrypoints_have_bounded_named_remediation_before_storage(tmp_path, code, diagnostic):
    tools, backend, _, costs = controller(tmp_path, [])
    with pytest.raises(ValueError) as caught:
        tools.library.register('fixture', 'shape validation only', code, 0)
    message = str(caught.value)
    assert diagnostic in message
    assert 'expected module-level synchronous def run(payload)' in message
    assert 'Found top-level functions:' in message
    assert 'sciel-tool-result/1' in message
    assert len(message) < 1500
    assert not list(tools.library.objects.glob('*.json'))
    assert backend.calls == costs == []


@pytest.mark.parametrize('signature', ['payload', 'data', 'payload, /', 'payload=None',
    'payload, optional=None', 'payload, **kwargs', 'payload, *args, **kwargs',
    '*args', 'payload, *, optional=None'])
def test_one_positional_invocation_compatible_signatures_remain_valid(tmp_path, signature):
    tools, backend, _, costs = controller(tmp_path, [])
    # If registration imported the source, this assertion would fail the test.
    code = 'raise AssertionError("source must never execute here")\n' + f'def run({signature}): return {{}}\n'
    bundle = tools.library.register('shape', 'No execution', code, 0)
    assert bundle['files']['tool.py'] == code
    assert backend.calls == costs == []


def test_last_definition_and_large_discovered_function_list_are_static_and_bounded():
    validate_tool_entrypoint('async def run(payload): return {}\ndef run(payload): return {}\n')
    source = '\n'.join('def f'+str(i)+'_'+'x'*200+'(): pass' for i in range(100))
    with pytest.raises(ValueError) as caught:
        validate_tool_entrypoint(source)
    assert '(+92 omitted)' in str(caught.value) and len(str(caught.value)) < 1500


@pytest.mark.parametrize('background', [False, True])
def test_minimum_prompt_preserves_scoped_contract_repair_and_input_projection(tmp_path, background):
    tools, backend, _, costs = controller(tmp_path, [result({'checked': True})],
        background=background, max_prompt_chars=8000)
    current = target_fixture('zero', huge=True)
    current.task_prompt = 'REQUIRED INTERFACE: candidate_solve, candidate_report.\n' + 'Task data ' * 6000
    current.directive = 'REFINEMENT: keep candidate family.\n' + 'Context ' * 6000
    before = deepcopy(current.research_targets)
    llm = RecordingLLM([observed_actions()[0], action()])
    episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    assert len(llm.prompts) == 2 and len(backend.calls) == 1 and costs == [.03, .03]
    for prompt in llm.prompts:
        assert len(prompt) <= 8000
        assert 'module-level synchronous def run(payload)' in prompt
        assert 'STUDY-SUBJECT DATA' in prompt and 'rules apply to candidates, not the tool API' in prompt
        assert 'run(payload) returns exactly five fields' in prompt
        assert_view_matches_inputs(rendered_view(prompt), backend.calls[0]['inputs'])
        if background:
            assert 'COMPUTE JOB CONTROLS:' in prompt and "payload['execution_budget']" in prompt
    assert 'expected module-level synchronous def run(payload)' in llm.prompts[1]
    assert current.research_targets == backend.calls[0]['inputs']['research_targets'] == before
    assert len(episode['registered_ids']) == 1


def test_two_candidate_style_replies_exhaust_budget_without_stopping_G(tmp_path, monkeypatch):
    tasks = tmp_path/'tasks'; task = task_fixture(tasks); provider = ResearchProvider(concurrent=1)
    evaluations, _ = evaluator(monkeypatch)
    engine = build_engine(config(task, tmp_path/'capacity', workers=1, num_islands=1),
        str(tasks), str(tmp_path/'search'), python=sys.executable, llm=provider)
    tools, backend, _, costs = controller(tmp_path/'tools', [])
    try:
        engine.log.run_start(); engine.seed(portfolio=False)
        engine.research_tools = tools; engine.cfg['research_tools'] = tools.config
        engine._parallel_io = IO(); engine._parallel_io.current = 'G:fixture'
        engine._parallel_io.sequences = {}; engine._parallel_wave = 1
        step = engine._prepare_step(0, spawn=False)
        step['snap'].tool_episode_plan = prepare(tools, step['snap'])
        proposals = RecordingLLM(observed_actions())
        class Transport:
            def generate(self, prompt, model, temperature=None, **options):
                return (proposals if prompt.startswith(PROMPT_MARKER) else provider).generate(
                    prompt, model, temperature, **options)
        outcome = engine._execute_step(step, llm=Transport())
        assert outcome.ok and provider.roles == ['G'] and len(evaluations) == 2
        assert len(proposals.prompts) == 2 and costs == [.03, .03]
        assert backend.calls == [] and outcome.meta['research_tools']['registered_ids'] == []
    finally:
        engine.close()
