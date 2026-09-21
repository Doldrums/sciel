"""A completed research trial yields to real ordinary search and durable replay.

The Engine, ResearchTools, scientific journal and cooperative WAL are real.
Provider/evaluator/isolated execution leaves return fixed fixture data; generated
candidate and tool source is never imported or executed.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from sciel.methods.models import canonical_bytes, content_hash
from sciel.research_tools.controller import ResearchTools
from test_parallel_engine import ResearchProvider, config, evaluator, events, task_fixture
from test_research_terminal_delivery_engine import delivered_cards
from test_research_tool_controller import IMAGE
from test_research_tool_result_flow import SavedExecutor, result
from test_sequential_research_episode import current_receipt, decision, finish


SAVED_STATE = {
    'incumbent': [[0.0, -0.0, 1.23456789012345], [1e-30, -2, 7]],
    'seed': 149,
    'rng_state': [4294967295, 0, 19],
    'conditions': {'precision': 'binary64', 'completed_evaluations': 11},
}
OBSERVATIONS = {'continuation_state': SAVED_STATE, 'measured_gain': 0.125,
                'fixture_marker': 'preserved research observation'}


class YieldProvider(ResearchProvider):
    def __init__(self, actions):
        super().__init__(concurrent=1)
        self.actions = list(actions)
        self.planner_prompts = []
        self.physical_order = []

    def generate(self, prompt, model, temperature=None, **options):
        if not prompt.startswith('SCIEL_RESEARCH_TOOL_EPISODE'):
            self.physical_order.append('ordinary')
            return super().generate(prompt, model, temperature, **options)
        self.physical_order.append('planner')
        self.planner_prompts.append(prompt)
        index = len(self.planner_prompts) - 1
        assert index < len(self.actions), 'Research issued another planner before yielding to ordinary G'
        action = self.actions[index]
        if callable(action):
            action = action(prompt)
        return LLMResponse(json.dumps(action), model, .03, 20, 20, finish_reason='stop')


def setup_run(tmp_path, monkeypatch, *, actions, outputs, workers=1,
              iterations=1, research_enabled=False):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    evaluations, _ = evaluator(monkeypatch)
    backend = SavedExecutor(outputs)
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor', lambda image: backend)
    cfg = config(task, tmp_path / 'capacity', workers=workers, num_islands=workers,
        pop_cap=1, max_iterations=iterations,
        research={'enabled': research_enabled, 'reflection_interval': 0, 'max_aux_calls': None},
        research_tools={'enabled': True, 'image_id': IMAGE, 'interval': 1,
            'proactive_every_attempts': 1, 'max_scientific_steps': 3,
            'yield_after_success': True, 'max_requests': 2,
            'max_context_chars': 4096, 'share_between_islands': False})
    provider = YieldProvider(actions)
    directory = tmp_path / 'run'

    def build(*, resume=False):
        engine = build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
        assert isinstance(engine.research_tools, ResearchTools)
        if resume:
            checkpoint.load(engine, str(directory / 'checkpoint.json'))
        return engine

    return build, provider, backend, evaluations, directory


def research_state(engine):
    states = engine.research_tools.state()['lifecycle']['research_state']
    assert len(states) == 1
    return next(iter(states.items()))


def generation_prompts(provider):
    return [prompt for prompt, role in zip(provider.calls, provider.roles) if role == 'G']


def assert_no_pending_physical_calls(directory):
    for marker in (directory / '.parallel').rglob('dispatched.json'):
        assert marker.with_name('receipt.json').is_file()


@pytest.mark.parametrize('research_enabled', [False, True])
def test_first_success_reaches_ordinary_generation_and_commits_before_another_planner(
        tmp_path, monkeypatch, research_enabled):
    build, provider, backend, evaluations, directory = setup_run(
        tmp_path, monkeypatch, actions=[decision()], outputs=[result(OBSERVATIONS)],
        research_enabled=research_enabled)
    engine = build()
    completed = engine.run()

    assert completed['iteration_no'] == completed['B_evals'] == 1
    assert len(provider.planner_prompts) == len(backend.calls) == 1
    assert len(evaluations) == 2  # Canonical seed and ordinary generated candidate.
    assert provider.physical_order[:2] == ['planner', 'ordinary']
    assert completed['cost_total'] == pytest.approx(.07 if research_enabled else .06)
    prompts = generation_prompts(provider)
    assert len(prompts) == 1
    card = delivered_cards(prompts[0], research_enabled)[0]
    assert card['observations']['status'] == 'success'
    assert 'preserved research observation' in card['observations']['output_excerpt']
    assert 'measured_gain' in card['observations']['output_excerpt']
    assert card['source']['invocation_id'] == backend.receipts[0]['invocation_id']
    assert card['question'] == decision()['question']
    assert not card.get('decision')  # A scheduler yield is not a scientific closure.
    assert engine.db.pop_best < 1  # Auxiliary observations are not canonical reward.

    cell_id, state = research_state(engine)
    assert state['status'] == 'open' and state['conclusion'] is None
    assert state['scientific_steps'] == 1 and state['known_llm_cost'] == pytest.approx(.03)
    assert state['resources']['cpu_seconds'] == .01
    saved = json.loads((directory / 'checkpoint.json').read_text())
    assert saved['research_tools'] == engine.research_tools.state()
    assert saved['research_tools']['lifecycle']['cells']['cells'][cell_id]['active'] is False
    assert state['steps'][0]['receipt_id'] == card['source']['receipt_id']
    rows = events(directory)
    assert len([row for row in rows if row.get('action') == 'job_committed']) == 1
    assert_no_pending_physical_calls(directory)


def test_later_opportunity_resumes_same_cell_on_changed_parent_with_exact_saved_state(
        tmp_path, monkeypatch):
    first_receipt = None
    first_cell = None

    def continue_stage(prompt):
        # The second planner cannot precede the first ordinary wave's commit.
        saved = json.loads((directory / 'checkpoint.json').read_text())
        assert saved['iteration_no'] == 1
        assert list(saved['research_tools']['lifecycle']['research_state']) == [first_cell]
        assert current_receipt(prompt) == first_receipt
        assert 'preserved research observation' in prompt
        return decision(code=decision()['code'] + '# next saved-data stage\n',
                        input_receipt_id=first_receipt)

    build, provider, backend, evaluations, directory = setup_run(tmp_path, monkeypatch,
        actions=[decision(), continue_stage,
                 finish(conclusion='inconclusive', next_decision='Try another measured starting state.')],
        outputs=[result(OBSERVATIONS), result({'measured_gain': 0})], iterations=3)
    first = build()
    first._stop_at = 1
    stopped = first.run()
    assert stopped['reason'] == 'paused' and stopped['iteration_no'] == 1
    first_cell, initial = research_state(first)
    first_receipt = initial['steps'][0]['receipt_id']
    persisted = deepcopy(first.research_tools.state())
    physical_before = (len(provider.planner_prompts), len(backend.calls), len(evaluations))

    second = build(resume=True)
    assert second.research_tools.state() == persisted
    assert (len(provider.planner_prompts), len(backend.calls), len(evaluations)) == physical_before
    second._stop_at = 2
    stopped = second.run()
    assert stopped['reason'] == 'paused' and stopped['iteration_no'] == 2
    current_cell, state = research_state(second)
    assert current_cell == first_cell and state['status'] == 'open'
    assert state['scientific_steps'] == 2 and state['known_llm_cost'] == pytest.approx(.06)
    assert state['resources']['cpu_seconds'] == .02
    assert len(provider.planner_prompts) == len(backend.calls) == 2
    prior, current = [call['inputs'] for call in backend.calls]
    assert prior['research_targets']['selected']['program_id'] != current['research_targets']['selected']['program_id']
    assert prior['parent_source'] != current['parent_source']
    stage = current['research_stage']
    assert stage['prior_receipt_id'] == first_receipt
    assert stage['target_binding']['candidate_id'] == prior['research_targets']['selected']['program_id']
    assert stage['target_binding']['candidate_id'] != current['research_targets']['selected']['program_id']
    assert canonical_bytes(stage['observations']) == canonical_bytes(OBSERVATIONS)
    assert canonical_bytes(current['continuation_state']) == canonical_bytes(SAVED_STATE)
    assert stage['observations_sha256'] == content_hash(OBSERVATIONS)
    assert stage['prior_input_sha256'] == content_hash(prior)
    assert stage['prior_output_sha256'] == backend.receipts[0]['output_sha256']
    assert stage['arguments'] == prior['arguments']

    third = build(resume=True)
    finished = third.run()
    final_cell, state = research_state(third)
    assert final_cell == first_cell and state['status'] == 'closed'
    assert state['scientific_steps'] == 2 and state['known_llm_cost'] == pytest.approx(.09)
    assert len(backend.calls) == 2 and len(provider.planner_prompts) == 3
    assert finished['iteration_no'] == finished['B_evals'] == 3
    assert finished['cost_total'] == pytest.approx(.18)
    assert len(evaluations) == 4 and len(generation_prompts(provider)) == 3
    assert_no_pending_physical_calls(directory)


@pytest.mark.parametrize('workers', [1, 2])
def test_partial_wave_pause_reuses_success_receipts_before_committing_ordinary_progress(
        tmp_path, monkeypatch, workers):
    build, provider, backend, evaluations, directory = setup_run(tmp_path, monkeypatch,
        actions=[decision()], outputs=[result(OBSERVATIONS)], workers=workers,
        iterations=workers)
    execute = backend.execute

    def stop_after_tool_receipt(*args, **kwargs):
        receipt = execute(*args, **kwargs)
        (directory / 'STOP').touch()
        return receipt

    monkeypatch.setattr(backend, 'execute', stop_after_tool_receipt)
    paused = build().run()
    assert paused['reason'] == 'paused'
    assert paused['iteration_no'] == paused['B_evals'] == 0
    assert len(provider.planner_prompts) == len(backend.calls) == 1
    saved = json.loads((directory / 'checkpoint.json').read_text())
    assert saved['research_tools']['lifecycle']['research_state'] == {}
    assert saved['research_tools']['registered'] == []
    assert saved['cost_total'] == 0  # Paid calls survive in WAL, not in the rolled-back baseline.
    assert not (directory / 'STOP').exists()
    assert_no_pending_physical_calls(directory)
    research_prompts = list(provider.planner_prompts)
    tool_receipts = deepcopy(backend.receipts)
    evaluated_before = list(evaluations)

    restored = build(resume=True)
    completed = restored.run()
    assert provider.planner_prompts == research_prompts and backend.receipts == tool_receipts
    assert len(backend.calls) == 1
    assert evaluations[:len(evaluated_before)] == evaluated_before
    assert len(evaluations) == len(set(evaluations)) == workers + 1
    assert len(generation_prompts(provider)) == workers
    assert completed['iteration_no'] == completed['B_evals'] == workers
    assert completed['cost_total'] == pytest.approx(.03 * (workers + 1))
    _, state = research_state(restored)
    assert state['status'] == 'open' and len(state['steps']) == state['scientific_steps'] == 1
    assert state['known_llm_cost'] == pytest.approx(.03)
    assert state['resources']['cpu_seconds'] == .01
    tool_state = restored.research_tools.state()
    assert len(tool_state['committed_episodes']) == len(tool_state['registered']) == 1
    assert sum(sum(by_island.values()) for by_island in tool_state['uses'].values()) == 1
    committed = [row['job_id'] for row in events(directory) if row.get('action') == 'job_committed']
    assert committed == [f'G:{index}' for index in range(1, workers + 1)]
    assert_no_pending_physical_calls(directory)
