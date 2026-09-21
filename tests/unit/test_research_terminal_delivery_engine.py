"""Terminal model interpretation reaches real ordinary G and survives resume.

The actual engine/controller/WAL run. Only provider and isolated execution leaves
are fakes; no generated candidate or tool code executes.
"""
from copy import deepcopy
import hashlib
import json
import sys

import pytest

from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from sciel.hypothesis.rendering import decode_evidence
from sciel.methods.models import content_hash
from test_parallel_engine import ResearchProvider, config, evaluator, task_fixture
from test_research_tool_controller import IMAGE
from test_research_tool_result_flow import SavedExecutor, result
from test_sequential_research_episode import decision, finish


INTERPRETATION = 'INVALID_OUTPUT: the reported objective is not a feasible candidate; the experiment is inconclusive.'
NEXT = 'Repair feasibility checks before testing a new starting state.'


def delivered_cards(prompt, research_enabled):
    marker = 'RESEARCH_CONTEXT_JSON:\n' if research_enabled else 'RESEARCH_CHECK_RESULTS_JSON:\n'
    value = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])[0]
    return decode_evidence(value)['research_cells_context']['cards'] if research_enabled else value


@pytest.mark.parametrize('research_enabled', [False, True])
def test_terminal_interpretation_reaches_ordinary_prompt_and_retained_prompt_after_resume(
        tmp_path, monkeypatch, research_enabled):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    evaluations, _ = evaluator(monkeypatch)
    backend = SavedExecutor([result({'reported_objective': 777, 'feasible': False})])
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor', lambda image: backend)

    class Provider(ResearchProvider):
        def __init__(self):
            super().__init__(concurrent=1)
            self.planner_prompts = []

        def generate(self, prompt, model, temperature=None, **options):
            if not prompt.startswith('SCIEL_RESEARCH_TOOL_EPISODE'):
                return super().generate(prompt, model, temperature, **options)
            self.planner_prompts.append(prompt)
            if len(self.planner_prompts) == 1:
                action = decision()
            elif len(self.planner_prompts) == 2:
                assert 'reported_objective' in prompt
                action = finish(conclusion='inconclusive', reason=INTERPRETATION, next_decision=NEXT)
            else:
                pytest.fail('Retained delivery must not repeat paid research work')
            return LLMResponse(json.dumps(action), model, .03, 20, 20, finish_reason='stop')

    cfg = config(task, tmp_path / 'capacity', workers=1, num_islands=1, max_iterations=1,
        research={'enabled': research_enabled, 'reflection_interval': 0, 'max_aux_calls': None},
        research_tools={'enabled': True, 'image_id': IMAGE, 'interval': 1,
            'proactive_every_attempts': 1, 'max_scientific_steps': 3,
            'max_requests': 2, 'max_context_chars': 4096, 'share_between_islands': False})
    provider = Provider()
    directory = tmp_path / 'run'
    engine = build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
    first = engine.run()
    assert first['iteration_no'] == 1
    assert len(backend.calls) == 1 and len(provider.planner_prompts) == 2
    first_prompt = next(prompt for prompt, role in zip(provider.calls, provider.roles) if role == 'G')
    first_card = delivered_cards(first_prompt, research_enabled)[0]
    assert first_card['schema'] == 'sciel-research-check/5'
    assert first_card['comparison_assessment']['status'] == 'unavailable'
    assert first_card['comparison_assessment']['receipt_id'] == first_card['source']['receipt_id']
    closure = first_card['decision']
    assert closure['conclusion'] == 'inconclusive'
    assert closure['interpretation'] == INTERPRETATION and closure['next_decision'] == NEXT
    assert first_card['observations']['status'] == 'success'  # Execution is not physical validity.
    assert first_card['question'] == decision()['question']
    assert first_card['source']['receipt_id'] in closure['receipt_ids']
    saved = deepcopy(engine.research_tools.state())
    stored = next(iter(saved['lifecycle']['cards'].values()))
    assert stored['decision'] == closure
    assert engine.db.pop_best < 1  # The tool's objective 777 is never canonical reward.

    continued_cfg = {**cfg, 'max_iterations': 2}
    restored = build_engine(continued_cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
    checkpoint.load(restored, str(directory / 'checkpoint.json'))
    assert restored.research_tools.state() == saved
    # Exercise subsequent ordinary retrieval without scheduling another episode.
    monkeypatch.setattr(restored.research_tools, 'prepare', lambda *args, **kwargs: None)
    completed = restored.run()
    prompts = [p for p, role in zip(provider.calls, provider.roles) if role == 'G']
    assert len(prompts) == 2 and len(backend.calls) == 1 and len(provider.planner_prompts) == 2
    second_card = delivered_cards(prompts[1], research_enabled)[0]
    assert second_card['decision'] == closure
    assert second_card['source'] == first_card['source']
    assert second_card['observations'] == first_card['observations']
    assert completed['iteration_no'] == 2 and len(evaluations) == 3
    assert completed['cost_total'] == pytest.approx(.14 if research_enabled else .12)

    trace = [json.loads(line) for line in (directory / 'artifacts/llm_calls.jsonl').read_text().splitlines()]
    prompt_hashes = {hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts}
    ordinary = [row for row in trace if row.get('event') == 'request' and row.get('prompt_sha256') in prompt_hashes]
    assert len(ordinary) == 2
    for request, prompt in zip(ordinary, prompts):
        wire = (directory / 'artifacts/llm_calls' / (request['call_id'] + '.prompt.txt')).read_bytes()
        assert hashlib.sha256(wire).hexdigest() == request['prompt_sha256']
        assert wire.decode() == prompt
    if research_enabled:
        followups = restored.research_tools.state()['followups']['records']
        assert len(followups) == 2
        for record, card, prompt in zip(followups, [first_card, second_card], prompts):
            exposure = record['exposures']['cards'][0]
            assert exposure['card_sha256'] == content_hash(card)
            assert exposure['observations_sha256'] == content_hash(card['observations'])
            assert record['exposures']['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
            assert record['model_use'] == record['causal_benefit'] == 'unknown'


def test_interleaved_first_evolution_runs_real_research_cell_before_all_fresh_starts(tmp_path, monkeypatch):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    evaluator(monkeypatch)
    backend = SavedExecutor([result({'reported_objective': 777, 'feasible': False})])
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor', lambda image: backend)

    class Provider(ResearchProvider):
        def __init__(self):
            super().__init__(concurrent=1)
            self.research_attempt_cursors = []

        def generate(self, prompt, model, temperature=None, **options):
            if not prompt.startswith('SCIEL_RESEARCH_TOOL_EPISODE'):
                return super().generate(prompt, model, temperature, **options)
            self.research_attempt_cursors.append(deepcopy(engine._portfolio_bootstrap))
            action = decision() if len(self.research_attempt_cursors) == 1 else finish(
                conclusion='inconclusive', reason=INTERPRETATION, next_decision=NEXT)
            return LLMResponse(json.dumps(action), model, .03, 20, 20, finish_reason='stop')

    cfg = config(task, tmp_path / 'capacity', workers=1, num_islands=3, max_iterations=1,
        portfolio_seed=True, research={'enabled': False},
        research_tools={'enabled': True, 'image_id': IMAGE, 'interval': 1,
            'proactive_every_attempts': 1, 'max_scientific_steps': 3, 'max_requests': 2,
            'max_context_chars': 4096, 'share_between_islands': False})
    cfg['parallel_search']['portfolio_schedule'] = 'interleaved'
    provider = Provider()
    directory = tmp_path / 'run'
    engine = build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
    completed = engine.run()
    assert completed['iteration_no'] == 1 and len(backend.calls) == 1
    assert provider.research_attempt_cursors == [dict(schema='sciel-portfolio-bootstrap/1', next_island=1, islands=3)] * 2
    ordinary = [prompt for prompt in provider.calls if 'RESEARCH_CHECK_RESULTS_JSON:\n' in prompt]
    assert len(ordinary) == 1
    assert delivered_cards(ordinary[0], False)[0]['decision']['interpretation'] == INTERPRETATION
    saved = json.loads((directory / 'checkpoint.json').read_text())
    assert saved['portfolio_bootstrap'] == dict(schema='sciel-portfolio-bootstrap/1', next_island=1, islands=3)
    assert len(saved['research_tools']['lifecycle']['research_state']) == 1
    rows = [json.loads(line) for line in (directory / 'sciel_events.jsonl').read_text().splitlines()]
    kinds = [row['wave_kind'] for row in rows if row.get('policy') == 'parallel_search'
             and row.get('action') == 'wave_complete']
    assert kinds == ['canonical_seed', 'portfolio', 'evolution']
