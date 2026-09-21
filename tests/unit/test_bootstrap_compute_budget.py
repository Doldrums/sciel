"""Bootstrap prompts carry task-derived limits; all model/evaluator work is local."""
import json
from pathlib import Path
import sys

import pytest
import yaml

from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.engine.parallel import _prepare_seed
from sciel.engine.prompt.build import compose_directive
from sciel.engine.worker import run_worker
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult


BASE = 'def run_search():\n    return 0\n'
CHILD = 'def run_search():\n    return 1\n'
ROUTES = ['legacy', 'novelty', 'research', 'parallel', 'parallel_research']


class Provider(MockLLM):
    def __init__(self, *, duplicate=False):
        super().__init__()
        self.prompts = []
        self.duplicate = duplicate

    def generate(self, prompt, model, temperature=None, **options):
        self.prompts.append(prompt)
        code = BASE if self.duplicate else CHILD
        hypothesis = dict(hypothesis='Change the returned fixture value.',
            predicted_low=0., predicted_high=1., reasoning='A local fixture prediction.', confidence=.5)
        response = 'HYPOTHESIS_JSON: ' + json.dumps(hypothesis) + '\n```python\n' + code + '```'
        return LLMResponse(response, model, .03, 20, 20)


def setup(tmp_path, monkeypatch, route, limits, *, duplicate=False):
    smoke, full, per_run, trials = limits
    task_dir = tmp_path/'tasks'/'fixture'/'search'
    task_dir.mkdir(parents=True)
    task = dict(suite='fixture', name='search', origin='math', primary_metric='quality',
        tiers=['smoke', 'full'], stochastic=False, confirm_n=1,
        smoke_timeout=smoke, full_timeout=full, per_run_timeout=per_run)
    (task_dir/'task.yaml').write_text(yaml.safe_dump(task))
    (task_dir/'initial_program.py').write_text(BASE)
    (task_dir/'evaluator.py').write_text('def evaluate(program):\n    assert hasattr(program, "run_search")\n')
    provider = Provider(duplicate=duplicate)
    cfg = dict(task='fixture/search', budget=None, max_iterations=1, schedule_horizon=10,
        num_islands=1, inflight_cap=1, dup_cap=4, random_seed=23,
        schedule_policy='round_robin', milestone_fracs=[], embed_grid=False, embed_niche=False,
        portfolio_seed=False, dynamic_islands=False, idea_island=False, crossover=False,
        policy={'enabled': False}, llm={'mock': True},
        candidate_novelty={'enabled': route != 'legacy', 'max_resamples': 2, 'neighbors': 3},
        research={'enabled': 'research' in route, 'max_aux_calls': None})
    evaluated = []

    def evaluate(spec, path, **kwargs):
        # The patch changes model information, never the evaluator's limits.
        assert (spec.smoke_timeout, spec.full_timeout, spec.per_run_timeout) == (smoke, full, per_run)
        code = Path(path).read_text()
        evaluated.append(code)
        score = .1 if code == BASE else .2
        return CascadeOutcome(EvalResult(True, score, {'quality': score}, timing=.01), score, 'full', False)

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    engine = build_engine(cfg, str(tmp_path/'tasks'), str(tmp_path/'run'), python=sys.executable, llm=provider)
    engine.seed()
    engine._observed_trials = trials
    # The test targets generation payloads; conclusions have their own protocol.
    monkeypatch.setattr(engine, '_complete_research_step', lambda: None)
    return engine, provider, evaluated


def propose(engine, route, origin='portfolio'):
    if route.startswith('parallel'):
        snapshot = _prepare_seed(engine, 0)['snap']
        return run_worker('full_rewrite', snapshot, engine.llm, engine._evaluate,
            dup_cap=1, base_temp=.9, novelty_check=engine._candidate_novelty_callback(0),
            research=engine.research)
    return engine._paradigm_child(BASE, 'a generic global search', 0, origin)


@pytest.mark.parametrize('route', ROUTES)
@pytest.mark.parametrize('limits,expected', [
    ((17., 120., None, 1), 17.),  # Smoke binds.
    ((23., 90., None, 9), 10.),  # Full stage contains multiple candidate trials.
    ((50., 150., 7., 1), 7.),    # Task-specific internal per-run cap binds.
])
def test_each_bootstrap_route_delivers_binding_budget_to_model(tmp_path, monkeypatch, route, limits, expected):
    engine, provider, evaluated = setup(tmp_path, monkeypatch, route, limits)
    try:
        propose(engine, route)
        assert len(provider.prompts) == 1 and len(evaluated) == 2
        prompt = provider.prompts[0]
        assert compose_directive(expected, None, plateau=False) in prompt
        assert 'EXPLICIT deadline guard' in prompt
        assert 'safety margin' in prompt
        assert 'REFINEMENT:' not in prompt and 'STAGNATION:' not in prompt
        assert ('CONSTRUCTIVE/GEOMETRIC' if route.startswith('parallel') else 'a generic global search') in prompt
    finally:
        engine.close()


@pytest.mark.parametrize('route', ROUTES[1:])
def test_budget_survives_every_novelty_retry_without_more_evaluations(tmp_path, monkeypatch, route):
    engine, provider, evaluated = setup(tmp_path, monkeypatch, route, (23., 90., None, 9), duplicate=True)
    try:
        propose(engine, route)
        assert len(provider.prompts) == 3  # Original request + two permitted retries.
        assert evaluated == [BASE]  # Duplicates never dispatch the evaluator.
        budget = compose_directive(10., None, plateau=False)
        assert all(budget in prompt for prompt in provider.prompts)
    finally:
        engine.close()


@pytest.mark.parametrize('route', ['legacy', 'novelty', 'research'])
def test_dynamic_spawn_uses_the_same_binding_budget(tmp_path, monkeypatch, route):
    engine, provider, _ = setup(tmp_path, monkeypatch, route, (50., 150., 7., 1))
    try:
        propose(engine, route, origin='spawn')
        assert compose_directive(7., None, plateau=False) in provider.prompts[0]
    finally:
        engine.close()
