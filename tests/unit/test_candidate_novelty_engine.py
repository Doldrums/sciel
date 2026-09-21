"""Whole engine novelty routing, without model calls or candidate execution."""
import json
from pathlib import Path
import sys

import pytest

from sciel.engine.factory import build_engine
from sciel.engine import checkpoint
from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.engine.llm.client import LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult


ROOT = Path(__file__).resolve().parents[2]


def settings(**overrides):
    cfg = dict(task='test/toy_quadratic', budget=None, max_iterations=1,
        schedule_horizon=10, num_islands=1, inflight_cap=1, dup_cap=4,
        random_seed=23, schedule_policy='round_robin', milestone_fracs=[],
        embed_grid=False, embed_niche=False, portfolio_seed=False,
        dynamic_islands=False, idea_island=False, crossover=False,
        op_fixed='full_rewrite', policy={'enabled': False}, llm={'mock': True},
        candidate_novelty={'enabled': True, 'max_resamples': 2, 'neighbors': 3})
    cfg.update(overrides)
    return cfg


class Provider(MockLLM):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.prompts = []

    def generate(self, prompt, model, temperature=None, **kwargs):
        self.prompts.append(prompt)
        text = '```python\n' + self.source + '\n```'
        return LLMResponse(text, model, .03, 20, 20)


def outcome(ok=True, score=.1):
    return CascadeOutcome(EvalResult(ok=ok, score=score, cases={}, timing=.01),
                          score, 'full', False)


def construct(path, cfg, provider):
    return build_engine(cfg, str(ROOT / 'tasks'), str(path),
                        python=sys.executable, llm=provider)


def fake_evaluator(monkeypatch, results=None):
    calls = []
    pending = iter(results) if results else None

    def evaluate(task, path, **kwargs):
        calls.append(Path(path).read_text())
        return next(pending) if pending else outcome()

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    return calls


def test_cross_island_seed_duplicates_never_reach_evaluator(tmp_path, monkeypatch):
    provider = Provider('')
    engine = construct(tmp_path, settings(num_islands=3), provider)
    provider.source = Path(engine.task.initial_program_path).read_text()
    calls = fake_evaluator(monkeypatch)
    result = engine.run()
    assert len(calls) == 1
    assert len(provider.prompts) == 3
    assert result['B_evals'] == 0
    assert result['cost_total'] == pytest.approx(.09)
    assert len(engine.db._all) == 3
    assert engine.candidate_novelty.inspect(provider.source)['accept'] is False
    events = [json.loads(row) for row in (tmp_path / 'sciel_events.jsonl').read_text().splitlines()]
    rejections = [r for r in events if r.get('policy') == 'candidate_novelty'
                  and r.get('action') == 'reject']
    assert len(rejections) == 3
    assert all(r['matched_program_id'].startswith('p') for r in rejections)


def test_portfolio_uses_same_registry_and_retry_budget(tmp_path, monkeypatch):
    provider = Provider('')
    engine = construct(tmp_path, settings(num_islands=3, portfolio_seed=True), provider)
    provider.source = Path(engine.task.initial_program_path).read_text()
    calls = fake_evaluator(monkeypatch)
    try:
        engine.seed()
    finally:
        engine.close()
    assert len(calls) == 1
    assert len(provider.prompts) == 9
    assert engine.cost_total == pytest.approx(.27)
    assert len(engine.db._all) == 3
    assert engine._portfolio_bootstrap['next_island'] == 3


def test_failed_candidate_can_be_remeasured_without_mutation_credit(tmp_path, monkeypatch):
    provider = Provider('def run_search():\n    return 312.0\n')
    engine = construct(tmp_path, settings(max_iterations=2), provider)
    calls = fake_evaluator(monkeypatch, [outcome(), outcome(False, 0), outcome(True, .8)])
    result = engine.run()
    assert len(calls) == 3
    assert len(provider.prompts) == 2
    assert engine.n_remeasurements == 1
    assert engine.n_remeasurement_successes == 1
    assert engine.record_stats()['records_expected'] == 0
    assert engine.n_events == engine.n_records == engine.n_postmortems == 0
    assert result['cost_total'] == pytest.approx(.06)
    assert engine.db.get('p2').origin == 'remeasurement'
    assert engine.db.get('p2').parent_id is None
    assert engine.candidate_novelty.inspect(provider.source)['accept'] is False
    saved = json.loads((tmp_path / 'checkpoint.json').read_text())
    assert saved['n_remeasurements'] == 1
    events = [json.loads(row) for row in (tmp_path / 'sciel_events.jsonl').read_text().splitlines()]
    measured = [r for r in events if r.get('policy') == 'evaluation']
    assert [r['remeasurement'] for r in measured] == [False, False, True]
    assert events[-1]['event'] == 'run_end'


def test_explicit_remeasurement_is_separate_from_mutation_gate(tmp_path, monkeypatch):
    engine = construct(tmp_path, settings(), Provider(''))
    calls = fake_evaluator(monkeypatch)
    source = 'def run_search():\n    return 4\n'
    try:
        engine._evaluate(source)
        with pytest.raises(ValueError, match='before evaluator'):
            engine._evaluate(source)
        engine._evaluate(source, remeasurement=True)
        assert len(calls) == 2
        assert engine.n_remeasurements == 1
    finally:
        engine.close()


def test_failed_seed_does_not_hide_later_successful_measurement(tmp_path, monkeypatch):
    provider = Provider('')
    engine = construct(tmp_path, settings(), provider)
    provider.source = Path(engine.task.initial_program_path).read_text()
    calls = fake_evaluator(monkeypatch, [outcome(False, 0), outcome(True, .8)])
    engine.run()
    assert len(calls) == 2 and engine.n_remeasurements == 1
    assert engine.db.get('p1').score == 0
    assert engine.db.get('p2').origin == 'remeasurement'
    assert engine.db.get('p2').parent_id is None
    assert engine.db.pop_best == .8
    assert engine.n_records == engine.n_events == engine.n_postmortems == 0


def test_evaluator_exception_does_not_permanently_ban_source(tmp_path, monkeypatch):
    engine = construct(tmp_path, settings(), Provider(''))
    source = 'def run_search():\n    return 4\n'
    def broken(*args, **kwargs):
        raise RuntimeError('offline evaluator fixture failure')
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', broken)
    try:
        with pytest.raises(RuntimeError, match='fixture failure'):
            engine._evaluate(source)
        decision = engine.candidate_novelty.inspect(source)
        assert decision['accept'] and decision['remeasurement']
        calls = fake_evaluator(monkeypatch)
        engine._evaluate(source)
        assert len(calls) == 1
    finally:
        engine.close()


def test_paid_generation_cost_survives_evaluator_exception(tmp_path, monkeypatch):
    source = 'def run_search():\n    return 312.0\n'
    provider = Provider(source)
    engine = construct(tmp_path, settings(), provider)
    calls = []
    def evaluate(task, path, **kwargs):
        calls.append(Path(path).read_text())
        if len(calls) > 1:
            raise RuntimeError('offline failure after paid generation')
        return outcome()
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    with pytest.raises(CampaignStopped):
        engine.run()
    saved = json.loads((tmp_path / 'checkpoint.json').read_text())
    assert saved['cost_total'] == pytest.approx(.03)
    assert engine.candidate_novelty.inspect(source)['remeasurement'] is True


def test_paid_duplicate_cost_survives_undispatched_retry(tmp_path, monkeypatch):
    class StopBeforeRetry(Provider):
        def generate(self, *args, **kwargs):
            if self.prompts:
                raise RequestNotDispatched('offline retry not sent')
            return super().generate(*args, **kwargs)
    provider = StopBeforeRetry('')
    engine = construct(tmp_path, settings(), provider)
    provider.source = Path(engine.task.initial_program_path).read_text()
    calls = fake_evaluator(monkeypatch)
    with pytest.raises(RequestNotDispatched):
        engine.run()
    saved = json.loads((tmp_path / 'checkpoint.json').read_text())
    assert len(calls) == 1
    assert saved['cost_total'] == pytest.approx(.03)
    assert not saved.get('_accounting_stopped_reason')


def test_measurement_ids_are_unique_across_segments(tmp_path, monkeypatch):
    engine = construct(tmp_path, settings(), Provider(''))
    calls = fake_evaluator(monkeypatch)
    try:
        engine.log.run_start()
        engine.log.decision(policy='fixture', action='setup')
        engine._evaluate('def run_search(): return 10\n')
        run_id = engine.log.run_id
        engine.log.run_start(restored_checkpoint={'provenance': {'run_id': run_id}})
        engine._evaluate('def run_search(): return 20\n')
        ids = [record['id'] for record in engine.candidate_novelty.state()['records']]
        assert len(calls) == len(set(ids)) == 2
        assert engine.log.run_id == run_id
    finally:
        engine.close()


@pytest.mark.parametrize('duplicate', [True, False])
def test_real_engine_resume_preserves_registry_and_cost(tmp_path, monkeypatch, duplicate):
    first_source = 'def run_search():\n    return 312.0\n'
    cfg = settings(max_iterations=2)
    class PauseAfterGeneration(Provider):
        def generate(self, *args, **kwargs):
            first.request_stop()
            return super().generate(*args, **kwargs)
    first = construct(tmp_path, cfg, PauseAfterGeneration(first_source))
    calls = fake_evaluator(monkeypatch)
    assert first.run()['reason'] == 'paused'
    provider = Provider(first_source if duplicate else first_source.replace('312', '313'))
    resumed = construct(tmp_path, cfg, provider)
    checkpoint.load(resumed, str(tmp_path / 'checkpoint.json'))
    result = resumed.run()
    assert result['reason'] == 'backstop'
    assert len(calls) == (2 if duplicate else 3)
    assert len(provider.prompts) == (3 if duplicate else 1)
    assert result['cost_total'] == pytest.approx(.12 if duplicate else .06)
    assert not resumed.candidate_novelty.inspect(first_source)['accept']
    assert result['n_remeasurements'] == 0
