"""Recovery and action attribution with no API calls or candidate execution."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMError, LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.engine.recovery import prepare_provider_recovery
from sciel.engine.types import NoChild, Result
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult

ROOT = Path(__file__).resolve().parents[2]


def settings(**overrides):
    cfg = dict(task='test/toy_quadratic', budget=None, max_iterations=None,
        schedule_horizon=10, checkpoint_interval=10, num_islands=1, inflight_cap=1,
        dup_cap=1, random_seed=23, schedule_policy='round_robin', milestone_fracs=[],
        embed_grid=False, embed_niche=False, portfolio_seed=False, dynamic_islands=False,
        idea_island=False, crossover=False, op_fixed='full_rewrite',
        policy={'enabled': False, 'plateau_shift': False}, llm={'mock': True})
    cfg.update(overrides)
    return cfg


def construct(path, cfg=None, provider=None):
    return build_engine(cfg or settings(), str(ROOT / 'tasks'), str(path),
                        python=sys.executable, llm=provider or MockLLM())


def outcome(score=.1, ok=True):
    return CascadeOutcome(EvalResult(ok=ok, score=score, cases={}, timing=.01),
                          score, 'full', False)


def rows(path):
    return [json.loads(line) for line in (path / 'sciel_events.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('failed_role', ['conclusion', 'reflection'])
def test_completed_preset_reward_survives_aux_failure_and_is_not_repeated(tmp_path, failed_role):
    class Provider(MockLLM):
        def __init__(self, resumed=False):
            super().__init__()
            self.resumed = resumed
        def generate(self, prompt, model, temperature=None, **options):
            role = ('reflection' if 'REFLECTION_JSON' in prompt else
                    'conclusion' if 'CONCLUSION_JSON' in prompt else 'generation')
            if role == failed_role and not self.resumed:
                raise LLMError(cause=ConnectionError())
            if role != 'generation':
                return LLMResponse('', model, 0., 10, 10)
            if self.resumed:
                engine.request_stop()
            hyp = dict(hypothesis='Improve a fixture value.', predicted_low=0.,
                       predicted_high=1., reasoning='Offline fixture.', confidence=.5)
            text = 'HYPOTHESIS_JSON: ' + json.dumps(hyp) + (
                '\n```python\ndef run_search():\n    return ' + str(3 + self.resumed) + '\n```')
            return LLMResponse(text, model, .03, 10, 10)
    cfg = settings(preset_selection=True, preset_adaptive=True,
        research={'enabled': True, 'max_aux_calls': None, 'reflection_interval': 1})
    engine = construct(tmp_path, cfg, Provider())
    prior = engine.preset_weights.state()
    scores = iter([.1, .7])
    engine._evaluate = lambda code: outcome(next(scores))
    with pytest.raises(CampaignStopped):
        engine.run()
    path = tmp_path / 'checkpoint.json'
    first = json.loads(path.read_bytes())
    assert first['iteration_no'] == first['B_evals'] == first['preset_weights']['updates'] == 1
    assert first['preset_weights']['log_weights'] != prior['log_weights']
    engine = construct(tmp_path, first['config'], Provider(resumed=True))
    engine._evaluate = lambda code: outcome(.7)
    checkpoint.load(engine, str(path))
    prepare_provider_recovery(engine, path, allow_legacy=False)
    assert engine.run()['reason'] == 'paused'
    final = json.loads(path.read_bytes())
    assert final['iteration_no'] == final['preset_weights']['updates'] == 2
    assert final['preset_weights']['log_weights'] == first['preset_weights']['log_weights']
    assert final['cost_total'] == pytest.approx(.06)


@pytest.mark.parametrize('kind', ['no_child', 'eval_failed', 'success'])
def test_last_plateau_action_is_never_credited_to_mode_bandit(tmp_path, monkeypatch, kind):
    cfg = settings(policy={'enabled': True, 'mode_bandit': True, 'plateau_shift': True,
                          'plateau_mix': {'explore': 1.}, 'mode_floor': 1e-6})
    engine = construct(tmp_path, cfg)
    engine.db.add('def run_search(): return 0\n', .1, {}, 0, 'seed')
    engine.iteration_no = 9
    engine.controller.plateau_until[0] = 10
    def worker(op, snapshot, *args, **kwargs):
        if kind == 'no_child':
            return NoChild(0, 'empty_code', {})
        return Result('def run_search(): return 1\n', .7, {}, kind == 'success', 0,
            snapshot.parent_id, 'gen', None,
            {'escalated': False, 'op_used': op, 'dup_attempts': 0},
            .7, 'full', False, '', '', 0., .01)
    monkeypatch.setattr('sciel.engine.loop.run_worker', worker)
    try:
        engine._step(0)
        assert engine.iteration_no == 10
        assert not any(engine.controller.mode_bandit.events.values())
        engine._step(0)
        assert sum(map(len, engine.controller.mode_bandit.events.values())) == 1
    finally:
        engine.close()


@pytest.mark.parametrize('pause_after', [0, 1, 2, 3])
def test_portfolio_resume_finishes_each_direction_once_without_reseeding(tmp_path, pause_after):
    cfg = settings(num_islands=3, portfolio_seed=True)
    calls, evaluations = [], []
    def install(engine, cutoff):
        def evaluate(code):
            evaluations.append(code)
            if cutoff == 0:
                engine.request_stop()
            return outcome()
        def child(initial, paradigm, island, origin):
            calls.append(island)
            engine.cost_total += .03
            if len(calls) >= cutoff:
                engine.request_stop()
            return False
        engine._evaluate = evaluate
        engine._paradigm_child = child
    engine = construct(tmp_path, cfg)
    install(engine, pause_after)
    assert engine.run()['reason'] == 'paused'
    path = tmp_path / 'checkpoint.json'
    initial = json.loads(path.read_bytes())
    assert initial['portfolio_bootstrap']['next_island'] == pause_after
    resumed = construct(tmp_path, initial['config'])
    install(resumed, 3)
    checkpoint.load(resumed, str(path))
    if pause_after == 3:
        resumed.request_stop()
    assert resumed.run()['reason'] == 'paused'
    assert calls == [0, 1, 2]
    assert len(evaluations) == 1  # canonical baseline is not executed again
    assert resumed.cost_total == pytest.approx(.09)
    assert len([r for r in rows(tmp_path) if r.get('action') == 'seed_program']) == 3


@pytest.mark.parametrize('cursor', [True, -1, 4, '1'])
def test_invalid_bootstrap_cursor_rejected_before_restore(tmp_path, cursor):
    engine = construct(tmp_path, settings(num_islands=3, portfolio_seed=True))
    engine._evaluate = lambda code: outcome()
    engine.request_stop()
    engine.run()
    path = tmp_path / 'checkpoint.json'
    saved = json.loads(path.read_bytes())
    saved['portfolio_bootstrap']['next_island'] = cursor
    path.write_text(json.dumps(saved))
    restored = construct(tmp_path, saved['config'])
    with pytest.raises(checkpoint.IdentityMismatch, match='bootstrap cursor'):
        checkpoint.load(restored, str(path))


@pytest.mark.parametrize('portfolio', [False, True])
@pytest.mark.parametrize('error', [LLMError(cause=TimeoutError()), RuntimeError('private-body')])
def test_nonresearch_unknown_dispatch_stops_and_preserves_known_cost(tmp_path, portfolio, error):
    class Provider(MockLLM):
        calls = 0
        def generate(self, prompt, model, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse('', model, .03, 10, 10)
            raise error
    provider = Provider()
    cfg = settings(num_islands=3 if portfolio else 1, portfolio_seed=portfolio, dup_cap=3)
    engine = construct(tmp_path, cfg, provider)
    engine._evaluate = lambda code: outcome()
    with pytest.raises(CampaignStopped, match='outcome and cost are unknown'):
        engine.run()
    saved = json.loads((tmp_path / 'checkpoint.json').read_bytes())
    assert saved['cost_total'] == pytest.approx(.03)
    assert saved['_accounting_stopped_reason']
    assert provider.calls == 2
    receipt = next(r for r in rows(tmp_path) if r.get('policy') == 'cost_accounting')
    assert receipt['generation_returned'] is False and receipt['generation_dispatch'] == 'unknown'
    assert receipt['cost'] is None
    assert 'private-body' not in json.dumps(rows(tmp_path))
    restored = construct(tmp_path, saved['config'], provider)
    checkpoint.load(restored, str(tmp_path / 'checkpoint.json'))
    with pytest.raises(CampaignStopped, match='reconcile'):
        restored.run()
    assert provider.calls == 2


def test_supervisor_requires_flag_for_initial_stop_before_starting_engine(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('supervisor_review', ROOT / 'scripts/continue_live_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'checkpoint.json').write_text(json.dumps({'config': settings()}))
    closed = []
    engine = SimpleNamespace(iteration_no=7, research=SimpleNamespace(stopped_reason='existing stop'),
                             close=lambda: closed.append(True))
    monkeypatch.setenv('OPENROUTER_API_KEY', 'unused-fixture')
    monkeypatch.setattr(sys, 'argv', ['continue_live_run.py', str(tmp_path)])
    monkeypatch.setattr(module, 'build_engine', lambda *args, **kwargs: engine)
    monkeypatch.setattr(module.checkpoint, 'load', lambda *args: None)
    monkeypatch.setattr(module, '_run_engine', lambda *args: pytest.fail('must not start without explicit flag'))
    monkeypatch.setattr(module, 'backoff', lambda *args: pytest.fail('initial stop is not a new failure'))
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 2 and closed == [True]


def test_provider_recovery_during_portfolio_skips_sealed_attempt_but_finishes_remaining(tmp_path):
    class Broken(MockLLM):
        def generate(self, *args, **kwargs):
            raise LLMError(cause=ConnectionError())
    cfg = settings(num_islands=3, portfolio_seed=True,
                   research={'enabled': True, 'max_aux_calls': None})
    first = construct(tmp_path, cfg, Broken())
    first._evaluate = lambda code: outcome()
    with pytest.raises(CampaignStopped):
        first.run()
    path = tmp_path / 'checkpoint.json'
    saved = json.loads(path.read_bytes())
    assert saved['portfolio_bootstrap']['next_island'] == 1
    class Remaining(MockLLM):
        calls = 0
        def generate(self, prompt, model, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                resumed.request_stop()
            return LLMResponse('invalid source', model, .02, 10, 10)
    provider = Remaining()
    resumed = construct(tmp_path, saved['config'], provider)
    resumed._evaluate = lambda code: pytest.fail('must not re-evaluate baseline or invalid source')
    checkpoint.load(resumed, str(path))
    prepare_provider_recovery(resumed, path, allow_legacy=False)
    assert resumed.run()['reason'] == 'paused'
    assert provider.calls == 2 and resumed.iteration_no == 0
    attempts = [r for r in rows(tmp_path) if r.get('phase') == 'attempt']
    assert [r['evidence']['island'] for r in attempts] == [0, 1, 2]
    assert resumed.cost_total == pytest.approx(.04)


def test_legacy_missing_portfolio_cursor_does_not_replay_paid_initialization(tmp_path):
    engine = construct(tmp_path, settings(num_islands=3, portfolio_seed=True))
    engine._evaluate = lambda code: outcome()
    engine.request_stop()
    engine.run()
    path = tmp_path / 'checkpoint.json'
    saved = json.loads(path.read_bytes())
    del saved['portfolio_bootstrap']
    path.write_text(json.dumps(saved))
    resumed = construct(tmp_path, saved['config'])
    checkpoint.load(resumed, str(path))
    resumed.request_stop()
    resumed._evaluate = lambda code: pytest.fail('legacy baseline must not replay')
    resumed._paradigm_child = lambda *args, **kwargs: pytest.fail('cannot invent legacy unattempted directions')
    assert resumed.run()['reason'] == 'paused'
    assert resumed._portfolio_bootstrap is None


@pytest.mark.parametrize('first_failure', [False, True])
def test_fresh_supervisor_starts_from_seed_and_recovers_only_new_failures(tmp_path, monkeypatch, first_failure):
    spec = importlib.util.spec_from_file_location('supervisor_fresh', ROOT / 'scripts/continue_live_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'new-run'
    config = tmp_path / 'fresh.json'
    cfg = settings(num_islands=3, portfolio_seed=True,
                   research={'enabled': True, 'max_aux_calls': None})
    config.write_text(json.dumps(cfg))
    engines, evaluated, backoffs, restored = [], [], [], []
    class Provider(MockLLM):
        calls = 0
        def generate(self, prompt, model, *args, **kwargs):
            self.calls += 1
            if first_failure and self.calls == 1:
                raise LLMError(cause=ConnectionError())
            if self.calls == 3:
                engines[-1].request_stop()
            assert self.calls <= 3, 'No unbounded test loop or ordinary mutation is allowed'
            return LLMResponse('invalid source', model, .02, 10, 10)
    provider = Provider()
    def build(*args, **kwargs):
        engine = build_engine(*args, **kwargs, llm=provider)
        assert engine.db.pop_best_id is None and engine.iteration_no == 0
        engine._evaluate = lambda code: evaluated.append(code) or outcome()
        engines.append(engine)
        return engine
    load = module.checkpoint.load
    def track_load(*args, **kwargs):
        restored.append(True)
        return load(*args, **kwargs)
    monkeypatch.setattr(sys, 'argv', ['continue_live_run.py', str(directory), '--fresh-config', str(config)])
    monkeypatch.setattr(module, 'build_engine', build)
    monkeypatch.setattr(module.checkpoint, 'load', track_load)
    monkeypatch.setattr(module, 'backoff', lambda path, delay: backoffs.append(delay) or True)
    assert module.main() == 0
    saved = json.loads((directory / 'checkpoint.json').read_bytes())
    assert evaluated == [(ROOT / 'tasks/test/toy_quadratic/initial_program.py').read_text()]
    assert provider.calls == 3 and saved['portfolio_bootstrap']['next_island'] == 3
    assert saved['iteration_no'] == saved['B_evals'] == 0
    assert len(restored) == len(saved.get('provider_recoveries', [])) == int(first_failure)
    assert backoffs == ([30] if first_failure else [])
    assert len([r for r in rows(directory) if r.get('action') == 'seed_program']) == 3


@pytest.mark.parametrize('invalid', ['nonempty', 'warm_start', 'finite', 'conflicting_flags'])
def test_fresh_supervisor_rejects_nonfresh_or_bounded_start(tmp_path, monkeypatch, invalid):
    spec = importlib.util.spec_from_file_location('supervisor_invalid', ROOT / 'scripts/continue_live_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / 'config.json'
    cfg = settings()
    directory = tmp_path / 'run'
    if invalid == 'nonempty':
        directory.mkdir()
        (directory / 'evidence').write_text('preserve')
    if invalid == 'warm_start':
        cfg['warm_start'] = {'fixture': True}
        monkeypatch.setattr(module, 'validate', lambda value: value)
    if invalid == 'finite':
        cfg['budget'] = 5
    config.write_text(json.dumps(cfg))
    argv = ['continue_live_run.py', str(directory), '--fresh-config', str(config)]
    if invalid == 'conflicting_flags':
        argv.append('--abandon-provider-failure')
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setattr(module, 'build_engine', lambda *args, **kwargs: pytest.fail('must reject before build'))
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 2
