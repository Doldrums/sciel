"""Failed search can request research without redefining evaluation statistics.

Real Engine/wave/ResearchTools paths, with local provider and evaluator stand-ins.
The tool action is skip, so no Docker, generated Python, network or paid call runs.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.engine.types import NoChild
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.research_tools.config import validate_config
from sciel.research_tools.controller import ResearchTools, PROMPT_MARKER
from sciel.research_tools.library import ToolLibrary

from test_parallel_engine import Provider, config, number, task_fixture
from test_research_tool_controller import Executor, IMAGE, snapshot


class SkipProvider(Provider):
    def generate(self, prompt, model, temperature=None, **options):
        if prompt.startswith(PROMPT_MARKER):
            self.calls.append(prompt)
            return LLMResponse('{"action":"skip","reason":"Recorded fixture check"}', model, .02, 3, 2)
        return super().generate(prompt, model, temperature, **options)


def fixture(tmp_path, monkeypatch, *, outcome='failed', workers=2, max_iterations=4,
            threshold=2, interval=1, portfolio_seed=False, trigger=True,
            proactive_every_attempts=None):
    tools_config = dict(enabled=True, image_id=IMAGE, interval=interval, max_scientific_steps=1,
                        attempt_stall_threshold=threshold,
                        proactive_every_attempts=proactive_every_attempts,
                        trigger_sources=['search_attempt_stall'] if trigger else ['population_stall'])
    executor = Executor()
    def tools_factory(config, run_dir, **kwargs):
        library = ToolLibrary(run_dir, {'image_id': IMAGE}, config['limits'], executor=executor)
        return ResearchTools(config, run_dir, library=library, **kwargs)
    monkeypatch.setattr('sciel.research_tools.controller.ResearchTools', tools_factory)
    tasks, directory = tmp_path / 'tasks', tmp_path / 'run'
    task = task_fixture(tasks)
    cfg = config(task, tmp_path / 'capacity', workers=workers,
                 max_iterations=max_iterations, refine_stall=1000,
                 portfolio_seed=portfolio_seed, research_tools=tools_config)
    provider, evaluations = SkipProvider('run_search'), []
    def evaluate(task, path, **options):
        value = number(Path(path).read_text())
        evaluations.append(value)
        ok = value == 0 or outcome == 'noop' or (outcome == 'record' and value >= 4)
        # The tool planner consumes provider call 3 before candidate value 4.
        score = (.2 if outcome == 'record' and value == 4 else .1) if ok else 0.
        result = EvalResult(ok, score, {task.primary_metric: score}, timing=.01,
                            artifacts={} if ok else {'error': 'fixture invalid candidate'})
        return CascadeOutcome(result, score, 'full', False)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    if outcome in {'dup_cap', 'empty_response', 'gate_rejected'}:
        def no_child(op, snapshot, *args, **kwargs):
            return NoChild(snapshot.island, outcome,
                           {'dup_attempts': 3, 'gate_rejections': 2}, cost=.03)
        monkeypatch.setattr('sciel.engine.loop.run_worker', no_child)
    def build(*, resume=False):
        engine = build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
        if resume:
            checkpoint.load(engine, str(directory / 'checkpoint.json'))
        return engine
    return build, provider, evaluations, executor, directory


def intents(directory):
    jobs = [job for path in (directory / '.parallel').glob('wave-*/jobs.json')
            for job in json.loads(path.read_text())['jobs'] if job['job_id'].startswith('G:')]
    return sorted(jobs, key=lambda j:int(j['job_id'].split(':')[1]))


@pytest.mark.parametrize('outcome', ['failed', 'noop', 'dup_cap', 'empty_response', 'gate_rejected'])
def test_failure_nochild_and_noop_streaks_open_research_after_committed_attempts(tmp_path, monkeypatch, outcome):
    build, provider, evaluations, executor, directory = fixture(tmp_path, monkeypatch, outcome=outcome)
    engine = build()
    result = engine.run()
    jobs = intents(directory)
    plans = [job['snapshot'].get('tool_episode_plan') for job in jobs]
    assert [plan is not None for plan in plans] == [False, False, True, True]
    assert result['iteration_no'] == 4 and engine._last_pop_improve_iteration == 0
    assert result['B_evals'] == (4 if outcome == 'noop' else 0)
    assert engine.n_records == 0
    for plan in plans[2:]:
        signal = plan['stagnation_signals']['search_attempt_stall']
        assert signal['elapsed_attempts'] == signal['threshold'] == 2
        assert signal['counter'] == 'iteration_no' and signal['threshold_reached']
        assert plan['trigger_sources'] == ['search_attempt_stall']
        assert not plan['stagnation_signals']['population_stall']['threshold_reached']
    assert len(engine.research_tools.state()['committed_episodes']) == 2
    assert result['cost_total'] == pytest.approx(.16)  # Four attempts, two skip calls.
    assert executor.calls == []
    assert sum(p.startswith(PROMPT_MARKER) for p in provider.calls) == 2
    assert all('search_attempt_stall' in p for p in provider.calls if p.startswith(PROMPT_MARKER))
    assert engine.research_tools.state()['registered'] == []
    if outcome in {'dup_cap', 'empty_response', 'gate_rejected'}:
        assert engine.n_nochild == 4 and len(evaluations) == 1
    elif outcome == 'failed':
        assert engine.n_eval_fail == 4 and len(evaluations) == 5


def test_population_record_resets_attempt_anchor_but_nonrecords_do_not(tmp_path, monkeypatch):
    build, _, _, _, directory = fixture(tmp_path, monkeypatch, outcome='record', workers=1)
    engine = build()
    result = engine.run()
    assert result['iteration_no'] == 4 and result['B_evals'] == 2
    assert engine.n_records == 1 and engine._last_pop_improve_iteration == 3
    assert engine._last_pop_improve == 1
    assert [bool(j['snapshot'].get('tool_episode_plan')) for j in intents(directory)] == [False, False, True, False]
    state = json.loads((directory / 'checkpoint.json').read_text())
    assert state['_last_pop_improve_iteration'] == 3
    assert state['provenance']['counters']['_last_pop_improve_iteration'] == 3
    resumed = build(resume=True)
    assert resumed._last_pop_improve_iteration == 3 and resumed.iteration_no == 4
    resumed.close()


def test_bootstrap_does_not_count_attempts_or_create_tools(tmp_path, monkeypatch):
    build, _, _, _, directory = fixture(tmp_path, monkeypatch, outcome='failed',
        portfolio_seed=True, max_iterations=2, threshold=2)
    engine = build()
    result = engine.run()
    assert engine._portfolio_bootstrap['next_island'] == 4
    assert result['iteration_no'] == 2 and engine._last_pop_improve_iteration == 0
    assert engine.research_tools.state()['committed_episodes'] == []
    assert all('tool_episode_plan' not in j['snapshot'] for j in intents(directory))


def test_attempt_trigger_remains_opt_in(tmp_path, monkeypatch):
    build, _, _, _, directory = fixture(tmp_path, monkeypatch, trigger=False)
    engine = build()
    engine.run()
    assert engine.iteration_no == 4 and engine.B_evals == 0
    assert engine.research_tools.state()['committed_episodes'] == []
    assert all('tool_episode_plan' not in j['snapshot'] for j in intents(directory))


def test_attempt_trigger_preserves_interval_cell_lease_and_active_cap(tmp_path, monkeypatch):
    build, provider, evaluations, executor, _ = fixture(tmp_path, monkeypatch, interval=4)
    engine = build()
    tools = engine.research_tools
    from research_target_fixtures import bind_snapshot
    def current(*args):
        return bind_snapshot(snapshot(*args), task_hash=engine.task.task_hash)
    signals = {'schema': 'sciel-stagnation-signals/1',
               'population_stall': {'threshold_reached': False, 'last_record_B_evals': 0},
               'search_attempt_stall': {'threshold_reached': True, 'elapsed_attempts': 8}}
    try:
        first = tools.prepare(current(0), signals=signals)
        assert first['opportunity'] == 0 and first['trigger_sources'] == ['search_attempt_stall']
        for opportunity in range(1, 5):
            # 1..3: interval; 4: the previous frozen episode still owns the cell.
            assert tools.prepare(current(0, opportunity), signals=signals) is None
        for island in range(1, 4):
            assert tools.prepare(current(island), signals=signals) is not None
        assert tools.prepare(current(4), signals=signals) is None  # Full active cap.
        tools.lifecycle.cells.deactivate(first['cell_id'])
        # The due opportunity4 was refused by the lease, so it did not reset
        # the last actual episode0 or force another modulo interval.
        resumed = tools.prepare(current(0, 5), signals=signals)
        assert resumed['opportunity'] == 5 and resumed['cell_id'] == first['cell_id']
        assert resumed['last_episode_opportunity'] == 0
        assert provider.calls == evaluations == executor.calls == []
    finally:
        engine.close()


def test_wave_replay_preserves_anchor_frozen_plans_and_exact_cost(tmp_path, monkeypatch):
    build, provider, evaluations, executor, directory = fixture(tmp_path, monkeypatch)
    original = Engine._commit_step
    crashed = False
    def crash(self, plan, *args, **kwargs):
        nonlocal crashed
        original(self, plan, *args, **kwargs)
        if plan['snap'].iteration_no == 2 and not crashed:
            crashed = True
            raise RuntimeError('fixture crash after attempt-stall commit')
    monkeypatch.setattr(Engine, '_commit_step', crash)
    with pytest.raises(CampaignStopped, match='fixture crash'):
        build().run()
    baseline = json.loads((directory / 'checkpoint.json').read_text())
    assert baseline['iteration_no'] == 2 and baseline['_last_pop_improve_iteration'] == 0
    before = (len(provider.calls), len(evaluations), len(executor.calls))
    frozen = deepcopy(intents(directory))
    engine = build(resume=True)
    result = engine.run()
    assert (len(provider.calls), len(evaluations), len(executor.calls)) == before
    assert intents(directory) == frozen
    assert engine.iteration_no == 4 and engine._last_pop_improve_iteration == 0
    assert result['cost_total'] == pytest.approx(.16)
    assert len(engine.research_tools.state()['committed_episodes']) == 2


@pytest.mark.parametrize('bad', [None, -1, 5, True, 1.5])
def test_checkpoint_missing_or_invalid_attempt_anchor_is_rejected(tmp_path, monkeypatch, bad):
    build, _, _, _, directory = fixture(tmp_path, monkeypatch)
    build().run()
    path = directory / 'checkpoint.json'
    state = json.loads(path.read_text())
    if bad is None:
        state.pop('_last_pop_improve_iteration')
    else:
        state['_last_pop_improve_iteration'] = bad
    path.write_text(json.dumps(state))
    with pytest.raises(checkpoint.IdentityMismatch, match='iteration anchor'):
        build(resume=True)


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_attempt_threshold_requires_positive_integer(value):
    with pytest.raises(ValueError, match='attempt_stall_threshold'):
        validate_config({'attempt_stall_threshold': value})
