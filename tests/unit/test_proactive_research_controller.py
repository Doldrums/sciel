"""Independent proactive cadence with real plans, checkpoints and wave replay."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.research_tools.controller import PROMPT_MARKER
from test_parallel_engine import number
from test_research_attempt_stall import fixture, intents
from test_research_tool_controller import setup, snapshot, LLM, IO


def healthy(committed):
    return {'schema': 'sciel-stagnation-signals/1', 'committed_iteration': committed,
        'population_stall': {'threshold_reached': False, 'last_record_B_evals': committed},
        'search_attempt_stall': {'threshold_reached': False, 'last_record_iteration': committed},
        'controller_plateau': {'enabled': True, 'active': False}}


def finish(tools, plan, current):
    episode = tools.execute(plan, current, LLM([{'action': 'skip'}]), io=IO())
    tools.commit(episode)


def test_global_cadence_survives_continuous_records_and_limits_one_wave(tmp_path):
    tools, backend, _, costs = setup(tmp_path, proactive_every_attempts=8, interval=100)
    for clock in (0, 8, 16):
        current = snapshot(0, clock)
        first = tools.prepare(current, signals=healthy(clock))
        assert first['stage'] == 'research' and first['trigger_sources'] == ['proactive']
        assert first['cadence_reason'] == 'proactive_research'
        assert first['proactive_research']['committed_iteration'] == clock
        frozen = deepcopy(first)
        for slot in range(1, 4):
            assert tools.prepare(snapshot(slot, clock + slot), signals=healthy(clock)) is None
        finish(tools, first, current)
        assert first == frozen
        # Both a new global record and a later planned slot leave the global
        # quota anchored to committed attempts, not the proposed iteration.
        assert tools.prepare(snapshot(0, clock + 7), signals=healthy(clock + 7)) is None
    assert tools.state()['last_proactive_iteration'] == 16
    assert costs == [.03] * 3 and backend.calls == []


def test_cap_and_lease_refusals_preserve_due_reservation(tmp_path):
    tools, _, _, _ = setup(tmp_path, proactive_every_attempts=8, max_active_cells=1)
    first = tools.prepare(snapshot(), signals=healthy(0))
    assert tools.state()['last_proactive_iteration'] == 0
    cells = deepcopy(tools.lifecycle.cells.state())
    assert tools.prepare(snapshot(1, 8), signals=healthy(8)) is None  # capacity
    assert tools.prepare(snapshot(0, 8), signals=healthy(8)) is None  # lease
    assert tools.state()['last_proactive_iteration'] == 0
    assert tools.lifecycle.cells.state() == cells
    finish(tools, first, snapshot())
    later = tools.prepare(snapshot(1, 9), signals=healthy(8))
    assert later['proactive_research']['committed_iteration'] == 8


def test_state_restore_and_rollback_preserve_reserved_plan(tmp_path):
    tools, _, _, _ = setup(tmp_path, proactive_every_attempts=8)
    baseline = tools.state()
    first = tools.prepare(snapshot(), signals=healthy(0))
    reserved = tools.state()
    tools.load_state(reserved)
    assert tools.prepare(snapshot(1, 1), signals=healthy(0)) is None
    tools.load_state(baseline)
    replay = tools.prepare(snapshot(), signals=healthy(0))
    assert replay == first
    finish(tools, replay, snapshot())
    saved = tools.state()
    restored, _, _, _ = setup(tmp_path, proactive_every_attempts=8)
    restored.load_state(saved)
    assert restored.state() == saved
    assert restored.prepare(snapshot(1, 7), signals=healthy(7)) is None
    assert restored.prepare(snapshot(1, 8), signals=healthy(8))['stage'] == 'research'


@pytest.mark.parametrize('clock', [None, True, -1, 1.0, '1'])
@pytest.mark.parametrize('independent', [False, True])
def test_missing_or_malformed_committed_clock_fails_without_reserving(tmp_path, clock, independent):
    tools, _, _, _ = setup(tmp_path, proactive_every_attempts=8,
        independent_research=independent, max_scientific_steps=2)
    before = tools.state()
    before_run_id = tools.lifecycle.run_id
    with pytest.raises(ValueError, match='committed ordinary-attempt clock'):
        tools.prepare(snapshot(), signals=healthy(clock))
    assert tools.state() == before
    assert tools.lifecycle.run_id == before_run_id


@pytest.mark.parametrize('value', [True, -1, '1', 1.0])
def test_invalid_checkpoint_cursor_rejected_atomically(tmp_path, value):
    tools, _, _, _ = setup(tmp_path, proactive_every_attempts=8)
    before = tools.state(); altered = deepcopy(before)
    altered['last_proactive_iteration'] = value
    with pytest.raises(ValueError, match='proactive research reservation cursor'):
        tools.load_state(altered)
    assert tools.state() == before


def test_fresh_result_review_precedes_due_research_without_spending_quota(tmp_path, monkeypatch):
    tools, _, _, _ = setup(tmp_path, proactive_every_attempts=8, jobs={'enabled': True})
    first = tools.prepare(snapshot(), signals=healthy(0))
    finish(tools, first, snapshot())
    monkeypatch.setattr(tools, '_island_jobs', lambda island: [
        {'collected': True, 'reviewed': False, 'job_id': 'a' * 64}])
    review = tools.prepare(snapshot(0, 8), signals=healthy(8))
    assert review is not None and 'proactive_research' not in review
    assert tools.state()['last_proactive_iteration'] == 0
    # A sibling without pending review can consume the still-due reservation.
    monkeypatch.setattr(tools, '_island_jobs', lambda island: [])
    assert tools.prepare(snapshot(1, 9), signals=healthy(8))['proactive_research']['committed_iteration'] == 8


def growing_evaluator(monkeypatch, evaluations):
    def evaluate(task, path, **options):
        value = number(Path(path).read_text()); evaluations.append(value)
        score = .1 + value / 100
        result = EvalResult(True, score, {task.primary_metric: score}, timing=.01)
        return CascadeOutcome(result, score, 'full', False)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)


@pytest.mark.parametrize('workers,expected', [(1, [0, 2, 4]), (2, [0, 2, 4])])
def test_real_engine_creates_research_despite_records_in_every_wave(tmp_path, monkeypatch, workers, expected):
    build, provider, evaluations, executor, directory = fixture(tmp_path, monkeypatch,
        workers=workers, max_iterations=6, threshold=1000, interval=100,
        proactive_every_attempts=2)
    growing_evaluator(monkeypatch, evaluations)
    engine = build(); result = engine.run()
    plans = [j['snapshot']['tool_episode_plan'] for j in intents(directory)
             if j['snapshot'].get('tool_episode_plan')]
    assert [p['proactive_research']['committed_iteration'] for p in plans] == expected
    assert all(p['stage'] == 'research' and p['trigger_sources'] == ['proactive'] for p in plans)
    assert result['iteration_no'] == 6 and engine.n_records >= 6 // workers
    assert engine._last_pop_improve_iteration >= 7 - workers
    for plan in plans[1:]:
        clock = plan['proactive_research']['committed_iteration']
        signal = plan['stagnation_signals']['search_attempt_stall']
        assert signal['last_record_iteration'] >= clock - workers + 1
        assert signal['threshold_reached'] is False
    assert sum(p.startswith(PROMPT_MARKER) for p in provider.calls) == 3
    assert result['cost_total'] == pytest.approx(.24)
    assert executor.calls == []


def test_crash_after_proactive_commit_replays_without_second_charge_or_dispatch(tmp_path, monkeypatch):
    build, provider, evaluations, executor, directory = fixture(tmp_path, monkeypatch,
        workers=2, max_iterations=2, threshold=1000, proactive_every_attempts=8)
    growing_evaluator(monkeypatch, evaluations)
    commit = Engine._commit_step; crashed = False
    def crash(self, *args, **kwargs):
        nonlocal crashed
        commit(self, *args, **kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError('fixture after proactive commit')
    monkeypatch.setattr(Engine, '_commit_step', crash)
    with pytest.raises(CampaignStopped, match='after proactive commit'):
        build().run()
    frozen = deepcopy(intents(directory)); calls = (len(provider.calls), len(evaluations))
    baseline = json.loads((directory / 'checkpoint.json').read_text())
    assert baseline['research_tools']['last_proactive_iteration'] is None
    restored = build(resume=True); result = restored.run()
    assert intents(directory) == frozen
    assert (len(provider.calls), len(evaluations)) == calls
    assert restored.research_tools.state()['last_proactive_iteration'] == 0
    assert result['cost_total'] == pytest.approx(.08) and executor.calls == []


def test_bootstrap_keeps_existing_protocol_then_first_ordinary_step_is_proactive(tmp_path, monkeypatch):
    build, provider, evaluations, executor, directory = fixture(tmp_path, monkeypatch,
        workers=2, max_iterations=2, threshold=1000, portfolio_seed=True,
        proactive_every_attempts=8)
    growing_evaluator(monkeypatch, evaluations)
    engine = build(); engine.run()
    assert engine._portfolio_bootstrap['next_island'] == 4
    all_jobs = [j for p in (directory / '.parallel').glob('wave-*/jobs.json')
                for j in json.loads(p.read_text())['jobs']]
    bootstrap = [j for j in all_jobs if j['job_id'].startswith('portfolio:')]
    assert len(bootstrap) == 4 and all('tool_episode_plan' not in j['snapshot'] for j in bootstrap)
    ordinary = intents(directory)
    assert ordinary[0]['snapshot']['tool_episode_plan']['proactive_research']['committed_iteration'] == 0
    assert 'tool_episode_plan' not in ordinary[1]['snapshot']
