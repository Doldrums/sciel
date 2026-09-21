"""Interleaved initial portfolio uses real coordinator/WAL, fake transports only."""
import json
import sys
from pathlib import Path

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.config import apply_defaults
from sciel.engine import checkpoint, parallel, portfolio_allocation
from sciel.engine.factory import build_engine
from sciel.engine.loop import Engine

from test_parallel_engine import Provider, config, evaluator, events, task_fixture, number


def setup(tmp_path, monkeypatch, *, islands=10, workers=4, max_iterations=12,
          budget=None, schedule='interleaved', **overrides):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    cfg = config(task, tmp_path / 'capacity', workers, num_islands=islands,
        max_iterations=max_iterations, budget=budget, portfolio_seed=True, **overrides)
    cfg['parallel_search']['portfolio_schedule'] = schedule
    provider = Provider('run_search', concurrent=min(workers, islands))
    evaluations, _ = evaluator(monkeypatch)
    directory = tmp_path / 'run'

    def build(resume=False):
        engine = build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
        if resume:
            checkpoint.load(engine, str(directory / 'checkpoint.json'))
        return engine
    return build, provider, evaluations, directory, cfg


def saved(directory):
    return json.loads((directory / 'checkpoint.json').read_text())


def waves(directory):
    return [row for row in events(directory) if row.get('action') == 'wave_dispatch']


def jobs(directory, wave):
    return json.loads((directory / '.parallel' / f'wave-{wave}' / 'jobs.json').read_text())['jobs']


def test_first_batch_evolution_then_interleaved_tail_and_all_evolution(tmp_path, monkeypatch):
    build, provider, evaluations, directory, _ = setup(tmp_path, monkeypatch, max_iterations=16)
    engine = build(); result = engine.run()
    dispatched = waves(directory)
    assert [row['wave_kind'] for row in dispatched] == [
        'portfolio','evolution','portfolio','evolution','portfolio','evolution','evolution']
    assert [row['jobs'] for row in dispatched] == [4,4,4,4,2,4,4]
    assert [row['portfolio_attempted'] for row in dispatched] == [0,4,4,8,8,10,10]
    assert result['iteration_no'] == result['B_evals'] == 16
    assert len(provider.calls) == 26 and len(evaluations) == 27
    assert result['cost_total'] == pytest.approx(26 * .03)
    for row in dispatched:
        if row['wave_kind'] == 'evolution':
            assert all(job['snapshot']['island'] < row['portfolio_attempted']
                       for job in jobs(directory, row['wave']))
    first = jobs(directory, 2)
    assert len(first) == 4 and {job['snapshot']['island'] for job in first} == set(range(4))
    assert any(job['snapshot']['parent_id'] not in {f'p{i}' for i in range(1,11)} for job in first)
    state = saved(directory)
    assert state['portfolio_bootstrap']['next_island'] == 10
    assert state['parallel_search']['allocation'] == dict(schema=portfolio_allocation.SCHEMA,
        policy='interleaved',next_kind='evolution',portfolio_waves=3,evolution_waves=4)


@pytest.mark.parametrize('budget,max_iterations,reason', [(3,99,'budget'),(None,3,'backstop')])
def test_finite_search_limit_stops_before_forcing_remaining_portfolio(tmp_path, monkeypatch, budget, max_iterations, reason):
    build, provider, evaluations, directory, _ = setup(tmp_path, monkeypatch,
        budget=budget,max_iterations=max_iterations)
    result = build().run()
    assert result['reason'] == reason
    assert result['iteration_no'] == result['B_evals'] == 3
    assert [row['wave_kind'] for row in waves(directory)] == ['portfolio','evolution']
    assert len(provider.calls) == 7 and len(evaluations) == 8
    assert saved(directory)['portfolio_bootstrap']['next_island'] == 4
    assert saved(directory)['parallel_search']['allocation']['next_kind'] == 'portfolio'


def test_failed_first_batch_keeps_canonical_parents_and_enters_evolution(tmp_path, monkeypatch):
    build, provider, evaluations, directory, _ = setup(tmp_path, monkeypatch,max_iterations=4)
    import sciel.engine.loop as loop
    original = loop.cascade_evaluate

    def fail_portfolio(task, path, **options):
        outcome = original(task, path, **options)
        if 1 <= number(Path(path).read_text()) <= 4:
            outcome.result.ok = False
            outcome.result.score = 0.
        return outcome

    monkeypatch.setattr(loop, 'cascade_evaluate', fail_portfolio)
    engine = build(); result = engine.run()
    assert [row['wave_kind'] for row in waves(directory)] == ['portfolio','evolution']
    assert all(job['snapshot']['parent_id'] in {'p1','p2','p3','p4'} for job in jobs(directory,2))
    assert saved(directory)['portfolio_bootstrap']['next_island'] == 4
    assert result['iteration_no'] == 4 and len(provider.calls) == 8 and len(evaluations) == 9


@pytest.mark.parametrize('paused_kind,wave,expected_cursor', [('portfolio',1,0),('evolution',2,4)])
def test_partial_wave_pause_resumes_exact_allocation_and_paid_calls_once(tmp_path, monkeypatch, paused_kind, wave, expected_cursor):
    build, provider, evaluations, directory, _ = setup(tmp_path, monkeypatch,max_iterations=4)
    generate = provider.generate
    triggered = False

    def stop_after_response(*args, **kwargs):
        nonlocal triggered
        result = generate(*args, **kwargs)
        allocation = directory / '.parallel' / f'wave-{wave}' / 'allocation.json'
        if allocation.exists():
            with provider.lock:
                if not triggered:
                    triggered = True
                    (directory / 'STOP').touch()
        return result

    monkeypatch.setattr(provider, 'generate', stop_after_response)
    first = build().run()
    assert first['reason'] == 'paused'
    baseline = saved(directory)
    assert baseline['portfolio_bootstrap']['next_island'] == expected_cursor
    assert baseline['parallel_search']['allocation']['next_kind'] == paused_kind
    allocation_file = directory / '.parallel' / f'wave-{wave}' / 'allocation.json'
    exact_allocation = allocation_file.read_bytes()
    paid_prompts = list(provider.calls)
    second = build(resume=True).run()
    assert second['iteration_no'] == 4
    assert allocation_file.read_bytes() == exact_allocation
    assert provider.calls[:len(paid_prompts)] == paid_prompts
    assert len(provider.calls) == 8 and len(evaluations) == 9
    assert second['cost_total'] == pytest.approx(.24)
    assert saved(directory)['portfolio_bootstrap']['next_island'] == 4
    assert saved(directory)['parallel_search']['allocation']['portfolio_waves'] == 1
    assert len([row for row in events(directory) if row.get('action') == 'job_committed']) == 8


def test_allocation_rolls_back_after_evolution_admission_crash(tmp_path, monkeypatch):
    build, provider, evaluations, directory, _ = setup(tmp_path, monkeypatch,max_iterations=4)
    original = Engine._commit_step
    crashed = False

    def crash(self, *args, **kwargs):
        nonlocal crashed
        original(self, *args, **kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError('after admission')

    monkeypatch.setattr(Engine, '_commit_step', crash)
    with pytest.raises(CampaignStopped,match='after admission'):
        build().run()
    baseline = saved(directory)
    assert baseline['portfolio_bootstrap']['next_island'] == 4
    assert baseline['parallel_search']['allocation']['next_kind'] == 'evolution'
    assert baseline['parallel_search']['allocation']['evolution_waves'] == 0
    count = len(provider.calls),len(evaluations)
    result = build(resume=True).run()
    assert (len(provider.calls),len(evaluations)) == count == (8,9)
    assert result['iteration_no'] == 4
    assert saved(directory)['parallel_search']['allocation']['evolution_waves'] == 1


def test_milestones_and_spawn_wait_for_initial_roster_but_not_evolution(tmp_path, monkeypatch):
    build, _, _, directory, _ = setup(tmp_path, monkeypatch, dynamic_islands=True)
    calls = []
    monkeypatch.setattr(Engine,'_maybe_spawn_island',lambda self: calls.append(('spawn',self._portfolio_bootstrap['next_island'])))
    monkeypatch.setattr(Engine,'_maybe_milestone',lambda self: calls.append(('milestone',self._portfolio_bootstrap['next_island'])))
    build().run()
    assert calls == [('spawn',10),('milestone',10)]
    assert [row['portfolio_attempted'] for row in waves(directory) if row['wave_kind']=='evolution'] == [4,8,10]


def test_blocking_default_preserves_old_all_bootstrap_before_search(tmp_path, monkeypatch):
    build, _, _, directory, cfg = setup(tmp_path, monkeypatch,schedule='blocking',max_iterations=4)
    del cfg['parallel_search']['portfolio_schedule']
    build().run()
    assert [row['wave_kind'] for row in waves(directory)] == ['portfolio','portfolio','portfolio','evolution']
    assert 'allocation' not in saved(directory)['parallel_search']
    assert not list((directory/'.parallel').glob('wave-*/allocation.json'))


def test_cooling_down_initial_pool_cannot_select_untouched_seed_aliases(tmp_path, monkeypatch):
    build, _, _, directory, _ = setup(tmp_path, monkeypatch,max_iterations=4,schedule_policy='sh')
    original = parallel._commit_seed
    def cooling(self, plan, result):
        original(self, plan, result)
        self.schedule.cooldown = {island:8 for island in range(10)}
    monkeypatch.setattr(parallel,'_commit_seed',cooling)
    build().run()
    assert {job['snapshot']['island'] for job in jobs(directory,2)} == {0,1,2,3}
    assert len({job['snapshot']['parent_id'] for job in jobs(directory,2)}) == 4


def test_smaller_roster_than_workers_finishes_first_batch_then_only_evolves(tmp_path, monkeypatch):
    build, provider, evaluations, directory, _ = setup(tmp_path,monkeypatch,islands=2,max_iterations=3)
    build().run()
    assert [row['wave_kind'] for row in waves(directory)] == ['portfolio','evolution','evolution']
    assert [row['jobs'] for row in waves(directory)] == [2,2,1]
    assert len(provider.calls) == 5 and len(evaluations) == 6


@pytest.mark.parametrize('corruption', ['missing_allocation','wrong_cursor','missing_cursor','wrong_next_kind'])
def test_checkpoint_rejects_corrupt_allocation_before_transport(tmp_path, monkeypatch, corruption):
    build, provider, evaluations, directory, _ = setup(tmp_path,monkeypatch,max_iterations=4)
    build().run()
    path = directory/'checkpoint.json'
    state = saved(directory)
    if corruption == 'missing_allocation':
        state['parallel_search'].pop('allocation')
    elif corruption == 'wrong_cursor':
        state['portfolio_bootstrap']['next_island'] = 8
    elif corruption == 'missing_cursor':
        state.pop('portfolio_bootstrap')
        state['parallel_search']['allocation'] = portfolio_allocation.initial()
    else:
        state['parallel_search']['allocation']['next_kind'] = 'evolution'
    path.write_text(json.dumps(state))
    counts = len(provider.calls),len(evaluations)
    engine = build()
    try:
        with pytest.raises(checkpoint.IdentityMismatch,match='portfolio'):
            checkpoint.load(engine,str(path),allow_identity_mismatch=True)
    finally:
        engine.close()
    assert (len(provider.calls),len(evaluations)) == counts


def test_interleaved_configuration_and_allocation_fail_closed():
    for changes in [dict(parallel_search={'portfolio_schedule':'adaptive'}),
                    dict(parallel_search={'portfolio_schedule':'interleaved'}),
                    dict(portfolio_seed=False,parallel_search={'enabled':True,'portfolio_schedule':'interleaved'})]:
        with pytest.raises(ValueError):
            apply_defaults(changes)
    cfg = dict(num_islands=10,parallel_search={'workers':4,'portfolio_schedule':'interleaved'})
    cursor = dict(islands=10,next_island=4)
    for value in [None,portfolio_allocation.initial(),dict(portfolio_allocation.initial(),portfolio_waves=1),
                  dict(portfolio_allocation.initial(),portfolio_waves=True,next_kind='evolution')]:
        with pytest.raises(ValueError):
            portfolio_allocation.validate(value,cfg,cursor)
