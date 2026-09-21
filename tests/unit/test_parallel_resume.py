"""Actual Engine wave crash/replay with fake provider and evaluator only."""
import json
from pathlib import Path
import sys

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint, parallel, parallel_calls, parallel_store
from sciel.engine.factory import build_engine
from sciel.engine.loop import Engine

from test_parallel_engine import Provider, config, evaluator, events, task_fixture


def setup_run(tmp_path, monkeypatch, *, same=False, portfolio=False, max_iterations=4):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    cfg = config(task, tmp_path / 'capacity', workers=4,
                 portfolio_seed=portfolio, max_iterations=max_iterations)
    provider = Provider('run_search', concurrent=4, same=same)
    evaluations, _ = evaluator(monkeypatch)
    directory = tmp_path / 'run'

    def build(*, resume=False):
        engine = build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
        if resume:
            checkpoint.load(engine, str(directory / 'checkpoint.json'))
        return engine

    return build, provider, evaluations, directory


def state(directory):
    return json.loads((directory / 'checkpoint.json').read_text())


@pytest.mark.parametrize('when', ['before', 'after'])
def test_replay_before_or_after_admission_never_repeats_returned_calls(tmp_path, monkeypatch, when):
    build, provider, evaluations, directory = setup_run(tmp_path, monkeypatch)
    commit = Engine._commit_step
    crashed = False

    def crash(self, plan, result, **options):
        nonlocal crashed
        if not crashed:
            crashed = True
            if when == 'after':
                commit(self, plan, result, **options)
            raise RuntimeError('injected admission crash')
        return commit(self, plan, result, **options)

    monkeypatch.setattr(Engine, '_commit_step', crash)
    with pytest.raises(CampaignStopped, match='admission crash'):
        build().run()
    assert len(provider.calls) == 4 and len(evaluations) == 5
    assert state(directory)['iteration_no'] == state(directory)['B_evals'] == 0
    assert not any(row.get('event') == 'iteration' for row in events(directory))
    restored = build(resume=True)
    outcome = restored.run()
    assert len(provider.calls) == 4 and len(evaluations) == 5
    assert outcome['B_evals'] == outcome['iteration_no'] == 4
    assert outcome['cost_total'] == pytest.approx(.12)
    assert [row['job_id'] for row in events(directory)
            if row.get('action') == 'job_committed'] == ['G:1', 'G:2', 'G:3', 'G:4']


def test_duplicate_retry_feedback_replays_exactly_across_new_segment(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup_run(tmp_path, monkeypatch, same=True)
    commit = Engine._commit_step
    first = True

    def crash(self, *args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise RuntimeError('crash after every duplicate retry returned')
        return commit(self, *args, **kwargs)

    monkeypatch.setattr(Engine, '_commit_step', crash)
    with pytest.raises(CampaignStopped):
        build().run()
    assert len(provider.calls) == 10 and len(evaluations) == 2
    prompts = list(provider.calls)
    assert sum('LOCAL CANDIDATE VALIDATION FEEDBACK' in prompt for prompt in prompts) == 6
    outcome = build(resume=True).run()
    assert provider.calls == prompts and len(evaluations) == 2
    assert outcome['B_evals'] == 1 and outcome['iteration_no'] == 4
    assert outcome['cost_total'] == pytest.approx(.30)
    assert len({row['segment_id'] for row in events(directory) if row['event'] == 'run_start'}) == 2


@pytest.mark.parametrize('append_bytes', [0, 17])
def test_engine_constructor_finishes_prepared_commit_before_loading_checkpoint(tmp_path, monkeypatch, append_bytes):
    build, provider, evaluations, directory = setup_run(tmp_path, monkeypatch)
    finish = parallel_store._finish_prepared
    crashed = False

    def crash(wave_dir, manifest, run_dir, **kwargs):
        nonlocal crashed
        if manifest['wave_id'] == '1' and not crashed:
            crashed = True
            if append_bytes:
                staged = (wave_dir / manifest['journal']).read_bytes()
                with open(run_dir / 'sciel_events.jsonl', 'ab') as stream:
                    stream.write(staged[:append_bytes])
            raise OSError('injected prepared commit crash')
        return finish(wave_dir, manifest, run_dir, **kwargs)

    monkeypatch.setattr(parallel_store, '_finish_prepared', crash)
    with pytest.raises(CampaignStopped):
        build().run()
    assert state(directory)['iteration_no'] == 0
    restored = build(resume=True)
    assert restored.iteration_no == restored.B_evals == 4
    outcome = restored.run()
    assert len(provider.calls) == 4 and len(evaluations) == 5
    assert outcome['cost_total'] == pytest.approx(.12)
    assert len([row for row in events(directory) if row['event'] == 'iteration']) == 4
    assert not list((directory / '.parallel').glob('wave-*/*.journal.jsonl'))


def test_unknown_dispatched_calls_fail_closed_without_provider_replay(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup_run(tmp_path, monkeypatch)
    write = parallel_calls.durable_json

    def lose_receipt(path, value):
        if (Path(path).name == 'receipt.json' and value['payload']['status'] == 'returned'
                and 'text' in value['payload'].get('value', {})):
            raise OSError('lost returned provider receipt')
        return write(path, value)

    monkeypatch.setattr(parallel_calls, 'durable_json', lose_receipt)
    with pytest.raises(CampaignStopped):
        build().run()
    assert len(provider.calls) == 4 and len(evaluations) == 1
    assert state(directory)['cost_total'] == 0
    unknown = [path.parent for path in (directory / '.parallel').rglob('dispatched.json')
               if not (path.parent / 'receipt.json').exists()]
    assert len(unknown) == 4
    monkeypatch.setattr(parallel_calls, 'durable_json', write)
    with pytest.raises(CampaignStopped):
        build(resume=True).run()
    assert len(provider.calls) == 4 and len(evaluations) == 1
    assert all(not (path / 'receipt.json').exists() for path in unknown)


def test_stop_drains_dispatched_generation_then_resumes_from_cached_receipts(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup_run(tmp_path, monkeypatch, max_iterations=20)
    generate = provider.generate
    stop_written = False

    def stop_after_dispatch(*args, **kwargs):
        nonlocal stop_written
        # The provider barrier establishes that all four calls were dispatched
        # before STOP. No next-round evaluation may start after it is observed.
        returned = generate(*args, **kwargs)
        with provider.lock:
            if not stop_written:
                (directory / 'STOP').touch()
                stop_written = True
        return returned

    monkeypatch.setattr(provider, 'generate', stop_after_dispatch)
    outcome = build().run()
    assert outcome['reason'] == 'paused'
    assert len(provider.calls) == 4 and len(evaluations) == 1
    assert outcome['B_evals'] == outcome['iteration_no'] == 0
    assert state(directory)['cost_total'] == 0  # Published baseline, not a claim of free calls.
    receipts = [json.loads(path.read_text())['payload']
                for path in (directory / '.parallel').rglob('receipt.json')]
    paid = [row for row in receipts if 'text' in row.get('value', {})]
    assert len(paid) == 4 and sum(row['value']['cost'] for row in paid) == pytest.approx(.12)
    assert not (directory / 'STOP').exists()
    assert all((path.parent / 'receipt.json').exists()
               for path in (directory / '.parallel').rglob('dispatched.json'))
    manifest = json.loads((directory / '.parallel' / 'wave-1' / 'transaction.json').read_text())
    assert manifest['phase'] == 'active'
    restored = build(resume=True)
    restored._stop_at = 4
    resumed = restored.run()
    assert resumed['reason'] == 'paused'
    assert len(provider.calls) == 4 and len(evaluations) == 5
    assert resumed['B_evals'] == resumed['iteration_no'] == 4
    assert resumed['cost_total'] == pytest.approx(.12)


def test_portfolio_resume_preserves_cursor_and_reuses_all_returned_calls(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup_run(tmp_path, monkeypatch, portfolio=True)
    commit = parallel._commit_seed
    crashed = False

    def crash(*args, **kwargs):
        nonlocal crashed
        commit(*args, **kwargs)
        if not crashed:
            crashed = True
            raise RuntimeError('portfolio after-admission crash')

    monkeypatch.setattr(parallel, '_commit_seed', crash)
    initial = build()
    initial._stop_at = 0
    with pytest.raises(CampaignStopped):
        initial.run()
    assert len(provider.calls) == 4 and len(evaluations) == 5
    assert state(directory)['portfolio_bootstrap']['next_island'] == 0
    restored = build(resume=True)
    restored._stop_at = 0
    outcome = restored.run()
    assert len(provider.calls) == 4 and len(evaluations) == 5
    assert outcome['reason'] == 'paused' and outcome['B_evals'] == 0
    assert outcome['cost_total'] == pytest.approx(.12)
    assert state(directory)['portfolio_bootstrap']['next_island'] == 4
    assert len([program for program in restored.db._all.values() if program.origin == 'portfolio']) == 4


def test_stop_cannot_hide_unknown_failure_of_already_dispatched_call(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup_run(tmp_path, monkeypatch)
    generate = provider.generate
    stopped = False

    def stop_with_unknown_call(*args, **kwargs):
        nonlocal stopped
        returned = generate(*args, **kwargs)
        fail = False
        with provider.lock:
            if not stopped:
                stopped = True
                (directory / 'STOP').touch()
                fail = True
        if fail:
            # All four calls passed the provider barrier; this one produced no
            # cost-bearing response for the engine. STOP cannot make it free or
            # turn an unknown paid outcome into a successful pause.
            raise ConnectionError('mock provider connection disappeared')
        return returned

    monkeypatch.setattr(provider, 'generate', stop_with_unknown_call)
    with pytest.raises(CampaignStopped):
        build().run()
    assert len(provider.calls) == 4 and len(evaluations) == 1
    runtime = json.loads((directory / 'runtime.json').read_text())
    assert runtime['status'] != 'paused'
    assert state(directory)['B_evals'] == state(directory)['iteration_no'] == 0
    receipts = [json.loads(path.read_text())['payload']
                for path in (directory / '.parallel').rglob('receipt.json')]
    failed = [row for row in receipts if row['status'] == 'error']
    assert len(failed) == 1
    assert failed[0]['error']['cause_type'] == 'ConnectionError'
    assert not failed[0]['not_dispatched']
    assert not any(row.get('event') == 'run_end' and row.get('reason') == 'paused'
                   for row in events(directory))
    # A subsequent invocation must replay that failure, not call the provider
    # again or silently abandon its unknown charge.
    with pytest.raises(CampaignStopped):
        build(resume=True).run()
    assert len(provider.calls) == 4 and len(evaluations) == 1
