"""Exclusive writers and provenance-bound fresh starts; no model or subprocess."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.cli import main as cli
from sciel.engine import checkpoint
from sciel.engine.llm.mock import MockLLM
from sciel.engine.loop import Engine
from sciel.engine.run_lock import RunDirectoryBusy, RunDirectoryLock
from sciel.engine.warm_start import from_run, validate_warm_start
from sciel.eval.registry import load_task

ROOT = Path(__file__).resolve().parents[2]


def construct(path, **changes):
    cfg = dict(task='test/toy_quadratic', random_seed=1, budget=1, max_iterations=1,
        num_islands=1, embed_grid=False, embed_niche=False, portfolio_seed=False,
        idea_island=False, dynamic_islands=False, crossover=False,
        policy={'enabled': False}, llm={'mock': True}, wandb={'enabled': False})
    cfg.update(changes)
    engine = Engine(cfg, load_task(str(ROOT / 'tasks/test/toy_quadratic')), str(path), MockLLM())
    engine.llm.generate = lambda *a, **k: pytest.fail('paid work must not run')
    engine._evaluate = lambda source: SimpleNamespace(result=SimpleNamespace(
        ok=True, score=.75, cases={}, timing=.01))
    engine._stop_at = 0
    return engine


def test_second_writer_is_rejected_before_config_or_journal_changes(tmp_path):
    first = construct(tmp_path)
    config = (tmp_path / 'config.yaml').read_bytes()
    journal = (tmp_path / 'sciel_events.jsonl').read_bytes()
    try:
        with pytest.raises(RunDirectoryBusy):
            construct(tmp_path, random_seed=12)
        assert (tmp_path / 'config.yaml').read_bytes() == config
        assert (tmp_path / 'sciel_events.jsonl').read_bytes() == journal
    finally:
        first.close()
    construct(tmp_path).close()


def test_existing_run_requires_resume_and_keeps_original_config(tmp_path):
    construct(tmp_path).run()
    original = {name: (tmp_path / name).read_bytes() for name in (
        'config.yaml', 'sciel_events.jsonl', 'checkpoint.json', 'runtime.json')}
    second = construct(tmp_path, random_seed=9)
    with pytest.raises(ValueError, match='use resume'):
        second.run()
    assert all((tmp_path / name).read_bytes() == data for name, data in original.items())
    resumed = construct(tmp_path)
    checkpoint.load(resumed, str(tmp_path / 'checkpoint.json'))
    assert resumed.run()['reason'] == 'paused'
    RunDirectoryLock(tmp_path).close()


def test_constructor_failure_releases_directory_lock(tmp_path, monkeypatch):
    def fail(*a, **k):
        raise ValueError('fixture construction failure')
    monkeypatch.setattr(Engine, '_initialize', fail)
    with pytest.raises(ValueError, match='fixture'):
        construct(tmp_path)
    RunDirectoryLock(tmp_path).close()


def test_run_exception_releases_directory_lock(tmp_path):
    first = construct(tmp_path)
    first._evaluate = lambda *_: (_ for _ in ()).throw(ValueError('fixture'))
    with pytest.raises(ValueError, match='fixture'):
        first.run()
    assert first.completed_result is None
    RunDirectoryLock(tmp_path).close()


def test_completed_result_survives_cleanup_failure_without_hiding_error(tmp_path):
    engine = construct(tmp_path)
    close = engine.log.close
    calls = []
    def fail_after_completion():
        calls.append(True)
        close()
        if len(calls) == 2:
            raise OSError('fixture final cleanup failure')
    engine.log.close = fail_after_completion
    with pytest.raises(OSError, match='cleanup failure'):
        engine.run()
    assert engine.completed_result['reason'] == 'paused'
    assert engine.completed_result['pop_best'] == .75
    saved = json.loads((tmp_path / 'checkpoint.json').read_text())
    assert saved['seed_score'] == engine.completed_result['pop_best']
    assert saved['iteration_no'] == engine.completed_result['iteration_no']
    # A failing resource close must still release the exclusive writer lock.
    RunDirectoryLock(tmp_path).close()


def test_cli_new_run_rejects_existing_directory_before_build(tmp_path, monkeypatch):
    (tmp_path / 'existing').mkdir()
    (tmp_path / 'existing/config.yaml').write_text('preserved')
    monkeypatch.setattr(cli, '_resolve_config', lambda *a: {'task': 'test/toy_quadratic', 'random_seed': 1})
    monkeypatch.setattr(cli, 'build_engine', lambda *a, **k: pytest.fail('must reject before build'))
    with pytest.raises(ValueError, match='not empty'):
        cli.cmd_run(SimpleNamespace(config='unused', seed=None, budget=None,
            outputs=str(tmp_path), run_id='existing', tasks_root=None))


def test_warm_start_transfers_only_source_and_remeasures_it(tmp_path):
    parent_dir = tmp_path / 'parent'
    parent = construct(parent_dir)
    source = 'def run_search():\n    return 0.91\n'
    parent.db.add(source, .91, {}, 0, 'gen')
    parent.cost_total = .37
    parent.run()
    original = {name: (parent_dir / name).read_bytes() for name in (
        'config.yaml', 'sciel_events.jsonl', 'checkpoint.json', 'runtime.json')}
    warm = from_run(parent_dir, 'test/toy_quadratic')
    assert warm['source'] == source
    assert warm['parent']['previous_known_cost'] == .37
    child = construct(tmp_path / 'child', warm_start=warm)
    seen = []
    evaluate = child._evaluate
    child._evaluate = lambda code: (seen.append(code), evaluate(code))[1]
    child.run()
    assert seen == [source]
    assert child.seed_score == child.db.pop_best == .75
    assert child.cost_total == child.iteration_no == child.B_evals == 0
    saved = json.loads((tmp_path / 'child/checkpoint.json').read_text())
    assert saved['identity']['warm_start'] == warm
    assert all((parent_dir / name).read_bytes() == data for name, data in original.items())
    events = [json.loads(line) for line in (tmp_path / 'child/sciel_events.jsonl').read_text().splitlines()]
    event = next(row for row in events if row.get('action') == 'warm_start')
    assert event['parent']['checkpoint_sha256'] == hashlib.sha256(original['checkpoint.json']).hexdigest()


def test_warm_start_rejects_live_parent_and_checkpoint_tampering(tmp_path):
    parent = construct(tmp_path)
    with pytest.raises(RunDirectoryBusy):
        from_run(tmp_path, 'test/toy_quadratic')
    parent.run()
    warm = from_run(tmp_path, 'test/toy_quadratic')
    changed = deepcopy(warm)
    changed['source'] += '# tampered\n'
    with pytest.raises(ValueError, match='source identity'):
        validate_warm_start(changed)
    path = tmp_path / 'checkpoint.json'
    saved = json.loads(path.read_text())
    saved['cost_total'] = 999
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='digest'):
        from_run(tmp_path, 'test/toy_quadratic')


@pytest.mark.parametrize('checkpoint_fails', [False, True])
@pytest.mark.parametrize('research_on', [False, True])
def test_unknown_charge_stops_heartbeat_releases_lock_and_prevents_resume(tmp_path, checkpoint_fails, research_on):
    import threading
    from sciel.campaign.budget import CampaignStopped
    settings = {'research': {'enabled': True, 'max_aux_calls': 0}} if research_on else {}
    engine = construct(tmp_path, **settings)
    engine._stop_at = None
    def unknown(*args, **kwargs):
        error = CampaignStopped('unknown charge fixture')
        error.known_cost = .02
        error.current_call_cost_unknown = True
        raise error
    engine._step = unknown
    if checkpoint_fails:
        engine._finish = lambda *_: (_ for _ in ()).throw(OSError('fixture persistence failure'))
    with pytest.raises(OSError if checkpoint_fails else CampaignStopped):
        engine.run()
    assert engine.cost_total == .02
    assert not any(t.name == 'sciel-runtime-heartbeat' for t in threading.enumerate())
    runtime = json.loads((tmp_path / 'runtime.json').read_text())
    assert runtime['status'] == ('failed' if checkpoint_fails else 'stopped')
    RunDirectoryLock(tmp_path).close()
    if not checkpoint_fails:
        resumed = construct(tmp_path, **settings)
        checkpoint.load(resumed, str(tmp_path / 'checkpoint.json'))
        assert resumed._accounting_stopped_reason
        with pytest.raises(CampaignStopped, match='reconcile'):
            resumed.run()
