"""Rollback restores the saved population, not only its common island prefix."""
import json
import sys
from pathlib import Path

from sciel.engine.db import Database
from sciel.engine.factory import build_engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from test_parallel_engine import Provider, config, task_fixture


def test_loading_smaller_population_discards_uncommitted_islands_and_evictions():
    db = Database(1, pop_cap=1)
    db.add('a', .5, {}, 0, 'seed')
    saved = db.state()
    island = db.add_island()
    db.add('b', .1, {}, island, 'mutation')
    db.add('c', .9, {}, island, 'mutation')
    db.evictions.append({'island': island, 'victim_id': 'uncommitted'})
    db.load_state(saved)
    assert db.state() == saved and db.evictions == []
    assert db.add_island() == 1


def test_pause_during_dynamic_spawn_restores_exact_saved_database(tmp_path, monkeypatch):
    task = task_fixture(tmp_path / 'tasks')
    def evaluate(task, path, **options):
        value = EvalResult(True, .1, {task.primary_metric: .1}, timing=.01)
        return CascadeOutcome(value, .1, 'full', False)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    class StopOnSpawn(Provider):
        def generate(self, *args, **kwargs):
            result = super().generate(*args, **kwargs)
            if len(self.calls) == 2:
                (tmp_path / 'run/STOP').touch()
            return result
    provider = StopOnSpawn('run_search')
    cfg = config(task, tmp_path / 'capacity', workers=1, num_islands=1,
                 dynamic_islands=True, max_islands=2, spawn_stall=1, max_iterations=3)
    engine = build_engine(cfg, str(tmp_path / 'tasks'), str(tmp_path / 'run'),
                          python=sys.executable, llm=provider)
    try:
        result = engine.run()
        saved = json.loads((tmp_path / 'run/checkpoint.json').read_text())['db']
        assert result['reason'] == 'paused' and len(provider.calls) == 2
        assert len(saved['islands']) == 1
        assert engine.db.state() == saved
        assert engine.db.evictions == []
    finally:
        engine.close()
