"""Real frozen-wave recovery for crossover, using only inert local fixtures."""
import json

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.mock import MockLLM
from sciel.engine.parallel_store import WaveTransaction
from sciel.engine.prompt.sources import source_digest
from test_parallel_engine import Provider, config, evaluator, task_fixture


def test_cross_fallback_then_real_donors_survive_crash_without_redispatch(tmp_path, monkeypatch):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    cfg = config(task, tmp_path / 'capacity', 2, num_islands=2,
                 crossover=True, op_fixed='cross', max_iterations=4)
    directory = tmp_path / 'run'
    provider = Provider('run_search', concurrent=2)
    calls, _ = evaluator(monkeypatch)
    engine = build_engine(cfg, str(tasks), str(directory), llm=provider)
    original = WaveTransaction.commit

    def crash(transaction):
        if transaction.wave_id == '2':
            raise RuntimeError('fixture crash after real crossover receipts')
        return original(transaction)

    with monkeypatch.context() as stopped:
        stopped.setattr(WaveTransaction, 'commit', crash)
        with pytest.raises(CampaignStopped, match='real crossover receipts'):
            engine.run()

    saved = {path: path.read_bytes() for path in (directory / '.parallel').rglob('jobs.json')}
    jobs = sorted((job for raw in saved.values() for job in json.loads(raw)['jobs']
                   if job['job_id'].startswith('G:')), key=lambda row: int(row['job_id'][2:]))
    assert len(jobs) == 4
    for job in jobs[:2]:
        snap = job['snapshot']
        assert snap['op'] == 'full_rewrite' and snap['inspirations'] == []
        assert snap['operator_decision']['requested'] == 'cross'
        assert snap['operator_decision']['fallback_reason'] == 'no_usable_crossover_donor'
        assert job['attribution']['op_prop'] is None
    for job in jobs[2:]:
        snap = job['snapshot']
        assert snap['op'] == 'cross' and len(snap['inspirations']) == 1
        donor = snap['inspirations'][0]
        assert donor['program_id'] != snap['parent_id']
        assert donor['source_sha256'] == source_digest(donor['code']) != source_digest(snap['parent_code'])
        assert snap['operator_decision']['effective'] == 'cross'
        assert job['attribution']['op_prop'] == 1.0
    assert len(provider.calls) == 4 and len(calls) == 5

    class ReplayOnly(MockLLM):
        def generate(self, *args, **kwargs):
            pytest.fail('frozen crossover replay must not repeat provider calls')

    def no_evaluation(*args, **kwargs):
        pytest.fail('frozen crossover replay must not repeat evaluations')

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', no_evaluation)
    resumed = build_engine(cfg, str(tasks), str(directory), llm=ReplayOnly())
    checkpoint.load(resumed, str(directory / 'checkpoint.json'))
    result = resumed.run()
    assert result['iteration_no'] == result['B_evals'] == 4
    assert result['cost_total'] == pytest.approx(.12)
    assert all(path.read_bytes() == raw for path, raw in saved.items())
    assert all(not values for values in resumed.op_bandit.events.values())
