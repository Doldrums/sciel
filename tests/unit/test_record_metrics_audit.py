import importlib.util
import json
from pathlib import Path
import socket

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('record_metrics_audit', ROOT / 'scripts/audit_review_record_metrics.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_summary_pools_record_counts_and_harmonics_instead_of_averaging_ratios():
    rows = [{'n':20, 'records':2, 'expected':2, 'ratio':1},
            {'n':100, 'records':18, 'expected':6, 'ratio':3}]
    summary = audit.summarize(rows)
    assert summary == {'runs':2, 'generated_evaluations':120, 'records':20,
                       'harmonic_sum':8, 'pooled_ratio':2.5, 'median_ratio':2,
                       'below_one':0}
    empty = audit.summarize([])
    assert empty['runs'] == 0 and empty['records'] == 0
    assert empty['pooled_ratio'] is None and empty['median_ratio'] is None


def test_offline_cli_uses_supplied_inventory_without_network_or_evaluation(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail('record metrics audit must not make network/API calls')
    monkeypatch.setattr(socket, 'create_connection', no_network)
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(socket.socket, 'connect_ex', no_network)
    checkout, corpus, out = tmp_path / 'checkout', tmp_path / 'corpus', tmp_path / 'report'
    task_root = checkout / 'tasks/oe/test'
    task_root.mkdir(parents=True)
    (task_root / 'task.yaml').write_text('suite: oe\nname: test\norigin: oe\nprimary_metric: score\n')
    for name in ('evaluator.py', 'initial_program.py'):
        (task_root / name).write_text("raise AssertionError('audit must not evaluate programs')\n")
    task_hash = audit.load_task(str(task_root)).task_hash
    (checkout / 'scripts').mkdir()
    for name in ('record_ratio.py', 'journal_data.py'):
        (checkout / 'scripts' / name).write_bytes((ROOT / 'scripts' / name).read_bytes())
    run = tmp_path / 'run'
    run.mkdir()
    events = [{'event':'run_start', 'task':'oe/test', 'task_hash':task_hash, 'seed':42}]
    events += [{'event':'iteration', 'iteration_no':i, 'B_evals':i, 'child_id':f'p{i}',
                'child_score':float(i), 'is_pop_event':i in (1, 5)} for i in range(1, 21)]
    events.append({'event':'run_end', 'B_evals':20, 'iteration_no':20, 'reason':'budget'})
    journal = run / 'sciel_events.jsonl'
    journal.write_text(''.join(json.dumps(event) + '\n' for event in events))
    checkpoint = run / 'checkpoint.json'
    checkpoint.write_text('{}')
    corpus.mkdir()
    (corpus / 'excluded_runs.json').write_text('[]')
    manifest = {'baseline_commit':'test-commit', 'excluded_run_count':0,
                'files':{'excluded_runs.json':audit.sha(corpus / 'excluded_runs.json')},
                'tasks':{'oe/test':{'task_hash':task_hash, 'file_sha256':{
                    name:audit.sha(task_root / name) for name in
                    ('task.yaml', 'evaluator.py', 'initial_program.py')}}},
                'runs':[{'run_id':'test-run', 'run_path':str(run), 'task':'oe/test',
                         'source_task_hash':task_hash, 'B_evals':20, 'end_reason':'budget',
                         'journal_sha256':audit.sha(journal),
                         'checkpoint_sha256':audit.sha(checkpoint), 'split':'test'}]}
    (corpus / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(audit, 'ROOT', checkout)
    assert audit.main(['--corpus', str(corpus), '--out', str(out)]) == 0
    result = json.loads((out / 'audit.json').read_text())
    assert result['inventory_runs'] == result['eligible_runs'] == 1
    assert result['short_runs'] == result['hygiene_excluded_runs'] == 0
    assert result['summary']['records'] == 2
    report = (out / 'report.md').read_text()
    assert 'Verified 1 journal hashes, 1 checkpoint hashes, 1 corpus artifact hashes' in report
    assert '1 runs qualify; 0 have fewer than 20' in report
    assert "corpus's 0 journal exclusions" in report
    assert '1 of 1 eligible runs have the same task hash' in report
    assert str(corpus) in report and str(out) in report
