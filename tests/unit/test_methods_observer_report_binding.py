"""Current observer terminal classification, using inert child/process hooks."""
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from types import SimpleNamespace

import pytest

from sciel.diagnostics import methods_observer as observer


def observe_terminal(tmp_path, monkeypatch, *, previous, current=None,
                     exit_code=2, worker_change=None, actual_parser=False):
    output = tmp_path/'run'
    output.mkdir()
    observer._atomic(output/'run-report.json', previous)
    terminal = tmp_path/'worker.json'
    calls = []
    pid = os.getpid() if actual_parser else 424242
    worker = dict(schema='sciel-method-worker-terminal/1', pid=pid,
                  exit_code=exit_code, controller_cpu_seconds=.02, error_type=None)
    if worker_change:
        worker.update(worker_change)

    def launch(*args, **kwargs):
        calls.append('fake process only')
        if current is not None:
            observer._atomic(output/'run-report.json', current)
        if actual_parser:
            # Actual CLI parser refusal, before credentials/executors can be read.
            config = tmp_path/'invalid.json'
            config.write_text('{')
            captured = StringIO()
            with redirect_stdout(captured):
                assert observer._worker(config, output, terminal) == 2
            assert json.loads(captured.getvalue())['error_type'] == 'JSONDecodeError'
        elif worker_change != {'missing': True}:
            observer._atomic(terminal, worker)
        return SimpleNamespace(pid=pid, returncode=exit_code, poll=lambda: exit_code)

    monkeypatch.setattr(observer.subprocess, 'Popen', launch)
    monkeypatch.setattr(observer, 'process_identity', lambda pid: {'pid': pid, 'fixture': True})
    monkeypatch.setattr(observer, 'process_matches', lambda _: False)
    monkeypatch.setattr(observer, '_sample_cpu', lambda _: None)
    monkeypatch.setattr(observer.signal, 'signal', lambda *args: None)
    value = observer.observe(['never executed'], output, resume=True, worker_terminal=terminal)
    assert calls == ['fake process only']
    assert json.loads((output/'runtime.json').read_text()) == value
    assert json.loads(next((output/'observation').glob('*/terminal.json')).read_text()) == value
    return value


@pytest.mark.parametrize('previous', [{'status':'budget_exhausted'}, {'status':'completed'}])
@pytest.mark.parametrize('actual_parser', [False, True])
def test_current_refusal_does_not_inherit_old_terminal_report(tmp_path, monkeypatch, previous, actual_parser):
    value = observe_terminal(tmp_path, monkeypatch, previous=previous, actual_parser=actual_parser)
    assert value['status'] == 'failed'
    assert value['reason'] == 'controller_nonzero_exit'
    assert value['controller_measurement_terminal'] is True


@pytest.mark.parametrize('report,expected', [
    ({'status':'budget_exhausted'}, ('stopped', 'budget_exhausted')),
    ({'status':'accounting_stopped'}, ('unknown', 'accounting_stopped')),
])
def test_fresh_identical_report_rewrite_is_current(tmp_path, monkeypatch, report, expected):
    value = observe_terminal(tmp_path, monkeypatch, previous=report, current=report)
    assert (value['status'], value['reason']) == expected


@pytest.mark.parametrize('worker_change', [{'missing':True}, {'pid':123}, {'exit_code':1}])
def test_fresh_report_without_matching_worker_stays_unknown(tmp_path, monkeypatch, worker_change):
    value = observe_terminal(tmp_path, monkeypatch, previous={'status':'completed'},
        current={'status':'budget_exhausted'}, worker_change=worker_change)
    assert value['status'] == 'unknown'
    assert value['controller_measurement_terminal'] is False


def test_zero_exit_does_not_certify_an_unchanged_success_report(tmp_path, monkeypatch):
    value = observe_terminal(tmp_path, monkeypatch, previous={'status':'completed'}, exit_code=0)
    assert value['status'] == 'unknown'
    assert value['reason'] == 'missing_or_inconsistent_run_report'


def test_new_completed_report_is_current(tmp_path, monkeypatch):
    value = observe_terminal(tmp_path, monkeypatch, previous={'status':'budget_exhausted'},
        current={'status':'completed'}, exit_code=0)
    assert (value['status'], value['reason']) == ('completed', None)
