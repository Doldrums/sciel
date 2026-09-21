"""Mocked terminal process receipts: no child or candidate is executed."""
import json
import hashlib
from pathlib import Path

import pytest

from sciel.eval import sandbox
from sciel.eval.facade import EvalResult


def run_terminal(monkeypatch, returncode, payload):
    cleanups = []

    class Process:
        pid = 999999999
        stderr = None

        def __init__(self):
            self.returncode = returncode

        def wait(self, timeout=None):
            return self.returncode

    class Tail:
        def __init__(self, stream):
            pass

        def finish(self):
            return 'terminal stderr'

    def popen(argv, **kwargs):
        if payload is None:
            Path(argv[5]).unlink()
        else:
            Path(argv[5]).write_text(payload)
        return Process()

    monkeypatch.setattr(sandbox.subprocess, 'Popen', popen)
    monkeypatch.setattr(sandbox, '_StderrTail', Tail)
    monkeypatch.setattr(sandbox, '_kill_group', lambda proc: cleanups.append(proc.pid))
    monkeypatch.setattr(sandbox, '_evaluation_env', lambda: {})
    result = sandbox.run_stage('never-read-evaluator', 'full', 'never-read-candidate', 1.)
    assert cleanups, 'mocked cleanup contract must still run'
    return result


@pytest.mark.parametrize('returncode', [1, 137, -9])
@pytest.mark.parametrize('reported_ok', [True, False])
def test_nonzero_exit_cannot_admit_saved_result(monkeypatch, returncode, reported_ok):
    payload = EvalResult(reported_ok, .9, cases={'combined_score': .9},
                         artifacts={'observation': 'kept'}, stderr='saved stderr',
                         timing=2., cost=.1, confirmed_sampled=True).to_json()
    result = run_terminal(monkeypatch, returncode, json.dumps(payload))
    assert result.ok is False and result.score == 0
    assert result.cases == {} and result.confirmed_sampled is False
    assert result.timing == 2. and result.cost == .1
    assert result.artifacts['error_type'] == 'EvaluatorProcessExit'
    assert result.artifacts['returncode'] == str(returncode)
    assert json.loads(result.artifacts['saved_result_json']) == payload
    assert result.artifacts['saved_result_sha256'] == hashlib.sha256(
        result.artifacts['saved_result_json'].encode('utf-8')).hexdigest()
    assert 'saved stderr' in result.stderr and 'terminal stderr' in result.stderr
    assert EvalResult.from_json(result.to_json()).ok is False


@pytest.mark.parametrize('payload', [None, '', 'not JSON', '{"ok":true,"score":1,"score":2}'])
def test_nonzero_exit_with_missing_or_malformed_payload_keeps_exit_diagnostic(monkeypatch, payload):
    result = run_terminal(monkeypatch, 1, payload)
    assert result.ok is False
    assert result.artifacts['error_type'] == 'EvaluatorProcessExit'
    assert result.artifacts['returncode'] == '1'
    assert result.artifacts['saved_result_error']


def test_large_saved_payload_is_diagnostic_only_and_bounded(monkeypatch):
    payload = EvalResult(True, .9, artifacts={'observation': '測' * 20000}).to_json()
    result = run_terminal(monkeypatch, -9, json.dumps(payload))
    assert result.ok is False and result.score == 0
    assert 'saved_result_json' not in result.artifacts
    assert len(result.artifacts['saved_result_excerpt'].encode('utf-8')) <= 8192
    assert result.artifacts['saved_result_truncated'] == 'true'
    assert len(result.artifacts['saved_result_sha256']) == 64
    assert EvalResult.from_json(result.to_json()).ok is False


def test_zero_exit_preserves_existing_success_contract(monkeypatch):
    payload = EvalResult(True, .9, artifacts={'observation': 'kept'}, timing=2., cost=.1).to_json()
    result = run_terminal(monkeypatch, 0, json.dumps(payload))
    assert result.ok is True and result.score == .9
    assert result.artifacts == payload['artifacts']
    assert result.timing == 2. and result.cost == .1


@pytest.mark.parametrize('payload', [
    '{"ok":true,"score":1,"score":2}', '{"ok":"false","score":1}',
    '{"ok":true,"score":NaN}',
])
def test_zero_exit_does_not_weaken_strict_result_reader(monkeypatch, payload):
    result = run_terminal(monkeypatch, 0, payload)
    assert result.ok is False and result.score == 0
    assert result.artifacts['error_type'] == 'ResultCorrupt'
