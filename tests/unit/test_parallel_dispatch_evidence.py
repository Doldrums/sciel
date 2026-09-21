"""Cached outcomes cannot substitute for physical dispatch provenance."""
from contextlib import nullcontext
import json
import os

import pytest

from sciel.engine.llm.client import LLMResponse
from sciel.engine.parallel_calls import CallLedger, durable_json
from sciel.engine.parallel_recovery import _calls
from sciel.engine.recovery import ProviderRecoveryError
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult


class Capacity:
    def acquire(self, *args, **kwargs):
        return nullcontext()


@pytest.mark.parametrize('kind', ['llm', 'evaluation', 'tool_execution'])
@pytest.mark.parametrize('damage', ['missing', 'list', 'request', 'pid', 'timestamp', 'symlink', 'fifo'])
def test_normal_and_recovery_replay_reject_damaged_dispatch(tmp_path, kind, damage):
    ledger = CallLedger(tmp_path, Capacity(), 'fixture')
    directory, request_hash, _ = ledger.prepare('saved', kind, {})
    value = (LLMResponse('ok', 'mock', .01, 1, 1) if kind == 'llm' else
             CascadeOutcome(EvalResult(True, 1., {'score': 1.}, timing=.01), 1., 'full', True)
             if kind == 'evaluation' else {'result': 'retained'})
    ledger.perform(directory, request_hash, kind, lambda: value)
    receipt = (directory / 'receipt.json').read_bytes()
    marker = directory / 'dispatched.json'
    record = json.loads(marker.read_text())
    marker.unlink()
    if damage == 'list':
        durable_json(marker, [])
    elif damage in ('request', 'pid', 'timestamp'):
        record[{'request': 'request_sha256', 'pid': 'pid', 'timestamp': 'timestamp'}[damage]] = None
        durable_json(marker, record)
    elif damage == 'symlink':
        target = tmp_path / 'alias.json'
        durable_json(target, record)
        marker.symlink_to(target)
    elif damage == 'fifo':
        os.mkfifo(marker)
    with pytest.raises(ValueError, match='dispatch evidence'):
        ledger.prepare('saved', kind, {})
    with pytest.raises(ProviderRecoveryError, match='dispatch'):
        _calls(tmp_path)
    assert (directory / 'receipt.json').read_bytes() == receipt
    if damage == 'missing':
        assert not marker.exists()


def test_valid_terminal_history_is_replayed_without_dispatch(tmp_path):
    ledger = CallLedger(tmp_path, Capacity(), 'fixture')
    directory, request_hash, _ = ledger.prepare('saved', 'llm', {})
    original = ledger.perform(directory, request_hash, 'llm',
                              lambda: LLMResponse('saved', 'mock', .01, 1, 1))
    _, _, replay = ledger.prepare('saved', 'llm', {})
    assert replay == original
    assert _calls(tmp_path)[0]['status'] == 'returned'
