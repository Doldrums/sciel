"""Offline outage regressions: no API dispatch, local evaluators or model loading."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier, Event
import traceback
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError
import pytest

from sciel.campaign.budget import BudgetedLLM, CampaignStopped, CostLedger
from sciel.engine.loop import Engine
from sciel.engine.observed_changes import ObservedChanges
from sciel.engine.worker import run_worker


SECRET = "fixture-only-secret-request-and-body"
CFG = {'models':['m'], 'weights':[1], 'max_tokens':100,
       'api_base':'https://openrouter.ai/api/v1'}
PRICING = {'models':[{'id':'m', 'pricing':{'prompt':'.000001', 'completion':'.000002'}}]}


def wrapped(path, run='r'):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        id='gen-offline', usage=SimpleNamespace(cost=.0001, prompt_tokens=5, completion_tokens=8),
        choices=[SimpleNamespace(message=SimpleNamespace(content='code'))])
    ledger = CostLedger(path, 1)
    return BudgetedLLM(CFG, PRICING, ledger, run, 'unused-key', client), client, ledger


def failure(kind):
    request = httpx.Request('POST', 'https://example.invalid',
                            headers={'Authorization':SECRET}, content=SECRET)
    if kind == 'timeout':
        exc = APITimeoutError(request=request)
    elif kind == 'connection':
        exc = APIConnectionError(message=SECRET, request=request)
    elif type(kind) is int:
        exc = APIStatusError(SECRET, response=httpx.Response(kind, request=request),
                             body={'error':SECRET})
    else:
        exc = RuntimeError(SECRET)
    exc.__cause__ = ConnectionRefusedError(SECRET)
    return exc


@pytest.mark.parametrize('kind', ['timeout', 'connection', 401, 429, 500, 503, 'unexpected'])
def test_unknown_api_outcome_halts_once_without_exposing_remote_content(tmp_path, kind):
    llm, client, ledger = wrapped(tmp_path/'ledger.json')
    original = failure(kind)
    client.chat.completions.create.side_effect = original
    with pytest.raises(CampaignStopped, match='outcome is unknown') as caught:
        llm.generate('prompt', 'm')
    persisted = json.loads(ledger.path.read_text())
    record, = persisted['calls'].values()
    assert record['status'] == 'unknown' and record['cost_usd'] is None
    assert record['error_type'] == type(original).__name__
    assert record['error_status_code'] == (kind if type(kind) is int else None)
    assert record['error_cause_type'] == 'ConnectionRefusedError'
    assert Decimal(ledger.snapshot()['reserved_or_unknown_usd']) == Decimal(record['reservation_usd'])
    assert ledger.snapshot()['known_cost_usd'] == '0'
    assert SECRET not in ledger.path.read_text()
    assert SECRET not in ''.join(traceback.format_exception(caught.value))
    assert caught.value.__suppress_context__ is True
    # Simulate the old engine's repeated iterations: every later attempt is
    # rejected before the client, even though the dollar cap has ample room.
    for _ in range(5):
        with pytest.raises(CampaignStopped, match='outcome is unknown'):
            llm.generate('another prompt', 'm')
    assert client.chat.completions.create.call_count == 1
    assert ledger.snapshot()['calls'] == 1


@pytest.mark.parametrize('status', [True, '503', 99, 600, None])
def test_only_integer_http_status_is_persisted(tmp_path, status):
    llm, client, ledger = wrapped(tmp_path/'ledger.json')
    exc = RuntimeError(SECRET)
    exc.status_code = status
    client.chat.completions.create.side_effect = exc
    with pytest.raises(CampaignStopped):
        llm.generate('prompt', 'm')
    record, = json.loads(ledger.path.read_text())['calls'].values()
    assert record['error_status_code'] is None and record['error_cause_type'] is None


def test_broken_exception_status_accessor_cannot_prevent_durable_stop(tmp_path):
    class BrokenStatus(RuntimeError):
        @property
        def status_code(self):
            raise RuntimeError(SECRET)
    llm, client, ledger = wrapped(tmp_path/'ledger.json')
    client.chat.completions.create.side_effect = BrokenStatus(SECRET)
    with pytest.raises(CampaignStopped):
        llm.generate('prompt', 'm')
    assert ledger.snapshot()['halted']
    assert SECRET not in ledger.path.read_text()


def test_inflight_peer_settles_known_cost_then_stops_without_another_dispatch(tmp_path):
    first, first_client, ledger = wrapped(tmp_path/'ledger.json', 'A')
    second, second_client, _ = wrapped(tmp_path/'ledger.json', 'B')
    entered = Barrier(2)
    release_peer = Event()
    successful_response = second_client.chat.completions.create.return_value
    def disconnect(**kwargs):
        entered.wait(timeout=5)
        raise failure('timeout')
    def complete_after_disconnect(**kwargs):
        entered.wait(timeout=5)
        assert release_peer.wait(timeout=5)
        return successful_response
    first_client.chat.completions.create.side_effect = disconnect
    second_client.chat.completions.create.side_effect = complete_after_disconnect
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first.generate, 'prompt', 'm')
        b = pool.submit(second.generate, 'prompt', 'm')
        try:
            with pytest.raises(CampaignStopped):
                a.result(timeout=5)
            assert ledger.snapshot()['halted']
        finally:
            release_peer.set()
        with pytest.raises(CampaignStopped):
            b.result(timeout=5)
    records = json.loads(ledger.path.read_text())['calls'].values()
    assert sorted(r['status'] for r in records) == ['settled', 'unknown']
    assert Decimal(ledger.snapshot()['known_cost_usd']) == Decimal('.0001')
    unknown, = [r for r in records if r['status'] == 'unknown']
    assert Decimal(ledger.snapshot()['reserved_or_unknown_usd']) == Decimal(unknown['reservation_usd'])
    for llm, client in ((first, first_client), (second, second_client)):
        with pytest.raises(CampaignStopped):
            llm.generate('next', 'm')
        assert client.chat.completions.create.call_count == 1


def test_new_process_observes_durable_halt_even_for_zero_cost_reservation(tmp_path):
    llm, client, ledger = wrapped(tmp_path/'ledger.json')
    client.chat.completions.create.side_effect = failure('connection')
    with pytest.raises(CampaignStopped):
        llm.generate('prompt', 'm')
    code = '''
import sys
from sciel.campaign.budget import CampaignStopped, CostLedger
ledger = CostLedger(sys.argv[1], 1)
try:
    ledger.reserve(0, run_id='restarted-peer', model='m')
except CampaignStopped:
    print('halted before dispatch')
else:
    raise AssertionError('shared halt was lost across process restart')
'''
    result = subprocess.run([sys.executable, '-c', code, str(ledger.path)],
                            env={**os.environ, 'PYTHONPATH':str(Path(__file__).resolve().parents[2]/'src')},
                            check=True, capture_output=True, text=True, timeout=10)
    assert result.stdout.strip() == 'halted before dispatch'
    assert ledger.snapshot()['calls'] == 1


def test_regular_worker_does_not_turn_campaign_stop_into_retryable_nochild(monkeypatch):
    monkeypatch.setattr('sciel.engine.worker.build', lambda *args: 'prompt')
    llm = Mock()
    llm.generate.side_effect = CampaignStopped('shared stop')
    evaluate = Mock()
    with pytest.raises(CampaignStopped, match='generation outcome and cost are unknown') as stopped:
        run_worker('diff', SimpleNamespace(model='m', island=0), llm, evaluate, dup_cap=200)
    assert stopped.value.current_call_cost_unknown is True
    assert stopped.value.generation_dispatch == 'unknown' and stopped.value.known_cost == 0
    llm.generate.assert_called_once()
    evaluate.assert_not_called()


@pytest.mark.parametrize('origin', ['portfolio', 'spawn'])
def test_paradigm_generation_cannot_swallow_campaign_stop(origin):
    llm = Mock()
    llm.generate.side_effect = CampaignStopped('shared stop')
    engine = SimpleNamespace(llm=llm, task_prompt='task', rng=Mock(), _evaluate=Mock(),
                             observed_changes=ObservedChanges(), _per_run_budget=lambda: 6.)
    with pytest.raises(CampaignStopped, match='shared stop'):
        Engine._paradigm_child(engine, 'def solve(): return 1', 'constructive', 0, origin)
    llm.generate.assert_called_once()
    engine._evaluate.assert_not_called()
