"""429 scheduling hints, without billing claims or hidden request replay."""
from datetime import datetime, timezone
import json
import multiprocessing
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.campaign.budget import RequestNotDispatched
from sciel.engine import capacity
from sciel.engine.llm import client, errors


class RateLimitError(Exception):
    status_code = 429

    def __init__(self, header):
        class Headers:
            def get(self, key):
                assert key in {'retry-after', 'Retry-After'}
                return header
        self.response = SimpleNamespace(headers=Headers())

    @property
    def body(self):
        raise AssertionError('private body must not be read')

    def __str__(self):
        raise AssertionError('private exception text must not be read')


@pytest.mark.parametrize('header,expected', [
    ('12', {'retry_after_seconds': 12.0}),
    (' 0.25 ', {'retry_after_seconds': .25}),
    ('0', {'retry_after_seconds': 0.0}),
    ('7200', {'retry_after_seconds': 3600.0, 'retry_after_clamped': True}),
    ('-3', {}), ('NaN', {}), ('Infinity', {}), ('private-provider-message', {}),
    ('9' * 300, {}), (None, {}), (12, {}),
])
def test_retry_after_numeric_sanitized_and_bounded(header, expected):
    metadata = client.llm_error_metadata(client.LLMError(cause=RateLimitError(header)))
    assert {key: value for key, value in metadata.items() if key.startswith('retry_after')} == expected
    assert metadata['http_status'] == 429
    assert 'private' not in json.dumps(metadata)


def test_http_date_and_elapsed_date(monkeypatch):
    now = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(errors, 'datetime', SimpleNamespace(now=lambda zone: now))
    future = client.llm_error_metadata(client.LLMError(cause=RateLimitError('Sat, 19 Sep 2026 12:00:17 GMT')))
    past = client.llm_error_metadata(client.LLMError(cause=RateLimitError('Sat, 19 Sep 2026 11:59:00 GMT')))
    assert future['retry_after_seconds'] == 17
    assert past['retry_after_seconds'] == 0


def test_hint_not_read_for_other_status_or_response_stage():
    error = RateLimitError('12')
    error.status_code = 503
    assert 'retry_after_seconds' not in client.llm_error_metadata(client.LLMError(cause=error))
    error.status_code = 429
    assert 'retry_after_seconds' not in client.llm_error_metadata(
        client.LLMError(cause=error, stage='response'))
    malicious = client.LLMError()
    malicious._metadata = dict(stage='request', reason='http_error', cause_type='RateLimitError',
                               http_status=429, retry_after_seconds='private-secret')
    assert 'retry_after_seconds' not in client.llm_error_metadata(malicious)


def test_ensemble_single_attempt_still_unknown_not_notdispatched():
    llm = client.EnsembleLLM.__new__(client.EnsembleLLM)
    llm.temperature, llm.max_tokens, llm.max_retries = .7, 100, 1
    llm._client = Mock()
    llm._client.chat.completions.create.side_effect = RateLimitError('15')
    with pytest.raises(client.LLMError) as failure:
        llm.generate('local mock prompt', 'mock')
    assert not isinstance(failure.value, RequestNotDispatched)
    assert llm._client.chat.completions.create.call_count == 1
    assert failure.value.metadata['retry_after_seconds'] == 15
    assert 'cost' not in failure.value.metadata


def test_provider_cooldown_shared_and_does_not_hold_global_permit(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(capacity.time, 'time', lambda: clock[0])
    first = capacity.HostCapacity(tmp_path, llm_slots=1, evaluator_slots=1)
    second = capacity.HostCapacity(tmp_path, llm_slots=1, evaluator_slots=1)
    receipt = first.defer_provider('provider-a', 20)
    assert receipt['until_unix'] == 1020
    with pytest.raises(capacity.CapacityTimeout):
        second.acquire('llm', provider='provider-a', timeout=0)
    with second.acquire('llm', provider='provider-b', timeout=0):
        pass
    with second.acquire('evaluator', timeout=0):
        pass
    # A shorter concurrent hint cannot shorten another run's longer cooldown.
    assert second.defer_provider('provider-a', 2)['until_unix'] == 1020
    clock[0] = 1021
    with second.acquire('llm', provider='provider-a', timeout=0):
        pass


def _cross_process_wait(root, pipe):
    manager = capacity.HostCapacity(root, llm_slots=1, evaluator_slots=1)
    try:
        lease = manager.acquire('llm', provider='provider-a', timeout=0)
    except capacity.CapacityTimeout:
        pipe.send('cooldown-observed')
    else:
        lease.release()
        pipe.send('unexpected-grant')
    finally:
        pipe.close()


def test_provider_cooldown_visible_to_independent_process(tmp_path):
    manager = capacity.HostCapacity(tmp_path, llm_slots=1, evaluator_slots=1)
    manager.defer_provider('provider-a', 30)
    context = multiprocessing.get_context('spawn')
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_cross_process_wait, args=(str(tmp_path), writer))
    process.start()
    writer.close()
    try:
        assert reader.poll(5)
        assert reader.recv() == 'cooldown-observed'
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        reader.close()


def test_cooldown_wait_remains_cancellable(tmp_path):
    import threading
    manager = capacity.HostCapacity(tmp_path, llm_slots=1, evaluator_slots=1)
    manager.defer_provider('provider-a', 3600)
    stop = threading.Event()
    stop.set()
    with pytest.raises(capacity.CapacityCancelled):
        manager.acquire('llm', provider='provider-a', cancel=stop)
    with pytest.raises(capacity.CapacityTimeout):
        manager.acquire('llm', provider='provider-a', timeout=.005)


@pytest.mark.parametrize('value', [-1, 3601, float('nan'), float('inf'), True, '20'])
def test_invalid_cooldown_cannot_create_shared_state(tmp_path, value):
    manager = capacity.HostCapacity(tmp_path)
    with pytest.raises(ValueError):
        manager.defer_provider('provider', value)
    assert not list(tmp_path.glob('*.cooldown.json'))
