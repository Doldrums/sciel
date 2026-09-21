import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.campaign.budget import RequestNotDispatched
from sciel.engine.llm.client import LLMError, LLMResponse
from sciel.engine.llm.trace import TracedLLM


def rows(path):
    return [json.loads(line) for line in (path / 'artifacts/llm_calls.jsonl').read_text().splitlines()]


def test_request_durable_before_provider_and_full_response_round_trip(tmp_path):
    prompt, reply = 'Complete prompt\n' + 'x' * 70000, 'Полный ответ\n' + 'y' * 70000
    response = LLMResponse(reply, 'fixture', .4, 12, 15)
    def call(actual, model, temperature, **kwargs):
        request, = rows(tmp_path)
        assert request['event'] == 'request'
        assert request['role'] == 'C' and request['journal_event_seq'] == 42
        saved = tmp_path / f"artifacts/llm_calls/{request['call_id']}.prompt.txt"
        assert saved.read_text() == prompt == actual
        assert kwargs['response_format'] == {'type': 'json_object'}
        return response
    provider = SimpleNamespace(generate=Mock(side_effect=call), weights=[1.0])
    traced = TracedLLM(provider, tmp_path, lambda p: dict(role='C', journal_event_seq=42))
    assert traced.generate(prompt, 'fixture', .2, response_format={'type': 'json_object'}) is response
    request, result = rows(tmp_path)
    assert result['call_id'] == request['call_id']
    assert result['event'] == 'response' and result['cost'] == .4
    assert result['response_chars'] == len(reply)
    assert result['response_sha256'] == hashlib.sha256(reply.encode()).hexdigest()
    assert (tmp_path / f"artifacts/llm_calls/{result['call_id']}.response.txt").read_text() == reply
    traced.weights = [.5]
    assert provider.weights == [.5]


def test_error_unknown_cost_sanitized_and_no_retry(tmp_path):
    secret = 'Bearer private-provider-token'
    error = LLMError(cause=ConnectionError(secret))
    provider = SimpleNamespace(generate=Mock(side_effect=error))
    traced = TracedLLM(provider, tmp_path, lambda p: dict(role='G'))
    with pytest.raises(LLMError) as caught:
        traced.generate('request text', 'fixture')
    assert caught.value is error and provider.generate.call_count == 1
    result = rows(tmp_path)[-1]
    assert result['event'] == 'error' and result['cost'] is None
    assert result['current_call_cost_unknown'] is True
    assert result['llm_error']['reason'] == 'connection_error'
    assert secret not in json.dumps(rows(tmp_path))
    assert not list((tmp_path / 'artifacts/llm_calls').glob('*.response.txt'))


def test_failed_request_persistence_prevents_dispatch(tmp_path, monkeypatch):
    provider = SimpleNamespace(generate=Mock())
    traced = TracedLLM(provider, tmp_path, lambda p: {})
    monkeypatch.setattr(TracedLLM, '_emit', Mock(side_effect=OSError('disk full')))
    with pytest.raises(RequestNotDispatched):
        traced.generate('prompt', 'fixture')
    provider.generate.assert_not_called()


def test_response_persistence_failure_keeps_paid_response_for_accounting(tmp_path, monkeypatch):
    response = LLMResponse('answer', 'fixture', .7, 2, 3)
    traced = TracedLLM(SimpleNamespace(generate=lambda *a: response), tmp_path, lambda p: {})
    original = TracedLLM._text
    def fail_reply(self, ident, kind, text):
        if kind == 'response':
            raise OSError('disk full')
        return original(self, ident, kind, text)
    monkeypatch.setattr(TracedLLM, '_text', fail_reply)
    assert traced.generate('prompt', 'fixture') is response
