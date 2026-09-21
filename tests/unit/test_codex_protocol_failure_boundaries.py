"""Typed response failures cannot hide fatal siblings or unclassified SSE errors."""
import json

import pytest

from sciel.engine import parallel_recovery
from sciel.engine.llm.client import LLMError, accounted_cost, llm_error_metadata
from sciel.engine.llm.codex_protocol import CODES, RECOVERABLE_CODES
from sciel.engine.parallel_calls import _decode
from sciel.engine.recovery import ProviderRecoveryError
from test_codex_parallel_recovery import ProtocolProvider
from test_codex_responses import MODEL, provider as fake_codex
from test_parallel_provider_recovery import physical_files, stopped_wave


def physical_receipts(directory):
    rows = []
    for path in (directory / '.parallel').rglob('receipt.json'):
        intent = json.loads(path.with_name('intent.json').read_text())
        payload = json.loads(path.read_text())['payload']
        rows.append((intent, payload))
    return rows


def assert_one_typed_eof(rows, provider):
    errors = [payload['error'] for _,payload in rows if payload['status'] == 'error']
    typed = [error for error in errors if error.get('codex_protocol')]
    assert len(typed) == len(provider.opener.calls) == 1
    assert typed[0]['codex_protocol']['code'] == 'stream_eof'
    assert parallel_recovery._protocol_failure(typed[0]) is not None
    return errors


def unknown_trace(directory):
    rows = [json.loads(line) for line in
            (directory / 'artifacts/llm_calls.jsonl').read_text().splitlines()]
    unknown = [row for row in rows if row.get('current_call_cost_unknown')]
    assert len(unknown) == 2 and all(row['cost'] is None for row in unknown)
    return unknown


def test_typed_eof_cannot_hide_a_local_worker_failure(tmp_path, monkeypatch):
    provider = ProtocolProvider(role='C')
    provider.mixed_failure = True  # A different dispatched C raises plain ValueError.
    first, _, _, _, directory = stopped_wave(
        tmp_path, monkeypatch, provider, llm=provider.codex_config)
    try:
        saved = physical_files(directory)
        errors = assert_one_typed_eof(physical_receipts(directory), provider)
        local = [error for error in errors if error.get('cause_type') == 'ValueError']
        assert local == [{'stage':'request','reason':'request_failed','cause_type':'ValueError'}]
        assert {row['event'] for row in unknown_trace(directory)} == {'error'}
        stops = list((directory / '.parallel/wave-1/provider_stops').glob('*.json'))
        assert len(stops) == 1
        classifications = json.loads(stops[0].read_text())['errors']
        assert {'provider','fatal'} <= {row['kind'] for row in classifications}
        with pytest.raises(ProviderRecoveryError, match='non-provider failure'):
            parallel_recovery.inspect_parallel_stop(directory / 'checkpoint.json')
        assert not list((directory / '.parallel').rglob('abandonment.json'))
        assert physical_files(directory) == saved and len(provider.opener.calls) == 1
    finally:
        first.close()


def test_typed_eof_cannot_hide_invalid_returned_subscription_accounting(tmp_path, monkeypatch):
    class InvalidAccounting(ProtocolProvider):
        damaged = False

        def generate(self, prompt, *args, **kwargs):
            response = super().generate(prompt, *args, **kwargs)
            if 'Review this one experiment.' in prompt[:160]:
                with self.lock:
                    if not self.damaged:
                        self.damaged = True
                        # Null USD is legitimate only with the exact typed
                        # subscription marker; this distinct returned C lacks it.
                        response.cost_provenance['endpoint_sha256'] = '0' * 64
            return response

    provider = InvalidAccounting(role='C')
    first, _, _, _, directory = stopped_wave(
        tmp_path, monkeypatch, provider, llm=provider.codex_config)
    try:
        saved = physical_files(directory)
        rows = physical_receipts(directory)
        assert_one_typed_eof(rows, provider)
        invalid = [payload for intent,payload in rows
            if intent['kind'] == 'llm' and payload['status'] == 'returned'
            and payload['value']['cost'] is None and not payload['value'].get('cost_provenance')]
        assert provider.damaged and len(invalid) == 1
        assert {row['event'] for row in unknown_trace(directory)} == {'error','response'}
        with pytest.raises(ValueError, match='cost'):
            accounted_cost(_decode('llm', invalid[0]['value']))
        # Sealing a provider stop itself refuses malformed returned accounting;
        # an EOF sibling cannot turn the unusable returned response into free work.
        with pytest.raises(ProviderRecoveryError, match='returned receipt is unusable'):
            parallel_recovery._calls(directory / '.parallel/wave-1')
        with pytest.raises(ProviderRecoveryError):
            parallel_recovery.inspect_parallel_stop(directory / 'checkpoint.json')
        assert not list((directory / '.parallel').rglob('abandonment.json'))
        assert physical_files(directory) == saved and len(provider.opener.calls) == 1
        baseline = json.loads((directory / 'checkpoint.json').read_text())
        assert baseline['cost_total'] == 0 and baseline['llm_unpriced_calls'] == 0
    finally:
        first.close()


@pytest.mark.parametrize('error_code', ['unauthorized', 'invalid_api_key', 'unclassified_future_error'])
def test_generic_sse_error_remains_fatal_even_with_a_typed_diagnostic(error_code):
    llm, opener = fake_codex([{'type':'error','code':error_code,
        'message':'private provider error detail','param':'private parameter'}])
    with pytest.raises(LLMError) as caught:
        llm.generate('p', MODEL)
    metadata = llm_error_metadata(caught.value)
    assert metadata['codex_protocol']['code'] == 'stream_error'
    assert 'stream_error' in CODES and 'stream_error' not in RECOVERABLE_CODES
    assert parallel_recovery._protocol_failure(metadata) is None
    assert parallel_recovery._classify(caught.value)['kind'] == 'fatal'
    assert len(opener.calls) == 1 and 'private' not in json.dumps(metadata)
