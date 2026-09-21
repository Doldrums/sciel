"""Real G/C WAL recovery after typed read failures; no HTTP or real credentials."""
from collections import Counter
import hashlib
from http.client import IncompleteRead
import json
import ssl
from urllib.error import URLError

import pytest

from sciel.engine import parallel_recovery
from sciel.engine.recovery import ProviderRecoveryError

from test_codex_parallel_recovery import ProtocolProvider
from test_parallel_provider_recovery import (
    events, physical_files, stop_after_publication, stopped_wave,
)


class InterruptedStream:
    """Deliver valid partial assistant output, then a stdlib transport error."""

    def __init__(self, code, wrapped):
        self.code, self.wrapped = code, wrapped
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read1(self, size):
        self.reads += 1
        if self.reads == 1:
            rows = [
                {'type': 'response.created', 'response': {
                    'id': 'offline-response', 'model': 'mock-model',
                    'status': 'in_progress', 'output': []}},
                {'type': 'response.output_item.added', 'output_index': 0,
                 'item': {'id': 'offline-message', 'type': 'message',
                          'role': 'assistant', 'phase': 'final_answer',
                          'status': 'in_progress', 'content': []}},
                {'type': 'response.output_text.delta', 'output_index': 0,
                 'content_index': 0, 'item_id': 'offline-message',
                 'delta': 'PRIVATE_UNCONFIRMED_TEXT'},
            ]
            return b''.join(b'data: ' + json.dumps(row).encode() + b'\n\n'
                            for row in rows)
        if self.code == 'transport_incomplete_read':
            error = IncompleteRead(b'PRIVATE_PARTIAL_BYTES', 123)
        else:
            error = ssl.SSLEOFError(8, 'PRIVATE_TLS_DETAIL')
        raise URLError(error) if self.wrapped else error


class InterruptedOpener:
    def __init__(self, code, wrapped):
        self.code, self.wrapped = code, wrapped
        self.calls = []

    def open(self, request, timeout):
        # Deliberately no urllib/network operation or authentication-file read.
        self.calls.append((request, timeout))
        return InterruptedStream(self.code, self.wrapped)


class TransportProvider(ProtocolProvider):
    def __init__(self, role, code, wrapped=False):
        super().__init__(role=role)
        self.opener = InterruptedOpener(code, wrapped)
        self.adapter._opener = self.opener


def unknown_trace(directory):
    return [row for row in (
        json.loads(line) for line in
        (directory / 'artifacts/llm_calls.jsonl').read_text().splitlines())
        if row.get('current_call_cost_unknown')]


@pytest.mark.parametrize('role', ['G', 'C'])
@pytest.mark.parametrize(('code', 'wrapped'), [
    ('transport_incomplete_read', False),
    ('transport_tls_eof', True),
])
def test_typed_read_failure_cold_recovery_never_reissues_failed_g_or_c(
        tmp_path, monkeypatch, role, code, wrapped):
    provider = TransportProvider(role, code, wrapped)
    first, build, _, evaluations, directory = stopped_wave(
        tmp_path, monkeypatch, provider, llm=provider.codex_config)
    path = directory / 'checkpoint.json'
    try:
        baseline = json.loads(path.read_text())
        assert baseline['iteration_no'] == baseline['B_evals'] == 0
        assert baseline['llm_unpriced_calls'] == 0
        recovery = parallel_recovery.inspect_parallel_stop(path)
        assert recovery['unknown_cost'] is True and recovery['retry'] is False
        assert len(recovery['failed_calls']) == len(provider.opener.calls) == 1
        failed = recovery['failed_calls'][0]
        assert failed['key'].startswith(role + ':')
        assert failed['not_dispatched'] is False
        diagnostic = failed['error']['codex_protocol']
        assert diagnostic['code'] == code
        assert diagnostic['body_sha256'] == hashlib.sha256(
            provider.opener.calls[0][0].data).hexdigest()
        assert diagnostic['model'] == 'mock-model'
        assert diagnostic['events_seen'] == 3
        assert diagnostic['last_event'] == 'response.output_text.delta'
        assert diagnostic['terminal_status'] == 'none'
        assert 'PRIVATE_' not in json.dumps(failed['error'])
        saved = physical_files(directory)
        unknown = unknown_trace(directory)
        assert len(unknown) == 1 and unknown[0]['cost'] is None
        assert unknown[0]['event'] == 'error'
        parallel_recovery.authorize_parallel_recovery(first, path, recovery)
        markers = {str(p): p.read_bytes() for p in
                   (directory / '.parallel').rglob('abandonment.json')}
        parallel_recovery.authorize_parallel_recovery(first, path, recovery)
        assert markers == {str(p): p.read_bytes() for p in
                           (directory / '.parallel').rglob('abandonment.json')}
    finally:
        first.close()

    generate = provider.generate

    def remaining_conclusions_only(prompt, *args, **kwargs):
        assert role == 'G' and 'Review this one experiment.' in prompt[:160], (
            'an old generation or conclusion was physically reissued')
        return generate(prompt, *args, **kwargs)

    monkeypatch.setattr(provider, 'generate', remaining_conclusions_only)
    if role == 'C':
        monkeypatch.setattr('sciel.engine.loop.cascade_evaluate',
                            lambda *a, **k: pytest.fail('completed evaluator replayed'))
    stop_after_publication(monkeypatch, directory)
    resumed = build(resume=True)
    result = resumed.run()
    assert result['reason'] == 'paused' and result['iteration_no'] == 4
    assert result['B_evals'] == (3 if role == 'G' else 4)
    assert len(evaluations) == (4 if role == 'G' else 5)  # Includes canonical seed.
    assert Counter(provider.roles) == {'G': 4, 'C': 3 if role == 'G' else 4}
    assert result['cost_total'] == 0  # Known subtotal, not an assertion of free work.
    assert resumed.llm_unpriced_calls == (6 if role == 'G' else 7)
    assert len(provider.opener.calls) == 1
    assert all(physical_files(directory)[key] == value for key, value in saved.items())
    assert unknown_trace(directory) == unknown  # Unknown failed charge is not erased/relogged.
    assert len(resumed.research.knowledge.lessons) == 3
    phase = 'generation_abandoned' if role == 'G' else 'aux_abandoned'
    abandoned = [row for row in events(directory) if row.get('phase') == phase]
    assert len(abandoned) == 1
    assert abandoned[0]['cost'] is None and abandoned[0]['current_call_cost_unknown'] is True
    assert abandoned[0]['provider_abandonment']['request_sha256'] == failed['request_sha256']
    if role == 'C':
        skipped = [c for c in resumed.research.conclusions.values() if c['status'] == 'skipped']
        assert len(skipped) == 1 and skipped[0]['llm'] is None
        assert skipped[0]['accepted_lesson_ids'] == []
    else:
        assert resumed.n_nochild == 1
        assert abandoned[0]['experiment_id'] not in resumed.research.conclusions
        assert not any(abandoned[0]['experiment_id'] in lesson['evidence_ids']
                       for lesson in resumed.research.knowledge.lessons)

    # A second cold load reuses the committed result without new physical work,
    # duplicate unknown-cost records, lessons, or accounting.
    committed_files = physical_files(directory)
    monkeypatch.setattr(provider, 'generate', lambda *a, **k: pytest.fail('cold redispatch'))
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate',
                        lambda *a, **k: pytest.fail('cold reevaluation'))
    again = build(resume=True)
    again._stop_at = result['B_evals']
    again.run()
    assert again.llm_unpriced_calls == resumed.llm_unpriced_calls
    assert len(again.research.knowledge.lessons) == 3
    assert physical_files(directory) == committed_files
    assert unknown_trace(directory) == unknown
    assert len([row for row in events(directory) if row.get('phase') == phase]) == 1


def test_typed_read_failure_cannot_authorize_a_mixed_local_worker_error(tmp_path, monkeypatch):
    provider = TransportProvider('C', 'transport_incomplete_read')
    provider.mixed_failure = True
    first, _, _, _, directory = stopped_wave(
        tmp_path, monkeypatch, provider, llm=provider.codex_config)
    try:
        saved = physical_files(directory)
        errors = []
        for path in (directory / '.parallel').rglob('receipt.json'):
            payload = json.loads(path.read_text())['payload']
            if payload['status'] == 'error':
                errors.append(payload['error'])
        typed = [e for e in errors if e.get('codex_protocol')]
        assert len(typed) == len(provider.opener.calls) == 1
        assert typed[0]['codex_protocol']['code'] == 'transport_incomplete_read'
        assert parallel_recovery._protocol_failure(typed[0]) is not None
        assert [e for e in errors if e.get('cause_type') == 'ValueError'] == [
            {'stage': 'request', 'reason': 'request_failed', 'cause_type': 'ValueError'}]
        unknown = unknown_trace(directory)
        assert len(unknown) == 2 and all(row['cost'] is None for row in unknown)
        with pytest.raises(ProviderRecoveryError, match='non-provider failure'):
            parallel_recovery.inspect_parallel_stop(directory / 'checkpoint.json')
        assert not list((directory / '.parallel').rglob('abandonment.json'))
        assert physical_files(directory) == saved
    finally:
        first.close()
