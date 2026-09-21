"""Flat explicit Codex server errors through the real adapter and Engine WAL.

All physical leaves are inert local fixtures. No authentication file, network,
provider, container, evaluator process, or generated candidate code is used.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import parallel_recovery, parallel_store
from sciel.engine.llm.client import LLMError, llm_error_metadata
from sciel.engine.llm.codex import CodexResponsesLLM
from sciel.engine.llm.codex_protocol import SCHEMA, validate_codex_protocol
from sciel.engine.recovery import ProviderRecoveryError

from test_codex_parallel_recovery import ProtocolProvider
import test_codex_research_recovery as research_fixture
from test_codex_responses import Opener, config
from test_codex_transport_recovery import unknown_trace
from test_parallel_provider_recovery import (
    events, physical_files, stop_after_publication, stopped_wave,
)


PRIVATE = 'PRIVATE_REMOTE_MESSAGE fixture-secret private-response-id'


def server_error(**changes):
    return dict(type='error', code='server_error', message=PRIVATE, param=None,
                **changes)


def stream(event=None):
    return [
        {'type': 'response.created', 'response': {
            'id': 'private-response-id', 'model': 'mock-model',
            'status': 'in_progress', 'output': []}},
        {'type': 'response.in_progress', 'response': {
            'id': 'private-response-id', 'model': 'mock-model',
            'status': 'in_progress', 'output': []}},
        server_error() if event is None else event,
    ]


class ServerErrorProvider(ProtocolProvider):
    def __init__(self, role='C', *, event=None, mode='stream_eof'):
        super().__init__(role=role, mode=mode if mode not in ('wrong_transport', 'legacy_v1') else 'stream_eof')
        self.legacy_v1 = mode == 'legacy_v1'
        self.wrong_transport = mode == 'wrong_transport'
        self.opener = Opener(stream(event))
        self.adapter._opener = self.opener

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except LLMError as error:
            if self.legacy_v1:
                diagnostic = error._metadata['codex_protocol']
                diagnostic['schema'] = 'sciel-codex-protocol-diagnostic/1'
                diagnostic['code'] = 'stream_error'
                diagnostic.pop('provider_error')
            raise

    def request_metadata(self, *args, **kwargs):
        result = super().request_metadata(*args, **kwargs)
        if self.wrong_transport:
            result['schema'] = 'sciel-codex-responses/0'
        return result


def stopped(tmp_path, monkeypatch, **kwargs):
    provider = ServerErrorProvider(**kwargs)
    return stopped_wave(tmp_path, monkeypatch, provider, llm=provider.codex_config)


def assert_diagnostic(provider, failed):
    diagnostic = failed['error']['codex_protocol']
    assert diagnostic['schema'] == SCHEMA
    assert diagnostic['code'] == 'provider_server_error'
    assert diagnostic['last_event'] == 'error' and diagnostic['terminal_status'] == 'none'
    assert diagnostic['events_seen'] == 3
    detail = diagnostic['provider_error']
    assert (detail['layout'], detail['code'], detail['param_kind']) == ('flat', 'server_error', 'null')
    assert detail['message_chars'] == len(PRIVATE)
    canonical = lambda value: json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('ascii')
    assert detail['event_sha256'] == hashlib.sha256(canonical(provider.opener.rows[-1])).hexdigest()
    assert detail['message_sha256'] == hashlib.sha256(canonical(PRIVATE)).hexdigest()
    assert diagnostic['model'] == 'mock-model'
    assert diagnostic['body_sha256'] == hashlib.sha256(provider.opener.calls[0][0].data).hexdigest()
    assert validate_codex_protocol(diagnostic) == diagnostic
    assert parallel_recovery._protocol_failure(failed['error']) == diagnostic
    encoded = json.dumps(failed['error'])
    assert 'PRIVATE_' not in encoded and 'fixture-secret' not in encoded
    assert 'private-response-id' not in encoded and len(encoded) < 1500
    return diagnostic


@pytest.mark.parametrize('role', ['G', 'C'])
def test_flat_server_error_abandons_only_missing_role_and_cold_replays_paid_siblings(
        tmp_path, monkeypatch, role):
    first, build, provider, evaluated, directory = stopped(tmp_path, monkeypatch, role=role)
    path = directory / 'checkpoint.json'
    try:
        baseline = path.read_bytes()
        assert json.loads(baseline)['iteration_no'] == 0
        authorization = parallel_recovery.inspect_parallel_stop(path)
        assert authorization['unknown_cost'] is True and authorization['retry'] is False
        assert len(authorization['failed_calls']) == len(provider.opener.calls) == 1
        failed = authorization['failed_calls'][0]
        assert failed['key'].startswith(role + ':') and failed['not_dispatched'] is False
        assert_diagnostic(provider, failed)
        saved = physical_files(directory)
        unknown = unknown_trace(directory)
        assert len(unknown) == 1 and unknown[0]['cost'] is None
        assert unknown[0]['llm_error']['codex_protocol'] == failed['error']['codex_protocol']
        parallel_recovery.authorize_parallel_recovery(first, path, authorization)
        markers = {str(p): p.read_bytes() for p in (directory / '.parallel').rglob('abandonment.json')}
        parallel_recovery.authorize_parallel_recovery(first, path, authorization)
        assert markers == {str(p): p.read_bytes() for p in (directory / '.parallel').rglob('abandonment.json')}
        assert path.read_bytes() == baseline
    finally:
        first.close()

    generate = provider.generate

    def new_conclusion_only(prompt, *args, **kwargs):
        assert role == 'G' and 'Review this one experiment.' in prompt[:160], 'paid request redispatched'
        return generate(prompt, *args, **kwargs)

    monkeypatch.setattr(provider, 'generate', new_conclusion_only)
    if role == 'C':
        monkeypatch.setattr('sciel.engine.loop.cascade_evaluate',
                            lambda *a, **k: pytest.fail('paid evaluator redispatched'))
    stop_after_publication(monkeypatch, directory)
    resumed = build(resume=True)
    result = resumed.run()
    assert result['reason'] == 'paused' and result['iteration_no'] == 4
    assert result['B_evals'] == (3 if role == 'G' else 4)
    assert len(evaluated) == (4 if role == 'G' else 5)
    assert Counter(provider.roles) == {'G': 4, 'C': 3 if role == 'G' else 4}
    assert result['cost_total'] == 0  # Known USD subtotal; failed physical charge remains unknown.
    assert resumed.llm_unpriced_calls == (6 if role == 'G' else 7)
    assert len(provider.opener.calls) == 1
    assert all(physical_files(directory)[key] == value for key, value in saved.items())
    assert unknown_trace(directory) == unknown
    assert len(resumed.research.knowledge.lessons) == 3
    phase = 'generation_abandoned' if role == 'G' else 'aux_abandoned'
    abandoned = [row for row in events(directory) if row.get('phase') == phase]
    assert len(abandoned) == 1 and abandoned[0]['cost'] is None
    assert abandoned[0]['current_call_cost_unknown'] is True
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

    committed = physical_files(directory)
    monkeypatch.setattr(provider, 'generate', lambda *a, **k: pytest.fail('second-cold redispatch'))
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', lambda *a, **k: pytest.fail('second-cold eval'))
    again = build(resume=True)
    again._stop_at = result['B_evals']
    again.run()
    assert again.llm_unpriced_calls == resumed.llm_unpriced_calls
    assert len(again.research.knowledge.lessons) == 3
    assert physical_files(directory) == committed and unknown_trace(directory) == unknown
    assert len([r for r in events(directory) if r.get('phase') == phase]) == 1


def test_flat_server_error_research_planner_closes_unknown_without_tool_or_lesson(
        tmp_path, monkeypatch):
    class Planner(research_fixture.PlannerProtocolProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.opener = Opener(stream(server_error(sequence_number=2)))
            self.adapter._opener = self.opener

    monkeypatch.setattr(research_fixture, 'PlannerProtocolProvider', Planner)
    first, build, provider, backend, evaluated, directory = research_fixture.stopped_planner(
        tmp_path, monkeypatch)
    path = directory / 'checkpoint.json'
    try:
        authorization = parallel_recovery.inspect_parallel_stop(path)
        assert authorization['unknown_cost'] is True and authorization['retry'] is False
        assert len(authorization['failed_calls']) == len(provider.opener.calls) == 1
        failed = authorization['failed_calls'][0]
        assert failed['key'].startswith('research:')
        assert_diagnostic(provider, failed)
        assert len(evaluated) == 2 and not backend.calls
        saved = physical_files(directory)
        unknown = unknown_trace(directory)
        assert len(unknown) == 1 and unknown[0]['cost'] is None
        parallel_recovery.authorize_parallel_recovery(first, path, authorization)
        parallel_recovery.authorize_parallel_recovery(first, path, authorization)
    finally:
        first.close()
    generate = provider.generate

    def only_new_conclusion(prompt, *args, **kwargs):
        assert 'Review this one experiment.' in prompt[:160], 'paid planner or G redispatched'
        return generate(prompt, *args, **kwargs)

    monkeypatch.setattr(provider, 'generate', only_new_conclusion)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', lambda *a, **k: pytest.fail('paid eval replay'))
    stop_after_publication(monkeypatch, directory)
    resumed = build(resume=True)
    result = resumed.run()
    assert result['reason'] == 'paused' and result['iteration_no'] == result['B_evals'] == 1
    assert result['cost_total'] == 0 and resumed.llm_unpriced_calls == 2
    assert Counter(provider.roles) == {'G': 1, 'C': 1}
    assert len(provider.opener.calls) == len(provider.planner_prompts) == 1 and not backend.calls
    assert all(physical_files(directory)[key] == value for key, value in saved.items())
    assert unknown_trace(directory) == unknown
    abandoned = [r for r in events(directory) if r.get('phase') == 'llm_abandoned']
    assert len(abandoned) == 1 and abandoned[0]['current_call_cost_unknown'] is True
    assert abandoned[0]['cost'] is None
    artifact = resumed.research_tools._read_artifact(abandoned[0]['receipt_id'], 'proposal_abandonment')
    assert artifact['llm_error'] == failed['error'] and artifact['prompt'] == provider.planner_prompts[0]
    state = resumed.research_tools.state()
    scientific = next(iter(state['lifecycle']['research_state'].values()))
    assert scientific['status'] == 'provider_abandoned' and scientific['unknown_llm_calls'] == 1
    assert scientific['known_llm_cost'] == 0 and scientific['scientific_steps'] == 0
    assert scientific['conclusion'] is None and scientific['steps'][0]['status'] == 'unknown'
    assert not state['registered'] and len(state['committed_episodes']) == 1
    queued = next(iter(state['continuations']['items'].values()))
    assert queued['status'] == 'closed' and queued['reason'] == 'provider_abandoned'
    assert len(resumed.research.knowledge.lessons) == 1  # Only the measured ordinary sibling.
    monkeypatch.setattr(provider, 'generate', lambda *a, **k: pytest.fail('cold planner redispatch'))
    again = build(resume=True)
    again._stop_at = result['B_evals']
    again.run()
    assert again.research_tools.state() == state and again.llm_unpriced_calls == 2
    assert unknown_trace(directory) == unknown and len(provider.opener.calls) == 1
    assert len([r for r in events(directory) if r.get('phase') == 'llm_abandoned']) == 1


@pytest.mark.parametrize('boundary', ['stop', 'publication_crash'])
def test_authorized_server_error_respects_stop_and_commit_crash(tmp_path, monkeypatch, boundary):
    first, build, provider, _, directory = stopped(tmp_path, monkeypatch)
    path = directory / 'checkpoint.json'
    try:
        authorization = parallel_recovery.inspect_parallel_stop(path)
        saved = physical_files(directory)
        if boundary == 'stop':
            (directory / 'STOP').touch()
            with pytest.raises(ProviderRecoveryError, match='STOP'):
                parallel_recovery.authorize_parallel_recovery(first, path, authorization)
            assert not list((directory / '.parallel').rglob('abandonment.json'))
            assert physical_files(directory) == saved
            return
        parallel_recovery.authorize_parallel_recovery(first, path, authorization)
    finally:
        first.close()
    unknown = unknown_trace(directory)
    monkeypatch.setattr(provider, 'generate', lambda *a, **k: pytest.fail('paid request redispatch'))
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', lambda *a, **k: pytest.fail('paid eval redispatch'))

    def crash(transaction):
        raise RuntimeError('synthetic publication crash')

    with monkeypatch.context() as local:
        local.setattr(parallel_store.WaveTransaction, 'commit', crash)
        engine = build(resume=True)
        with pytest.raises(CampaignStopped, match='publication crash'):
            engine.run()
    assert physical_files(directory) == saved and unknown_trace(directory) == unknown
    stop_after_publication(monkeypatch, directory)
    resumed = build(resume=True)
    assert resumed.run()['reason'] == 'paused'
    assert resumed.llm_unpriced_calls == 7 and len(resumed.research.knowledge.lessons) == 3
    assert physical_files(directory) == saved and unknown_trace(directory) == unknown


def malformed_events():
    valid = server_error()
    return [
        pytest.param({'type': 'error'}, id='generic'),
        pytest.param({**valid, 'code': 'authentication_error'}, id='wrong-code'),
        pytest.param({k: v for k, v in valid.items() if k != 'param'}, id='missing-param'),
        pytest.param({**valid, 'param': 'model'}, id='nonnull-param'),
        pytest.param({**valid, 'message': {'private': 'nested'}}, id='nonstring-message'),
        pytest.param({**valid, 'error': {'code': 'server_error'}}, id='nested-error'),
        pytest.param({**valid, 'response': None}, id='response-member'),
        pytest.param({**valid, 'sequence_number': True}, id='bool-sequence'),
        pytest.param({**valid, 'sequence_number': -1}, id='negative-sequence'),
        pytest.param({**valid, 'sequence_number': 1.5}, id='fractional-sequence'),
    ]


@pytest.mark.parametrize('event', malformed_events())
def test_ambiguous_sse_errors_remain_fatal_in_adapter_and_real_wal(tmp_path, monkeypatch, event):
    first, _, provider, _, directory = stopped(tmp_path, monkeypatch, event=event)
    try:
        saved = physical_files(directory)
        errors = [json.loads(p.read_text())['payload']['error']
                  for p in (directory / '.parallel').rglob('receipt.json')
                  if json.loads(p.read_text())['payload']['status'] == 'error']
        assert len(errors) == len(provider.opener.calls) == 1
        error = errors[0]
        assert error['codex_protocol']['code'] == 'stream_error'
        assert parallel_recovery._protocol_failure(error) is None
        assert 'PRIVATE_' not in json.dumps(error)
        with pytest.raises(ProviderRecoveryError):
            parallel_recovery.inspect_parallel_stop(directory / 'checkpoint.json')
        assert not list((directory / '.parallel').rglob('abandonment.json'))
        assert physical_files(directory) == saved
    finally:
        first.close()


@pytest.mark.parametrize('mode', ['wrong_body', 'wrong_model', 'wrong_transport', 'legacy', 'legacy_v1'])
def test_server_error_recovery_requires_exact_request_and_typed_current_evidence(
        tmp_path, monkeypatch, mode):
    first, _, provider, _, directory = stopped(tmp_path, monkeypatch, mode=mode)
    try:
        saved = physical_files(directory)
        with pytest.raises(ProviderRecoveryError):
            parallel_recovery.inspect_parallel_stop(directory / 'checkpoint.json')
        assert len(provider.opener.calls) == 1
        if mode == 'legacy_v1':
            error = next(json.loads(p.read_text())['payload']['error']
                         for p in (directory / '.parallel').rglob('receipt.json')
                         if json.loads(p.read_text())['payload']['status'] == 'error')
            diagnostic = error['codex_protocol']
            assert diagnostic['schema'] == 'sciel-codex-protocol-diagnostic/1'
            assert diagnostic['code'] == 'stream_error'
            assert validate_codex_protocol(diagnostic) == diagnostic
            assert parallel_recovery._protocol_failure(error) is None
        assert not list((directory / '.parallel').rglob('abandonment.json'))
        assert physical_files(directory) == saved
    finally:
        first.close()


def test_safe_provider_envelope_cannot_be_downgraded_or_erase_terminal():
    opener = Opener(stream())
    adapter = CodexResponsesLLM({**config(), 'models': ['mock-model']}, 'fixture-secret', opener=opener)
    with pytest.raises(LLMError) as caught:
        adapter.generate('private prompt', 'mock-model')
    metadata = llm_error_metadata(caught.value)
    diagnostic = metadata['codex_protocol']
    assert diagnostic['code'] == 'provider_server_error'
    assert parallel_recovery._protocol_failure(metadata) == diagnostic
    for changes in [
        {'schema': 'sciel-codex-protocol-diagnostic/1'},
        {'last_event': 'response.in_progress'},
        {'terminal_status': 'completed'},
        {'terminal_status': 'failed'},
    ]:
        forged = deepcopy(metadata)
        forged['codex_protocol'].update(changes)
        assert parallel_recovery._protocol_failure(forged) is None
    # A historical generic stream_error must never acquire the new disposition.
    legacy = deepcopy(metadata)
    legacy['codex_protocol']['schema'] = 'sciel-codex-protocol-diagnostic/1'
    legacy['codex_protocol']['code'] = 'stream_error'
    legacy['codex_protocol'].pop('provider_error', None)
    assert parallel_recovery._protocol_failure(legacy) is None


def test_server_error_does_not_hide_a_local_worker_failure(tmp_path, monkeypatch):
    provider = ServerErrorProvider()
    provider.mixed_failure = True
    first, _, _, _, directory = stopped_wave(tmp_path, monkeypatch, provider, llm=provider.codex_config)
    try:
        saved = physical_files(directory)
        assert len(provider.opener.calls) == 1 and len(unknown_trace(directory)) == 2
        with pytest.raises(ProviderRecoveryError, match='non-provider failure'):
            parallel_recovery.inspect_parallel_stop(directory / 'checkpoint.json')
        assert not list((directory / '.parallel').rglob('abandonment.json'))
        assert physical_files(directory) == saved
    finally:
        first.close()
