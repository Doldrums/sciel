"""Prospective overload semantic cores; real local Engine/WAL, never HTTP.

The historical V14 shape is deliberately not reconstructed from its hash.
These fixtures specify the newly supported wire shapes, and preserve the old
diagnostic/3 disposition separately. Evaluators inspect inert VALUE constants;
neither generated source nor research-tool code is executed.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.engine import parallel_recovery, parallel_store
from sciel.engine.llm.client import LLMError, llm_error_metadata
from sciel.engine.llm.codex import CodexResponsesLLM, CONTRACT
from sciel.engine.llm.codex_protocol import SCHEMA, _ProtocolState, validate_codex_protocol
from sciel.engine.recovery import ProviderRecoveryError

import test_codex_research_recovery as planner_cases
import test_codex_server_error_recovery as server_cases
from test_codex_responses import Opener, config
from test_codex_transport_recovery import unknown_trace
from test_parallel_provider_recovery import events, physical_files, stopped_wave


CODE = 'server_is_overloaded'
PRIVATE = server_cases.PRIVATE


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode('ascii')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def overload_event(**outer):
    return dict(type='error', error=dict(code=CODE, message=PRIVATE, param=None), **outer)


def extended_overload_event(**outer):
    event = overload_event(**outer)
    event.update(event_id='PRIVATE_EVENT_ID', request_id='PRIVATE_OUTER_REQUEST',
                 metadata={'unrelated': ['PRIVATE_EXTENSION', 23, None]}, status='error')
    event['error'].update(request_id='PRIVATE_INNER_REQUEST', extra='PRIVATE_EXTRA',
                          type='server_error', status='failed', sequence_number=7)
    return event


def parse(event):
    opener = Opener(server_cases.stream(event))
    adapter = CodexResponsesLLM({**config(), 'models': ['mock-model']}, 'fixture-only', opener=opener)
    with pytest.raises(LLMError) as caught:
        adapter.generate('private prompt', 'mock-model')
    return opener, llm_error_metadata(caught.value)


def assert_diagnostic(provider, failed):
    diagnostic = failed['error']['codex_protocol']
    assert diagnostic['schema'] == SCHEMA == 'sciel-codex-protocol-diagnostic/4'
    assert diagnostic['code'] == 'provider_overloaded'
    assert diagnostic['last_event'] == 'error' and diagnostic['terminal_status'] == 'none'
    detail = diagnostic['provider_error']
    assert detail['layout'] == 'nested_overload'
    assert detail['code'] is None  # Do not change old finite-category membership.
    assert detail['code_identifier'] == CODE and detail['code_sha256'] == digest(CODE)
    assert detail['param_kind'] == 'null'
    assert detail['event_sha256'] == digest(provider.opener.rows[-1])
    assert detail['message_sha256'] == digest(PRIVATE)
    assert detail['message_chars'] == len(PRIVATE)
    assert diagnostic['body_sha256'] == hashlib.sha256(provider.opener.calls[0][0].data).hexdigest()
    assert validate_codex_protocol(diagnostic) == diagnostic
    assert parallel_recovery._protocol_failure(failed['error']) == diagnostic
    assert PRIVATE not in json.dumps(failed['error']) and 'fixture-only' not in json.dumps(failed['error'])
    for private in ('PRIVATE_EVENT_ID', 'PRIVATE_OUTER_REQUEST', 'PRIVATE_EXTENSION',
                    'PRIVATE_INNER_REQUEST', 'PRIVATE_EXTRA'):
        assert private not in json.dumps(failed['error'])
    return diagnostic


@pytest.mark.parametrize('inner_type', [False, True])
@pytest.mark.parametrize('sequence', [None, 0, 2])
@pytest.mark.parametrize('extended', [False, True])
def test_exact_prospective_overload_has_bounded_typed_diagnostic(inner_type, sequence, extended):
    make = extended_overload_event if extended else overload_event
    event = make(**({} if sequence is None else {'sequence_number': sequence}))
    if inner_type:
        event['error']['type'] = 'server_error'
    opener, metadata = parse(event)
    assert len(opener.calls) == 1 and CONTRACT == 'sciel-codex-responses/6'
    assert_diagnostic(SimpleNamespace(opener=opener), {'error': metadata})


def invalid_events():
    for code in ['SERVER_IS_OVERLOADED', 'server_is_overloaded ', 'upstream_overloaded',
                 'server_error', 'rate_limit_exceeded', 'insufficient_quota',
                 'invalid_api_key', 'context_length_exceeded', None, 429]:
        event = overload_event()
        event['error']['code'] = code
        yield pytest.param(event, id='code-' + str(code))
    for field in ['code', 'message', 'param']:
        event = overload_event()
        event['error'].pop(field)
        yield pytest.param(event, id='missing-' + field)
    for field, value in [('message', {}), ('message', None), ('param', 'model'),
                         ('type', 'authentication_error'), ('type', None), ('error', None),
                         ('error', {'code': 'invalid_api_key'}), ('response', None),
                         ('response', {'status': 'completed'}), ('sequence_number', True),
                         ('sequence_number', -1)]:
        event = overload_event()
        event['error'][field] = value
        yield pytest.param(event, id='inner-' + field + '-' + str(value))
    for field, value in [('code', CODE), ('response', None), ('param', None), ('message', PRIVATE),
                         ('sequence_number', True), ('sequence_number', -1),
                         ('sequence_number', 1.5)]:
        event = overload_event()
        event[field] = value
        yield pytest.param(event, id='outer-' + field + '-' + str(value))
    for carrier in ['outer', 'inner']:
        for status in ['completed', 'incomplete', 'in_progress', 'queued', 'cancelled',
                       'unknown', None, True, {}, []]:
            event = extended_overload_event()
            target = event if carrier == 'outer' else event['error']
            target['status'] = status
            yield pytest.param(event, id=carrier + '-status-' + str(status))
    yield pytest.param(dict(type='error', code=CODE, message=PRIVATE, param=None), id='flat-overload')
    yield pytest.param(dict(type='error', message=CODE), id='message-only')


@pytest.mark.parametrize('event', list(invalid_events()))
def test_ambiguous_or_other_error_categories_are_still_fatal(event):
    opener, metadata = parse(event)
    assert len(opener.calls) == 1
    assert metadata['codex_protocol']['code'] == 'stream_error'
    assert parallel_recovery._protocol_failure(metadata) is None
    assert PRIVATE not in json.dumps(metadata)


@pytest.mark.parametrize('terminal', ['completed', 'incomplete', 'failed'])
def test_overload_does_not_override_an_observed_terminal(terminal):
    state = _ProtocolState()
    state.event('response.' + terminal)
    state.event('error')
    assert state.server_error(overload_event()) == 'stream_error'
    assert state.terminal_status == terminal
    _, metadata = parse(overload_event())
    metadata['codex_protocol']['terminal_status'] = terminal
    assert parallel_recovery._protocol_failure(metadata) is None


@pytest.mark.parametrize('version', [1, 2, 3])
def test_old_generic_overload_keeps_original_bytes_and_fatal_disposition(version):
    _, metadata = parse(overload_event())
    old = deepcopy(metadata)
    diagnostic = old['codex_protocol']
    diagnostic.update(schema=f'sciel-codex-protocol-diagnostic/{version}', code='stream_error')
    if version == 1:
        diagnostic.pop('provider_error')
    else:
        diagnostic['provider_error']['layout'] = 'nested'
        if version == 2:
            diagnostic['provider_error'].pop('code_identifier')
    original = canonical(old)
    assert validate_codex_protocol(diagnostic) == diagnostic
    assert parallel_recovery._protocol_failure(old) is None
    assert canonical(old) == original
    diagnostic['code'] = 'provider_overloaded'
    assert validate_codex_protocol(diagnostic) is None


@pytest.mark.parametrize('change', [
    {'layout': 'nested'}, {'layout': 'flat'}, {'code': CODE}, {'code_identifier': 'provider_busy'},
    {'code_sha256': '0' * 64}, {'param_kind': 'missing'}, {'param_kind': 'present'},
    {'message_sha256': None, 'message_chars': None},
])
def test_typed_overload_requires_exact_projection_and_fingerprints(change):
    _, metadata = parse(overload_event())
    metadata['codex_protocol']['provider_error'].update(change)
    assert validate_codex_protocol(metadata['codex_protocol']) is None
    assert parallel_recovery._protocol_failure(metadata) is None


def overload_scenarios(monkeypatch, *, extended=False):
    """Use the same real G/C/planner/STOP replay scenarios with a new wire event."""
    monkeypatch.setattr(server_cases, 'server_error', extended_overload_event if extended else overload_event)
    monkeypatch.setattr(server_cases, 'assert_diagnostic', assert_diagnostic)


@pytest.mark.parametrize('role', ['G', 'C'])
@pytest.mark.parametrize('extended', [False, True])
def test_overload_real_generation_and_conclusion_abandon_exactly_once(tmp_path, monkeypatch, role, extended):
    overload_scenarios(monkeypatch, extended=extended)
    server_cases.test_flat_server_error_abandons_only_missing_role_and_cold_replays_paid_siblings(
        tmp_path, monkeypatch, role)


def test_overload_real_planner_abandons_without_tool_or_efficacy_lesson(tmp_path, monkeypatch):
    overload_scenarios(monkeypatch)
    server_cases.test_flat_server_error_research_planner_closes_unknown_without_tool_or_lesson(
        tmp_path, monkeypatch)


@pytest.mark.parametrize('boundary', ['stop', 'publication_crash'])
def test_overload_respects_stop_and_survives_publication_crash(tmp_path, monkeypatch, boundary):
    overload_scenarios(monkeypatch)
    server_cases.test_authorized_server_error_respects_stop_and_commit_crash(tmp_path, monkeypatch, boundary)


@pytest.mark.parametrize('mode', ['wrong_body', 'wrong_model', 'wrong_transport', 'legacy', 'legacy_v1'])
def test_overload_recovery_requires_bound_current_request(tmp_path, monkeypatch, mode):
    overload_scenarios(monkeypatch)
    server_cases.test_server_error_recovery_requires_exact_request_and_typed_current_evidence(
        tmp_path, monkeypatch, mode)


@pytest.mark.parametrize('mixed', ['worker', 'evaluation'])
def test_overload_cannot_hide_independent_worker_or_evaluator_failure(tmp_path, monkeypatch, mixed):
    class Planner(planner_cases.PlannerProtocolProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.opener = Opener(server_cases.stream(overload_event()))
            self.adapter._opener = self.opener

    monkeypatch.setattr(planner_cases, 'PlannerProtocolProvider', Planner)
    planner_cases.test_typed_planner_error_cannot_hide_independent_worker_or_evaluator_failure(
        tmp_path, monkeypatch, mixed)


@pytest.mark.parametrize('extended', [False, True])
def test_portfolio_overload_replays_paid_siblings_then_ordinary_evolution_commits(tmp_path, monkeypatch, extended):
    event = extended_overload_event() if extended else overload_event()
    provider = server_cases.ServerErrorProvider(role='G', event=event)
    first, build, _, evaluated, directory = stopped_wave(tmp_path, monkeypatch, provider,
        llm=provider.codex_config, portfolio_seed=True, portfolio_schedule='interleaved')
    path = directory / 'checkpoint.json'
    try:
        authorization = parallel_recovery.inspect_parallel_stop(path)
        assert authorization['unknown_cost'] is True and authorization['retry'] is False
        failed, = authorization['failed_calls']
        assert failed['key'].startswith('portfolio:') and failed['not_dispatched'] is False
        assert_diagnostic(provider, failed)
        assert json.loads(path.read_bytes())['iteration_no'] == 0
        saved, unknown = physical_files(directory), unknown_trace(directory)
        paid_prompts = set(provider.calls)
        parallel_recovery.authorize_parallel_recovery(first, path, authorization)
        parallel_recovery.authorize_parallel_recovery(first, path, authorization)
    finally:
        first.close()

    generate = provider.generate

    def only_new_requests(prompt, *args, **kwargs):
        assert prompt not in paid_prompts, 'portfolio paid request redispatched'
        return generate(prompt, *args, **kwargs)

    monkeypatch.setattr(provider, 'generate', only_new_requests)
    commit = parallel_store.WaveTransaction.commit

    def publish_two_then_stop(transaction):
        result = commit(transaction)
        if transaction.wave_id == '2':
            (directory / 'STOP').touch()
        return result

    monkeypatch.setattr(parallel_store.WaveTransaction, 'commit', publish_two_then_stop)
    resumed = build(resume=True)
    result = resumed.run()
    assert result['reason'] == 'paused' and result['iteration_no'] == result['B_evals'] == 4
    assert resumed._portfolio_bootstrap['next_island'] == 4
    assert Counter(p.origin for p in resumed.db._all.values()) == {'seed': 4, 'portfolio': 3, 'gen': 4}
    assert Counter(provider.roles) == {'G': 8, 'C': 7}
    assert len(evaluated) == 8  # One canonical seed + three portfolio + four ordinary.
    assert len(provider.opener.calls) == 1 and resumed.llm_unpriced_calls == 14
    assert result['cost_total'] == 0  # Known subtotal only, not the unknown failed charge.
    assert all(physical_files(directory)[key] == value for key, value in saved.items())
    assert unknown_trace(directory) == unknown and len(unknown) == 1 and unknown[0]['cost'] is None
    abandoned = [row for row in events(directory) if row.get('phase') == 'generation_abandoned']
    assert len(abandoned) == 1 and abandoned[0]['current_call_cost_unknown'] is True
    assert abandoned[0]['provider_abandonment']['request_sha256'] == failed['request_sha256']
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
    assert physical_files(directory) == committed and unknown_trace(directory) == unknown
    assert len([row for row in events(directory) if row.get('phase') == 'generation_abandoned']) == 1
