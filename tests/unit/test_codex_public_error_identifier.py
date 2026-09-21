"""Public error identifiers are observability, never extra recovery authority.

Real Codex parser/metadata/Engine-WAL paths use inert local SSE leaves only.
No authentication file, network, provider, evaluator or candidate code executes.
"""
from copy import deepcopy
import hashlib
import io
import json

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import parallel_recovery
from sciel.engine.llm.client import LLMError, llm_error_metadata
from sciel.engine.llm.codex import CodexResponsesLLM
from sciel.engine.llm.codex_protocol import validate_codex_protocol
from sciel.engine.recovery import ProviderRecoveryError
from test_codex_responses import MODEL, Opener, config
from test_codex_server_error_recovery import ServerErrorProvider
from test_codex_transport_recovery import unknown_trace
from test_parallel_provider_recovery import physical_files, stopped_wave

MESSAGE = 'Do not expose this message: fixture-secret@example.invalid /private/user/path'
PARAM = {'private_parameter': 'Bearer fixture-secret-token'}


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('ascii')


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def error_event(code, *, nested=False, param=None):
    payload = dict(code=code, message=MESSAGE, param=param)
    return dict(type='error', error=payload) if nested else dict(type='error', **payload)


def parsed(code, *, nested=False, param=None):
    event = error_event(code, nested=nested, param=param)
    class EscapedOpener(Opener):
        def open(self, request, timeout):
            self.calls.append((request, timeout))
            return io.BytesIO(b'data: ' + canonical(event) + b'\n\n')

    opener = EscapedOpener([])
    adapter = CodexResponsesLLM(config(), 'fixture-auth-secret', opener=opener)
    with pytest.raises(LLMError) as caught:
        adapter.generate('private prompt content', MODEL)
    metadata = llm_error_metadata(caught.value)
    assert len(opener.calls) == 1
    diagnostic = metadata['codex_protocol']
    assert diagnostic['schema'] == 'sciel-codex-protocol-diagnostic/4'
    assert diagnostic['body_sha256'] == hashlib.sha256(opener.calls[0][0].data).hexdigest()
    assert diagnostic['model'] == MODEL
    assert diagnostic['last_event'] == 'error' and diagnostic['terminal_status'] == 'none'
    assert validate_codex_protocol(diagnostic) == diagnostic
    detail = diagnostic['provider_error']
    assert detail['event_sha256'] == sha(event) and detail['event_bytes'] == len(canonical(event))
    assert detail['message_sha256'] == sha(MESSAGE) and detail['message_chars'] == len(MESSAGE)
    assert detail['code_sha256'] == (sha(code) if type(code) is str else None)
    exposed = json.dumps(metadata)
    for private in (MESSAGE, 'fixture-auth-secret', 'fixture-secret@example.invalid',
                    '/private/user/path', 'private prompt content', 'private_parameter',
                    'Bearer fixture-secret-token'):
        assert private not in exposed
    assert len(exposed) < 1800
    return caught.value, metadata


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('code', [
    'upstream_overloaded', 'provider.private_category', 'TEMPORARY-Failure.2',
    '429', '123456', 'a', 'g' * 24, 'a' * 21 + '_' + 'b' * 21 + '_' + 'c' * 20,
])
def test_bounded_structured_unknown_identifier_is_visible_but_fatal(code, nested):
    _, metadata = parsed(code, nested=nested)
    diagnostic = metadata['codex_protocol']
    assert diagnostic['code'] == 'stream_error'
    detail = diagnostic['provider_error']
    assert detail['code_identifier'] == code and detail['code'] is None
    assert detail['layout'] == ('nested' if nested else 'flat')
    assert parallel_recovery._protocol_failure(metadata) is None


@pytest.mark.parametrize('code', [
    '', 'a' * 65, 'a' * 25, 'g' * 25,
    'g' * 21 + '_' + 'h' * 21 + '_' + 'j' * 21,
    '1234567', '0123456789abcdef',
    '00112233-4455-6677-8899-aabbccddeeff',
    'a__b', 'a..b', '-bad', 'trailing_', 'has space', 'has\nnewline',
    'email@example.invalid', 'https://provider.invalid/error', 'unicodé_error',
    'code\ud800', 'sk-short', 'SK_short', 'sess-short', 'SeSs_short',
    'BearerToken', 'eyJshort', 'ghp_short', 'github_pat_short', 'xoxb-short', 'xoxp-short',
    None, True, 429, ['server_error'], {'code': 'server_error'},
])
def test_prose_opaque_ids_and_conspicuous_credentials_are_not_exposed(code):
    _, metadata = parsed(code, nested=True, param=PARAM)
    diagnostic = metadata['codex_protocol']
    assert diagnostic['code'] == 'stream_error'
    assert diagnostic['provider_error']['code_identifier'] is None
    assert diagnostic['provider_error']['code'] is None
    assert diagnostic['provider_error']['param_kind'] == 'present'
    assert parallel_recovery._protocol_failure(metadata) is None


@pytest.mark.parametrize(('code', 'nested', 'recoverable'), [
    ('server_error', False, True), ('server_error', True, False),
    ('invalid_api_key', False, False), ('rate_limit_exceeded', False, False),
    ('context_length_exceeded', True, False),
])
def test_identifier_does_not_broaden_existing_finite_recovery_categories(code, nested, recoverable):
    _, metadata = parsed(code, nested=nested)
    diagnostic = metadata['codex_protocol']
    assert diagnostic['provider_error']['code_identifier'] == diagnostic['provider_error']['code'] == code
    assert (parallel_recovery._protocol_failure(metadata) is not None) is recoverable
    assert diagnostic['code'] == ('provider_server_error' if recoverable else 'stream_error')


def test_public_projection_is_detached_and_identifier_cannot_relabel_category_or_hash():
    error, metadata = parsed('provider_busy', nested=True)
    original = deepcopy(metadata)
    detail = metadata['codex_protocol']['provider_error']
    detail['code_identifier'] = 'server_error'
    error.metadata['codex_protocol']['provider_error']['code_identifier'] = 'forged_code'
    assert llm_error_metadata(error) == original
    changes = [
        {'code_identifier': 'different_identifier'},
        {'code_identifier': 'server_error'},
        {'code': 'server_error'},
        {'code_identifier': 'sk-secret'},
        {'code_identifier': ['provider_busy']},
        {'code_identifier': 'provider_busy', 'code_sha256': '0' * 64},
        {'code_identifier': 'server_error', 'code_sha256': sha('server_error')},
        {'code': 'server_error', 'code_sha256': sha('server_error')},
    ]
    for change in changes:
        forged = deepcopy(original)
        forged['codex_protocol']['provider_error'].update(change)
        assert validate_codex_protocol(forged['codex_protocol']) is None
        assert parallel_recovery._protocol_failure(forged) is None
    # Even a fully self-consistent *known* identifier in a nested layout remains fatal.
    forged = deepcopy(original)
    forged['codex_protocol']['provider_error'].update(
        code='server_error', code_identifier='server_error', code_sha256=sha('server_error'))
    assert validate_codex_protocol(forged['codex_protocol']) == forged['codex_protocol']
    assert parallel_recovery._protocol_failure(forged) is None


@pytest.mark.parametrize(('version', 'code', 'nested', 'recoverable'), [
    (1, 'provider_busy', True, False),
    (2, 'provider_busy', True, False),
    (2, 'server_error', False, True),
    (2, 'server_error', True, False),
])
def test_exact_legacy_v1_v2_shape_retains_original_disposition_and_detaches(
        version, code, nested, recoverable):
    _, metadata = parsed(code, nested=nested)
    legacy = deepcopy(metadata['codex_protocol'])
    legacy['schema'] = 'sciel-codex-protocol-diagnostic/' + str(version)
    if version == 1:
        legacy.pop('provider_error')
    else:
        legacy['provider_error'].pop('code_identifier')
    saved = deepcopy(legacy)
    detached = validate_codex_protocol(legacy)
    assert detached == saved
    if version == 2:
        detached['provider_error']['code'] = 'authentication_error'
    else:
        detached['model'] = 'changed-model'
    assert legacy == saved
    historical_metadata = {**metadata, 'codex_protocol': legacy}
    assert (parallel_recovery._protocol_failure(historical_metadata) is not None) is recoverable
    if version == 2:
        legacy['provider_error']['code_identifier'] = code
        assert validate_codex_protocol(legacy) is None  # Do not silently reinterpret mixed schemas.


@pytest.mark.parametrize('version', [1, 2])
def test_legacy_unknown_hash_never_invents_a_public_identifier(version):
    _, metadata = parsed('provider_busy', nested=True)
    old = deepcopy(metadata['codex_protocol'])
    old['schema'] = 'sciel-codex-protocol-diagnostic/' + str(version)
    if version == 1:
        old.pop('provider_error')
    else:
        old['provider_error'].pop('code_identifier')
    historical = LLMError('historical fixture')
    historical._metadata = {**metadata, 'codex_protocol': old}
    restored = llm_error_metadata(historical)
    assert restored['codex_protocol'] == old
    assert 'provider_busy' not in json.dumps(restored)
    assert parallel_recovery._protocol_failure(restored) is None


@pytest.mark.parametrize('nested', [False, True])
def test_unknown_identifier_real_wal_stays_fatal_across_cold_load_without_redispatch(
        tmp_path, monkeypatch, nested):
    provider = ServerErrorProvider(role='C', event=error_event('provider_busy', nested=nested))
    first, build, _, _, directory = stopped_wave(
        tmp_path, monkeypatch, provider, llm=provider.codex_config)
    checkpoint_path = directory / 'checkpoint.json'
    try:
        saved = physical_files(directory)
        unknown = unknown_trace(directory)
        assert len(provider.opener.calls) == len(unknown) == 1
        assert unknown[0]['cost'] is None and unknown[0]['current_call_cost_unknown'] is True
        metadata = unknown[0]['llm_error']
        diagnostic = metadata['codex_protocol']
        assert diagnostic['schema'] == 'sciel-codex-protocol-diagnostic/4'
        assert diagnostic['code'] == 'stream_error'
        assert diagnostic['provider_error']['code_identifier'] == 'provider_busy'
        assert diagnostic['provider_error']['code'] is None
        assert diagnostic['body_sha256'] == hashlib.sha256(provider.opener.calls[0][0].data).hexdigest()
        errors = [json.loads(p.read_text())['payload']['error']
                  for p in (directory / '.parallel').rglob('receipt.json')
                  if json.loads(p.read_text())['payload']['status'] == 'error']
        assert errors == [metadata]  # Sanitized trace and sealed WAL agree.
        assert 'private_parameter' not in json.dumps(errors) and MESSAGE not in json.dumps(errors)
        before = json.loads(checkpoint_path.read_bytes())
        assert before['iteration_no'] == before['B_evals'] == before['llm_unpriced_calls'] == 0
        assert before['research']['knowledge']['lessons'] == []
        with pytest.raises(ProviderRecoveryError):
            parallel_recovery.inspect_parallel_stop(checkpoint_path)
        assert not list((directory / '.parallel').rglob('abandonment.json'))
    finally:
        first.close()

    monkeypatch.setattr(provider, 'generate', lambda *a, **k: pytest.fail('fatal physical call redispatched'))
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate',
                        lambda *a, **k: pytest.fail('paid sibling reevaluated'))
    resumed = build(resume=True)
    with pytest.raises(CampaignStopped):
        resumed.run()
    assert resumed._writer_lock._file.closed
    assert len(provider.opener.calls) == 1
    assert physical_files(directory) == saved and unknown_trace(directory) == unknown
    after = json.loads(checkpoint_path.read_bytes())
    assert after['iteration_no'] == after['B_evals'] == after['llm_unpriced_calls'] == 0
    assert after['research']['knowledge']['lessons'] == []
    assert not list((directory / '.parallel').rglob('abandonment.json'))
    with pytest.raises(ProviderRecoveryError):
        parallel_recovery.inspect_parallel_stop(checkpoint_path)
