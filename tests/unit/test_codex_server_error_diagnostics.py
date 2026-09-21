"""Typed provider errors remain safe, bounded and distinct from unknown failures."""
from copy import deepcopy
import hashlib
import io
import json

import pytest

from sciel.engine.llm.client import LLMError, llm_error_metadata
from sciel.engine.llm.codex import CodexResponsesLLM
from sciel.engine.llm.codex_protocol import (
    LEGACY_SCHEMA, SCHEMA, validate_codex_protocol)
from sciel.engine.parallel_recovery import _protocol_failure
from test_codex_responses import MODEL, config, provider


def metadata(event):
    llm, opener = provider([event])
    with pytest.raises(LLMError) as caught:
        llm.generate('private prompt', MODEL)
    assert len(opener.calls) == 1
    return caught.value, llm_error_metadata(caught.value)


def server_error(**changes):
    return dict(type='error', code='server_error', message='private provider message',
                param=None, **changes)


def test_flat_server_error_has_safe_typed_receipt_and_detached_nested_fields():
    event = server_error(sequence_number=3)
    error, result = metadata(event)
    diagnostic = result['codex_protocol']
    detail = diagnostic['provider_error']
    encoded = json.dumps(event, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('ascii')
    assert diagnostic['schema'] == SCHEMA
    assert diagnostic['code'] == 'provider_server_error'
    assert diagnostic['last_event'] == 'error' and diagnostic['terminal_status'] == 'none'
    assert detail['layout'] == 'flat' and detail['code'] == 'server_error'
    assert detail['param_kind'] == 'null'
    assert detail['event_sha256'] == hashlib.sha256(encoded).hexdigest()
    assert detail['event_bytes'] == len(encoded)
    assert detail['message_chars'] == len(event['message'])
    assert _protocol_failure(result) == diagnostic
    assert 'private' not in json.dumps(result) and 'fixture-secret' not in json.dumps(result)
    assert len(json.dumps(result)) < 1536
    # Both raw exception metadata and re-projected public metadata are detached.
    error.metadata['codex_protocol']['provider_error']['code'] = 'invalid_api_key'
    detail['code'] = 'invalid_api_key'
    assert llm_error_metadata(error)['codex_protocol']['provider_error']['code'] == 'server_error'


@pytest.mark.parametrize('event,code,layout', [
    ({'type':'error','code':'invalid_api_key','message':'private detail','param':None},
     'invalid_api_key', 'flat'),
    ({'type':'error','code':'future-provider-code','message':'private detail','param':None},
     None, 'flat'),
    ({'type':'error','error':{'code':'server_error','message':'private detail','param':None}},
     'server_error', 'nested'),
    ({'type':'error','message':'private detail'}, None, 'other'),
    ({'type':'error','code':'server_error','message':'private detail','param':'private parameter'},
     'server_error', 'flat'),
    ({'type':'error','code':'server_error','message':'private detail','param':None,
      'error':{'code':'invalid_api_key'}}, 'server_error', 'other'),
])
def test_other_error_shapes_are_observable_but_cannot_enable_recovery(event, code, layout):
    _, result = metadata(event)
    diagnostic = result['codex_protocol']
    assert diagnostic['code'] == 'stream_error'
    assert diagnostic['provider_error']['code'] == code
    assert diagnostic['provider_error']['layout'] == layout
    assert _protocol_failure(result) is None
    assert 'private' not in json.dumps(result)


@pytest.mark.parametrize('mutation', [
    {'provider_error':None}, {'last_event':'other'}, {'terminal_status':'failed'},
    {'provider_error':{'extra':'private'}},
])
def test_incomplete_provider_diagnostic_is_rejected(mutation):
    _, result = metadata(server_error())
    value = result['codex_protocol']
    value.update(mutation)
    assert validate_codex_protocol(value) is None


@pytest.mark.parametrize('mutation', [
    {'layout':'nested'}, {'code':'invalid_api_key'}, {'code':None}, {'param_kind':'present'},
    {'event_bytes':True}, {'event_bytes':0}, {'event_sha256':'F'*64},
    {'code_sha256':'0'*64}, {'message_chars':True}, {'message_chars':None},
    {'message_sha256':None}, {'message_chars':-1}, {'private':'extra'},
])
def test_tampered_or_weaker_detail_does_not_recover(mutation):
    _, result = metadata(server_error())
    result['codex_protocol']['provider_error'].update(mutation)
    assert validate_codex_protocol(result['codex_protocol']) is None
    assert _protocol_failure(result) is None


def test_legacy_error_has_no_new_server_classification():
    _, result = metadata(server_error())
    legacy = deepcopy(result['codex_protocol'])
    legacy['schema'] = LEGACY_SCHEMA
    del legacy['provider_error']
    assert validate_codex_protocol(legacy) is None
    legacy['code'] = 'stream_error'
    assert validate_codex_protocol(legacy) == legacy
    result['codex_protocol'] = legacy
    assert _protocol_failure(result) is None


def test_unpaired_unicode_and_unknown_code_are_fingerprinted_without_content():
    event = {'type':'error', 'code':'private\ud800code', 'message':'private\ud800message', 'param':None}
    class EscapedJSON:
        calls = 0
        def open(self, request, timeout):
            self.calls += 1
            return io.BytesIO(b'data: ' + json.dumps(event, ensure_ascii=True).encode('ascii') + b'\n\n')
    opener = EscapedJSON()
    llm = CodexResponsesLLM(config(), 'fixture-secret', opener=opener)
    with pytest.raises(LLMError) as caught:
        llm.generate('p', MODEL)
    result = llm_error_metadata(caught.value)
    assert opener.calls == 1
    detail = result['codex_protocol']['provider_error']
    assert detail['code'] is None and len(detail['code_sha256']) == 64
    assert detail['message_chars'] == len(event['message'])
    assert 'private' not in json.dumps(result)
