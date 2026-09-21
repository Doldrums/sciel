"""In-memory SSE only: no credentials, provider, evaluator or generated code."""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import io
import json
from types import SimpleNamespace

import pytest

from sciel.engine.llm.client import LLMError, llm_error_metadata, response_accounting
from sciel.engine.llm.codex import CodexResponsesLLM, ENDPOINT
from sciel.engine.llm.codex_protocol import (
    ENDPOINT_SHA256, MAX_BYTES, RECOVERABLE_CODES, SCHEMA, validate_codex_protocol)
from sciel.engine.parallel_calls import CallLedger, CooperativeIO
from test_codex_responses import MODEL, Opener, config, events, provider


@pytest.mark.parametrize('placeholder', ['', '{'])
@pytest.mark.parametrize('final_output', [False, True])
@pytest.mark.parametrize('chunk', [1, 65536])
def test_added_text_is_provisional_until_done(placeholder, final_output, chunk):
    rows = events('{"ok":true,"unicode":"雪🙂"}', final_output=final_output)
    rows[1]['item']['content'] = [{'type':'output_text', 'text':placeholder}]
    llm, opener = provider(rows); opener.chunk = chunk
    response = llm.generate('p', MODEL)
    assert response.text == '{"ok":true,"unicode":"雪🙂"}'
    assert response_accounting(response) == {'known_usd':0., 'unpriced_calls':1}
    assert len(opener.calls) == 1


def test_documented_content_part_added_and_multiple_final_parts_do_not_duplicate():
    rows = events('one')
    rows.insert(2, {'type':'response.content_part.added', 'output_index':0,
        'content_index':0, 'item_id':'message1',
        'part':{'type':'output_text','text':'','annotations':[]}})
    item_done = rows[-2]
    item_done['item']['content'].append({'type':'output_text','text':'two'})
    rows[-2:-2] = [
        {'type':'response.output_text.delta','output_index':0,'content_index':1,'delta':'two'},
        {'type':'response.output_text.done','output_index':0,'content_index':1,'text':'two'},
    ]
    # A provisional commentary item must never enter final output or refusal.
    commentary = {'id':'commentary','type':'message','role':'assistant','phase':'commentary',
                  'status':'in_progress','content':[{'type':'output_text','text':''}]}
    rows[1:1] = [{'type':'response.output_item.added','output_index':1,'item':commentary}]
    rows[-1:-1] = [{'type':'response.output_item.done','output_index':1,
                   'item':{**commentary,'status':'completed','content':[
                       {'type':'output_text','text':'private commentary fixture'}]}}]
    llm, opener = provider(rows)
    assert llm.generate('p',MODEL).text == 'onetwo'
    assert len(opener.calls) == 1


def failure_rows(case):
    rows = events()
    if case == 'stream_eof': return rows[:-1]
    if case == 'invalid_event': return [{'type':23}]
    if case == 'invalid_index': rows[2]['output_index'] = True
    if case == 'invalid_item': rows[1]['item'] = []
    if case == 'invalid_content': rows[1]['item']['content'] = [42]
    if case == 'invalid_text': rows[4]['text'] = {'private':'do not serialize'}
    if case == 'invalid_delta': rows[2]['delta'] = {'private':'do not serialize'}
    if case == 'invalid_refusal':
        rows[2] = {'type':'response.refusal.done','output_index':0,'content_index':0,
                   'refusal':{'private':'do not serialize'}}
    if case == 'invalid_terminal': rows[-1]['response']['status'] = 'in_progress'
    if case == 'stream_error': rows = [{'type':'error','message':'private provider body'}]
    if case == 'response_identity': rows[-1]['response']['id'] = 'private changed id'
    if case == 'item_identity': rows[-2]['item']['id'] = 'private changed id'
    if case == 'model_identity': rows[-1]['response']['model'] = 'private wrong model'
    if case == 'text_conflict': rows[4]['text'] = 'private conflicting text'
    if case == 'delta_after_done': rows.insert(-1,deepcopy(rows[2]))
    if case == 'provisional_text_unconfirmed':
        rows[1]['item']['content'] = [{'type':'output_text','text':'private provisional text'}]
        rows = [row for row in rows if row['type'] not in (
            'response.output_text.done', 'response.output_item.done')]
    return rows


@pytest.mark.parametrize('case', [
    'stream_eof','invalid_event','invalid_index','invalid_item','invalid_content',
    'invalid_text','invalid_delta','invalid_refusal','invalid_terminal','stream_error',
    'response_identity','item_identity','model_identity','text_conflict',
    'delta_after_done','provisional_text_unconfirmed',
])
def test_typed_failures_are_bounded_safe_and_bound_to_exact_wire_request(case):
    llm, opener = provider(failure_rows(case))
    with pytest.raises(LLMError) as caught:
        llm.generate('private prompt', MODEL)
    assert len(opener.calls) == 1
    metadata = llm_error_metadata(caught.value)
    assert metadata['stage'] == 'response' and metadata['reason'] == 'malformed_response'
    assert metadata['cause_type'] == '_CodexProtocolError'
    diagnostic = metadata['codex_protocol']
    assert diagnostic['code'] == case
    assert validate_codex_protocol(diagnostic) == diagnostic
    assert diagnostic['model'] == MODEL and diagnostic['endpoint_sha256'] == ENDPOINT_SHA256
    assert diagnostic['body_sha256'] == hashlib.sha256(opener.calls[0][0].data).hexdigest()
    assert diagnostic['schema'] == SCHEMA and diagnostic['bytes_read'] > 0
    assert len(json.dumps(metadata)) < 1536
    assert 'private' not in json.dumps(metadata) and 'fixture-secret' not in json.dumps(metadata)
    if case in ('response_identity','item_identity','model_identity','text_conflict',
                'delta_after_done','provisional_text_unconfirmed'):
        assert case not in RECOVERABLE_CODES
    # Caller mutation must not corrupt the exception's sealed metadata.
    diagnostic['code'] = 'private arbitrary code'
    assert llm_error_metadata(caught.value)['codex_protocol']['code'] == case


@pytest.mark.parametrize('raw,code', [
    (b'data: [DONE]\n\n','done_without_terminal'),
    (b'data: {\xff}\n\n','invalid_utf8'),
    (b'data: {private body}\n\n','invalid_json'),
    (b':' + b'x'*MAX_BYTES, 'stream_limit'),
])
def test_framing_failures_never_retain_wire_bytes(raw, code):
    class RawOpener(Opener):
        def open(self, *args, **kwargs):
            super().open(*args, **kwargs)
            return io.BytesIO(raw)
    opener = RawOpener()
    llm = CodexResponsesLLM(config(), 'fixture-secret', opener=opener)
    with pytest.raises(LLMError) as caught: llm.generate('p',MODEL)
    diagnostic = llm_error_metadata(caught.value)['codex_protocol']
    assert diagnostic['code'] == code and diagnostic['terminal_status'] == 'none'
    assert len(opener.calls) == 1 and 'private' not in json.dumps(diagnostic)


@pytest.mark.parametrize('location', ['opener', 'parser'])
def test_unexpected_local_valueerror_is_not_a_recoverable_protocol_error(monkeypatch, location):
    error = ValueError('private local implementation bug')
    opener = Opener(error=error if location == 'opener' else None)
    if location == 'parser':
        def broken_parser(*args, **kwargs): raise error
        monkeypatch.setattr('sciel.engine.llm.codex._consume', broken_parser)
    llm = CodexResponsesLLM(config(), 'fixture-secret', opener=opener)
    with pytest.raises(LLMError) as caught: llm.generate('p',MODEL)
    assert llm_error_metadata(caught.value) == {
        'stage':'request','reason':'request_failed','cause_type':'ValueError'}
    assert len(opener.calls) == 1


@pytest.mark.parametrize('mutation', [
    {'extra':'private'}, {'code':'private'}, {'last_event':'private'},
    {'terminal_status':'private'}, {'bytes_read':True}, {'events_seen':-1},
    {'bytes_read':MAX_BYTES+65537}, {'endpoint_sha256':'0'*64},
    {'body_sha256':'F'*64}, {'model':'private\nheader'},
])
def test_untrusted_or_malformed_diagnostic_is_dropped_by_sanitizer(mutation):
    llm,_ = provider(failure_rows('stream_eof'))
    with pytest.raises(LLMError) as caught: llm.generate('p',MODEL)
    value = caught.value.metadata
    value['codex_protocol'].update(mutation)
    class Untrusted(LLMError):
        @property
        def metadata(self): return value
    sanitized = llm_error_metadata(Untrusted())
    assert 'codex_protocol' not in sanitized and 'private' not in json.dumps(sanitized)


def test_protocol_diagnostic_survives_durable_error_receipt_without_redispatch(tmp_path):
    llm,opener = provider(failure_rows('invalid_delta'))
    capacity = SimpleNamespace(acquire=lambda *a,**k:nullcontext())
    request = {'provider_request':llm.request_metadata('p',MODEL), 'model':MODEL}
    def run():
        io_calls = CooperativeIO(CallLedger(tmp_path/'wave',capacity,'chatgpt.com'),1)
        try:
            return io_calls.run([('G:fixture',lambda:io_calls.call(
                'llm', request, lambda:llm.generate('p',MODEL)))])
        finally: io_calls.close()
    _,errors = run()
    assert errors and len(opener.calls) == 1
    saved = {str(p):p.read_bytes() for p in tmp_path.rglob('*.json')}
    receipt = json.loads(next(tmp_path.rglob('receipt.json')).read_text())
    assert receipt['payload']['status'] == 'error'
    assert receipt['payload']['error']['codex_protocol']['code'] == 'invalid_delta'
    _,again = run()
    assert again and len(opener.calls) == 1
    assert saved == {str(p):p.read_bytes() for p in tmp_path.rglob('*.json')}
