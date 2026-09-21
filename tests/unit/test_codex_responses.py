"""Offline Codex SSE/transport contracts; never read real auth or call a provider."""
from contextlib import nullcontext
from copy import deepcopy
from email.message import Message
import hashlib
import json
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from sciel.campaign.budget import RequestNotDispatched
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.factory import build_llm
from sciel.engine.llm.client import LLMError, known_cost, response_accounting
from sciel.engine.llm.codex import CodexResponsesLLM, ENDPOINT
from sciel.engine.llm.trace import TracedLLM
from sciel.engine.parallel import WaveLLM
from sciel.engine.parallel_calls import CallLedger, CooperativeIO

MODEL = 'gpt-5.6-luna'
FORMAT = {'type':'json_schema','json_schema':{'name':'result','strict':True,'schema':{
    'type':'object','additionalProperties':False,'properties':{'ok':{'type':'boolean'}},'required':['ok']}}}


def config(**extra):
    return dict(provider='codex_responses', models=[MODEL], weights=[1.], api_base=ENDPOINT,
                max_tokens=32000, temperature=.7, **extra)


def events(text='{"ok":true}', *, status='completed', final_output=False):
    item = {'id':'message1','type':'message','role':'assistant','phase':'final_answer',
            'status':'completed','content':[{'type':'output_text','text':text}]}
    response = {'id':'response1','model':MODEL,'status':status,'output':[item] if final_output else [],
                'usage':{'input_tokens':45,'output_tokens':14},'error':None}
    return [{'type':'response.created','response':{**response,'status':'in_progress','output':[],'usage':None}},
        {'type':'response.output_item.added','output_index':0,'item':{**item,'status':'in_progress','content':[]}},
        {'type':'response.output_text.delta','output_index':0,'content_index':0,'item_id':'message1','delta':text[:2]},
        {'type':'response.output_text.delta','output_index':0,'content_index':0,'item_id':'message1','delta':text[2:]},
        {'type':'response.output_text.done','output_index':0,'content_index':0,'item_id':'message1','text':text},
        {'type':'response.output_item.done','output_index':0,'item':item},
        {'type':'response.'+status,'response':response}]


class Stream:
    def __init__(self, rows, chunk=7, error=None):
        raw = b': heartbeat\r\n\r\n' + b''.join(b'data: '+json.dumps(r,ensure_ascii=False).encode()+b'\r\n\r\n' for r in rows)
        self.chunks = iter(raw[n:n+chunk] for n in range(0,len(raw),chunk))
        self.error = error
        self.headers = Message(); self.headers['Content-Type'] = 'text/event-stream; charset=utf-8'
    def read1(self, size):
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        return next(self.chunks, b'')
    def __enter__(self): return self
    def __exit__(self, *args): pass


class Opener:
    def __init__(self, rows=None, *, error=None, chunk=7):
        self.rows = events() if rows is None else rows
        self.error, self.chunk, self.calls = error, chunk, []
    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error is not None: raise self.error
        return Stream(self.rows, self.chunk)


def provider(rows=None, **cfg):
    opener = Opener(rows)
    return CodexResponsesLLM(config(**cfg), 'fixture-secret', opener=opener), opener


@pytest.mark.parametrize('final_output', [False, True])
@pytest.mark.parametrize('chunk', [1, 7, 65536])
def test_native_schema_stream_deduplicates_done_final_and_preserves_unpriced_usage(final_output, chunk):
    text = '{"ok":true,"unicode":"雪🙂"}'
    llm, opener = provider(events(text, final_output=final_output)); opener.chunk=chunk
    response = llm.generate('exact prompt', MODEL, .2, response_format=FORMAT, max_tokens=64)
    assert response.text == text and response.finish_reason == 'stop'
    assert response.prompt_tokens == 45 and response.completion_tokens == 14 and response.cost is None
    assert response_accounting(response) == {'known_usd':0.0,'unpriced_calls':1}
    with pytest.raises(ValueError): known_cost(response)
    assert len(opener.calls) == 1
    request = opener.calls[0][0]; body = json.loads(request.data)
    assert request.full_url == ENDPOINT and request.get_header('Authorization') == 'Bearer fixture-secret'
    assert body['text']['format'] == {'type':'json_schema', **FORMAT['json_schema']}
    assert body['input'][0]['content'][0]['text'] == 'exact prompt'
    assert body['store'] is False and body['stream'] is True
    assert 'temperature' not in body and 'max_output_tokens' not in body and 'max_tokens' not in body
    metadata = llm.request_metadata('exact prompt',MODEL,.2,response_format=FORMAT,max_tokens=64)
    assert metadata['body_sha256'] == hashlib.sha256(request.data).hexdigest()
    assert metadata['output_token_cap'] == {'requested':64,'sent':False,'enforced':False}
    assert metadata['temperature'] == {'requested':.2,'sent':False}
    assert 'fixture-secret' not in json.dumps(metadata)


@pytest.mark.parametrize('status,finish', [('incomplete','length'),('failed','error')])
def test_received_noncomplete_terminal_preserves_text_tokens_but_is_not_priced_completion(status,finish):
    llm, opener = provider(events('partial',status=status))
    response = llm.generate('p',MODEL)
    assert response.text == 'partial' and response.finish_reason == finish
    assert response.completion_tokens == 14 and response.cost is None
    with pytest.raises(ValueError): response_accounting(response)
    assert len(opener.calls) == 1


def test_valid_looking_json_without_terminal_is_not_a_completed_response():
    llm, opener = provider(events()[:-1])
    with pytest.raises(LLMError,match='LLM request failed'): llm.generate('p',MODEL,response_format=FORMAT)
    assert len(opener.calls) == 1


@pytest.mark.parametrize('content_type', [None, 'text/plain', 'application/octet-stream'])
def test_complete_sse_without_event_stream_mime_is_accepted(content_type):
    class MissingMIME(Opener):
        def open(self, *args, **kwargs):
            stream = super().open(*args, **kwargs)
            del stream.headers['Content-Type']
            if content_type is not None:
                stream.headers['Content-Type'] = content_type
            return stream
    opener = MissingMIME()
    llm = CodexResponsesLLM(config(), 'fixture-secret', opener=opener)
    response = llm.generate('p', MODEL)
    assert json.loads(response.text) == {'ok': True}
    assert response_accounting(response)['unpriced_calls'] == 1
    assert len(opener.calls) == 1


@pytest.mark.parametrize('body', [b'<html>unavailable</html>', b'{"ok":true}'])
def test_non_sse_body_without_content_type_cannot_look_completed(body):
    class NotSSE(Opener):
        def open(self, *args, **kwargs):
            stream = super().open(*args, **kwargs)
            del stream.headers['Content-Type']
            stream.chunks = iter([body])
            return stream
    opener = NotSSE()
    llm = CodexResponsesLLM(config(), 'fixture-secret', opener=opener)
    with pytest.raises(LLMError):
        llm.generate('p', MODEL)
    assert len(opener.calls) == 1


@pytest.mark.parametrize('failure', [TimeoutError('private error'), HTTPError(ENDPOINT,400,'private body',{},None),
    HTTPError(ENDPOINT,429,'private body',{'Retry-After':'3'},None)])
def test_errors_never_retry_fallback_or_expose_remote_body(failure):
    opener = Opener(error=failure); llm = CodexResponsesLLM(config(), 'fixture-secret', opener=opener)
    with pytest.raises(LLMError) as caught: llm.generate('p',MODEL,response_format=FORMAT)
    assert len(opener.calls) == 1
    assert 'private' not in str(caught.value) and 'fixture-secret' not in json.dumps(caught.value.metadata)
    if getattr(failure,'code',None) == 429:
        assert caught.value.metadata['retry_after_seconds'] == 3


def test_auth_file_is_explicit_rotates_per_request_and_does_not_change_wire_identity(tmp_path):
    path = tmp_path/'auth.json'; path.write_text(json.dumps({'tokens':{'access_token':'first'}}))
    llm, opener = provider(codex_auth_file=str(path))
    identity = llm.transport_identity; metadata = llm.request_metadata('p',MODEL)
    llm.generate('p',MODEL)
    path.write_text(json.dumps({'tokens':{'access_token':'second'}}))
    llm.generate('p',MODEL)
    assert [c[0].get_header('Authorization') for c in opener.calls] == ['Bearer first','Bearer second']
    assert llm.transport_identity == identity and llm.request_metadata('p',MODEL) == metadata
    path.write_text('{}')
    with pytest.raises(RequestNotDispatched): llm.generate('p',MODEL)
    assert len(opener.calls) == 2


def test_factory_explicit_auth_file_skips_env_and_selects_only_codex(monkeypatch,tmp_path):
    monkeypatch.setattr('sciel.engine.factory.load_env',lambda: pytest.fail('explicit auth file does not read dotenv'))
    llm = build_llm({'llm':config(codex_auth_file=str(tmp_path/'later.json'))})
    assert isinstance(llm,CodexResponsesLLM)
    # Construction and pure identity do not read credentials.
    assert llm.transport_identity['auth']['strategy'] == 'file_per_request'


@pytest.mark.parametrize('change',[{'api_base':'https://openrouter.ai/api/v1'}, {'max_retries':2},
    {'model_capabilities':{MODEL:{'temperature':True}}}, {'provider':'unknown'}])
def test_bad_provider_configuration_rejected_before_dispatch(change):
    with pytest.raises(ValueError): apply_defaults({'llm':{**config(),**change}})


def test_checkpoint_binds_backend_auth_strategy_and_wire_semantics_not_token():
    llm,_=provider()
    engine=SimpleNamespace(task=SimpleNamespace(qualified_name='fixture',task_hash='task'),
                           llm=llm,cfg={'llm':config()})
    saved={'identity':checkpoint._identity(engine)}
    assert checkpoint.check_identity(engine,saved)==[]
    llm._api_key='rotated'
    assert checkpoint.check_identity(engine,saved)==[]
    llm.config['reasoning_effort']='high'
    with pytest.raises(checkpoint.IdentityMismatch):checkpoint.check_identity(engine,saved)
    engine.cfg['llm']['reasoning_effort']='high'
    assert any('llm_transport' in p for p in checkpoint.check_identity(engine,saved))


def test_trace_and_wal_bind_exact_wire_and_replay_without_post_or_double_callback(tmp_path):
    llm,opener=provider();accounted=[]
    engine=SimpleNamespace(run_dir=tmp_path,_parallel_wave=1,
        _llm_trace_context=lambda _: {'role':'C','reference':'r1','experiment_id':None},
        _record_llm_accounting=lambda response:accounted.append(response_accounting(response)))
    capacity=SimpleNamespace(acquire=lambda *a,**k:nullcontext())
    def run():
        io=CooperativeIO(CallLedger(tmp_path/'wave',capacity,'chatgpt.com'),1)
        try:
            return io.run([('C:r1',lambda:WaveLLM(engine,llm,io).generate('exact',MODEL,.2,response_format=FORMAT))])
        finally:io.close()
    result,errors=run();assert not errors
    # Simulate wave baseline restore: scientific accounting rolls back, physical I/O does not.
    accounted.clear(); saved={str(p):p.read_bytes() for p in (tmp_path/'wave').rglob('*.json')}
    again,errors=run();assert not errors and result==again
    assert len(opener.calls)==1 and accounted==[{'known_usd':0.0,'unpriced_calls':1}]
    assert saved=={str(p):p.read_bytes() for p in (tmp_path/'wave').rglob('*.json')}
    intent=json.loads(next((tmp_path/'wave').rglob('intent.json')).read_text())
    trace=[json.loads(line) for line in (tmp_path/'artifacts/llm_calls.jsonl').read_text().splitlines()]
    assert intent['request']['provider_request']==trace[0]['provider_request']
    assert trace[-1]['cost'] is None and trace[-1]['unpriced_calls']==1
    assert trace[-1]['current_call_cost_unknown'] is False
    llm.config['reasoning_effort']='high'
    _,errors=run();assert errors and len(opener.calls)==1
