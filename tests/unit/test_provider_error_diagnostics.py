"""Offline structured provider-error capture; no network or candidate execution."""
from copy import deepcopy
from decimal import Decimal
import json
from unittest.mock import Mock

import httpx
from openai import BadRequestError
import pytest

from sciel.campaign.budget import (BudgetedLLM, CampaignStopped, CostLedger,
                                  provider_error_metadata, request_cost_bound)

PROMPT = 'Private experiment premise αβγ: translate these exact candidate coordinates.'
KEY = 'fixture-opaque-credential-without-provider-prefix'
REQUEST = {'messages': [{'role': 'user', 'content': PROMPT}]}


def error(body, *, httpx_only=False):
    request = httpx.Request('POST', 'https://fixture.invalid/private-path',
                            headers={'Authorization': 'Bearer '+KEY, 'X-Other': 'never-copy'},
                            json=REQUEST)
    response = httpx.Response(400, request=request, json=body,
                              headers={'x-request-id': 'req-fixture-400', 'set-cookie': 'never-copy'})
    if httpx_only:
        return httpx.HTTPStatusError('Do not persist '+PROMPT+KEY, request=request, response=response)
    return BadRequestError('Do not persist '+PROMPT+KEY, response=response, body=body)


def details():
    return {'error': {'code': 'unsupported_parameter', 'param': 'reasoning.effort',
        'message': "Requested effort 'none' is unsupported.", 'ignored': KEY,
        'metadata': {'provider_name': 'unused', 'raw': json.dumps({'error': {
            'code': 400, 'param': 'thinking_budget', 'message': 'A positive budget is required.',
            'request': REQUEST, 'headers': {'Authorization': KEY}}})}}}


@pytest.mark.parametrize('httpx_only', [False, True])
def test_allowlisted_sdk_and_httpx_json_details_without_raw_envelope(httpx_only):
    body=details(); before=deepcopy(body)
    exc=error(body,httpx_only=httpx_only)
    result=provider_error_metadata(exc,request=REQUEST)
    fields=result['error_details']
    assert fields['code']=='unsupported_parameter' and fields['param']=='reasoning.effort'
    assert fields['message']=="Requested effort 'none' is unsupported."
    assert fields['provider']=={'code':400,'param':'thinking_budget','message':'A positive budget is required.'}
    if not httpx_only:
        assert fields['request_id']=='req-fixture-400' and result['error_status_code']==400
    assert body==before
    saved=json.dumps(result)
    assert all(text not in saved for text in (KEY,PROMPT,'Authorization','headers','private-path','never-copy','ignored'))
    assert len(saved)<2000


@pytest.mark.parametrize('field,value', [
    ('message', 'Authorization: Bearer '+KEY),
    ('param', KEY),
    ('request_id', KEY),
    ('code', 'sk-fixture-long-secret'),
    ('message', 'Server repeated '+PROMPT[8:40]),
    ('message', json.dumps(REQUEST,ensure_ascii=True)),
    ('message', ''.join('\\u%04x'%ord(char) for char in KEY)),
])
def test_malicious_reflection_redacts_complete_affected_field(field,value):
    body={'error':{field:value,'metadata':{'raw':json.dumps({'error':{field:value}})}}}
    exc=error(body)
    if field=='request_id':exc.request_id=value
    result=provider_error_metadata(exc,request=REQUEST)['error_details']
    assert result[field]=='[redacted]' and result['provider'][field]=='[redacted]'
    assert KEY not in json.dumps(result) and PROMPT not in json.dumps(result)


def test_diagnostics_bounds_and_broken_accessors_are_safe():
    exc=error({'message':'x'*600,'param':'p'*9000,'code':True,'metadata':{'raw':'{'*20000}})
    result=provider_error_metadata(exc,request=REQUEST)['error_details']
    assert len(result['message'])==512 and result['message'].endswith('...')
    assert result['param']=='[omitted: oversized diagnostic]' and 'code' not in result and 'provider' not in result
    class Broken(RuntimeError):
        @property
        def body(self):raise RuntimeError(KEY)
        @property
        def response(self):raise RuntimeError(KEY)
        @property
        def status_code(self):raise RuntimeError(KEY)
    assert provider_error_metadata(Broken(KEY),request=REQUEST)=={
        'error_type':'Broken','error_status_code':None,'error_cause_type':None}


def test_legacy_callers_and_unstructured_exception_have_exact_old_shape():
    exc=error(details())
    assert provider_error_metadata(exc)=={'error_type':'BadRequestError','error_status_code':400,'error_cause_type':None}
    exc=TimeoutError(PROMPT+KEY)
    assert provider_error_metadata(exc,request=REQUEST)=={
        'error_type':'TimeoutError','error_status_code':None,'error_cause_type':None}


def test_ledger_diagnostics_precede_unknown_settlement_and_no_retry(tmp_path,monkeypatch):
    sdk=Mock();sdk.chat.completions.create.side_effect=error(details())
    ledger=CostLedger(tmp_path/'ledger.json',1)
    cfg={'models':['m'],'weights':[1],'max_tokens':100,'api_base':'https://openrouter.ai/api/v1'}
    rates={'prompt':'.000001','completion':'.000002','request':'0'}
    llm=BudgetedLLM(cfg,{'models':[{'id':'m','pricing':rates}]},ledger,'fixture',KEY,sdk)
    settle=ledger.settle
    def durable(ident,*args,**kwargs):
        saved=provider_error_metadata(sdk.chat.completions.create.side_effect,
                                      request=sdk.chat.completions.create.call_args.kwargs)
        assert all(kwargs[key]==value for key,value in saved.items())
        return settle(ident,*args,**kwargs)
    monkeypatch.setattr(ledger,'settle',durable)
    with pytest.raises(CampaignStopped,match='unknown') as stopped:
        llm.generate(PROMPT,'m')
    state=json.loads(ledger.path.read_text());record,=state['calls'].values()
    assert record['status']=='unknown' and record['cost_usd'] is None and state['halted']
    assert Decimal(record['reservation_usd'])==request_cost_bound(PROMPT,100,rates)
    assert record['error_details']==provider_error_metadata(
        sdk.chat.completions.create.side_effect,
        request=sdk.chat.completions.create.call_args.kwargs)['error_details']
    assert KEY not in ledger.path.read_text() and PROMPT not in ledger.path.read_text()
    assert KEY not in str(stopped.value) and PROMPT not in str(stopped.value)
    with pytest.raises(CampaignStopped):llm.generate('another request','m')
    assert sdk.chat.completions.create.call_count==1 and ledger.snapshot()['calls']==1


def test_broken_request_truthiness_cannot_prevent_diagnostic_or_unknown_settlement(tmp_path):
    class BrokenRequest:
        def __bool__(self):raise RuntimeError('request truthiness must never run')
    exc=RuntimeError('not captured');exc.body={'message':'Parameter unavailable.'};exc.request=BrokenRequest()
    sdk=Mock();sdk.chat.completions.create.side_effect=exc
    ledger=CostLedger(tmp_path/'ledger.json',1)
    cfg={'models':['m'],'weights':[1],'max_tokens':100,'api_base':'https://openrouter.ai/api/v1'}
    llm=BudgetedLLM(cfg,{'models':[{'id':'m','pricing':{'prompt':'.000001','completion':'.000002'}}]},
                    ledger,'fixture','unused',sdk)
    with pytest.raises(CampaignStopped,match='unknown'):llm.generate(PROMPT,'m')
    record,=json.loads(ledger.path.read_text())['calls'].values()
    assert record['status']=='unknown' and record['error_details']['message']=='Parameter unavailable.'
    assert ledger.snapshot()['halted'] and sdk.chat.completions.create.call_count==1


@pytest.fixture(scope='module')
def structured_request():
    """A strict request with short schema literals that also occur in errors."""
    return {
        'model': 'fixture-model',
        'messages': [{'role': 'user', 'content': PROMPT}],
        'temperature': .2,
        'max_tokens': 4096,
        'extra_body': {'reasoning': {'effort': 'none'}, 'provider': {
            'allow_fallbacks': False, 'require_parameters': True,
            'max_price': {'prompt': 3.0, 'completion': 15.0, 'request': 0.0}}},
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'fixture_result', 'strict': True, 'schema': {
                'type': 'object', 'properties': {'result': {'type': 'string'}},
                'required': ['result'], 'additionalProperties': False}}},
    }


def assert_hint(node, category):
    assert node['message_hints']=={
        'schema':'sciel-provider-message-hints/1', 'categories':[category],
        'source':'bounded_provider_message_pattern', 'cause_verified':False}


@pytest.mark.parametrize('message,category',[
    ('Invalid schema: expected object', 'schema_invalid'),
    ('Invalid JSON schema: expected object', 'schema_invalid'),
    ('Schema is invalid: object required', 'schema_invalid'),
    ('Schema is too complex for compilation.', 'schema_complexity_limit'),
    ('JSON schema is too complex for compilation.', 'schema_complexity_limit'),
    ('Schema contains too many enum values.', 'schema_complexity_limit'),
    ('Unsupported parameter: fixture-only-field', 'parameter_unsupported'),
    ('Unrecognized request argument: fixture-only-field', 'parameter_unsupported'),
    ('Context length exceeded.', 'context_limit'),
    ("This model's maximum context length is 123 tokens.", 'context_limit'),
])
def test_fixed_provider_assertions_survive_schema_request_context(structured_request,message,category):
    before=deepcopy(structured_request)
    result=provider_error_metadata(error({'message':message}),request=structured_request)
    fields=result['error_details'];assert_hint(fields,category)
    if message=='Invalid schema: expected object':
        # Existing short-literal redaction is intentionally preserved.
        assert fields['message']=='[redacted]'
    assert structured_request==before
    assert len(json.dumps(fields['message_hints']))<256


def test_redacted_secrets_and_prompt_echo_keep_only_fixed_unverified_hints(structured_request):
    prompt=structured_request['messages'][0]['content']
    body={'error':{'message':'Invalid schema: expected object. '+KEY,
        'message_hints':{'categories':[KEY],'cause_verified':True},
        'metadata':{'raw':json.dumps({'error':{
            'message':'Schema is too complex for compilation. '+prompt[:180]+' Authorization: Bearer '+KEY}})}}}
    original=deepcopy(body)
    fields=provider_error_metadata(error(body),request=structured_request)['error_details']
    assert fields['message']==fields['provider']['message']=='[redacted]'
    assert_hint(fields,'schema_invalid');assert_hint(fields['provider'],'schema_complexity_limit')
    saved=json.dumps(fields)
    assert all(text not in saved for text in (KEY,prompt[:40],'Authorization','expected object'))
    assert body==original and len(saved)<1024


@pytest.mark.parametrize('message',[
    'The schema is valid; another parameter is invalid.',
    'Your input said: Invalid schema: expected object',
    '"Schema is too complex for compilation."',
    'Provider rejected something; schema and invalid appear in separate clauses.',
    '[redacted]',
    'No error: Context length exceeded is a quoted example.',
])
def test_nonassertions_quoted_input_and_lost_history_are_not_classified(structured_request,message):
    fields=provider_error_metadata(error({'message':message}),request=structured_request)['error_details']
    assert 'message_hints' not in fields


@pytest.mark.parametrize('message',[
    'Invalid schema: '+KEY+'x'*8192,
    {'message':'Invalid schema: expected object'},
    ['Schema is too complex for compilation.'],
    b'Context length exceeded', True, 400,
])
def test_oversized_and_malformed_messages_have_no_hints(structured_request,message):
    # SDK body may be a malformed object; it is never stringified by the hint path.
    exc=RuntimeError('never capture');exc.body={'message':message}
    result=provider_error_metadata(exc,request=structured_request)
    assert 'message_hints' not in result.get('error_details',{})
    assert KEY not in json.dumps(result)
    if type(message) is str:
        assert result['error_details']['message']=='[omitted: oversized diagnostic]'


def test_hint_bounds_do_not_serialize_hostile_values_or_parse_oversized_nested_json(structured_request):
    class Hostile:
        def __str__(self):pytest.fail('diagnostic must never stringify arbitrary objects')
    exc=RuntimeError('never capture');exc.body={'message':Hostile(),
        'metadata':{'raw':'{"message":"Invalid schema: '+'x'*17000+'"}'}}
    assert 'error_details' not in provider_error_metadata(exc,request=structured_request)
    exc.body={'message':'Invalid schema: '+('\\u0073'*1360)}
    fields=provider_error_metadata(exc,request=structured_request)['error_details']
    assert_hint(fields,'schema_invalid')
    assert len(json.dumps(fields))<1024


def test_legacy_no_request_shape_unchanged_even_for_recognized_error():
    result=provider_error_metadata(error({'message':'Schema is too complex for compilation.'}))
    assert result=={'error_type':'BadRequestError','error_status_code':400,'error_cause_type':None}


def test_structured_failure_persists_hints_without_settling_or_retrying(structured_request,tmp_path):
    sdk=Mock();sdk.chat.completions.create.side_effect=error({'message':'Invalid schema: expected object. '+KEY})
    request=structured_request;model=request['model'];prompt=request['messages'][0]['content']
    cfg={'models':[model],'weights':[1],'max_tokens':4096,'temperature':.2,'api_base':'https://openrouter.ai/api/v1'}
    rates={'prompt':'.000003','completion':'.000015','request':'0'}
    ledger=CostLedger(tmp_path/'cost_ledger.json',1)
    llm=BudgetedLLM(cfg,{'models':[{'id':model,'pricing':rates}]},ledger,'synthetic-hint-only',KEY,sdk)
    with pytest.raises(CampaignStopped,match='unknown'):
        llm.generate(prompt,model,.2,response_format=request['response_format'])
    assert sdk.chat.completions.create.call_count==1
    assert sdk.chat.completions.create.call_args.kwargs==request
    saved=provider_error_metadata(sdk.chat.completions.create.side_effect,request=request)
    state=json.loads(ledger.path.read_text());row,=state['calls'].values()
    assert row['error_details']==saved['error_details'];assert_hint(row['error_details'],'schema_invalid')
    assert row['status']=='unknown' and row['cost_usd'] is None and state['halted']
    assert Decimal(row['reservation_usd'])==request_cost_bound(
        prompt,4096,rates,response_format=request['response_format'])
    assert KEY not in ledger.path.read_text() and prompt[:80] not in ledger.path.read_text()
    with pytest.raises(CampaignStopped):llm.generate(prompt,model,.2,response_format=request['response_format'])
    assert sdk.chat.completions.create.call_count==1 and ledger.snapshot()['calls']==1
