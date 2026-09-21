"""Provider transport/cost plumbing only: fake completions, no model or code execution."""
import copy
from decimal import Decimal
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.campaign.budget import BudgetedLLM, CampaignStopped, CostLedger, request_cost_bound
from sciel.engine.llm.client import (EnsembleLLM, LLMError, LLMResponse,
    response_format_bytes, validate_response_format)


def schema():
    return {'type': 'json_schema', 'json_schema': {'name': 'fixture_v1', 'strict': True,
        'schema': {'type': 'object', 'properties': {'body': {'type': 'string', 'description': 'Точный код 😀'}},
                   'required': ['body'], 'additionalProperties': False}}}


RATES = {'prompt': '0.000001', 'completion': '0.000002', 'request': '0.000003'}


@pytest.fixture(params=['ensemble', 'budgeted'])
def provider(request, tmp_path):
    sdk = Mock()
    sdk.chat.completions.create.return_value = SimpleNamespace(id='fake-return',
        choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content='raw response', refusal=None))],
        usage=SimpleNamespace(cost=.0001, prompt_tokens=7, completion_tokens=9))
    ledger = None
    if request.param == 'ensemble':
        instance = EnsembleLLM.__new__(EnsembleLLM)
        instance.temperature, instance.max_tokens, instance.max_retries = .7, 100, 1
        instance.api_base, instance._client = 'https://openrouter.ai/api/v1', sdk
    else:
        ledger = CostLedger(tmp_path / 'ledger.json', 1)
        instance = BudgetedLLM({'models': ['m'], 'weights': [1], 'max_tokens': 100,
            'api_base': 'https://openrouter.ai/api/v1'},
            {'models': [{'id': 'm', 'pricing': RATES}]}, ledger, 'run', 'unused', sdk)
    return SimpleNamespace(kind=request.param, llm=instance, sdk=sdk, ledger=ledger)


def expected_legacy(kind):
    result = {'model': 'm', 'messages': [{'role': 'user', 'content': 'prompt'}],
              'temperature': .7, 'max_tokens': 100}
    if kind == 'budgeted':
        result['extra_body'] = {'reasoning': {'effort': 'none'}, 'provider': {
            'allow_fallbacks': False, 'require_parameters': True,
            'max_price': {'prompt': 1., 'completion': 2., 'request': .000003}}}
    return result


def test_none_preserves_exact_old_kwargs_and_legacy_dto_defaults(provider):
    assert LLMResponse('raw', 'm', 0., 0, 0).finish_reason is None
    assert LLMResponse('raw', 'm', 0., 0, 0).refusal is None
    for options in ({}, {'response_format': None}):
        provider.llm.generate('prompt', 'm', **options)
        assert provider.sdk.chat.completions.create.call_args.kwargs == expected_legacy(provider.kind)


def test_structured_format_is_detached_and_text_is_not_parsed_or_repaired(provider):
    requested = schema(); before = copy.deepcopy(requested)
    raw = '  {not even valid JSON}\r\n```python\nprovider raw bytes\n```\x00'
    response = provider.sdk.chat.completions.create.return_value
    response.choices[0].message.content = raw
    def create(**kwargs):
        assert kwargs['response_format'] == before and kwargs['response_format'] is not requested
        requested['json_schema']['schema']['properties']['body']['description'] = 'mutated externally'
        assert kwargs['response_format'] == before
        return response
    provider.sdk.chat.completions.create.side_effect = create
    result = provider.llm.generate('prompt', 'm', .2, response_format=requested)
    assert result.text == raw and result.cost == .0001
    assert result.finish_reason == 'stop' and result.refusal is None
    assert provider.sdk.chat.completions.create.call_count == 1
    policy = provider.sdk.chat.completions.create.call_args.kwargs['extra_body']['provider']
    assert policy['require_parameters'] is True and policy['allow_fallbacks'] is False
    if provider.ledger:
        saved = json.loads(provider.ledger.path.read_text())['calls']
        assert Decimal(next(iter(saved.values()))['reservation_usd']) == request_cost_bound(
            'prompt', 100, RATES, response_format=before)


@pytest.mark.parametrize('finish,refusal', [('length', None), ('content_filter', 'No.'), (None, None),
    ('stop', ''), ('stop', {'unsafe_sdk_object': 'not serialized'})])
def test_returned_metadata_is_honest_and_charge_is_preserved(provider, finish, refusal):
    response = provider.sdk.chat.completions.create.return_value
    response.choices[0].finish_reason = finish
    response.choices[0].message.refusal = refusal
    response.choices[0].message.content = ''
    result = provider.llm.generate('prompt', 'm', response_format=schema())
    assert result.cost == .0001 and result.text == '' and result.finish_reason == finish
    assert result.refusal == (refusal if refusal is None or type(refusal) is str
                              else '[unsupported refusal metadata]')
    if provider.ledger:
        assert Decimal(provider.ledger.snapshot()['known_cost_usd']) == Decimal('.0001')


def invalid_formats():
    cases = [False, [], {'type': 'json_object'}, {'type': 'json_schema', 'json_schema': {}},
             {'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': {}, 'strict': True}}]
    for field, value in [('strict', False), ('strict', 1), ('name', 'bad name'), ('name', 'x'*65),
                         ('schema', []), ('description', 3)]:
        item = schema(); item['json_schema'][field] = value; cases.append(item)
    for value in [float('nan'), float('inf'), (1, 2), object(), {1: 'key'}, '\ud800', 'x'*131072]:
        item = schema(); item['json_schema']['schema']['custom'] = value; cases.append(item)
    item = schema(); item['healing'] = True; cases.append(item)
    item = schema(); item['json_schema']['schema']['cycle'] = item; cases.append(item)
    return cases


@pytest.mark.parametrize('value', invalid_formats())
def test_invalid_format_rejected_before_reservation_or_dispatch(provider, value):
    before = provider.ledger.path.read_bytes() if provider.ledger else None
    with pytest.raises(ValueError):
        provider.llm.generate('prompt', 'm', response_format=value)
    provider.sdk.chat.completions.create.assert_not_called()
    if before is not None:
        assert provider.ledger.path.read_bytes() == before


def test_pure_bound_counts_exact_canonical_utf8_schema_and_keeps_old_formula():
    wanted = schema(); before = copy.deepcopy(wanted)
    encoded = json.dumps(wanted, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    assert response_format_bytes(wanted) == encoded
    old = ((len('π'.encode()) + 1024) * Decimal(RATES['prompt'])
           + 100 * Decimal(RATES['completion']) + Decimal(RATES['request'])) * Decimal('1.10')
    assert request_cost_bound('π', 100, RATES) == old
    assert request_cost_bound('π', 100, RATES, response_format=wanted) == (
        old + len(encoded)*Decimal(RATES['prompt'])*Decimal('1.10'))
    assert wanted == before and validate_response_format(wanted) == before


def test_schema_overhead_can_block_before_provider_when_old_bound_would_fit(tmp_path):
    cap = request_cost_bound('prompt', 100, RATES)
    ledger = CostLedger(tmp_path/'ledger.json', cap)
    sdk = Mock()
    llm = BudgetedLLM({'models': ['m'], 'weights': [1], 'max_tokens': 100,
        'api_base': 'https://openrouter.ai/api/v1'},
        {'models': [{'id': 'm', 'pricing': RATES}]}, ledger, 'run', 'unused', sdk)
    with pytest.raises(CampaignStopped):
        llm.generate('prompt', 'm', response_format=schema())
    sdk.chat.completions.create.assert_not_called()
    assert ledger.snapshot()['calls'] == 0


@pytest.mark.parametrize('cost', [None, 'invalid'])
def test_structured_unknown_cost_still_holds_full_schema_bound(tmp_path, cost):
    sdk = Mock()
    sdk.chat.completions.create.return_value = SimpleNamespace(id='fake-return',
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"body":"raw"}'))],
        usage=SimpleNamespace(cost=cost, prompt_tokens=1, completion_tokens=1))
    ledger = CostLedger(tmp_path/'ledger.json', 1)
    llm = BudgetedLLM({'models': ['m'], 'weights': [1], 'max_tokens': 100,
        'api_base': 'https://openrouter.ai/api/v1'},
        {'models': [{'id': 'm', 'pricing': RATES}]}, ledger, 'run', 'unused', sdk)
    with pytest.raises(CampaignStopped): llm.generate('prompt', 'm', response_format=schema())
    assert sdk.chat.completions.create.call_count == 1
    assert Decimal(ledger.snapshot()['reserved_or_unknown_usd']) == request_cost_bound(
        'prompt', 100, RATES, response_format=schema())


def test_structured_network_failure_is_one_call_even_with_legacy_retry_setting(provider, monkeypatch):
    provider.llm.max_retries = 3
    provider.sdk.chat.completions.create.side_effect = TimeoutError('fake failure')
    with pytest.raises((LLMError, CampaignStopped)):
        provider.llm.generate('prompt', 'm', response_format=schema())
    assert provider.sdk.chat.completions.create.call_count == 1
    if provider.ledger:
        assert Decimal(provider.ledger.snapshot()['reserved_or_unknown_usd']) > 0


def test_legacy_ensemble_setting_cannot_replay_an_unknown_physical_dispatch():
    llm = EnsembleLLM.__new__(EnsembleLLM)
    llm.temperature, llm.max_tokens, llm.max_retries, llm._client = .7, 100, 2, Mock()
    llm._client.chat.completions.create.side_effect = TimeoutError('fake failure')
    with pytest.raises(LLMError): llm.generate('prompt', 'm')
    assert llm._client.chat.completions.create.call_count == 1


def test_generic_compatible_endpoint_receives_native_schema_without_router_fields():
    llm = EnsembleLLM.__new__(EnsembleLLM)
    llm.temperature, llm.max_tokens, llm.max_retries, llm._client = .7, 100, 3, Mock()
    llm.api_base = 'https://compatible.invalid/v1'
    llm._client.chat.completions.create.return_value = SimpleNamespace(
        usage=SimpleNamespace(cost=.0001, prompt_tokens=0, completion_tokens=0),
        choices=[SimpleNamespace(message=SimpleNamespace(content=' exact raw \r\n'))])
    result = llm.generate('prompt', 'm', response_format=schema())
    assert llm._client.chat.completions.create.call_args.kwargs == {
        **expected_legacy('ensemble'), 'response_format': schema()}
    assert result.text == ' exact raw \r\n' and result.finish_reason is None and result.refusal is None
