"""Prospective pricing provenance, with offline SDK bodies and durable replay."""
from contextlib import nullcontext
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.engine.llm.client import (COST_PROVENANCE_SCHEMA, EnsembleLLM, LLMResponse,
                                     known_cost, response_cost_provenance)
from sciel.engine.llm.trace import TracedLLM
from sciel.engine.parallel_calls import CallLedger, _decode, _encode


ABSENT = object()


def provider(usage=ABSENT, *, model='google/gemini-2.5-flash', choices=True):
    """Use the installed SDK's real non-strict deserializer, never its network."""
    from openai._models import construct_type
    from openai.types.chat import ChatCompletion

    body = dict(id='offline-fixture', created=0, model=model, object='chat.completion',
                choices=[dict(index=0, finish_reason='error',
                    message=dict(role='assistant', content='partial saved output'))] if choices else [])
    if usage is not ABSENT:
        body['usage'] = usage
    sdk_response = construct_type(type_=ChatCompletion, value=body)
    llm = EnsembleLLM.__new__(EnsembleLLM)
    llm.temperature, llm.max_tokens, llm.max_retries = .7, 100, 1
    llm._client = Mock()
    llm._client.chat.completions.create.return_value = sdk_response
    return llm


@pytest.mark.parametrize('usage', [ABSENT, None, {}, {'prompt_tokens': 0}])
def test_missing_usage_stays_unknown_with_unavailable_provenance(usage):
    llm = provider(usage)
    result = llm.generate('prompt', 'google/gemini-2.5-flash')
    assert result.cost is None and result.completion_tokens is None
    assert result.cost_provenance == {
        'schema': COST_PROVENANCE_SCHEMA, 'source': 'unavailable'}
    assert response_cost_provenance(result) == result.cost_provenance
    assert llm._client.chat.completions.create.call_count == 1
    with pytest.raises(ValueError, match='unknown'):
        known_cost(result)


@pytest.mark.parametrize('provider_cost', [ABSENT, 0.0, .002])
def test_explicit_and_estimated_cost_survive_durable_receipt_replay_and_trace(tmp_path, provider_cost):
    usage = dict(prompt_tokens=0, completion_tokens=0)
    if provider_cost is not ABSENT:
        usage['cost'] = provider_cost
    llm = provider(usage)
    traced = TracedLLM(llm, tmp_path / 'trace', lambda _: {'role': 'G'})
    ledger = CallLedger(tmp_path / 'wave', SimpleNamespace(acquire=lambda *a, **k: nullcontext()), 'local')
    request = {'prompt': 'same saved prompt'}
    directory, digest, cached = ledger.prepare('G:1:0', 'llm', request)
    assert cached is None
    payload = ledger.perform(directory, digest, 'llm', lambda: traced.generate(request['prompt'], 'google/gemini-2.5-flash'))
    first = ledger.result('llm', payload)
    # A fresh ledger loads the immutable receipt and performs no provider call.
    again = CallLedger(tmp_path / 'wave', SimpleNamespace(acquire=Mock(side_effect=AssertionError('no dispatch'))), 'local')
    _, _, cached = again.prepare('G:1:0', 'llm', request)
    replay = again.result('llm', cached)
    assert replay == first and llm._client.chat.completions.create.call_count == 1
    assert replay.cost == (0.0 if provider_cost is ABSENT else provider_cost)
    assert replay.finish_reason == 'error'
    trace = [json.loads(x) for x in (tmp_path / 'trace/artifacts/llm_calls.jsonl').read_text().splitlines()][-1]
    assert trace['cost_provenance'] == replay.cost_provenance == payload['value']['cost_provenance']
    assert trace['current_call_cost_unknown'] is False
    if provider_cost is ABSENT:
        assert replay.cost_provenance == dict(schema=COST_PROVENANCE_SCHEMA,
            source='token_estimate', pricing_key='gemini-2.5-flash',
            prompt_usd_per_1k=.0003, completion_usd_per_1k=.0025)
    else:
        assert replay.cost_provenance == dict(schema=COST_PROVENANCE_SCHEMA,
            source='provider_reported', reported_cost=provider_cost)


def test_estimate_records_rates_and_validates_historical_rates_without_current_lookup(monkeypatch):
    result = provider({'prompt_tokens': 1000, 'completion_tokens': 2000}).generate(
        'p', 'google/gemini-2.5-flash')
    assert result.cost == .0053
    # Pricing changes cannot relabel or invalidate the inputs of an old estimate.
    monkeypatch.setitem(__import__('sciel.engine.llm.client', fromlist=['PRICE_PER_1K']).PRICE_PER_1K,
                        'gemini-2.5-flash', (.9, .9))
    assert response_cost_provenance(result)['completion_usd_per_1k'] == .0025
    fallback = provider({'prompt_tokens': 1000, 'completion_tokens': 2000}, model='unlisted').generate('p', 'unlisted')
    assert fallback.cost == pytest.approx(.0018) and fallback.cost_provenance['pricing_key'] == 'default'


def test_provider_cost_zero_without_counts_and_malformed_paid_body_keep_provenance():
    result = provider({'cost': 0.0}, choices=False).generate('prompt', 'google/gemini-2.5-flash')
    assert result.cost == 0 and result.prompt_tokens is None and result.completion_tokens is None
    assert result.response_error['reason'] == 'empty_choices'
    assert response_cost_provenance(result)['source'] == 'provider_reported'


def test_unavailable_cost_survives_receipt_and_trace_as_unknown(tmp_path):
    llm = provider()
    result = TracedLLM(llm, tmp_path, lambda _: {}).generate('prompt', 'google/gemini-2.5-flash')
    replay = _decode('llm', _encode('llm', result))
    assert replay.cost is None and replay.prompt_tokens is None and replay.completion_tokens is None
    assert replay.cost_provenance == dict(schema=COST_PROVENANCE_SCHEMA, source='unavailable')
    row = json.loads((tmp_path / 'artifacts/llm_calls.jsonl').read_text().splitlines()[-1])
    assert row['cost'] is None and row['current_call_cost_unknown'] is True
    assert row['cost_provenance'] == replay.cost_provenance


@pytest.mark.parametrize('value', [
    {'schema': COST_PROVENANCE_SCHEMA, 'source': 'provider_reported', 'reported_cost': .1},
    {'schema': COST_PROVENANCE_SCHEMA, 'source': 'provider_reported', 'reported_cost': float('nan')},
    {'schema': COST_PROVENANCE_SCHEMA, 'source': 'provider_reported', 'reported_cost': 0, 'headers': {'Authorization': 'secret'}},
    {'schema': COST_PROVENANCE_SCHEMA, 'source': 'token_estimate', 'pricing_key': 'default',
     'prompt_usd_per_1k': float('inf'), 'completion_usd_per_1k': .0008},
    {'schema': COST_PROVENANCE_SCHEMA, 'source': 'unavailable'},
    {'schema': 'unknown', 'source': 'provider_reported'},
])
def test_malformed_optional_provenance_never_changes_cost_or_claims_provider(value, tmp_path):
    result = LLMResponse('text', 'fixture', 0.0, 0, 0, cost_provenance=value)
    assert response_cost_provenance(result) is None
    encoded = _encode('llm', result)
    assert 'cost_provenance' not in encoded and encoded['cost'] == 0.0
    # Even a legacy/damaged optional field on decode is not trusted or upgraded.
    decoded = _decode('llm', {**encoded, 'cost_provenance': value})
    assert decoded.cost_provenance is None and decoded.cost == 0.0
    trace = TracedLLM(SimpleNamespace(generate=lambda *a: result), tmp_path, lambda _: {})
    assert trace.generate('prompt', 'fixture') is result
    saved = (tmp_path / 'artifacts/llm_calls.jsonl').read_text()
    assert 'cost_provenance' not in saved and 'Authorization' not in saved and 'secret' not in saved


def test_inconsistent_estimate_and_unavailable_provenance_are_not_upgraded():
    details = dict(schema=COST_PROVENANCE_SCHEMA, source='token_estimate', pricing_key='default',
                   prompt_usd_per_1k=.0002, completion_usd_per_1k=.0008)
    response = LLMResponse('x', 'm', .01, 1, 1, cost_provenance=details)
    assert response_cost_provenance(response) is None
    response.cost = None
    assert response_cost_provenance(response) is None


def test_legacy_receipt_remains_unannotated_and_unchanged():
    old = dict(text='saved', model='m', cost=0.0, prompt_tokens=0,
               completion_tokens=0, finish_reason='stop', refusal=None)
    before = json.dumps(old, sort_keys=True)
    result = _decode('llm', old)
    assert result.cost_provenance is None and _encode('llm', result) == old
    assert json.dumps(old, sort_keys=True) == before


@pytest.mark.parametrize('reported_cost', [0.0, .0001])
def test_budgeted_adapter_marks_only_accepted_actual_provider_cost(tmp_path, reported_cost):
    from test_campaign_budget import llm as make_llm

    llm, sdk, ledger = make_llm(tmp_path)
    sdk.chat.completions.create.return_value.usage.cost = reported_cost
    response = llm.generate('prompt', 'm')
    assert response.cost == reported_cost
    assert response_cost_provenance(response) == dict(schema=COST_PROVENANCE_SCHEMA,
        source='provider_reported', reported_cost=reported_cost)
    assert sdk.chat.completions.create.call_count == 1
