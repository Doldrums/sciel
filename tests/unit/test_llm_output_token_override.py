"""Actual SDK adapters honor per-call output caps and reserve/trace that cap."""
from decimal import Decimal
import json

import pytest

from sciel.campaign.budget import CampaignStopped, RequestNotDispatched, request_cost_bound
from sciel.engine.llm.client import LLMError
from sciel.engine.llm.trace import TracedLLM
from test_llm_structured_transport import provider, schema, RATES


@pytest.mark.parametrize('structured', [False, True])
def test_actual_adapter_uses_override_without_mutating_default(provider, tmp_path, structured):
    traced = TracedLLM(provider.llm, tmp_path / 'trace', lambda _: {'role': 'C'})
    options = {'response_format': schema()} if structured else {}
    traced.generate('prompt', 'm', max_tokens=40, **options)
    assert provider.sdk.chat.completions.create.call_args.kwargs['max_tokens'] == 40
    assert provider.llm.max_tokens == 100
    if provider.ledger:
        row, = json.loads(provider.ledger.path.read_text())['calls'].values()
        assert Decimal(row['reservation_usd']) == request_cost_bound(
            'prompt', 40, RATES, response_format=options.get('response_format'))
    traced.generate('prompt', 'm', max_tokens=None, **options)
    assert provider.sdk.chat.completions.create.call_args.kwargs['max_tokens'] == 100
    requests = [r for r in map(json.loads, (tmp_path / 'trace/artifacts/llm_calls.jsonl').read_text().splitlines())
                if r['event'] == 'request']
    assert [r['max_tokens'] for r in requests] == [40, 100]
    assert provider.sdk.chat.completions.create.call_count == 2


@pytest.mark.parametrize('value', [True, False, 0, -1, 1.5, '40', [], {}])
def test_bad_override_is_rejected_before_sdk_or_reservation(provider, value):
    before = provider.ledger.path.read_bytes() if provider.ledger else None
    with pytest.raises(ValueError, match='max_tokens'):
        provider.llm.generate('prompt', 'm', max_tokens=value)
    provider.sdk.chat.completions.create.assert_not_called()
    if before is not None:
        assert provider.ledger.path.read_bytes() == before


@pytest.mark.parametrize('value', [0, True, -1, '40'])
def test_bad_traced_cap_is_not_recorded_as_unknown_dispatch(provider, tmp_path, value):
    trace_dir = tmp_path / 'never-dispatched'
    traced = TracedLLM(provider.llm, trace_dir, lambda _: {'role': 'C'})
    before = provider.ledger.path.read_bytes() if provider.ledger else None
    with pytest.raises(RequestNotDispatched, match='max_tokens'):
        traced.generate('prompt', 'm', max_tokens=value)
    assert not trace_dir.exists()
    provider.sdk.chat.completions.create.assert_not_called()
    if before is not None:
        assert provider.ledger.path.read_bytes() == before


def test_unknown_call_is_not_retried_and_holds_override_reservation(provider):
    provider.sdk.chat.completions.create.side_effect = TimeoutError('offline fixture')
    with pytest.raises((LLMError, CampaignStopped)):
        provider.llm.generate('prompt', 'm', max_tokens=40, response_format=schema())
    assert provider.sdk.chat.completions.create.call_count == 1
    if provider.ledger:
        assert Decimal(provider.ledger.snapshot()['reserved_or_unknown_usd']) == request_cost_bound(
            'prompt', 40, RATES, response_format=schema())
