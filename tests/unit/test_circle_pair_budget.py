"""One native pair is one budgeted request; all provider I/O is a fake."""
import copy
from decimal import Decimal
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from sciel.campaign.budget import (
    BudgetedLLM, CostLedger, RequestNotDispatched, request_cost_bound,
)
from sciel.hypothesis import circle_proposal_pair as pair


MODEL = 'fixture/flash'
RATES = {'prompt': '0.0000001', 'completion': '0.0000004', 'request': '0.000001'}
PROMPT = 'Use the same measured parent for both preferred candidates. π'
CHARGE = Decimal('0.0002')


def native_reply():
    candidates = []
    for index, body in enumerate(('return best_params.copy()\n', 'return initial_params.copy()\n')):
        candidates.append({'hypothesis': {
            'hypothesis': f'Prospective candidate {index}.',
            'predicted_low': -.001, 'predicted_high': .002,
            'reasoning': 'Unmeasured forecast relative to the same parent.',
            'confidence': .4}, 'body': body})
    # Noncanonical whitespace is intentionally preserved through the provider.
    return ' \n' + json.dumps({'candidates': candidates}, indent=2) + '\n'


def fixture(tmp_path, monkeypatch, *, limit, max_tokens=8192, prior_cost=None):
    ledger = CostLedger(tmp_path / 'temporary-ledger.json', limit)
    prior = None
    if prior_cost is not None:
        ident = ledger.reserve(prior_cost, run_id='earlier', model=MODEL)
        ledger.settle(ident, prior_cost, generation_id='earlier-return')
        prior = copy.deepcopy(json.loads(ledger.path.read_text())['calls'])
    reserve, settle = Mock(wraps=ledger.reserve), Mock(wraps=ledger.settle)
    monkeypatch.setattr(ledger, 'reserve', reserve)
    monkeypatch.setattr(ledger, 'settle', settle)
    sdk = Mock()
    raw = native_reply()
    returned = NS(id='pair-return', usage=NS(cost=str(CHARGE), prompt_tokens=41,
        completion_tokens=85), choices=[NS(finish_reason='stop',
        message=NS(content=raw, refusal=None))])
    held_at_dispatch = []
    def create(**_):
        state = json.loads(ledger.path.read_text())
        held_at_dispatch.append([copy.deepcopy(call) for call in state['calls'].values()
                                 if call['status'] == 'reserved'])
        return returned
    sdk.chat.completions.create.side_effect = create
    llm = BudgetedLLM({'models': [MODEL], 'weights': [1], 'max_tokens': max_tokens,
        'temperature': .2, 'api_base': 'https://openrouter.ai/api/v1'},
        {'models': [{'id': MODEL, 'pricing': RATES}]}, ledger, 'pair-run',
        'unused-fixture-key', sdk)
    return NS(ledger=ledger, sdk=sdk, llm=llm, raw=raw, reserve=reserve,
              settle=settle, held=held_at_dispatch, prior=prior)


@pytest.mark.parametrize('max_tokens', [1024, 8192])
def test_actual_pair_schema_uses_one_reserve_one_charge_and_unchanged_output_cap(
        tmp_path, monkeypatch, max_tokens):
    f = fixture(tmp_path, monkeypatch, limit=1, max_tokens=max_tokens)
    schema = pair.response_format()
    expected_schema = copy.deepcopy(schema)
    result = f.llm.generate(PROMPT, MODEL, response_format=schema)
    f.sdk.chat.completions.create.assert_called_once()
    sent = f.sdk.chat.completions.create.call_args.kwargs
    assert sent['response_format'] == expected_schema == pair.response_format()
    assert sent['response_format'] is not schema
    array = sent['response_format']['json_schema']['schema']['properties']['candidates']
    assert array['minItems'] == array['maxItems'] == 2
    assert sent['max_tokens'] == f.llm.max_tokens == max_tokens
    assert sent['messages'] == [{'role': 'user', 'content': PROMPT}]
    assert sent['extra_body']['provider']['require_parameters'] is True
    assert sent['extra_body']['provider']['allow_fallbacks'] is False
    assert result.text == f.raw and result.cost == float(CHARGE)
    decoded = pair.decode_response(result.text)
    assert decoded['error'] is None and len(decoded['entries']) == 2
    assert all(entry['proposal'] is not None for entry in decoded['entries'])
    assert f.reserve.call_count == f.settle.call_count == 1
    assert len(f.held) == len(f.held[0]) == 1
    schema_bytes = json.dumps(expected_schema, sort_keys=True, ensure_ascii=False,
                             separators=(',', ':'), allow_nan=False).encode('utf-8')
    expected = ((len(PROMPT.encode('utf-8')) + len(schema_bytes) + 1024)
                * Decimal(RATES['prompt']) + max_tokens * Decimal(RATES['completion'])
                + Decimal(RATES['request'])) * Decimal('1.10')
    assert Decimal(f.held[0][0]['reservation_usd']) == expected
    calls = json.loads(f.ledger.path.read_text())['calls']
    call, = calls.values()
    assert call['run_id'] == 'pair-run' and call['generation_id'] == 'pair-return'
    assert call['status'] == 'settled' and Decimal(call['cost_usd']) == CHARGE
    assert Decimal(f.ledger.snapshot()['known_cost_usd']) == CHARGE
    assert Decimal(f.ledger.snapshot()['reserved_or_unknown_usd']) == 0


@pytest.mark.parametrize('available_bounds', [Decimal('1'), Decimal('1.5')])
def test_remaining_shared_allowance_for_one_full_pair_not_two_dispatches_once(
        tmp_path, monkeypatch, available_bounds):
    bound = request_cost_bound(PROMPT, 8192, RATES, response_format=pair.response_format())
    remaining = available_bounds * bound
    assert bound <= remaining < 2 * bound
    f = fixture(tmp_path, monkeypatch, limit=5, prior_cost=Decimal(5) - remaining)
    assert Decimal(f.ledger.snapshot()['remaining_usd']) == remaining
    result = f.llm.generate(PROMPT, MODEL, response_format=pair.response_format())
    assert result.text == f.raw
    assert f.reserve.call_count == f.settle.call_count == 1
    f.sdk.chat.completions.create.assert_called_once()
    assert len(f.held[0]) == 1 and Decimal(f.held[0][0]['reservation_usd']) == bound
    saved = json.loads(f.ledger.path.read_text())['calls']
    assert {key: saved[key] for key in f.prior} == f.prior
    own = [call for key, call in saved.items() if key not in f.prior]
    assert len(own) == 1 and own[0]['status'] == 'settled'
    assert Decimal(f.ledger.snapshot()['known_cost_usd']) == Decimal(5) - remaining + CHARGE


def test_below_one_full_schema_bound_prevents_dispatch_without_new_paid_receipt(tmp_path, monkeypatch):
    full = request_cost_bound(PROMPT, 8192, RATES, response_format=pair.response_format())
    without_schema = request_cost_bound(PROMPT, 8192, RATES)
    remaining = (without_schema + full) / 2
    assert without_schema < remaining < full
    f = fixture(tmp_path, monkeypatch, limit=5, prior_cost=Decimal(5) - remaining)
    before = f.ledger.path.read_bytes()
    with pytest.raises(RequestNotDispatched, match='cost limit'):
        f.llm.generate(PROMPT, MODEL, response_format=pair.response_format())
    f.sdk.chat.completions.create.assert_not_called()
    assert f.reserve.call_count == 1 and f.settle.call_count == 0 and f.held == []
    assert f.ledger.path.read_bytes() == before
    assert json.loads(f.ledger.path.read_text())['calls'] == f.prior
    assert Decimal(f.ledger.snapshot()['reserved_or_unknown_usd']) == 0
