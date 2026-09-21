from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.campaign.budget import BudgetedLLM, CampaignStopped, CostLedger, validate_llm_config
from sciel.campaign.identity import file_sha, safe_config, verify_files
from sciel.engine.llm.client import EnsembleLLM


def test_parallel_reservations_and_restart_cannot_exceed_limit(tmp_path):
    path = tmp_path / 'cost.json'
    ledger = CostLedger(path, 1)
    def reserve(i):
        try:
            return ledger.reserve('.3', run_id=str(i), model='m')
        except CampaignStopped:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = [i for i in pool.map(reserve, range(8)) if i]
    assert len(ids) == 3
    ledger = CostLedger(path, 1)
    assert Decimal(ledger.snapshot()['reserved_or_unknown_usd']) == Decimal('.9')
    ledger.settle(ids[0], '.1')
    ledger.reserve('.3', run_id='next', model='m')
    with pytest.raises(ValueError, match='limit cannot change'):
        CostLedger(path, 2)


def test_unknown_response_keeps_reservation_and_double_settlement_is_rejected(tmp_path):
    ledger = CostLedger(tmp_path / 'cost.json', 1)
    i = ledger.reserve('.8', run_id='r', model='m')
    ledger.settle(i, error_type='Timeout')
    with pytest.raises(CampaignStopped):
        ledger.reserve('.3', run_id='r', model='m')
    with pytest.raises(ValueError, match='already settled'):
        ledger.settle(i, 0)
    assert ledger.snapshot()['known_cost_usd'] == '0'


def test_provider_overrun_stops_all_future_requests(tmp_path):
    ledger = CostLedger(tmp_path / 'cost.json', 1)
    i = ledger.reserve('.1', run_id='r', model='m')
    with pytest.raises(CampaignStopped, match='exceeded'):
        ledger.settle(i, '.2')
    with pytest.raises(CampaignStopped, match='exceeded'):
        ledger.reserve('.1', run_id='another', model='m')


def llm(tmp_path, limit=1):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        id='gen-1', usage=SimpleNamespace(cost=.0001, prompt_tokens=5, completion_tokens=8),
        choices=[SimpleNamespace(message=SimpleNamespace(content='code'))])
    ledger = CostLedger(tmp_path / 'cost.json', limit)
    cfg = {'models':['m'], 'weights':[1], 'max_tokens':100,
           'api_base':'https://openrouter.ai/api/v1'}
    prices = {'models':[{'id':'m','pricing':{'prompt':'0.000001','completion':'0.000002'}}]}
    return BudgetedLLM(cfg, prices, ledger, 'run', 'test-key', client), client, ledger


def test_cost_limit_is_checked_before_api_call(tmp_path):
    wrapped, client, _ = llm(tmp_path, limit=.00001)
    with pytest.raises(CampaignStopped):
        wrapped.generate('prompt', 'm')
    client.chat.completions.create.assert_not_called()


def test_provider_charge_and_caps_are_used(tmp_path):
    wrapped, client, ledger = llm(tmp_path)
    result = wrapped.generate('prompt', 'm')
    assert result.cost == .0001
    assert Decimal(ledger.snapshot()['known_cost_usd']) == Decimal('.0001')
    body = client.chat.completions.create.call_args.kwargs['extra_body']
    assert body['provider']['max_price'] == {'prompt':1.0, 'completion':2.0, 'request':0.0}
    assert body['provider']['allow_fallbacks'] is False


def test_network_failure_is_not_automatically_retried_or_counted_free(tmp_path):
    wrapped, client, ledger = llm(tmp_path)
    client.chat.completions.create.side_effect = TimeoutError('no response')
    with pytest.raises(CampaignStopped, match='outcome is unknown'):
        wrapped.generate('prompt', 'm')
    assert client.chat.completions.create.call_count == 1
    assert Decimal(ledger.snapshot()['reserved_or_unknown_usd']) > 0
    assert ledger.snapshot()['halted']


def test_missing_cost_stops_campaign_with_reservation_intact(tmp_path):
    wrapped, client, ledger = llm(tmp_path)
    client.chat.completions.create.return_value.usage.cost = None
    with pytest.raises(CampaignStopped, match='actual charge'):
        wrapped.generate('prompt', 'm')
    assert Decimal(ledger.snapshot()['reserved_or_unknown_usd']) > 0
    assert 'actual charge' in ledger.snapshot()['halted']
    # Another process/restart must see the same stop before calling the provider.
    wrapped.ledger = CostLedger(tmp_path / 'cost.json', 1)
    client.reset_mock()
    client.chat.completions.create.return_value.usage.cost = .0001
    with pytest.raises(CampaignStopped, match='actual charge'):
        wrapped.generate('next prompt', 'm')
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize('reported', [float('nan'), float('inf'), -1, True, 'invalid'])
def test_invalid_actual_cost_also_halts_shared_campaign(tmp_path, reported):
    wrapped, client, ledger = llm(tmp_path)
    client.chat.completions.create.return_value.usage.cost = reported
    with pytest.raises(CampaignStopped, match='actual charge'):
        wrapped.generate('prompt', 'm')
    assert ledger.snapshot()['halted']
    assert Decimal(ledger.snapshot()['reserved_or_unknown_usd']) > 0


def valid_settings():
    return ({'models':['m'], 'weights':[1], 'max_tokens':100,
             'api_base':'https://openrouter.ai/api/v1'},
            {'models':[{'id':'m','pricing':{'prompt':'0.000001','completion':'0.000002'}}]})


@pytest.mark.parametrize('field,value', [
    ('models', []), ('models', ['m', 'm']), ('models', 'm'), ('models', [None]),
    ('weights', []), ('weights', [0]), ('weights', [-1]), ('weights', [True]),
    ('weights', ['1']), ('weights', [float('nan')]), ('weights', [float('inf')]),
    ('max_tokens', 0), ('max_tokens', True), ('max_tokens', 1.5),
    ('timeout', 0), ('timeout', float('inf')), ('temperature', float('nan')),
    ('api_base', None), ('api_base', 'https://other.example/v1'),
])
def test_invalid_config_rejected_before_dispatch(tmp_path, field, value):
    cfg, pricing = valid_settings()
    cfg[field] = value
    client, ledger = Mock(), Mock()
    with pytest.raises(ValueError):
        BudgetedLLM(cfg, pricing, ledger, 'run', 'unused', client)
    assert not client.mock_calls
    assert not ledger.mock_calls


@pytest.mark.parametrize('change', ['empty', 'duplicate', 'missing_model', 'missing_prompt',
                                  'negative', 'nonfinite', 'malformed', 'request_bool'])
def test_bad_pricing_rejected_before_dispatch(change):
    cfg, pricing = valid_settings()
    row = pricing['models'][0]
    if change == 'empty': pricing['models'] = []
    elif change == 'duplicate': pricing['models'].append(dict(row))
    elif change == 'missing_model': row['id'] = 'different'
    elif change == 'missing_prompt': del row['pricing']['prompt']
    elif change == 'negative': row['pricing']['completion'] = '-1'
    elif change == 'nonfinite': row['pricing']['prompt'] = 'NaN'
    elif change == 'malformed': row['pricing']['completion'] = 'unknown'
    elif change == 'request_bool': row['pricing']['request'] = True
    with pytest.raises(ValueError):
        validate_llm_config(cfg, pricing)


def test_zero_prices_valid_and_price_snapshot_is_not_mutable_through_inputs(tmp_path):
    cfg, pricing = valid_settings()
    pricing['models'][0]['pricing'].update(prompt=0, completion=0)
    wrapped = BudgetedLLM(cfg, pricing, Mock(), 'run', 'unused', Mock())
    pricing['models'][0]['pricing']['prompt'] = 99
    cfg['models'][0] = 'other'
    cfg['weights'][0] = 0
    assert wrapped.models == ['m'] and wrapped.weights == [1]
    assert wrapped.prices['m']['prompt'] == 0


def test_settlement_metadata_cannot_replace_reservation(tmp_path):
    ledger = CostLedger(tmp_path / 'cost.json', 1)
    ident = ledger.reserve('.5', run_id='r', model='m')
    with pytest.raises(ValueError, match='accounting fields'):
        ledger.settle(ident, '.6', reservation_usd='1')
    assert ledger.snapshot()['reserved_or_unknown_usd'] == '0.5'


@pytest.mark.parametrize('reported', [0.0, .0123])
def test_regular_client_prefers_provider_actual_charge(reported):
    client = EnsembleLLM.__new__(EnsembleLLM)
    client.temperature, client.max_tokens, client.max_retries = .7, 100, 1
    client._client = Mock()
    client._client.chat.completions.create.return_value = SimpleNamespace(
        usage=SimpleNamespace(cost=reported, prompt_tokens=100, completion_tokens=50),
        choices=[SimpleNamespace(message=SimpleNamespace(content='code'))])
    assert client.generate('prompt', 'google/gemini-2.5-flash').cost == reported


def test_file_identity_detects_drift_and_config_redacts_credentials(tmp_path):
    p = tmp_path / 'candidate.py'
    p.write_text('first')
    expected = {'candidate.py': file_sha(p)}
    verify_files(tmp_path, expected)
    p.write_text('changed')
    with pytest.raises(ValueError, match='drift'):
        verify_files(tmp_path, expected)
    cfg = safe_config({'llm':{'api_key':'secret','api_key_env':'MY_KEY'},'token':'hidden'})
    assert cfg == {'llm':{'api_key':'<redacted>','api_key_env':'MY_KEY'},'token':'<redacted>'}


@pytest.mark.parametrize('tree,suffixes', [('src', ('.py', '.txt')), ('frozen', None)])
def test_exact_frozen_file_set_rejects_additions(tmp_path, tree, suffixes):
    folder = tmp_path / tree
    folder.mkdir()
    p = folder / 'module.py'
    p.write_text('VALUE = 1\n')
    frozen = {str(p.relative_to(tmp_path)): file_sha(p)}
    verify_files(tmp_path, frozen, exact_root=tree, suffixes=suffixes)
    # Namespace package initializers are executable source, even when all
    # previously frozen module hashes still match.
    (folder / '__init__.py').write_text('VALUE = 2\n')
    with pytest.raises(ValueError, match='file set changed'):
        verify_files(tmp_path, frozen, exact_root=tree, suffixes=suffixes)
