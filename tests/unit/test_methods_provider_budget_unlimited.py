from decimal import Decimal
import json

import pytest

from sciel.campaign.budget import CampaignStopped, CostLedger, RequestNotDispatched
from sciel.methods.provider_budget import UnlimitedCostLedger


def test_provider_unlimited_cap_stays_null_and_preserves_charges_across_resume(tmp_path):
    ledger = UnlimitedCostLedger(tmp_path / 'provider.json')
    for _ in range(2):
        ident = ledger.reserve(Decimal('1e40'), run_id='run', model='fixture')
        ledger.settle(ident, Decimal('1e40'))
    snap = ledger.snapshot()
    assert snap['limit_usd'] is None and snap['remaining_usd'] is None
    assert Decimal(snap['known_cost_usd']) == Decimal('2e40')
    assert snap['reserved_or_unknown_usd'] == '0'
    assert snap['calls'] == 2 and snap['halted'] is None
    assert UnlimitedCostLedger(ledger.path).snapshot() == snap
    raw = json.loads(ledger.path.read_text())
    assert raw['limit_usd'] is None and len(raw['calls']) == 2
    assert 'Infinity' not in ledger.path.read_text()


def test_existing_finite_provider_ledger_cannot_be_upgraded_silently(tmp_path):
    path = tmp_path / 'provider.json'
    CostLedger(path, 1)
    original = path.read_bytes()
    with pytest.raises(ValueError, match='cannot change'):
        UnlimitedCostLedger(path)
    assert path.read_bytes() == original
    with pytest.raises(ValueError):
        UnlimitedCostLedger(tmp_path / 'other', 1)
    unbounded = UnlimitedCostLedger(tmp_path / 'unbounded')
    with pytest.raises(ValueError):
        CostLedger(unbounded.path, 1)


@pytest.mark.parametrize('bad', [None, True, float('nan'), float('inf'), '-1'])
def test_per_call_bounds_and_costs_are_finite_with_unlimited_total(tmp_path, bad):
    ledger = UnlimitedCostLedger(tmp_path / 'provider.json')
    with pytest.raises(ValueError):
        ledger.reserve(bad, run_id='r', model='m')
    if bad is None:
        return  # None actual is the separately tested retained unknown outcome.
    ident = ledger.reserve('.1', run_id='r', model='m')
    with pytest.raises(ValueError):
        ledger.settle(ident, bad)
    assert ledger.snapshot()['reserved_or_unknown_usd'] == '0.1'


@pytest.mark.parametrize('halt_reason', [None, 'provider charge unknown'])
def test_unknown_outcome_remains_held_and_blocks_more_provider_dispatch(tmp_path, halt_reason):
    ledger = UnlimitedCostLedger(tmp_path / 'provider.json')
    ident = ledger.reserve('.3', run_id='r', model='m')
    ledger.settle(ident, None, halt_reason=halt_reason)
    restored = UnlimitedCostLedger(ledger.path)
    assert restored.snapshot()['reserved_or_unknown_usd'] == '0.3'
    with pytest.raises(RequestNotDispatched):
        restored.reserve('.2', run_id='r', model='m')
    with pytest.raises(ValueError, match='already settled'):
        restored.settle(ident, 0)


def test_request_overrun_records_actual_and_halts_despite_null_total_cap(tmp_path):
    ledger = UnlimitedCostLedger(tmp_path / 'provider.json')
    ident = ledger.reserve('.1', run_id='r', model='m')
    with pytest.raises(CampaignStopped, match='exceeded'):
        ledger.settle(ident, '.2')
    assert ledger.snapshot()['known_cost_usd'] == '0.2'
    with pytest.raises(RequestNotDispatched, match='exceeded'):
        ledger.reserve('.1', run_id='r', model='m')


def test_local_pause_resume_preserves_every_charge_and_audit(tmp_path):
    ledger = UnlimitedCostLedger(tmp_path / 'provider.json')
    reason = 'local research pause: inspect updated procedure'
    ident = ledger.reserve('.2', run_id='r', model='m')
    ledger.settle(ident, '.1', halt_reason=reason)
    ledger.resume_local_pause(reason, resolution='explicit local inspection complete')
    state = json.loads(ledger.path.read_text())
    assert state['halted'] is None
    assert state['calls'][ident]['cost_usd'] == '0.1'
    assert state['local_pause_resolutions'][0]['limit_usd'] is None
    assert state['local_pause_resolutions'][0]['known_cost_usd'] == '0.1'


def test_local_pause_cannot_forgive_an_unknown_provider_charge(tmp_path):
    ledger = UnlimitedCostLedger(tmp_path / 'provider.json')
    reason = 'local research pause: inspect'
    ident = ledger.reserve('.2', run_id='r', model='m')
    ledger.settle(ident, None, halt_reason=reason)
    with pytest.raises(CampaignStopped, match='unknown'):
        ledger.resume_local_pause(reason, resolution='not proof of cost')
    assert ledger.snapshot()['reserved_or_unknown_usd'] == '0.2'
