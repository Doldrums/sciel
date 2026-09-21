"""Unbounded total admission does not weaken finite per-dispatch accounting."""
from decimal import Decimal
import json

import pytest

from sciel.methods.budget import (BudgetExceeded, BudgetUnknown, RESOURCES,
                                  ResourceLedger, _limit_vector, _vector)


def unlimited(tmp_path):
    ledger = ResourceLedger(tmp_path / 'ledger.jsonl', dict.fromkeys(RESOURCES, None))
    ledger.create_account('lab', {})
    return ledger


def test_explicit_unbounded_caps_persist_without_infinity_or_magic_number(tmp_path):
    ledger = unlimited(tmp_path)
    bound = dict(dollars='123456.789', cpu_seconds=900, gpu_seconds=0, wall_seconds=950,
                 storage_bytes=123456789, steps=100)
    for index in range(2):
        row = ledger.reserve('lab', 'call-' + str(index), bound, 'experiment')
        assert row['dispatch_allowed'] is True
        ledger.settle(row['id'], bound)
    snap = ledger.snapshot()
    assert snap['limits'] == snap['remaining'] == snap['shared_remaining'] == dict.fromkeys(RESOURCES, None)
    assert Decimal(snap['spent']['dollars']) == Decimal('246913.578')
    assert snap['spent']['storage_bytes'] == 246913578
    assert snap['spent']['steps'] == 200
    assert set(snap['overrun'].values()) <= {'0', 0}
    assert not snap['halted']
    assert 'Infinity' not in ledger.path.read_text()
    original = ledger.path.read_bytes()
    restored = ResourceLedger(ledger.path, dict.fromkeys(RESOURCES, None))
    assert restored.snapshot() == snap
    assert restored.snapshot('lab')['limits'] == dict.fromkeys(RESOURCES, None)
    assert ledger.path.read_bytes() == original
    cached = restored.reserve('lab', 'call-0', bound, 'experiment')
    assert cached['reused'] is True and cached['dispatch_allowed'] is False


def test_null_cap_does_not_mean_unlimited_for_omitted_dimensions(tmp_path):
    ledger = ResourceLedger(tmp_path / 'ledger', {'dollars': None})
    ledger.create_account('lab', {})
    snap = ledger.snapshot('lab')
    assert snap['limits']['dollars'] is None
    assert snap['limits']['cpu_seconds'] == '0'
    assert snap['limits']['steps'] == 0
    with pytest.raises(BudgetExceeded, match='cpu_seconds'):
        ledger.reserve('lab', 'cpu', {'cpu_seconds': 1}, 'work')


def test_finite_account_cap_remains_binding_under_unlimited_campaign(tmp_path):
    ledger = unlimited(tmp_path)
    ledger.create_account('small', {'dollars': 1, 'steps': 2})
    row = ledger.reserve('small', 'first', {'dollars': '.6', 'steps': 1}, 'generation')
    ledger.settle(row['id'], {'dollars': '.6', 'steps': 1})
    with pytest.raises(BudgetExceeded, match='account dollars'):
        ledger.reserve('small', 'second', {'dollars': '.5'}, 'generation')
    assert ledger.snapshot()['remaining']['dollars'] is None
    assert ledger.snapshot('small')['remaining']['dollars'] == '0.4'
    assert ledger.snapshot('small')['remaining']['cpu_seconds'] is None


def test_finite_campaign_does_not_allow_unbounded_account_to_reset_its_cap(tmp_path):
    ledger = ResourceLedger(tmp_path / 'ledger', {'dollars': 1})
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match='account cap exceeds'):
        ledger.create_account('lab', {'dollars': None})
    assert ledger.path.read_bytes() == before


def test_total_cap_changes_are_refused_both_directions_on_resume(tmp_path):
    path = tmp_path / 'ledger'
    ledger = ResourceLedger(path, {'dollars': None})
    ledger.create_account('lab', {'dollars': 2})
    with pytest.raises(ValueError, match='campaign limits cannot change'):
        ResourceLedger(path, {'dollars': 2})
    with pytest.raises(ValueError, match='account limits cannot change'):
        ledger.create_account('lab', {'dollars': None})
    finite = ResourceLedger(tmp_path / 'finite', {'dollars': 2})
    with pytest.raises(ValueError, match='campaign limits cannot change'):
        ResourceLedger(finite.path, {'dollars': None})


@pytest.mark.parametrize('bad', [None, float('inf'), float('nan'), '-Infinity', True, -1])
def test_unlimited_totals_still_require_finite_reservations_and_actuals(tmp_path, bad):
    ledger = unlimited(tmp_path)
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError):
        ledger.reserve('lab', 'bad', {'dollars': bad}, 'generation')
    assert ledger.path.read_bytes() == before
    row = ledger.reserve('lab', 'known', {'dollars': 1}, 'generation')
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError):
        ledger.settle(row['id'], {'dollars': bad})
    assert ledger.path.read_bytes() == before


def test_unknown_call_keeps_reserve_and_blocks_its_account_even_when_unlimited(tmp_path):
    ledger = unlimited(tmp_path)
    row = ledger.reserve('lab', 'lost', {'dollars': 2}, 'generation')
    ledger.settle(row['id'], {'dollars': '.25'}, status='unknown')
    snapshot = ledger.snapshot('lab')
    assert snapshot['spent']['dollars'] == '0.25'
    assert snapshot['reserved']['dollars'] == '1.75'
    assert snapshot['remaining']['dollars'] is None
    assert snapshot['unknown_reservations'] == [row['id']]
    with pytest.raises(BudgetUnknown):
        ledger.reserve('lab', 'next', {'dollars': '.1'}, 'generation')
    with pytest.raises(BudgetUnknown):
        ledger.reserve('lab', 'lost', {'dollars': 2}, 'generation')
    ledger.resolve_unknown(row['id'], {'dollars': '.5'}, reason='provider receipt recovered',
                           evidence_refs=['receipt:known'])
    assert ledger.snapshot()['reserved']['dollars'] == '0'
    assert ledger.reserve('lab', 'next', {'dollars': '.1'}, 'generation')['dispatch_allowed']


def test_observed_request_overrun_halts_even_with_all_total_caps_unbounded(tmp_path):
    ledger = unlimited(tmp_path)
    row = ledger.reserve('lab', 'call', {'dollars': 1}, 'generation')
    with pytest.raises(BudgetExceeded, match='charge recorded'):
        ledger.settle(row['id'], {'dollars': '1.1'})
    snap = ledger.snapshot()
    assert snap['spent']['dollars'] == '1.1'
    assert snap['halted'] == [row['id']]
    assert snap['reservations'][0]['violations'] == ['reservation bound exceeded: dollars']
    assert snap['overrun']['dollars'] == '0'  # No total ceiling; per-call bound still exceeded.
    with pytest.raises(BudgetExceeded, match='halted'):
        ledger.reserve('lab', 'again', {'dollars': 1}, 'generation')


def test_limit_normalizer_is_separate_from_finite_usage_vectors():
    assert _limit_vector({'steps': None})['steps'] is None
    assert _limit_vector({}, defaults={'dollars': None})['dollars'] is None
    with pytest.raises(ValueError):
        _vector({'steps': None})
    with pytest.raises(ValueError):
        _limit_vector({'steps': float('inf')})
    with pytest.raises(ValueError):
        _limit_vector({'tokens': None})
