"""One verified ledger projection; raw-byte checks still detect every rewrite."""
from decimal import Decimal
import json
import os
import time

import pytest

from sciel.methods.budget import ResourceLedger, RESOURCES


def ledger_at(tmp_path):
    ledger = ResourceLedger(tmp_path / 'resources.jsonl', dict.fromkeys(RESOURCES, None))
    ledger.create_account('lab', {})
    return ledger


def charge(ledger, key, amount='.01'):
    row = ledger.reserve('lab', key, {'cpu_seconds': 1}, 'fixture')
    ledger.settle(row['id'], {'cpu_seconds': amount})
    return row


def test_own_append_and_snapshot_do_not_replay_existing_events(tmp_path, monkeypatch):
    ledger = ledger_at(tmp_path)
    original = ResourceLedger._read
    calls = []
    def read(fd):
        calls.append(fd)
        return original(fd)
    monkeypatch.setattr(ResourceLedger, '_read', staticmethod(read))
    for index in range(10):
        charge(ledger, 'call-' + str(index))
    assert ledger.snapshot()['spent']['cpu_seconds'] == '0.1'
    assert calls == []
    resumed = ResourceLedger(ledger.path, ledger.limits)
    assert len(calls) == 1  # A fresh process always verifies the complete chain.
    assert resumed.snapshot() == ledger.snapshot()
    assert len(calls) == 1


def test_peer_append_revalidates_and_preserves_cached_accounting(tmp_path, monkeypatch):
    first = ledger_at(tmp_path)
    second = ResourceLedger(first.path, first.limits)
    charge(first, 'first')
    assert second.snapshot()['spent']['cpu_seconds'] == '0.01'
    charge(second, 'second')
    assert first.snapshot()['spent']['cpu_seconds'] == '0.02'
    assert first.snapshot()['journal'] == second.snapshot()['journal']


def test_full_byte_hash_detects_same_length_restored_timestamp_tamper(tmp_path):
    ledger = ledger_at(tmp_path)
    charge(ledger, 'first')
    ledger.snapshot()
    before = ledger.path.stat()
    raw = ledger.path.read_bytes()
    assert b'"cpu_seconds":"0.01"' in raw
    ledger.path.write_bytes(raw.replace(b'"cpu_seconds":"0.01"', b'"cpu_seconds":"0.02"'))
    os.utime(ledger.path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert ledger.path.stat().st_size == before.st_size
    with pytest.raises(ValueError, match='hash chain mismatch'):
        ledger.snapshot()
    assert ledger._read_cache is None


def test_failed_persistence_cannot_leave_phantom_reserved_usage_in_cache(tmp_path, monkeypatch):
    ledger = ledger_at(tmp_path)
    before = ledger.snapshot()
    original = os.write
    def failed(fd, value):
        raise OSError('fixture disk error before first byte')
    monkeypatch.setattr(os, 'write', failed)
    with pytest.raises(OSError):
        ledger.reserve('lab', 'lost', {'cpu_seconds': 1}, 'fixture')
    assert ledger._read_cache is None
    monkeypatch.setattr(os, 'write', original)
    assert ledger.snapshot() == before
    assert ledger.reserve('lab', 'lost', {'cpu_seconds': 1}, 'fixture')['dispatch_allowed']


def test_partial_write_is_not_hidden_by_cached_prior_valid_state(tmp_path, monkeypatch):
    ledger = ledger_at(tmp_path)
    original = os.write
    calls = []
    def partial(fd, value):
        if calls:
            raise OSError('fixture partial write')
        calls.append(True)
        return original(fd, value[:10])
    monkeypatch.setattr(os, 'write', partial)
    with pytest.raises(OSError):
        ledger.reserve('lab', 'partial', {'cpu_seconds': 1}, 'fixture')
    assert ledger._read_cache is None
    monkeypatch.setattr(os, 'write', original)
    with pytest.raises(ValueError, match='incomplete'):
        ledger.snapshot()


def test_cached_snapshot_returns_no_shared_mutable_accounting(tmp_path):
    ledger = ledger_at(tmp_path)
    charge(ledger, 'first')
    view = ledger.snapshot()
    view['reservations'][0]['actual']['cpu_seconds'] = '999'
    view['limits']['cpu_seconds'] = '0'
    view['accounts']['lab']['cpu_seconds'] = '0'
    assert ledger.snapshot()['spent']['cpu_seconds'] == '0.01'
    assert ledger.snapshot()['limits']['cpu_seconds'] is None


def test_thousand_settled_reservations_keep_hot_snapshot_within_host_bound(tmp_path):
    ledger = ledger_at(tmp_path)
    for index in range(1000):
        charge(ledger, 'call-' + str(index))
    started = time.process_time()
    result = ledger.snapshot()
    elapsed = time.process_time() - started
    assert elapsed < 10, 'Cached snapshot exceeds the actual finite host-operation CPU reservation'
    assert len(result['reservations']) == 1000
    assert Decimal(result['spent']['cpu_seconds']) == Decimal('10')
    assert not result['halted'] and not result['pending_reservations']
    assert result['limits']['cpu_seconds'] is None and result['remaining']['cpu_seconds'] is None
