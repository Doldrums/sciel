"""Offline resource accounting and funding-policy regressions."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import multiprocessing
import os

import pytest

from sciel.methods.budget import BudgetExceeded, BudgetUnknown, FundingPolicy, RESOURCES, ResourceLedger


def make_ledger(tmp_path, **limits):
    ledger = ResourceLedger(tmp_path / "ledger.jsonl", limits or {"dollars": 1})
    ledger.create_account("lab", {})
    return ledger


def _process_reservation(args):
    path, index = args
    ledger = ResourceLedger(path, {"dollars": 1})
    try:
        return ledger.reserve(f"lab-{index}", f"request-{index}", {"dollars": ".3"}, "generation")["id"]
    except BudgetExceeded:
        return None


def test_campaign_pool_is_shared_across_processes_and_accounts(tmp_path):
    path = tmp_path / "pool.jsonl"
    ledger = ResourceLedger(path, {"dollars": 1})
    for i in range(8):
        ledger.create_account(f"lab-{i}", {"dollars": 1})
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork")) as pool:
        successful = list(filter(None, pool.map(_process_reservation, [(str(path), i) for i in range(8)])))
    assert len(successful) == 3
    snap = ResourceLedger(path, {"dollars": 1}).snapshot()
    assert snap["reserved"]["dollars"] == "0.9"
    assert snap["remaining"]["dollars"] == "0.1"


@pytest.mark.parametrize("key", RESOURCES)
def test_every_resource_enforces_account_and_campaign_bounds(tmp_path, key):
    ledger = make_ledger(tmp_path, **{key: 10})
    ledger.create_account("limited", {key: 3})
    with pytest.raises(BudgetExceeded, match="account"):
        ledger.reserve("limited", "too-big", {key: 4}, "evaluation")
    a = ledger.reserve("limited", "first", {key: 3}, "evaluation")
    ledger.settle(a["id"], {key: 2})
    ledger.reserve("lab", "second", {key: 8}, "evaluation")
    with pytest.raises(BudgetExceeded, match="campaign"):
        ledger.reserve("limited", "last", {key: 1}, "evaluation")
    assert Decimal(str(ledger.snapshot()["remaining"][key])) == 0


def test_failed_known_calls_charge_and_cached_receipts_do_not_recharge(tmp_path):
    ledger = make_ledger(tmp_path)
    reservation = ledger.reserve("lab", "req", {"dollars": ".8"}, "repair")
    assert reservation["dispatch_allowed"] and not reservation["reused"]
    with pytest.raises(BudgetUnknown, match="dispatched again"):
        ledger.reserve("lab", "req", {"dollars": ".8"}, "repair")
    settled = ledger.settle(reservation["id"], {"dollars": ".2"}, status="failed")
    count = ledger.snapshot()["journal"]["events"]
    assert ledger.settle(reservation["id"], {"dollars": ".20"}, status="failed") == settled
    replay = ResourceLedger(ledger.path, {"dollars": "1.0"}).reserve("lab", "req", {"dollars": ".8"}, "repair")
    assert replay["status"] == "failed" and replay["reused"] and not replay["dispatch_allowed"]
    assert ledger.snapshot()["spent"]["dollars"] == "0.2"
    assert ledger.snapshot()["journal"]["events"] == count
    with pytest.raises(ValueError, match="different reservation"):
        ledger.reserve("lab", "req", {"dollars": ".8"}, "generation")


def test_unknown_holds_unmeasured_resources_after_restart_until_audited_resolution(tmp_path):
    ledger = make_ledger(tmp_path, dollars=1, cpu_seconds=10)
    ledger.create_account("other", {})
    row = ledger.reserve("lab", "lost", {"dollars": ".8", "cpu_seconds": 6}, "reproduction")
    ledger.settle(row["id"], {"cpu_seconds": 2}, status="unknown")
    ledger = ResourceLedger(ledger.path, {"dollars": 1, "cpu_seconds": 10})
    snap = ledger.snapshot("lab")
    assert snap["spent"]["cpu_seconds"] == "2"
    assert snap["reserved"]["cpu_seconds"] == "4"
    assert snap["reserved"]["dollars"] == "0.8"
    with pytest.raises(BudgetUnknown):
        ledger.reserve("lab", "new-id", {"dollars": ".1"}, "generation")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("other", "other-big", {"dollars": ".3"}, "generation")
    ledger.reserve("other", "other-small", {"dollars": ".1"}, "generation")
    with pytest.raises(BudgetUnknown, match="explicit audited"):
        ledger.settle(row["id"], {"dollars": ".3", "cpu_seconds": 2})
    with pytest.raises(ValueError, match="evidence"):
        ledger.resolve_unknown(row["id"], {"dollars": ".3", "cpu_seconds": 2}, reason="guess", evidence_refs=[])
    resolved = ledger.resolve_unknown(row["id"], {"dollars": ".3", "cpu_seconds": 2},
                                     reason="provider receipt reconciled", evidence_refs=["receipt-sha256:abc"])
    assert resolved["resolution"]["reason"] == "provider receipt reconciled"
    assert ledger.snapshot()["spent"]["dollars"] == "0.3"
    assert ledger.snapshot()["reserved"]["dollars"] == "0.1"
    ledger.reserve("lab", "resumed", {"dollars": ".2"}, "generation")


def test_observed_overrun_is_durable_and_blocks_new_work_without_losing_charge(tmp_path):
    ledger = make_ledger(tmp_path)
    first = ledger.reserve("lab", "first", {"dollars": ".4"}, "generation")
    sibling = ledger.reserve("lab", "sibling", {"dollars": ".4"}, "generation")
    with pytest.raises(BudgetExceeded, match="charge recorded") as error:
        ledger.settle(first["id"], {"dollars": "1.2"})
    assert error.value.record["actual"]["dollars"] == "1.2"
    restored = ResourceLedger(ledger.path, {"dollars": 1})
    snap = restored.snapshot()
    assert snap["spent"]["dollars"] == "1.2"
    assert snap["reserved"]["dollars"] == "0.4"
    assert snap["overrun"]["dollars"] == "0.6"
    with pytest.raises(BudgetExceeded, match="halted"):
        restored.reserve("lab", "forbidden", {}, "generation")
    # Already dispatched siblings still persist their true cost after the halt.
    with pytest.raises(BudgetExceeded):
        restored.settle(sibling["id"], {"dollars": ".1"})
    assert restored.snapshot()["spent"]["dollars"] == "1.3"
    assert restored.snapshot()["reserved"]["dollars"] == "0"


def test_missing_actual_does_not_turn_unmeasured_usage_into_zero(tmp_path):
    ledger = make_ledger(tmp_path, dollars=1, cpu_seconds=10, steps=3)
    row = ledger.reserve("lab", "req", {"dollars": ".5", "cpu_seconds": 4, "steps": 1}, "experiment")
    with pytest.raises(ValueError, match="missing reserved"):
        ledger.settle(row["id"], {"dollars": ".1", "steps": 1})
    assert ledger.snapshot()["pending_reservations"] == [row["id"]]
    ledger.settle(row["id"], {"dollars": ".1", "cpu_seconds": 2}, status="unknown")
    with pytest.raises(ValueError, match="cannot be reduced"):
        ledger.resolve_unknown(row["id"], {"dollars": 0, "cpu_seconds": 2, "steps": 1},
                               reason="unsupported cancellation", evidence_refs=["r"])


def test_limits_and_existing_account_cannot_reset_on_revision(tmp_path):
    ledger = make_ledger(tmp_path)
    row = ledger.reserve("lab", "req", {"dollars": ".7"}, "generation")
    ledger.settle(row["id"], {"dollars": ".7"})
    ledger.create_account("lab", {"dollars": "1.0"})
    assert ledger.snapshot("lab")["remaining"]["dollars"] == "0.3"
    with pytest.raises(ValueError, match="account limits cannot change"):
        ledger.create_account("lab", {"dollars": 2})
    with pytest.raises(ValueError, match="campaign limits cannot change"):
        ResourceLedger(ledger.path, {"dollars": 2})
    ledger.create_account("fork", {"dollars": 1})
    with pytest.raises(BudgetExceeded, match="campaign"):
        ledger.reserve("fork", "fork-req", {"dollars": ".4"}, "generation")


@pytest.mark.parametrize("values", [{"tokens": 1}, {"steps": 1.5}, {"storage_bytes": True},
                                   {"dollars": float("nan")}, {"dollars": float("inf")},
                                   {"cpu_seconds": -1}, {"dollars": None}])
def test_invalid_resource_vectors_fail_without_events(tmp_path, values):
    ledger = make_ledger(tmp_path)
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError):
        ledger.reserve("lab", "invalid", values, "generation")
    assert ledger.path.read_bytes() == before


def test_decimal_accounting_is_exact_and_default_resources_have_no_budget(tmp_path):
    ledger = make_ledger(tmp_path)
    for i in range(10):
        row = ledger.reserve("lab", str(i), {"dollars": ".1"}, "generation")
        ledger.settle(row["id"], {"dollars": ".1"})
    assert ledger.snapshot()["spent"]["dollars"] == "1"
    with pytest.raises(BudgetExceeded):
        ledger.reserve("lab", "step", {"steps": 1}, "generation")


def test_actual_timer_float_is_not_rejected_for_representation_precision(tmp_path):
    ledger = make_ledger(tmp_path, cpu_seconds=1)
    row = ledger.reserve("lab", "cpu", {"cpu_seconds": 1}, "measurement")
    actual = 0.00011299999999999599
    ledger.settle(row["id"], {"cpu_seconds": actual})
    assert Decimal(ledger.snapshot()["spent"]["cpu_seconds"]) == Decimal(str(actual))


@pytest.mark.parametrize("damage", ["truncate", "change", "reorder"])
def test_corrupted_journal_fails_closed_without_rewriting(tmp_path, damage):
    ledger = make_ledger(tmp_path)
    ledger.reserve("lab", "one", {"dollars": ".2"}, "generation")
    content = ledger.path.read_bytes()
    if damage == "truncate":
        content = content[:-7]
    elif damage == "change":
        content = content.replace(b'"0.2"', b'"0.1"')
    else:
        lines = content.splitlines(keepends=True)
        content = b"".join([lines[0], lines[2], lines[1]])
    ledger.path.write_bytes(content)
    with pytest.raises(ValueError):
        ResourceLedger(ledger.path, {"dollars": 1})
    assert ledger.path.read_bytes() == content


def test_short_writes_are_completed_before_success_and_snapshots_are_detached(tmp_path, monkeypatch):
    original = os.write
    monkeypatch.setattr(os, "write", lambda fd, data: original(fd, data[:17]))
    ledger = make_ledger(tmp_path)
    row = ledger.reserve("lab", "req", {"dollars": ".2"}, "generation")
    row["bounds"]["dollars"] = "0"
    snap = ledger.snapshot()
    snap["accounts"]["lab"]["dollars"] = "100"
    assert ResourceLedger(ledger.path, {"dollars": 1}).snapshot()["reserved"]["dollars"] == "0.2"


def test_funding_lineage_survives_revisions_and_replication_does_not_penalize(tmp_path):
    policy = FundingPolicy(initial_quota=4)
    policy.create_lineage("line")
    policy.bind_version("v1", "line")
    policy.record_block("line", "b1", informative=False, evidence_refs=["same-result"])
    policy.bind_version("v2", "line", parent_versions=["v1"])
    policy.record_block("line", "replication", informative=False, replication=True, evidence_refs=["reproduced"])
    assert policy.snapshot("line")["no_information_streak"] == 1
    policy.record_block("line", "b2", informative=False, evidence_refs=["same-result-2"])
    assert policy.snapshot("line")["next_quota"] == 2
    state = policy.state()
    restored = FundingPolicy.from_state(json.loads(json.dumps(state)))
    assert restored.snapshot() == policy.snapshot()
    restored.create_lineage("new-name")
    with pytest.raises(ValueError, match="inherit"):
        restored.bind_version("v3", "new-name", parent_versions=["v2"])
    restored.bind_version("v3", "line", parent_versions=["v2"])
    assert restored.snapshot("line")["next_quota"] == 2


def test_informative_negative_evidence_retained_and_dormancy_requires_bounded_rescue():
    policy = FundingPolicy(initial_quota=1)
    policy.create_lineage("line")
    policy.record_block("line", "one", informative=False, evidence_refs=["known"])
    policy.record_block("line", "negative", informative=True, evidence_refs=["new-counterexample"])
    assert policy.snapshot("line")["no_information_streak"] == 0
    for block in ("two", "three"):
        policy.record_block("line", block, informative=False, evidence_refs=[block])
    assert policy.snapshot("line")["status"] == "dormant"
    with pytest.raises(ValueError, match="explicit new trial"):
        policy.record_block("line", "four", informative=True, evidence_refs=["four"])
    with pytest.raises(ValueError, match="bounded"):
        policy.authorize_trial("line", 2, reason="new question", evidence_refs=["question"])
    policy.authorize_trial("line", 1, reason="new counterexample", evidence_refs=["new-counterexample"])
    assert policy.snapshot("line")["status"] == "trial"
    assert any(e.get("evidence_refs") == ["new-counterexample"] for e in policy.state()["events"])
    assert FundingPolicy.from_state(policy.state()).state() == policy.state()


def test_funding_state_tamper_and_duplicate_block_conflicts_fail_closed():
    policy = FundingPolicy()
    policy.create_lineage("line")
    policy.record_block("line", "one", informative=False, evidence_refs=["receipt"])
    state = policy.state()
    policy.record_block("line", "one", informative=False, evidence_refs=["receipt"])
    assert policy.state() == state
    with pytest.raises(ValueError, match="different evidence"):
        policy.record_block("line", "one", informative=True, evidence_refs=["receipt"])
    tampered = deepcopy(state)
    tampered["initial_quota"] = 16
    with pytest.raises(ValueError, match="identity"):
        FundingPolicy.from_state(tampered)
    # Even a recomputed outer hash cannot make duplicate/noncanonical events valid.
    tampered = deepcopy(state)
    tampered["events"].append(deepcopy(tampered["events"][-1]))
    body = {k: v for k, v in tampered.items() if k != "sha256"}
    tampered["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    with pytest.raises(ValueError, match="noncanonical"):
        FundingPolicy.from_state(tampered)
