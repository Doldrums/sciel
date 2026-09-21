"""A scientific protocol fix may resume only completely settled local pauses."""
import json

import pytest

from sciel.campaign.budget import CampaignStopped, CostLedger

REASON = "local research pause: invalid measurement"


def paused(tmp_path, cost="0.2", status="settled"):
    ledger = CostLedger(tmp_path / "ledger.json", 1)
    ident = ledger.reserve("0.3", run_id="old", model="mock")
    if status == "settled":
        ledger.settle(ident, cost)
    elif status == "unknown":
        ledger.settle(ident, None)
    with ledger._locked() as state:
        state["halted"] = REASON
    return ledger, ident


def test_resume_keeps_all_prior_costs_and_remaining_cap(tmp_path):
    ledger, ident = paused(tmp_path)
    previous = json.loads(ledger.path.read_text())["calls"]
    ledger.resume_local_pause(REASON, resolution="Frozen corrected protocol abc passed preflight")
    state = json.loads(ledger.path.read_text())
    assert state["calls"] == previous
    assert state["halted"] is None
    assert state["local_pause_resolutions"][0]["known_cost_usd"] == "0.2"
    assert ledger.snapshot()["remaining_usd"] == "0.8"
    with pytest.raises(CampaignStopped):
        ledger.reserve("0.81", run_id="new", model="mock")
    new = ledger.reserve("0.8", run_id="new", model="mock")
    assert new != ident


@pytest.mark.parametrize("status", ["reserved", "unknown"])
def test_unsettled_calls_prevent_resume(tmp_path, status):
    ledger, _ = paused(tmp_path, status=status)
    before = ledger.path.read_bytes()
    with pytest.raises(CampaignStopped, match="unsettled"):
        ledger.resume_local_pause(REASON, resolution="fix")
    assert ledger.path.read_bytes() == before


def test_provider_halt_cannot_be_mistaken_for_local_pause(tmp_path):
    ledger, _ = paused(tmp_path)
    with ledger._locked() as state:
        state["halted"] = "provider request outcome is unknown"
    before = ledger.path.read_bytes()
    with pytest.raises(CampaignStopped, match="changed"):
        ledger.resume_local_pause(REASON, resolution="fix")
    assert ledger.path.read_bytes() == before
    with pytest.raises(ValueError):
        ledger.resume_local_pause("provider request outcome is unknown", resolution="fix")


def test_overrun_cannot_be_cleared_by_relabelling_halt(tmp_path):
    ledger, ident = paused(tmp_path)
    with ledger._locked() as state:
        state["calls"][ident]["cost_usd"] = "0.4"
    before = ledger.path.read_bytes()
    with pytest.raises(CampaignStopped, match="overrun"):
        ledger.resume_local_pause(REASON, resolution="fix")
    assert ledger.path.read_bytes() == before


@pytest.mark.parametrize("resolution", ["", None])
def test_resolution_must_be_recorded(tmp_path, resolution):
    ledger, _ = paused(tmp_path)
    with pytest.raises(ValueError):
        ledger.resume_local_pause(REASON, resolution=resolution)
    assert ledger.snapshot()["halted"] == REASON
