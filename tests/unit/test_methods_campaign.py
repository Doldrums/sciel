from copy import deepcopy

import pytest

from sciel.methods.budget import ResourceLedger
from sciel.methods.campaign import (arm_config, campaign_report, charge_upfront,
                                    freeze_campaign, load_campaign, validate_arm)
from sciel.methods.models import content_hash


def common(**changes):
    return dict(task=dict(id="fixture", identity="a" * 64, metric="objective", direction="max"),
                model=dict(model="offline-fixture", temperature=0, max_tokens=100),
                procedure="b" * 64, toolbox=["c" * 64], runtime_lock={"image_id": "sha256:" + "d" * 64},
                splits={"development": ["dev-1"], "heldout": ["test-1", "test-2"]}, lab_count=3, concurrency=2,
                budget=dict(dollars=10, cpu_seconds=100, gpu_seconds=0, wall_seconds=100, storage_bytes=10000, steps=10),
                seed_contract={"kind": "declared_fixture", "seeds": [1, 2]}, initial_knowledge=[],
                upfront_costs=[dict(id="e" * 64, resources={"dollars": 2, "cpu_seconds": 4}, evidence_refs=["f" * 64])],
                **changes)


def run_record(tmp_path, plan, arm="A", repeat=0, *, upfront=True, score=3, unknown=False):
    cfg = arm_config(plan, arm, repeat)
    ledger = ResourceLedger(tmp_path / f"{arm}-{repeat}.jsonl", plan["common"]["budget"])
    if upfront:
        charge_upfront(ledger, plan)
    ledger.create_account("lab-1", {})
    row = ledger.reserve("lab-1", f"{arm}-{repeat}/eval", {"dollars": 1, "steps": 1}, "evaluation_use")
    ledger.settle(row["id"], {"dollars": 1, "steps": 1}, status="completed")
    obs = dict(receipt_id=content_hash({"arm": arm, "repeat": repeat, "kind": "evaluation"}),
               reservation_id=row["id"], task_id="fixture", case_id="test-1", split="heldout", valid=True, score=score)
    if unknown:
        pending = ledger.reserve("lab-1", f"{arm}-{repeat}/unknown", {"dollars": 2}, "generation")
        ledger.settle(pending["id"], {}, status="unknown")
    return dict(arm=arm, repeat=repeat, config=cfg, ledger=ledger.snapshot(), observations=[obs]), ledger


def test_freeze_once_and_exact_common_configuration_for_every_arm(tmp_path):
    shared = common()
    path = tmp_path / "plan.json"
    plan = freeze_campaign(path, shared, repeats=2)
    shared["model"]["max_tokens"] = 200
    assert load_campaign(path) == plan
    assert freeze_campaign(path, common(), repeats=2) == plan
    with pytest.raises(ValueError, match="cannot change"):
        freeze_campaign(path, shared, repeats=2)
    configs = [arm_config(plan, arm, 0) for arm in ("A", "B", "C")]
    for cfg in configs:
        validate_arm(plan, cfg["arm"], 0, cfg)
        assert cfg["model"] == plan["common"]["model"]
        assert cfg["seed_contract"] == plan["common"]["seed_contract"]
        assert cfg["required_checks"] == dict(contract_checks=True, compatibility_checks=True, final_evaluator=True, recipient_trial=True)
    configs[0]["toolbox"].append("0" * 64)
    assert len(plan["common"]["toolbox"]) == 1


@pytest.mark.parametrize("field,replacement", [("model", {"model": "stronger"}), ("lab_count", 4),
                                               ("concurrency", 8), ("procedure", "1" * 64),
                                               ("required_checks", {"final_evaluator": False})])
def test_no_silent_arm_feature_budget_or_control_drift(tmp_path, field, replacement):
    plan = freeze_campaign(tmp_path / "plan", common(), repeats=1)
    cfg = arm_config(plan, "C", 0)
    cfg[field] = replacement
    with pytest.raises(ValueError, match="drifts"):
        validate_arm(plan, "C", 0, cfg)


def test_controls_keep_validation_and_unsupported_planner_is_explicitly_gated(tmp_path):
    plan = freeze_campaign(tmp_path / "plan", common(), repeats=1,
                           controls=["unreplayed", "solution_only", "shared_planner"])
    unreplayed = arm_config(plan, "unreplayed", 0)
    assert unreplayed["mode"] == "C" and unreplayed["library_policy"] == "unreplayed"
    assert all(unreplayed["required_checks"].values())
    solution = arm_config(plan, "solution_only", 0)
    assert solution["transfer_policy"] == "solution_only" and all(solution["required_checks"].values())
    with pytest.raises(NotImplementedError, match="pinned"):
        arm_config(plan, "shared_planner", 0)
    shared = common()
    shared["planner_implementations"] = {"cost_aware_shared_library": "1" * 64}
    implemented = freeze_campaign(tmp_path / "implemented", shared, repeats=1, controls=["shared_planner"])
    assert arm_config(implemented, "shared_planner", 0)["planner_implementation"] == "1" * 64


@pytest.mark.parametrize("mutation", ["overlap", "missing_resource", "treatment_override", "too_much_upfront"])
def test_invalid_experiment_design_rejected_before_writing(tmp_path, mutation):
    spec = common()
    if mutation == "overlap":
        spec["splits"]["heldout"].append("dev-1")
    elif mutation == "missing_resource":
        del spec["budget"]["steps"]
    elif mutation == "treatment_override":
        spec["mode"] = "C"
    else:
        spec["upfront_costs"][0]["resources"]["dollars"] = 11
    with pytest.raises(ValueError):
        freeze_campaign(tmp_path / "plan", spec, repeats=1)
    assert not (tmp_path / "plan").exists()


def test_upfront_creation_charged_once_system_wide_and_uses_each_charge(tmp_path):
    plan = freeze_campaign(tmp_path / "plan", common(), repeats=1)
    run, ledger = run_record(tmp_path, plan)
    charge_upfront(ledger, plan)
    charge_upfront(ledger, plan)
    ledger.create_account("lab-2", {})
    use = ledger.reserve("lab-2", "second-use", {"dollars": 1}, "method_use")
    ledger.settle(use["id"], {"dollars": ".5"})
    run["ledger"] = ledger.snapshot()
    # Reading a cached measurement cannot manufacture a second observation.
    run["observations"].append(deepcopy(run["observations"][0]))
    report = campaign_report(plan, [run])
    row = report["units"][0]
    assert row["costs"]["known_total"]["dollars"] == "3.5"
    assert row["costs"]["attributed_upfront"]["dollars"] == "2"
    assert row["costs"]["operating"]["dollars"] == "1.5"
    assert row["declared_accounting_complete"]
    assert row["cost_status"] == "known_lower_bound_or_unresolved"
    assert row["upfront_provenance"]["status"] == "unverified"
    assert row["quality"]["heldout"]["valid_observations"] == 1
    assert row["quality"]["heldout"]["best_by_case"] == {"test-1": 3}
    assert report["claims"] == "descriptive_only" and report["analysis_unit"] == "whole_campaign"
    assert report["expected_units"] == 3 and report["reported_units"] == 1


def test_unknown_or_missing_prepaid_cost_never_reported_as_complete(tmp_path):
    plan = freeze_campaign(tmp_path / "plan", common(), repeats=1)
    unknown, _ = run_record(tmp_path, plan, "C", unknown=True)
    missing, _ = run_record(tmp_path, plan, "B", upfront=False)
    reports = campaign_report(plan, [unknown, missing])["units"]
    assert all(row["cost_status"] == "known_lower_bound_or_unresolved" for row in reports)
    assert reports[0]["costs"]["reserved"]["dollars"] == "2"
    assert reports[1]["missing_upfront_costs"] == ["e" * 64]
    spec = common()
    del spec["upfront_costs"]
    undeclared = freeze_campaign(tmp_path / "undeclared", spec, repeats=1)
    run, _ = run_record(tmp_path, undeclared, "A", upfront=False)
    assert not campaign_report(undeclared, [run])["units"][0]["upfront_declared"]


def test_heldout_assignment_no_free_measurements_and_no_repeated_receipt_pseudoreplication(tmp_path):
    plan = freeze_campaign(tmp_path / "plan", common(), repeats=2)
    first, _ = run_record(tmp_path, plan, "A", 0)
    second, _ = run_record(tmp_path, plan, "A", 1)
    assert campaign_report(plan, [first, second])["reported_units"] == 2
    bad = deepcopy(first)
    bad["observations"][0]["case_id"] = "dev-1"
    with pytest.raises(ValueError, match="split"):
        campaign_report(plan, [bad])
    bad = deepcopy(first)
    bad["observations"][0]["reservation_id"] = "fake-unpaid"
    with pytest.raises(ValueError, match="no charged"):
        campaign_report(plan, [bad])
    bad = deepcopy(second)
    bad["ledger"] = deepcopy(first["ledger"])
    with pytest.raises(ValueError, match="reuse the same ledger"):
        campaign_report(plan, [first, bad])
    bad = deepcopy(second)
    bad["observations"][0]["receipt_id"] = first["observations"][0]["receipt_id"]
    with pytest.raises(ValueError, match="independent campaign repeat"):
        campaign_report(plan, [first, bad])
    with pytest.raises(ValueError, match="duplicate whole-campaign"):
        campaign_report(plan, [first, first])


def test_task_metric_direction_invalidity_and_accounting_tamper(tmp_path):
    spec = common()
    spec["task"]["direction"] = "min"
    plan = freeze_campaign(tmp_path / "plan", spec, repeats=1)
    run, ledger = run_record(tmp_path, plan, score=0)
    for suffix, score in [("second", -1), ("invalid", None)]:
        row = ledger.reserve("lab-1", suffix, {"steps": 1}, "evaluation_use")
        ledger.settle(row["id"], {"steps": 1}, status="completed" if score is not None else "failed")
        run["observations"].append(dict(receipt_id=content_hash(suffix), reservation_id=row["id"],
            task_id="fixture", case_id="test-1", split="heldout", valid=score is not None, score=score))
    run["ledger"] = ledger.snapshot()
    quality = campaign_report(plan, [run])["units"][0]["quality"]["heldout"]
    assert quality == dict(valid_observations=2, invalid_observations=1, best_by_case={"test-1": -1})
    run["ledger"]["spent"]["dollars"] = "0"
    with pytest.raises(ValueError, match="totals disagree"):
        campaign_report(plan, [run])


def test_empty_upfront_list_with_toolbox_is_not_complete_historical_accounting(tmp_path):
    spec = common()
    assert spec['toolbox']
    spec['upfront_costs'] = []
    plan = freeze_campaign(tmp_path/'plan', spec, repeats=1)
    run, _ = run_record(tmp_path, plan)
    report = campaign_report(plan, [run])['units'][0]
    assert report['declared_accounting_complete']
    assert report['cost_status'] == 'known_lower_bound_or_unresolved'
    assert report['upfront_provenance']['status'] == 'unverified'
    assert not report['upfront_provenance']['independently_verified']
    assert report['costs']['attributed_upfront']['dollars'] == '0'


def test_explicit_historical_attestation_is_labelled_not_independently_certified(tmp_path):
    spec = common()
    spec['upfront_provenance'] = dict(status='attested', evidence_refs=['3'*64],
                                    statement='Caller declares coverage of this exact frozen toolbox preparation scope.')
    plan = freeze_campaign(tmp_path/'plan', spec, repeats=1)
    run, _ = run_record(tmp_path, plan)
    report = campaign_report(plan, [run])['units'][0]
    assert report['declared_accounting_complete']
    assert report['cost_status'] == 'declared_costs_settled_attested_scope'
    assert report['upfront_provenance']['status'] == 'attested'
    assert not report['upfront_provenance']['independently_verified']
