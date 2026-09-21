"""Frozen comparisons and descriptive accounting; no experiment is run here.

One (arm, repeat) is an entire independent campaign, not an individual child.
Reports consume host-verified ledger snapshots and evaluation records; a hash
checks binding, not the truth of arbitrary caller-supplied scientific claims.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, localcontext
import json
import math
import os
from pathlib import Path
import stat
import tempfile

from .budget import RESOURCES, _vector, _wire
from .models import canonical_bytes, content_hash, require_hash, validate_runtime_lock

_SCHEMA = "sciel-method-campaign/1"
_ARMS = {
    "A": dict(mode="A", library_policy="verified", transfer_policy="methods", planner="lab"),
    "B": dict(mode="B", library_policy="verified", transfer_policy="methods", planner="lab"),
    "C": dict(mode="C", library_policy="verified", transfer_policy="methods", planner="lab"),
    "unreplayed": dict(mode="C", library_policy="unreplayed", transfer_policy="methods", planner="lab"),
    "solution_only": dict(mode="C", library_policy="verified", transfer_policy="solution_only", planner="lab"),
    "shared_planner": dict(mode="C", library_policy="verified", transfer_policy="methods", planner="cost_aware_shared_library"),
}
_GUARDS = {"contract_checks": True, "compatibility_checks": True,
           "final_evaluator": True, "recipient_trial": True}
_RESERVED = set(_ARMS["A"]) | {"campaign_id", "arm", "repeat", "analysis_unit_id", "required_checks", "planner_implementation"}


def _name(value, label):
    if type(value) is not str or not value.strip() or len(value) > 256:
        raise ValueError(f"invalid {label}")
    return value


def upfront_provenance(config):
    """A declaration/receipt reference cannot certify all historical costs."""
    value = config.get("upfront_provenance")
    if value is None:
        return dict(status="unverified", evidence_refs=[], statement=None,
                    uncertainty="Historical toolbox construction coverage is not attested; declared charges are a lower bound.",
                    independently_verified=False)
    if (type(value) is not dict or set(value) != {"status", "evidence_refs", "statement"}
            or value["status"] not in ("attested", "unverified")
            or type(value["evidence_refs"]) is not list or type(value["statement"]) is not str
            or not value["statement"].strip()):
        raise ValueError("upfront provenance requires status, evidence_refs and a coverage statement")
    for ref in value["evidence_refs"]:
        require_hash(ref, "upfront coverage evidence")
    if value["status"] == "attested" and not value["evidence_refs"]:
        raise ValueError("attested historical coverage requires evidence references")
    return dict(deepcopy(value), independently_verified=False,
                uncertainty="Caller-attested scope; referenced historical costs are not independently verified by this reporter."
                if value["status"] == "attested" else "Historical cost coverage remains unverified; charges are a lower bound.")


def _common(value):
    canonical_bytes(value)
    required = {"task", "model", "procedure", "toolbox", "runtime_lock", "splits", "lab_count", "concurrency", "budget"}
    if type(value) is not dict or not required <= value.keys() or _RESERVED.intersection(value):
        raise ValueError("common configuration is missing fields or overrides treatment fields")
    common = deepcopy(value)
    task = common["task"]
    if type(task) is not dict or set(task) != {"id", "identity", "metric", "direction"}:
        raise ValueError("task requires id, identity, metric and direction")
    _name(task["id"], "task ID")
    _name(task["metric"], "task metric")
    require_hash(task["identity"], "task/evaluator identity")
    if task["direction"] not in ("max", "min"):
        raise ValueError("task direction must be max or min")
    if type(common["model"]) is not dict or not common["model"]:
        raise ValueError("model must contain the complete frozen generator configuration")
    require_hash(common["procedure"], "search procedure identity")
    validate_runtime_lock(common["runtime_lock"])
    for key in ("lab_count", "concurrency"):
        if type(common[key]) is not int or common[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if type(common["toolbox"]) is not list:
        raise ValueError("toolbox must list exact method version hashes")
    if len(set(require_hash(i, "toolbox version") for i in common["toolbox"])) != len(common["toolbox"]):
        raise ValueError("duplicate toolbox version")
    splits = common["splits"]
    if type(splits) is not dict or set(splits) != {"development", "heldout"}:
        raise ValueError("freeze development and heldout assignments explicitly")
    seen = set()
    for label, items in splits.items():
        if type(items) is not list or not items:
            raise ValueError(f"{label} must contain assigned case IDs")
        for case_id in items:
            _name(case_id, "case ID")
            if case_id in seen:
                raise ValueError("case assignments overlap or repeat")
            seen.add(case_id)
    if type(common["budget"]) is not dict or set(common["budget"]) != set(RESOURCES):
        raise ValueError("the full resource budget must explicitly specify all six counters")
    common["budget"] = _vector(common["budget"])
    if not any(Decimal(str(v)) > 0 for v in common["budget"].values()):
        raise ValueError("campaign budget cannot be entirely zero")
    # Empty means no charges declared, not evidence that preparation was free.
    upfront_provenance(common)
    if "upfront_costs" in common:
        if type(common["upfront_costs"]) is not list:
            raise ValueError("upfront_costs must be an explicit list")
        ids = set()
        for item in common["upfront_costs"]:
            if type(item) is not dict or set(item) != {"id", "resources", "evidence_refs"}:
                raise ValueError("upfront cost requires id, resources and evidence_refs")
            require_hash(item["id"], "upfront cost identity")
            if item["id"] in ids:
                raise ValueError("duplicate upfront cost identity")
            ids.add(item["id"])
            item["resources"] = _vector(item["resources"])
            if type(item["evidence_refs"]) is not list or not item["evidence_refs"]:
                raise ValueError("upfront cost requires source receipt references")
            for ref in item["evidence_refs"]:
                require_hash(ref, "upfront source receipt")
        with localcontext() as ctx:
            ctx.prec = 160
            for key in RESOURCES:
                total = sum((Decimal(str(item["resources"][key])) for item in common["upfront_costs"]), Decimal(0))
                if total > Decimal(str(common["budget"][key])):
                    raise ValueError("upfront creation costs exceed the full campaign budget")
    return common


def _plan(common, repeats, controls):
    if type(repeats) is not int or not 1 <= repeats <= 10000:
        raise ValueError("repeats must be a bounded positive count of whole campaigns")
    if type(controls) not in (list, tuple) or len(set(controls)) != len(controls):
        raise ValueError("controls must be a unique list")
    if any(name not in ("unreplayed", "solution_only", "shared_planner") for name in controls):
        raise ValueError("unknown campaign control")
    body = dict(schema=_SCHEMA, common=_common(common), repeats=repeats,
                arms={name: deepcopy(_ARMS[name]) for name in ["A", "B", "C", *sorted(controls)]},
                required_checks=deepcopy(_GUARDS), analysis_unit="whole_campaign")
    return dict(body, id=content_hash(body))


def _validate(plan):
    if type(plan) is not dict or set(plan) != {"schema", "common", "repeats", "arms", "required_checks", "analysis_unit", "id"}:
        raise ValueError("invalid frozen campaign shape")
    expected = _plan(plan["common"], plan["repeats"], [name for name in plan["arms"] if name not in ("A", "B", "C")])
    if plan != expected:
        raise ValueError("campaign identity or fixed arm policy mismatch")
    return expected


def load_campaign(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_size > 8 * 1024 * 1024:
            raise ValueError("campaign manifest must be a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            plan = json.load(stream)
    finally:
        os.close(fd)
    return _validate(plan)


def freeze_campaign(path, common, *, repeats, controls=()):
    """Create once; an identical retry succeeds, changing a frozen plan fails."""
    plan = _plan(common, repeats, controls)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".campaign-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_bytes(plan) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if load_campaign(path) != plan:
                raise ValueError("frozen campaign cannot change")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return plan


def arm_config(plan, arm, repeat):
    plan = _validate(plan)
    if arm not in plan["arms"] or type(repeat) is not int or not 0 <= repeat < plan["repeats"]:
        raise ValueError("arm/repeat is not in the frozen campaign")
    result = dict(deepcopy(plan["common"]), **plan["arms"][arm], campaign_id=plan["id"],
                  arm=arm, repeat=repeat, analysis_unit_id=f"{plan['id']}/{arm}/{repeat}",
                  required_checks=deepcopy(plan["required_checks"]))
    if arm == "shared_planner":
        # Naming a control does not implement it. Execution requires a concrete
        # frozen planner identity; there is deliberately no fallback to lab mode.
        identity = plan["common"].get("planner_implementations", {}).get("cost_aware_shared_library")
        if identity is None:
            raise NotImplementedError("shared_planner needs a pinned cost_aware_shared_library implementation")
        result["planner_implementation"] = require_hash(identity, "planner implementation")
    return result


def validate_arm(plan, arm, repeat, config):
    if canonical_bytes(config) != canonical_bytes(arm_config(plan, arm, repeat)):
        raise ValueError("arm configuration drifts from the frozen common settings or treatment")


def charge_upfront(ledger, plan):
    """Allocate known prepaid creation costs once per system ledger, never per lab.

    These are attributed historical costs with source receipts, not new provider
    calls. Reports keep them separate from operating charges. Unknown receipts
    are not eligible for a fabricated known upfront allocation.
    """
    plan = _validate(plan)
    common = plan["common"]
    if "upfront_costs" not in common:
        raise ValueError("upfront costs must be explicitly declared before charging")
    if ledger.snapshot()["limits"] != common["budget"]:
        raise ValueError("ledger ceiling differs from the frozen full budget")
    ledger.create_account("campaign.setup", {})
    receipts = []
    for item in common["upfront_costs"]:
        row = ledger.reserve("campaign.setup", "upfront:" + item["id"], item["resources"], "upfront_creation")
        if row["dispatch_allowed"]:
            row = ledger.settle(row["id"], item["resources"])
        receipts.append(row)
    return receipts


def _costs(snapshot, limits):
    if type(snapshot) is not dict or snapshot.get("schema") != "sciel-resource-ledger/1" or snapshot.get("limits") != limits:
        raise ValueError("report requires the campaign-wide ledger with the frozen limits")
    if snapshot.get("account_id") is not None:
        raise ValueError("an account subset is not full campaign accounting")
    require_hash(snapshot["journal"]["head_sha256"], "ledger anchor")
    require_hash(snapshot["journal"]["genesis_sha256"], "ledger identity")
    spent, held = {k: Decimal(0) for k in RESOURCES}, {k: Decimal(0) for k in RESOURCES}
    rows, requests = {}, set()
    with localcontext() as ctx:
        ctx.prec = 160
        for row in snapshot["reservations"]:
            if row["id"] in rows or row["request_id"] in requests:
                raise ValueError("duplicate accounting reservation")
            if row["status"] not in ("reserved", "unknown", "completed", "failed"):
                raise ValueError("invalid accounting state")
            rows[row["id"]] = row
            requests.add(row["request_id"])
            actual, bounds = _vector(row["actual"]), _vector(row["bounds"])
            if row["status"] in ("completed", "failed") and any(
                    Decimal(str(bounds[k])) > 0 and k not in row["actual"] for k in RESOURCES):
                raise ValueError("terminal accounting omits measured resource usage")
            for key in RESOURCES:
                value = Decimal(str(actual[key]))
                spent[key] += value
                if row["status"] in ("reserved", "unknown"):
                    held[key] += max(Decimal(0), Decimal(str(bounds[key])) - value)
    known = {k: _wire(v, k) for k, v in spent.items()}
    reserved = {k: _wire(v, k) for k, v in held.items()}
    if snapshot["spent"] != known or snapshot["reserved"] != reserved:
        raise ValueError("ledger totals disagree with its reservation records")
    return rows, known, reserved


def _campaign_report(plan, runs):
    """Describe caller-verified observations/costs without claiming causal benefit.

    Each run: {arm,repeat,config,ledger,observations}. Each observation:
    {receipt_id,reservation_id,task_id,case_id,split,valid,score}. Receipt IDs are
    immutable external-evaluator artifact hashes. Re-reading an observation does
    not add evidence; conflicting records for one invocation/case are rejected.
    """
    plan = _validate(plan)
    if type(runs) is not list:
        raise ValueError("runs must be a list of whole-campaign records")
    reports, seen, ledgers, receipt_units = [], set(), set(), {}
    for run in runs:
        arm, repeat = run["arm"], run["repeat"]
        validate_arm(plan, arm, repeat, run["config"])
        unit = (arm, repeat)
        if unit in seen:
            raise ValueError("duplicate whole-campaign analysis unit")
        seen.add(unit)
        rows, known, reserved = _costs(run["ledger"], plan["common"]["budget"])
        ledger_id = run["ledger"]["journal"]["genesis_sha256"]
        if ledger_id in ledgers:
            raise ValueError("independent campaign units cannot reuse the same ledger")
        ledgers.add(ledger_id)
        upfront = {k: Decimal(0) for k in RESOURCES}
        missing = []
        for item in plan["common"].get("upfront_costs", []):
            match = [r for r in rows.values() if r["request_id"] == "upfront:" + item["id"]]
            if (not match or match[0]["status"] != "completed" or match[0]["category"] != "upfront_creation"
                    or match[0]["actual"] != item["resources"]):
                missing.append(item["id"])
            else:
                for k in RESOURCES:
                    upfront[k] += Decimal(str(match[0]["actual"][k]))
        quality = {split: {"valid_observations": 0, "invalid_observations": 0, "best_by_case": {}}
                   for split in ("development", "heldout")}
        observations = {}
        for obs in run["observations"]:
            if type(obs) is not dict or set(obs) != {"receipt_id", "reservation_id", "task_id", "case_id", "split", "valid", "score"}:
                raise ValueError("invalid evaluation observation fields")
            require_hash(obs["receipt_id"], "evaluation receipt")
            if obs["receipt_id"] in receipt_units and receipt_units[obs["receipt_id"]] != unit:
                raise ValueError("an evaluation receipt cannot count as an independent campaign repeat")
            receipt_units[obs["receipt_id"]] = unit
            if (obs["task_id"] != plan["common"]["task"]["id"] or obs["split"] not in quality
                    or obs["case_id"] not in plan["common"]["splits"][obs["split"]]):
                raise ValueError("observation is outside frozen task/split assignments")
            key = (obs["reservation_id"], obs["case_id"])
            if key in observations:
                if observations[key] != obs:
                    raise ValueError("conflicting observation for one invocation/case")
                continue
            if obs["reservation_id"] not in rows:
                raise ValueError("observation has no charged or reserved invocation")
            if type(obs["valid"]) is not bool:
                raise ValueError("validity must be explicitly boolean")
            if obs["valid"] and (rows[obs["reservation_id"]]["status"] != "completed"
                    or type(obs["score"]) not in (int, float) or not math.isfinite(obs["score"])):
                raise ValueError("valid score requires a completed invocation and finite measurement")
            if not obs["valid"] and obs["score"] is not None:
                raise ValueError("invalid measurements must not supply a usable score")
            observations[key] = obs
            out = quality[obs["split"]]
            out["valid_observations" if obs["valid"] else "invalid_observations"] += 1
            if obs["valid"]:
                best = out["best_by_case"].get(obs["case_id"])
                choose = max if plan["common"]["task"]["direction"] == "max" else min
                out["best_by_case"][obs["case_id"]] = obs["score"] if best is None else choose(best, obs["score"])
        unresolved = [r["id"] for r in rows.values() if r["status"] in ("reserved", "unknown")]
        within = all(Decimal(str(known[k])) + Decimal(str(reserved[k])) <= Decimal(str(plan["common"]["budget"][k])) for k in RESOURCES)
        complete = "upfront_costs" in plan["common"] and not missing and not unresolved and not run["ledger"]["halted"] and within
        provenance = upfront_provenance(plan["common"])
        reports.append(dict(arm=arm, repeat=repeat, analysis_unit_id=run["config"]["analysis_unit_id"],
            config_sha256=content_hash(run["config"]), ledger_anchor=deepcopy(run["ledger"]["journal"]),
            declared_accounting_complete=complete, upfront_provenance=provenance,
            cost_status="declared_costs_settled_attested_scope" if complete and provenance["status"] == "attested"
                        else "known_lower_bound_or_unresolved",
            costs=dict(known_total=known, reserved=reserved,
                       attributed_upfront={k: _wire(v, k) for k, v in upfront.items()},
                       operating={k: _wire(Decimal(str(known[k])) - upfront[k], k) for k in RESOURCES}),
            missing_upfront_costs=missing, upfront_declared="upfront_costs" in plan["common"],
            unresolved_reservations=unresolved, within_declared_budget=within, quality=quality))
    return dict(schema="sciel-method-campaign-report/1", campaign_id=plan["id"], analysis_unit="whole_campaign",
                expected_units=len(plan["arms"]) * plan["repeats"], reported_units=len(reports), units=reports,
                claims="descriptive_only", notes=[
                    "Upfront figures are attributed historical creation costs, not new provider calls.",
                    "Settled declared charges do not establish that every historical construction expense was captured; empty upfront lists do not prove free preparation.",
                    "Each independent repeat must start from the frozen initial library and knowledge; no cross-repeat warm starts.",
                    "Case IDs freeze assignments; they do not prove task-family independence or absence of semantic leakage.",
                    "Reporter assumes host-verified receipts; hashes do not authenticate arbitrary supplied measurements.",
                    "No quality average combines different cases; no significance or recursive-improvement claim is inferred."])


def campaign_report(plan, runs):
    """Report frozen whole-campaign units; all resource sums retain exact decimals.

    See `_campaign_report` for the run/observation input contract. The caller
    supplies host-verified receipts, never model-authored claims of validity.
    """
    with localcontext() as ctx:
        ctx.prec = 160
        return _campaign_report(plan, runs)
