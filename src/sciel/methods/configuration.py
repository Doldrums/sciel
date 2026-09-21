"""Validation of standalone research-run configuration."""
from __future__ import annotations

import math
from copy import deepcopy
from decimal import Decimal, localcontext

from .budget import RESOURCES, _limit_vector, _vector
from .campaign import upfront_provenance
from .executor import _limits
from .models import canonical_bytes, require_hash, validate_runtime_lock


def _no_credentials(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"api_key", "authorization", "password", "secret", "access_token", "credentials"}:
                raise ValueError("credentials must come from OPENROUTER_API_KEY, never configuration")
            _no_credentials(item)
    elif isinstance(value, list):
        for item in value:
            _no_credentials(item)


def validate_config(config):
    from ..campaign.budget import validate_llm_config
    canonical_bytes(config)
    _no_credentials(config)
    required = {"schema", "mode", "task", "image_id", "llm", "pricing", "model", "budget", "resources",
                "max_steps", "max_cases", "calendar_seconds", "storage_bytes", "labs"}
    optional = {"common_methods", "common_toolbox", "library_policy", "transfer_policy", "routing", "claim", "upfront_costs", "upfront_provenance", "capacity_wait_seconds", "checkpoint_interval"}
    if type(config) is not dict or not required <= config.keys() or set(config) - required - optional:
        raise ValueError("run config requires the documented bounded fields; unknown fields are rejected")
    if config["schema"] != "sciel-method-run/1" or config["mode"] not in ("A", "B", "C"):
        raise ValueError("unsupported run schema or mode")
    if config["task"] not in ("sort", "circle26"):
        raise ValueError("task must be sort or circle26")
    validate_runtime_lock({"image_id": config["image_id"]})
    if type(config["budget"]) is not dict or set(config["budget"]) != set(RESOURCES):
        raise ValueError("all six resource budget counters must be explicitly specified")
    out = deepcopy(config)
    continuous = config.get("claim") == "continuous_run"
    normalize_limits = _limit_vector if continuous else _vector
    out["budget"] = normalize_limits(config["budget"])
    if out["budget"]["dollars"] is not None and float(out["budget"]["dollars"]) <= 0:
        raise ValueError("real provider mode needs a positive dollar ceiling or explicit continuous unlimited mode")
    _limits(config["resources"], config["resources"])
    for key in ("max_steps", "max_cases", "storage_bytes"):
        if continuous and key != "max_cases" and config[key] is None:
            continue
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not (continuous and config["calendar_seconds"] is None) and (type(config["calendar_seconds"]) not in (int, float) or not math.isfinite(config["calendar_seconds"]) or not 0 < config["calendar_seconds"] <= 86400):
        raise ValueError("calendar_seconds must be positive and at most one day")
    if "capacity_wait_seconds" in config:
        wait = config["capacity_wait_seconds"]
        if not (continuous and wait is None) and (type(wait) not in (int, float) or not math.isfinite(wait) or not 0 < wait <= 300):
            raise ValueError("capacity_wait_seconds must be finite and positive or null in continuous mode")
    if "checkpoint_interval" in config and (type(config["checkpoint_interval"]) is not int or config["checkpoint_interval"] < 1):
        raise ValueError("checkpoint_interval must be a positive integer")
    validate_llm_config(config["llm"], config["pricing"])
    if config["model"] not in config["llm"]["models"]:
        raise ValueError("selected model must have frozen provider pricing")
    if config["llm"].get("max_retries", 1) != 1:
        raise ValueError("method runner permits one provider dispatch per reserved intent")
    out.setdefault("library_policy", "verified")
    out.setdefault("transfer_policy", "methods")
    out.setdefault("common_toolbox", "none")
    out.setdefault("common_methods", [])
    out.setdefault("claim", "local_episode")
    if out["library_policy"] not in ("verified", "unreplayed") or out["transfer_policy"] not in ("methods", "solution_only"):
        raise ValueError("unsupported library/transfer control")
    if out["common_toolbox"] not in ("none", "sort", "circle26") or (out["common_toolbox"] != "none" and out["common_toolbox"] != out["task"]):
        raise ValueError("common toolbox must match the task")
    if type(out["common_methods"]) is not list:
        raise ValueError("common_methods must contain immutable bundle objects")
    if type(out["labs"]) is not list or not out["labs"]:
        raise ValueError("at least one explicitly funded lab is required")
    names, accounts = set(), set()
    for lab in out["labs"]:
        if type(lab) is not dict or set(lab) != {"id", "agenda", "account_id", "limits"}:
            raise ValueError("labs require id, agenda, account_id and explicit resource limits")
        if any(type(lab[k]) is not str or not lab[k].strip() for k in ("id", "agenda", "account_id")):
            raise ValueError("lab identity and agenda must be nonempty")
        if lab["id"] in names or lab["account_id"] in accounts or lab["account_id"] == "campaign.setup":
            raise ValueError("lab/account IDs must be distinct and cannot reuse the setup account")
        names.add(lab["id"]); accounts.add(lab["account_id"])
        if type(lab["limits"]) is not dict or set(lab["limits"]) != set(RESOURCES):
            raise ValueError("each lab must have explicit caps for all six resources")
        lab["limits"] = normalize_limits(lab["limits"])
        if any(out["budget"][key] is not None and (lab["limits"][key] is None or Decimal(str(lab["limits"][key])) > Decimal(str(out["budget"][key]))) for key in RESOURCES):
            raise ValueError("lab cap exceeds the shared campaign ceiling")
    routing = out.get("routing")
    if routing is not None:
        keys = {"author", "reviewer", "recipient", "control", "transfer_inputs", "max_transfers"}
        if type(routing) is not dict or set(routing) != keys:
            raise ValueError("routing requires fixed author/reviewer/recipient/control/transfer_inputs/max_transfers")
        if any(routing[key] not in names for key in ("author", "reviewer", "recipient")) or len({routing[k] for k in ("author", "reviewer", "recipient")}) != 3:
            raise ValueError("routing needs three distinct explicitly funded labs")
        if type(routing["control"]) is not list or not routing["control"]:
            raise ValueError("routing control requires exact common method IDs")
        for ident in routing["control"]:
            require_hash(ident)
        if type(routing["transfer_inputs"]) is not list or not 1 <= len(routing["transfer_inputs"]) <= out["max_cases"]:
            raise ValueError("routing requires a bounded preassigned recipient input set")
        limit = routing["max_transfers"]
        if not (continuous and limit is None) and (type(limit) is not int or limit < 1 or (out["max_steps"] is not None and limit > out["max_steps"])):
            raise ValueError("routing max_transfers must be within the step cap")
    if out["claim"] not in ("local_episode", "bounded_arm", "continuous_run"):
        raise ValueError("CLI can report local_episode, bounded_arm or continuous_run, not comparative scientific claims")
    if out["claim"] in ("bounded_arm", "continuous_run") and out["mode"] in ("B", "C") and routing is None:
        raise ValueError("B/C bounded_arm requires explicit replay/adoption/transfer routing")
    if out["claim"] in ("bounded_arm", "continuous_run") and "upfront_costs" not in out:
        raise ValueError("bounded_arm requires an explicit upfront_costs declaration, including an empty list")
    fees = out.get("upfront_costs", [])
    upfront_provenance(out)
    if type(fees) is not list:
        raise ValueError("upfront_costs must be an explicit list of historical allocations")
    fee_ids = set()
    for fee in fees:
        if type(fee) is not dict or set(fee) != {"id", "resources", "evidence_refs", "description"}:
            raise ValueError("upfront allocations require id, resources, evidence_refs and description")
        require_hash(fee["id"])
        if fee["id"] in fee_ids:
            raise ValueError("duplicate upfront allocation ID")
        fee_ids.add(fee["id"])
        if type(fee["resources"]) is not dict or set(fee["resources"]) != set(RESOURCES):
            raise ValueError("upfront allocations require all six resource counters")
        fee["resources"] = _vector(fee["resources"])
        if (type(fee["description"]) is not str or not fee["description"].strip()
                or type(fee["evidence_refs"]) is not list or not fee["evidence_refs"]
                or any(type(ref) is not str or not ref.strip() for ref in fee["evidence_refs"])):
            raise ValueError("upfront allocations require explicit evidence references and a description")
    with localcontext() as ctx:
        ctx.prec = 160
        if any(out["budget"][key] is not None and sum((Decimal(str(fee["resources"][key])) for fee in fees), Decimal(0)) > Decimal(str(out["budget"][key]))
               for key in RESOURCES):
            raise ValueError("declared upfront costs exceed the campaign ceiling")
    return out
