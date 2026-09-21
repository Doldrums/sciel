"""Validate research observations and derive evidence-bound lesson statistics."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from ..hypothesis.measurement import validate_metric_context, validate_cases, validate_evaluation
from ..hypothesis.controlled_changes import validate_controlled_change
from ..hypothesis.circle_proposal_group import validate_group
from .paired_comparisons import SCHEMA as COMPARISON_SCHEMA
from .eligibility import efficacy_eligible, output_repeat

SCHEMA = "sciel-research-knowledge/1"


INITIALIZER_REGISTRY_SCHEMA = "sciel-initializer-registry/1"


INITIALIZER_BINDING_SCHEMA = "sciel-initializer-observation-binding/1"


_FIELDS = {"id", "branch", "parent_id", "parent_score", "child_id", "source_sha256",
           "proposed_sha256", "hypothesis", "status", "actual_delta", "score",
           "forecast_verdict", "injected_ids", "code_fragment", "source_repaired", "error",
           "island", "iteration", "parent_source_sha256", "attempt", "operator",
           "generation_cost", "failure", "change_excerpt", "change_excerpt_may_be_truncated",
           "metric", "parent_cases", "evaluation", "generation_dispatch", "controlled_change", "proposal_group",
           "restart_prediction_binding", "method_exposures", "generation_response_sha256", "output_novelty",
           "generation_cost_accounting", "generation_cost_provenance", "generation_model_identity"}


_MEASUREMENT_FIELDS = {"metric", "parent_cases", "evaluation"}


_DISPATCH_FIELDS = {"generation_dispatch"}


_CONTROLLED_FIELDS = {"controlled_change"}


_RESTART_PREDICTION_FIELDS = {"restart_prediction_binding"}


_METHOD_FIELDS = {"method_exposures", "generation_response_sha256"}


_COST_FIELDS = {"generation_cost_accounting", "generation_cost_provenance"}


_MODEL_FIELDS = {"generation_model_identity"}


_SNAPSHOT_FIELDS = frozenset((_FIELDS - {"code_fragment"}) | {"code_fragment_sha256"})


_SNAPSHOT_REQUIRED_FIELDS = _SNAPSHOT_FIELDS - (
    _MEASUREMENT_FIELDS | _DISPATCH_FIELDS | _CONTROLLED_FIELDS
    | _RESTART_PREDICTION_FIELDS | _METHOD_FIELDS | _COST_FIELDS | _MODEL_FIELDS
    | {"proposal_group", "output_novelty"})


_LESSON_FIELDS = {"id", "rule", "polarity", "scope", "branch", "evidence_ids",
                  "evidence_snapshots", "exposure_ids", "exposure_snapshots",
                  "inherited_exposures", "source_support_ids", "support_ids", "refute_ids", "confidence"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value, name, *, limit=4096):
    _require(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
             f"invalid {name}")
    return value


def _number(value, name, *, nullable=False):
    if value is None and nullable:
        return None
    _require(type(value) in (int, float), f"invalid finite {name}")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"invalid finite {name}") from exc
    _require(math.isfinite(value), f"invalid finite {name}")
    return value


def _hash(value, name, *, nullable=True):
    _require((nullable and value is None) or
             (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)), f"invalid {name}")
    return value


def _ids(values, name):
    _require(isinstance(values, list), f"{name} must be a list")
    for value in values:
        _text(value, name, limit=128)
    _require(len(values) == len(set(values)), f"duplicate {name}")
    return list(values)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _is_comparison(record):
    return isinstance(record, dict) and record.get("schema") == COMPARISON_SCHEMA


def _supports(record, polarity):
    if _is_comparison(record):
        return record["derived"]["outcome"] == ("improved" if polarity == "positive" else "regressed")
    sign = 1 if polarity == "positive" else -1
    return efficacy_eligible(record) and sign * record["actual_delta"] > 0


def _valid_snapshot_fields(keys):
    # Legacy snapshots may omit each complete measurement/accounting group.
    # Method exposure alone is historical; a response digest requires exposure.
    return (_SNAPSHOT_REQUIRED_FIELDS <= keys <= _SNAPSHOT_FIELDS
            and (not keys & _MEASUREMENT_FIELDS or _MEASUREMENT_FIELDS <= keys)
            and (not keys & _COST_FIELDS or _COST_FIELDS <= keys)
            and ("generation_response_sha256" not in keys or "method_exposures" in keys))


def _snapshot(record):
    if _is_comparison(record):
        return copy.deepcopy(record)
    result = {k: copy.deepcopy(v) for k, v in record.items() if k != "code_fragment"}
    result["code_fragment_sha256"] = (hashlib.sha256(record["code_fragment"].encode()).hexdigest()
                                        if record["code_fragment"] else None)
    return result


def _experiment(raw):
    _require(isinstance(raw, dict) and not set(raw) - _FIELDS, "unknown experiment fields")
    r = copy.deepcopy(raw)
    for key in ("id", "branch", "status"):
        _text(r.get(key), key, limit=128)
    _require(re.fullmatch(r"[a-z][a-z0-9_]*", r["status"]), "invalid experiment status")
    for key in ("parent_id", "child_id"):
        r.setdefault(key, None)
        if r[key] is not None:
            _text(r[key], key, limit=128)
    for key in ("parent_score", "score", "actual_delta"):
        r[key] = _number(r.get(key), key, nullable=True)
    for key in ("source_sha256", "proposed_sha256"):
        r[key] = _hash(r.get(key), key)
    r["parent_source_sha256"] = _hash(r.get("parent_source_sha256"), "parent_source_sha256")
    for key in ("island", "iteration", "attempt"):
        r.setdefault(key, None)
        _require(r[key] is None or (type(r[key]) is int and r[key] >= (1 if key == "attempt" else 0)),
                 f"invalid {key}")
    for key in ("operator", "failure"):
        r.setdefault(key, None)
        if r[key] is not None:
            _text(r[key], key)
    r["generation_cost"] = _number(r.get("generation_cost"), "generation_cost", nullable=True)
    _require(r["generation_cost"] is None or r["generation_cost"] >= 0, "negative generation_cost")
    r.setdefault("change_excerpt", "")
    _require(isinstance(r["change_excerpt"], str) and len(r["change_excerpt"]) <= 6000, "invalid change_excerpt")
    r.setdefault("change_excerpt_may_be_truncated", False)
    _require(type(r["change_excerpt_may_be_truncated"]) is bool, "invalid change excerpt flag")
    r.setdefault("source_repaired", False)
    _require(type(r["source_repaired"]) is bool, "source_repaired must be bool")
    r["injected_ids"] = _ids(r.get("injected_ids", []), "injected_ids")
    r.setdefault("error", None)
    if r["error"] is not None:
        _text(r["error"], "error")
    r.setdefault("code_fragment", "")
    _require(isinstance(r["code_fragment"], str) and len(r["code_fragment"].encode()) <= 262144,
             "invalid code_fragment")
    if r["code_fragment"]:
        _text(r["code_fragment"], "code_fragment", limit=262144)
        _require(hashlib.sha256(r["code_fragment"].encode()).hexdigest() == r["source_sha256"],
                 "code_fragment must be the exact evaluated source with matching SHA256")
    if "controlled_change" in r:
        r["controlled_change"] = validate_controlled_change(r["controlled_change"],
            parent_source_sha256=r["parent_source_sha256"], proposed_sha256=r["proposed_sha256"],
            source_sha256=r["source_sha256"], source_repaired=r["source_repaired"],
            source_code=r["code_fragment"] or None)
    h = r.get("hypothesis")
    if h is not None:
        fields = {"text", "predicted_low", "predicted_high", "reasoning", "confidence"}
        _require(isinstance(h, dict) and set(h) in (fields, fields | {'restart_predictions'}),
                 "invalid hypothesis schema")
        _text(h["text"], "hypothesis text")
        _text(h["reasoning"], "hypothesis reasoning")
        for key in ("predicted_low", "predicted_high", "confidence"):
            h[key] = _number(h[key], key)
        _require(h["predicted_low"] <= h["predicted_high"] and 0 <= h["confidence"] <= 1,
                 "invalid hypothesis bounds/confidence")
        if 'restart_predictions' in h:
            from ..hypothesis.restart_predictions import validate_spec
            h['restart_predictions'] = validate_spec(h['restart_predictions'])
    r["hypothesis"] = h
    if 'restart_prediction_binding' in r:
        binding = r['restart_prediction_binding']
        fields = {'schema', 'spec_sha256', 'response_sha256', 'proposed_sha256', 'source_sha256'}
        _require(type(binding) is dict and set(binding) == fields
                 and binding['schema'] == 'sciel-restart-prediction-binding/1',
                 'invalid restart prediction binding schema')
        for key in fields - {'schema'}:
            _hash(binding[key], 'restart prediction ' + key, nullable=False)
        _require(h is not None and 'restart_predictions' in h
                 and binding['spec_sha256'] == _digest(h['restart_predictions'])
                 and binding['proposed_sha256'] == r['proposed_sha256']
                 and binding['source_sha256'] == r['source_sha256'],
                 'restart prediction binding differs from hypothesis/source')
    measured = r["status"] == "evaluated"
    if measured:
        _require(all(r[k] is not None for k in ("parent_score", "score", "actual_delta", "source_sha256")),
                 "evaluated experiment needs measured parent/child scores, delta and source hash")
        _require(r["score"] - r["parent_score"] == r["actual_delta"],
                 "actual_delta differs from measured score difference")
    else:
        _require(r["score"] is None and r["actual_delta"] is None,
                 "unmeasured failure cannot carry a quality score or delta")
    if _MEASUREMENT_FIELDS & set(r):
        _require(_MEASUREMENT_FIELDS <= set(r), "incomplete experiment measurement context")
        r["metric"] = validate_metric_context(r["metric"])
        if r["parent_cases"] is not None:
            r["parent_cases"] = validate_cases(r["parent_cases"])
        r["evaluation"] = validate_evaluation(r["evaluation"])
        if r["evaluation"] is not None:
            observed = r["evaluation"]
            if "native_geometry" in observed:
                from ..eval.native_geometry import validate_binding
                _require(r["source_sha256"] is not None,
                         "native geometry requires a committed experiment source")
                validate_binding(observed["native_geometry"], observed,
                                 source_sha256=r["source_sha256"], metric=r["metric"])
            _require((measured and observed["ok"] and observed["score"] == r["score"])
                     or (r["status"] == "eval_failed" and not observed["ok"]),
                     "returned evaluation conflicts with experiment outcome")
    if "output_novelty" in r:
        from .eligibility import validate_output_observation
        r["output_novelty"] = validate_output_observation(r["output_novelty"],
            source_sha256=r["source_sha256"], metric=r.get("metric"), island=r.get("island"),
            evaluation=r.get("evaluation"), status=r["status"])
        if output_repeat(r):
            _require(measured and r["evaluation"]["ok"] and not r["code_fragment"],
                     "recorded output repeat cannot carry an efficacy fragment or failed evaluation")
    if "generation_dispatch" in r:
        _require(type(r["generation_dispatch"]) is str and r["generation_dispatch"] in
                 {"not_dispatched", "unknown", "returned"}, "invalid generation dispatch state")
        if r["generation_dispatch"] == "not_dispatched":
            _require(r["status"] == "generation_stopped" and r["hypothesis"] is None
                     and r["source_sha256"] is None and r["proposed_sha256"] is None
                     and r["generation_cost"] is None and r.get("evaluation") is None
                     and r["child_id"] is None and not r["source_repaired"]
                     and not r["code_fragment"] and not r["change_excerpt"],
                     "not-dispatched attempt cannot contain a generated or evaluated response")
    if 'generation_model_identity' in r:
        from ..engine.llm.client import validate_resolved_model
        identity = r['generation_model_identity']
        _require(type(identity) is dict and set(identity) == {'schema', 'requested_model', 'resolved_model'}
                 and identity['schema'] == 'sciel-llm-model-identity/1'
                 and all(validate_resolved_model(identity[key]) is not None
                         for key in ('requested_model', 'resolved_model'))
                 and r.get('generation_dispatch') == 'returned',
                 'invalid returned generation model identity')
    if _COST_FIELDS & set(r):
        from ..engine.llm.client import LLMResponse, response_cost_provenance
        _require(_COST_FIELDS <= set(r), 'incomplete generation cost accounting')
        accounting = r['generation_cost_accounting']
        _require(type(accounting) is dict and set(accounting) == {'known_usd', 'unpriced_calls'}
                 and type(accounting['known_usd']) in (int, float) and accounting['known_usd'] == 0
                 and type(accounting['unpriced_calls']) is int and accounting['unpriced_calls'] == 1,
                 'invalid unpriced generation accounting')
        provenance = r['generation_cost_provenance']
        response = LLMResponse('', 'evidence-validation', None, 0, 0, finish_reason='stop',
                               cost_provenance=provenance)
        checked = response_cost_provenance(response)
        _require(checked is not None and checked.get('source') == 'subscription_unpriced'
                 and checked == provenance and r.get('generation_dispatch') == 'returned'
                 and r['generation_cost'] == accounting['known_usd'],
                 'unpriced generation provenance or subtotal differs')
    if 'method_exposures' in r:
        from ..research_tools.followups import validate_exposures
        r['method_exposures'] = validate_exposures(r['method_exposures'])
        _require('generation_dispatch' in r, 'method exposure requires generation dispatch state')
    if 'generation_response_sha256' in r:
        _hash(r['generation_response_sha256'], 'generation response digest', nullable=False)
        _require('method_exposures' in r and r.get('generation_dispatch') == 'returned',
                 'generation response digest requires a returned method exposure')
    # Legacy records retain their exact historical shape and digest. Never infer
    # units or insert unknown fields into previously recorded measurements.
    comparable = (measured and h is not None and not r["source_repaired"]
                  and r["proposed_sha256"] == r["source_sha256"])
    verdict = ("CONFIRMED" if h["predicted_low"] <= r["actual_delta"] <= h["predicted_high"]
               else "REFUTED") if comparable else "INCONCLUSIVE"
    _require(r.get("forecast_verdict", verdict) == verdict, "forecast_verdict conflicts with measured evidence")
    r["forecast_verdict"] = verdict
    if "proposal_group" in r:
        r["proposal_group"] = validate_group(r["proposal_group"], experiment=r)
        if 'restart_prediction_binding' in r:
            _require(r['restart_prediction_binding']['response_sha256'] == r['proposal_group']['response_sha256'],
                     'restart prediction binding differs from proposal response')
    return r


def _lesson_key(lesson):
    # Rules can contain source snippets: operators, case and whitespace inside
    # literals change their meaning. Only trim exterior whitespace for dedup.
    rule = lesson["rule"].strip()
    _require(bool(re.search(r"\w", rule)), "rule must contain words")
    return rule, lesson["polarity"], lesson["scope"], lesson["branch"]


def _refresh(lesson):
    # A warning inferred from a regression also aims to IMPROVE later outcomes.
    # Source evidence truth is separate from benefit after injecting the advice.
    observations = {e["id"]: e for e in lesson["exposure_snapshots"]
                    + [x["experiment"] for x in lesson["inherited_exposures"]]}
    support, refute = [], []
    for eid, e in observations.items():
        if not efficacy_eligible(e):
            continue
        if e["actual_delta"] > 0:
            support.append(eid)
        elif e["actual_delta"] < 0:
            refute.append(eid)
    lesson.update(evidence_ids=[e["id"] for e in lesson["evidence_snapshots"]],
                  source_support_ids=[e["id"] for e in lesson["evidence_snapshots"]],
                  exposure_ids=[e["id"] for e in lesson["exposure_snapshots"]],
                  support_ids=support, refute_ids=refute,
                  confidence=(1 + len(support)) / (2 + len(support) + len(refute)))
