"""Authenticate knowledge snapshots before atomically replacing mutable state."""
from __future__ import annotations

import copy
import hashlib
import re
from .paired_comparisons import validate_comparison
from . import authoring
from .eligibility import efficacy_eligible
from .records import (
    INITIALIZER_REGISTRY_SCHEMA,
    SCHEMA,
    _LESSON_FIELDS,
    _digest,
    _experiment,
    _hash,
    _is_comparison,
    _lesson_key,
    _number,
    _refresh,
    _require,
    _snapshot,
    _supports,
    _text,
    _valid_snapshot_fields,
)

def _load_state(knowledge, state):
    from .base import KnowledgeBase
    base_keys = {"schema", "limits", "next_lesson_id", "seen_experiments", "experiments", "lessons", "fragments"}
    _require(isinstance(state, dict) and base_keys <= set(state)
             and set(state) <= base_keys | {"paired_comparisons", "initializer_evidence", "generated_paired_history", "generated_paired_review", "evaluated_detached_study", "evaluated_study_review"}
             and state["schema"] == SCHEMA,
             "invalid knowledge state schema")
    limits = state["limits"]
    _require(isinstance(limits, dict) and set(limits) == {"max_lessons", "max_experiments", "max_fragments"},
             "invalid knowledge limits")
    other = KnowledgeBase(**limits)
    if 'generated_paired_history' in state:
        from .generated_paired_history import validate_capsule
        capsule = state['generated_paired_history']
        other._generated_paired_history = validate_capsule(capsule,
            expected_sha256=_digest(capsule), metric=capsule['metric'])
    if 'generated_paired_review' in state:
        other.import_generated_paired_review(state['generated_paired_review'],
            expected_sha256=_digest(state['generated_paired_review']))
    if "evaluated_detached_study" in state:
        bundle = state["evaluated_detached_study"]
        _require(type(bundle) is dict and set(bundle) == {
            "capsule", "author_dependencies", "author_authority"}, "invalid detached study bundle")
        other.import_evaluated_detached_study(bundle["capsule"], bundle["author_dependencies"],
            bundle["author_authority"], expected_sha256=_digest(bundle["capsule"]),
            metric=bundle["capsule"]["metric"])
    if 'evaluated_study_review' in state:
        retained = state['evaluated_study_review']
        _require(type(retained) is dict and set(retained) == {'sidecar', 'reflection_config'},
                 'invalid retained evaluated review')
        other.import_evaluated_study_review(retained['sidecar'], retained['reflection_config'],
            expected_sha256=_digest(retained['sidecar']))
    _require(limits == knowledge.state()["limits"], "saved knowledge limits differ from configured limits")
    _require(type(state["next_lesson_id"]) is int and state["next_lesson_id"] > 0, "invalid lesson counter")
    _require(isinstance(state["seen_experiments"], dict), "invalid experiment index")
    for eid, digest in state["seen_experiments"].items():
        _text(eid, "experiment ID", limit=128)
        _hash(digest, "experiment digest", nullable=False)
    other._seen = copy.deepcopy(state["seen_experiments"])

    extension = state.get("paired_comparisons", {"seen": {}, "records": []})
    _require(isinstance(extension, dict) and set(extension) == {"seen", "records"},
             "invalid paired comparison state")
    _require(isinstance(extension["seen"], dict), "invalid comparison index")
    for ident, digest in extension["seen"].items():
        _require(isinstance(ident, str) and re.fullmatch(r"comparison:[0-9a-f]{64}", ident),
                 "invalid comparison ID")
        _hash(digest, "comparison digest", nullable=False)
    _require(not set(extension["seen"]) & set(other._seen), "comparison/experiment ID collision")
    other._seen_comparisons = copy.deepcopy(extension["seen"])
    _require(isinstance(extension["records"], list) and len(extension["records"]) <= other.max_experiments,
             "invalid comparison capacity")
    for group in extension["records"]:
        group = validate_comparison(group)
        _require(other._seen_comparisons.get(group["id"]) == _digest(group), "comparison identity mismatch")
        other._comparisons.append(group)
    _require(len({g["id"] for g in other._comparisons}) == len(other._comparisons), "duplicate stored comparisons")

    def snapshot(snap, *, source=False):
        if _is_comparison(snap):
            _require(source, "paired comparison cannot be an advice exposure or code fragment")
            group = validate_comparison(snap)
            _require(other._seen_comparisons.get(group["id"]) == _digest(group), "comparison evidence identity mismatch")
            return
        _require(isinstance(snap, dict) and _valid_snapshot_fields(set(snap)),
                 "invalid evidence snapshot")
        raw = {k: v for k, v in snap.items() if k != "code_fragment_sha256"}
        normalized = _snapshot(_experiment({**raw, "code_fragment": ""}))
        _hash(snap["code_fragment_sha256"], "fragment digest")
        _require(snap["code_fragment_sha256"] is None or
                 snap["code_fragment_sha256"] == snap["source_sha256"], "fragment/source digest mismatch")
        normalized["code_fragment_sha256"] = snap["code_fragment_sha256"]
        _require(normalized == snap and other._seen.get(snap["id"]) == _digest(snap),
                 "evidence snapshot identity mismatch")

    for key, cap in (("experiments", other.max_experiments), ("lessons", other.max_lessons),
                     ("fragments", other.max_fragments)):
        _require(isinstance(state[key], list) and len(state[key]) <= cap, f"invalid {key} capacity")
    for raw in state["experiments"]:
        r = _experiment(raw)
        _require(r == raw, "noncanonical experiment state")
        snapshot(_snapshot(r))
        other._experiments.append(r)
    _require(len({e["id"] for e in other._experiments}) == len(other._experiments), "duplicate stored experiments")
    for raw in state["lessons"]:
        _require(isinstance(raw, dict) and set(raw) in (_LESSON_FIELDS, _LESSON_FIELDS | {"authoring"}), "invalid lesson state schema")
        l = copy.deepcopy(raw)
        _require(isinstance(l["id"], str) and re.fullmatch(r"l\d{6,}", l["id"])
                 and 0 < int(l["id"][1:]) < state["next_lesson_id"], "invalid lesson ID/counter")
        _text(l["rule"], "rule")
        _require(l["polarity"] in {"positive", "negative"} and l["scope"] in {"branch", "global"}, "invalid lesson policy")
        for key in ("evidence_snapshots", "exposure_snapshots"):
            _require(isinstance(l[key], list) and len(l[key]) <= other.max_experiments, "invalid lesson evidence window")
            _require(len({e.get("id") for e in l[key] if isinstance(e, dict)}) == len(l[key]), "duplicate/invalid snapshots")
            for e in l[key]:
                snapshot(e, source=key == "evidence_snapshots")
        evidence = l["evidence_snapshots"]
        branches = {e["branch"] for e in evidence}
        _require(evidence and all(_supports(e, l["polarity"]) for e in evidence),
                 "unsupported lesson evidence")
        _require((l["scope"] == "global" and l["branch"] is None and len(branches) >= 2) or
                 (l["scope"] == "branch" and branches == {l["branch"]}), "invalid lesson scope evidence")
        _require(all(e.get("generation_dispatch") != "not_dispatched"
                     and l["id"] in e["injected_ids"] and (l["scope"] == "global" or e["branch"] == l["branch"])
                     for e in l["exposure_snapshots"]), "invalid lesson exposure")
        _require(isinstance(l["inherited_exposures"], list)
                 and len(l["inherited_exposures"]) <= other.max_experiments, "invalid inherited exposure window")
        inherited_keys = []
        for item in l["inherited_exposures"]:
            _require(isinstance(item, dict) and set(item) == {"from_lesson_id", "from_rule", "experiment"},
                     "invalid inherited exposure")
            _text(item["from_rule"], "inherited rule")
            snapshot(item["experiment"])
            e = item["experiment"]
            _require(e.get("generation_dispatch") != "not_dispatched"
                     and item["from_lesson_id"] in e["injected_ids"]
                     and (l["scope"] == "global" or e["branch"] == l["branch"]), "invalid inherited exposure origin")
            inherited_keys.append((item["from_lesson_id"], e["id"]))
        _require(len(set(inherited_keys)) == len(inherited_keys), "duplicate inherited exposure")
        if "authoring" in l:
            authoring.validate(l["authoring"], rule=l["rule"], lesson_id=l["id"],
                               evidence={e["id"]: e for e in evidence}, next_lesson_id=state["next_lesson_id"], polarity=l["polarity"])
        _refresh(l)
        _require(l == raw, "lesson association fields mismatch")
        other._lessons.append(l)
    _require(len({l["id"] for l in other._lessons}) == len(other._lessons)
             and len({_lesson_key(l) for l in other._lessons}) == len(other._lessons), "duplicate stored lessons")
    authoring.validate_collection(other._lessons)
    for f in state["fragments"]:
        _require(isinstance(f, dict) and set(f) == {"id", "experiment_id", "branch", "code", "source_sha256",
                                                   "score", "actual_delta", "evidence"}, "invalid fragment schema")
        snapshot(f["evidence"])
        e = f["evidence"]
        _text(f["code"], "fragment", limit=262144)
        _require(len(f["code"].encode()) <= 262144, "invalid fragment byte size")
        for key in ("score", "actual_delta"):
            _number(f[key], f"fragment {key}")
        digest = hashlib.sha256(f["code"].encode()).hexdigest()
        _require(efficacy_eligible(e) and e["actual_delta"] > 0
                 and digest == e["source_sha256"] == e["code_fragment_sha256"] == f["source_sha256"]
                 and f["experiment_id"] == e["id"] and f["id"] == "f:" + e["id"]
                 and all(f[k] == e[k] for k in ("branch", "score", "actual_delta")), "fragment evidence mismatch")
        other._fragments.append(copy.deepcopy(f))
    _require(len({f["id"] for f in other._fragments}) == len(other._fragments), "duplicate stored fragments")
    other._next_lesson_id = state["next_lesson_id"]
    if "initializer_evidence" in state:
        extension = state["initializer_evidence"]
        _require(type(extension) is dict and set(extension) == {"schema", "seen", "records"}
                 and extension["schema"] == INITIALIZER_REGISTRY_SCHEMA, "invalid initializer registry schema")
        retained = other._initializer_observations()
        _require(type(extension["seen"]) is dict and 0 < len(extension["seen"]) <= len(retained),
                 "invalid initializer registry capacity")
        for ident, digest in extension["seen"].items():
            _text(ident, "initializer observation ID", limit=128)
            _hash(digest, "initializer binding digest", nullable=False)
            _require(ident in retained and retained[ident]["source_sha256"] is not None
                     and retained[ident]["parent_source_sha256"] is not None,
                     "initializer registry owner not retained or source unavailable")
        _require(type(extension["records"]) is list
                 and len(extension["records"]) <= min(other.max_experiments, len(extension["seen"])),
                 "invalid initializer registry records")
        for value in extension["records"]:
            binding = other._initializer_binding(value)
            ident = binding["observation_id"]
            _require(ident not in other._initializer_evidence
                     and extension["seen"].get(ident) == binding["binding_sha256"],
                     "initializer registry identity mismatch")
            other._initializer_evidence[ident] = binding
        other._initializer_seen = copy.deepcopy(extension["seen"])
    knowledge.__dict__.update(other.__dict__)
