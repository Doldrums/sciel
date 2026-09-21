"""Apply validated observations, lesson proposals and evidence-preserving retirement."""
from __future__ import annotations

import copy
from . import authoring
from .eligibility import efficacy_eligible
from .records import (
    _digest,
    _experiment,
    _ids,
    _lesson_key,
    _refresh,
    _require,
    _snapshot,
    _supports,
    _text,
)

def record(knowledge, experiment):
    r = _experiment(experiment)
    snap = _snapshot(r)
    _require(r["id"] not in knowledge._seen and r["id"] not in knowledge._seen_comparisons,
             "duplicate or conflicting experiment ID")
    known = {l["id"]: l for l in knowledge._lessons}
    _require(set(r["injected_ids"]) <= set(known), "injected lesson ID is unknown/retired")
    for lid in r["injected_ids"]:
        lesson = known[lid]
        _require(lesson["scope"] == "global" or lesson["branch"] == r["branch"],
                 "branch-local lesson injected into a foreign branch")
    # All validation above is side-effect free, including source/delta binding.
    knowledge._seen[r["id"]] = _digest(snap)
    knowledge._experiments = (knowledge._experiments + [r])[-knowledge.max_experiments:]
    # These IDs retain the intended prompt membership even if a budget gate
    # proves that the current request never left the process. Preserve that
    # audit record, but do not invent an exposure of the model to its advice.
    exposure_ids = [] if r.get("generation_dispatch") == "not_dispatched" else r["injected_ids"]
    for lid in exposure_ids:
        lesson = known[lid]
        lesson["exposure_snapshots"] = (lesson["exposure_snapshots"] + [copy.deepcopy(snap)])[-knowledge.max_experiments:]
        _refresh(lesson)
    if efficacy_eligible(r) and r["actual_delta"] > 0 and r["code_fragment"]:
        knowledge._fragments = (knowledge._fragments + [{"id": "f:" + r["id"], "experiment_id": r["id"],
            "branch": r["branch"], "code": r["code_fragment"], "source_sha256": r["source_sha256"],
            "score": r["score"], "actual_delta": r["actual_delta"], "evidence": snap}])[-knowledge.max_fragments:]
    knowledge._age_initializer_evidence()


def _evidence(knowledge):
    result = {e["id"]: _snapshot(e) for e in knowledge._experiments}
    result.update((g["id"], copy.deepcopy(g)) for g in knowledge._comparisons)
    for lesson in knowledge._lessons:
        result.update((e["id"], e) for e in lesson["evidence_snapshots"] + lesson["exposure_snapshots"])
        result.update((x["experiment"]["id"], x["experiment"]) for x in lesson["inherited_exposures"])
    result.update((f["experiment_id"], f["evidence"]) for f in knowledge._fragments)
    return result


def _proposals(knowledge, proposals, allowed_evidence_ids, writer_context=None):
    _require(isinstance(proposals, list), "proposals must be a list")
    if writer_context is not None:
        writer_context = authoring.validate_context(writer_context)
    allowed = set(_ids(allowed_evidence_ids, "allowed_evidence_ids"))
    evidence = knowledge._evidence()
    _require(allowed <= set(evidence), "allowed evidence is unknown or no longer retained")
    lessons, counter, result = copy.deepcopy(knowledge._lessons), knowledge._next_lesson_id, []
    for index, p in enumerate(proposals):
        required = {"rule", "polarity", "scope", "evidence_ids"}
        _require(isinstance(p, dict) and required <= set(p) <= required | {"branch"},
                 "invalid lesson proposal schema")
        _text(p["rule"], "lesson rule")
        _require(p["polarity"] in ("positive", "negative") and p["scope"] in ("branch", "global"),
                 "invalid lesson polarity/scope")
        ids = _ids(p["evidence_ids"], "evidence_ids")
        _require(bool(ids) and set(ids) <= allowed, "lesson evidence must be nonempty and allowed")
        snaps = [copy.deepcopy(evidence[eid]) for eid in ids]
        _require(all(_supports(e, p["polarity"]) for e in snaps),
                 "lesson polarity requires matching measured quality deltas")
        branches = {e["branch"] for e in snaps}
        _require(len(branches) >= 2 if p["scope"] == "global" else len(branches) == 1,
                 "global evidence needs two branches; branch evidence needs exactly one")
        lesson = {"rule": p["rule"].strip(), "polarity": p["polarity"], "scope": p["scope"],
                  "branch": next(iter(branches)) if p["scope"] == "branch" else None}
        # A model may copy this redundant field from an existing lesson.
        # The evidence determines scope; generated branch claims cannot.
        if "branch" in p:
            _require(p["branch"] == lesson["branch"], "proposal branch differs from evidence-derived branch")
        key = _lesson_key(lesson)  # Validate even when there are no prior lessons.
        existing = next((l for l in lessons if _lesson_key(l) == key), None)
        if existing is None:
            lesson.update(id=f"l{counter:06d}", evidence_snapshots=[], exposure_snapshots=[], inherited_exposures=[])
            counter += 1
            lessons.append(lesson)
        else:
            lesson = existing
        authoring.add_writer(lesson, writer_context, p["rule"], index, existed=existing is not None)
        merged = {e["id"]: e for e in lesson["evidence_snapshots"] + snaps}
        _require(len(merged) <= knowledge.max_experiments, "lesson evidence window capacity exceeded")
        lesson["evidence_snapshots"] = list(merged.values())
        _refresh(lesson)
        if "authoring" in lesson:
            authoring.validate(lesson["authoring"], rule=lesson["rule"], lesson_id=lesson["id"],
                               evidence=merged, next_lesson_id=counter, polarity=lesson["polarity"])
        result.append(lesson["id"])
    authoring.validate_collection(lessons)
    return lessons, counter, result


def add_lessons(knowledge, proposals, *, allowed_evidence_ids, writer_context=None):
    lessons, counter, result = knowledge._proposals(proposals, allowed_evidence_ids, writer_context)
    # Keep proposed lessons available; oldest other lessons yield capacity.
    protected = set(result)
    _require(len(protected) <= knowledge.max_lessons, "too many new lessons")
    while len(lessons) > knowledge.max_lessons:
        lessons.remove(next(l for l in lessons if l["id"] not in protected))
    knowledge._lessons, knowledge._next_lesson_id = lessons, counter
    knowledge._age_initializer_evidence()
    return result


def reflect(knowledge, proposals, retire_ids, *, allowed_evidence_ids, writer_context=None):
    retire = set(_ids(retire_ids, "retire_ids"))
    old = {l["id"]: l for l in knowledge._lessons}
    _require(retire <= set(old), "cannot retire unknown lesson")
    lessons, counter, ids = knowledge._proposals(proposals, allowed_evidence_ids, writer_context)
    replacements = [l for l in lessons if l["id"] in ids and l["id"] not in retire]
    for lid in sorted(retire):
        prior = old[lid]
        target = next((l for l in replacements if l["polarity"] == prior["polarity"]
            and (l["scope"] == "global" or l["branch"] == prior["branch"])
            and set(prior["evidence_ids"]) <= set(l["evidence_ids"])), None)
        _require(target is not None,
            "retirement requires a replacement preserving scope, polarity and evidence")
        authoring.inherit(target, prior, enabled=writer_context is not None)
        inherited = target["inherited_exposures"] + prior["inherited_exposures"] + [
            {"from_lesson_id": lid, "from_rule": prior["rule"], "experiment": e}
            for e in prior["exposure_snapshots"]]
        inherited = {(x["from_lesson_id"], x["experiment"]["id"]): x for x in inherited}
        target["inherited_exposures"] = list(inherited.values())[-knowledge.max_experiments:]
        _refresh(target)
    lessons = [l for l in lessons if l["id"] not in retire]
    _require(len(lessons) <= knowledge.max_lessons, "reflection exceeds lesson capacity; retire with evidence")
    authoring.validate_collection(lessons)
    knowledge._lessons, knowledge._next_lesson_id = lessons, counter
    knowledge._age_initializer_evidence()
    return {"lesson_ids": [lid for lid in ids if lid not in retire], "retired_ids": sorted(retire)}
