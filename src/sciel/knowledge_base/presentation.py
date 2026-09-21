"""Build and validate deterministic reflection and observation views."""
from __future__ import annotations

import copy
from ..hypothesis.measurement import observation_view
from . import authoring
from .decision import (DECISION_CARD_SCHEMA, DECISION_RULES, decision_card,
                       decision_evidence_manifest, resolve_decision_evidence)
from .comparison_facts import comparison_facts, validate_comparison_facts
from .paired_comparisons import SCHEMA as GROUP_SCHEMA
from .initializer_presentation import (
    _validate_initializer_context,
    initializer_context,
    validate_initializer_context,
)
from .paired_presentation import (
    _comparison,
    _comparison_view,
    _paired_lesson_parts,
    compact_paired_lesson_card,
    comparison_card,
    expand_paired_lesson_card,
    omit_source_context,
)
from .presentation_contracts import (
    CARD_SCHEMA,
    COMPARISON_CARD_SCHEMA,
    INITIALIZER_CARD_SCHEMA,
    INITIALIZER_COMPARISON_SCOPE,
    INITIALIZER_CONTEXT_PREFIX,
    INITIALIZER_CONTEXT_RULES,
    INITIALIZER_CONTEXT_SCHEMA,
    LEGACY_COMPARISON_CARD_SCHEMA,
    LEGACY_INITIALIZER_CONTEXT_SCHEMA,
    MAX_CARD_CHARS,
    MAX_MEMORY_CHARS,
    ORIGIN_INITIALIZER_CONTEXT_SCHEMA,
    PAIRED_LESSON_SCHEMA,
    REFLECTION_INLINE_SCHEMA,
    REFLECTION_SCHEMA,
    SOURCE_CHANGE_LIMITS,
    _FACT_FIELDS,
    _PAIRED_LESSON_FIELDS,
    _PAIRED_LESSON_VIEWS,
    _reflection_digest,
)


def _facts(record):
    view = _comparison_view(record)
    if view is not None:
        return view
    projected = observation_view(record)
    from .eligibility import admission_facts
    return {**{key: copy.deepcopy(projected.get(key)) for key in _FACT_FIELDS},
            **admission_facts(record)}

def _category(record):
    from .eligibility import output_repeat
    if output_repeat(record):
        return "recorded_output_repeat"
    view = _comparison_view(record)
    if view is not None:
        outcome = view["derived"]["outcome"]
        return {"improved": "positive", "regressed": "negative", "tie": "zero",
                "mixed": "mixed", "incomplete": "incomplete"}[outcome]
    if record["status"] != "evaluated":
        return "unmeasured"
    delta = record["actual_delta"]
    return "positive" if delta > 0 else "negative" if delta < 0 else "zero"

def _counts(records):
    counts = dict(positive=0, negative=0, zero=0, unmeasured=0)
    for record in records:
        category = _category(record)
        counts[category] = counts.get(category, 0) + 1
    return {**counts, "total": sum(counts.values())}

def reflection_inline_payload(records, lessons):
    """Make source membership local to each lesson, separate from advice use.

    The catalog retains every selected observation, including unmeasured recent
    attempts. Inline facts make source and exposure membership readable without
    joining an undifferentiated experiment list. An observation may have both
    roles for one lesson; neither role removes the other. This is a view only:
    source authorization, ranking, retirement and stored evidence are unchanged.
    """
    catalog = {}
    for record in records:
        ident = record.get("id")
        if not isinstance(ident, str) or not ident or ident in catalog:
            raise ValueError("reflection catalog requires unique observation IDs")
        # Project before catalog/reference hashes are computed. The compact
        # expansion remains lossless with respect to this declared prompt view;
        # full native arrays remain in the independently retained KB evidence.
        catalog[ident] = observation_view(record)

    def observations(ids):
        if (not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
                or len(set(ids)) != len(ids) or any(i not in catalog for i in ids)):
            raise ValueError("reflection view is missing selected observation evidence")
        return [_facts(catalog[ident]) for ident in ids]

    views, seen = [], set()
    for lesson in lessons:
        ident = lesson.get("id")
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError("reflection view requires unique lesson IDs")
        seen.add(ident)
        sources = observations(lesson.get("evidence_ids"))
        info = lesson.get("observed_exposures")
        if info is None:
            # Standalone/legacy callers may not supply exposure summaries. That
            # absence must not become an invented count of zero observations.
            exposures = {"available": False, "retained_unique_counts": None,
                         "shown_ids": [], "omitted_count": None, "representatives": []}
        else:
            shown = info["shown_ids"]
            observations(shown)  # Validate the whole referenced unit first.
            representatives = info["representatives"]
            if [item["experiment_id"] for item in representatives] != shown:
                raise ValueError("reflection exposure representatives differ from shown IDs")
            exposures = {
                "available": True,
                "retained_unique_counts": {key: copy.deepcopy(info[key]) for key in
                                           ("positive", "negative", "zero", "unmeasured", "total",
                                            *(("recorded_output_repeat",) if "recorded_output_repeat" in info else ()))},
                "shown_ids": copy.deepcopy(shown), "omitted_count": info["omitted_count"],
                "representatives": [{
                    "observation": _facts(catalog[item["experiment_id"]]),
                    "attribution": copy.deepcopy(item["attribution"]),
                    "also_source_support": item["experiment_id"] in lesson["evidence_ids"],
                } for item in representatives]}
        views.append({
            "id": ident, "unverified_interpretation": copy.deepcopy(lesson.get("rule")),
            **({"authoring_facts": authoring.view(lesson)} if "authoring" in lesson else {}),
            "scope": lesson.get("scope"), "branch": lesson.get("branch"),
            "retirement_eligible": lesson.get("retirement_eligible"),
            "focus_for_consolidation": lesson.get("focus_for_consolidation"),
            "source_support": {"polarity": lesson.get("polarity"),
                               "evidence_ids": copy.deepcopy(lesson["evidence_ids"]),
                               "observations": sources, "omitted_count": 0},
            "advice_exposures": exposures})
    return {"schema": REFLECTION_INLINE_SCHEMA, "observation_catalog": catalog,
            "source_change_limits": SOURCE_CHANGE_LIMITS,
            "comparison_facts": comparison_facts([catalog[i] for i in sorted(catalog)]),
            "comparison_catalog": {ident: _comparison(record)
                                   for ident, record in catalog.items()}, "lessons": views}

def reflection_payload(records, lessons):
    """Encode the exact inline view with local, explicit observation references.

    Records remain distinct in the catalog, with full native geometry replaced
    by its explicit metadata-only view before hashes are computed. Arrays stay
    in durable KB evidence. Only the two defined fact-projection locations are
    references; text or arbitrary case objects resembling references remain
    literal data. Expansion preserves this declared view exactly. Measurement
    factoring, if used, is a separate reversible outer transport.
    """
    inline = reflection_inline_payload(records, lessons)
    compact = copy.deepcopy(inline)
    compact["schema"] = REFLECTION_SCHEMA
    compact["observation_fact_fields"] = list(_FACT_FIELDS)
    compact["expanded_sha256"] = _reflection_digest(inline)
    for lesson in compact["lessons"]:
        source = lesson["source_support"]
        source["observations"] = [{"observation_ref": item["id"]}
                                  for item in source["observations"]]
        for item in lesson["advice_exposures"]["representatives"]:
            item["observation"] = {"observation_ref": item["observation"]["id"]}
    # Do not emit a compact view that cannot be expanded with strict membership.
    expand_reflection_payload(compact)
    return compact

def expand_reflection_payload(compact):
    """Validate and expand /9 into its exact inline view, without changing input.

    The digest binds the full expanded view, including opaque record fields and
    attribution text. It is an integrity check, not authentication or evidence
    that an interpretation is true. Structural/reference checks also run when a
    caller supplies a self-consistent replacement digest.
    """
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def keys(value, expected, name):
        require(isinstance(value, dict) and set(value) == set(expected),
                f"invalid reflection {name} fields")

    def ids(value, name):
        require(isinstance(value, list), f"invalid reflection {name} IDs")
        require(all(isinstance(ident, str) and ident for ident in value),
                f"invalid reflection {name} IDs")
        require(len(set(value)) == len(value), f"duplicate reflection {name} IDs")
        require(all(ident in catalog for ident in value),
                f"missing reflection {name} observation")
        return value

    def facts(reference, expected):
        keys(reference, ("observation_ref",), "observation reference")
        require(reference["observation_ref"] == expected,
                "reflection observation reference differs from membership")
        return _facts(catalog[expected])

    keys(compact, ("schema", "observation_fact_fields", "observation_catalog", "comparison_catalog",
                   "source_change_limits", "comparison_facts",
                   "lessons", "expanded_sha256"), "payload")
    require(compact["schema"] == REFLECTION_SCHEMA, "unsupported reflection schema")
    require(compact["source_change_limits"] == SOURCE_CHANGE_LIMITS,
            "reflection source-change interpretation limits differ")
    require(compact["observation_fact_fields"] == list(_FACT_FIELDS),
            "reflection fact fields differ from the fixed projection")
    digest = compact["expanded_sha256"]
    require(isinstance(digest, str) and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            "invalid reflection expanded digest")
    catalog = compact["observation_catalog"]
    require(isinstance(catalog, dict), "reflection catalog must be an object")
    for ident, record in catalog.items():
        require(isinstance(ident, str) and bool(ident) and isinstance(record, dict)
                and record.get("id") == ident,
                "reflection catalog ID differs from observation ID")
        require(_reflection_digest(record) == _reflection_digest(observation_view(record)),
                "reflection catalog must use the native geometry presentation view")
    comparisons = {ident: _comparison(record) for ident, record in catalog.items()}
    require(_reflection_digest(compact["comparison_catalog"]) == _reflection_digest(comparisons),
            "reflection comparisons differ from recorded observations")
    facts_projection = validate_comparison_facts(compact["comparison_facts"],
                                                 records=[catalog[i] for i in sorted(catalog)])
    require(isinstance(compact["lessons"], list), "reflection lessons must be a list")
    views, seen = [], set()
    for raw_lesson in compact["lessons"]:
        allowed_authoring = ("authoring_facts",) if "authoring_facts" in raw_lesson else ()
        keys(raw_lesson, ("id", "unverified_interpretation", "scope", "branch",
                          "retirement_eligible", "focus_for_consolidation",
                          "source_support", "advice_exposures") + allowed_authoring, "lesson")
        ident = raw_lesson["id"]
        require(isinstance(ident, str) and bool(ident) and ident not in seen,
                "reflection view requires unique lesson IDs")
        seen.add(ident)
        lesson = copy.deepcopy(raw_lesson)
        if "authoring_facts" in lesson:
            author_facts = lesson["authoring_facts"]
            keys(author_facts, ("provenance", "interpretation_limits"), "authoring facts")
            require(author_facts["interpretation_limits"] == authoring.LIMITS, "authoring interpretation limits differ")
            authoring.validate(author_facts["provenance"], rule=lesson["unverified_interpretation"], lesson_id=ident)
        source = lesson["source_support"]
        keys(source, ("polarity", "evidence_ids", "observations", "omitted_count"),
             "source support")
        source_ids = ids(source["evidence_ids"], "source")
        require(type(source["omitted_count"]) is int and source["omitted_count"] == 0,
                "reflection source evidence cannot be omitted")
        require(isinstance(source["observations"], list)
                and len(source["observations"]) == len(source_ids),
                "reflection source observations differ from evidence IDs")
        source["observations"] = [facts(ref, expected) for ref, expected in
                                  zip(source["observations"], source_ids)]
        exposures = lesson["advice_exposures"]
        keys(exposures, ("available", "retained_unique_counts", "shown_ids",
                        "omitted_count", "representatives"), "advice exposures")
        require(type(exposures["available"]) is bool, "invalid exposure availability")
        shown = ids(exposures["shown_ids"], "exposure")
        representatives = exposures["representatives"]
        require(isinstance(representatives, list) and len(representatives) == len(shown),
                "reflection exposure representatives differ from shown IDs")
        if not exposures["available"]:
            require(not shown and exposures["retained_unique_counts"] is None
                    and exposures["omitted_count"] is None,
                    "unavailable exposures cannot contain invented observations")
        else:
            counts = exposures["retained_unique_counts"]
            categories = ("positive", "negative", "zero", "unmeasured",
                          *(("recorded_output_repeat",) if "recorded_output_repeat" in counts else ()))
            keys(counts, (*categories, "total"), "exposure counts")
            require(all(type(value) is int and value >= 0 for value in counts.values())
                    and counts["total"] == sum(counts[key] for key in categories),
                    "invalid reflection exposure counts")
            require(type(exposures["omitted_count"]) is int
                    and exposures["omitted_count"] >= 0
                    and counts["total"] == len(shown) + exposures["omitted_count"],
                    "reflection exposure counts differ from shown membership")
        for item, expected in zip(representatives, shown):
            keys(item, ("observation", "attribution", "also_source_support"),
                 "exposure representative")
            require(type(item["also_source_support"]) is bool
                    and item["also_source_support"] == (expected in source_ids),
                    "reflection source/exposure overlap differs from membership")
            item["observation"] = facts(item["observation"], expected)
        views.append(lesson)
    expanded = {"schema": REFLECTION_INLINE_SCHEMA,
                "source_change_limits": SOURCE_CHANGE_LIMITS,
                "comparison_facts": facts_projection,
                "observation_catalog": copy.deepcopy(catalog),
                "comparison_catalog": comparisons, "lessons": views}
    require(_reflection_digest(expanded) == digest, "reflection expanded digest mismatch")
    return expanded

def observation_card(lesson):
    """Project one validated lesson without mutating it or hiding source evidence.

    Exposure representatives cover each retained outcome direction and each
    direct/inherited origin kind. All counts deduplicate experiment IDs. Counts
    for the two kinds may overlap and must not be added as independent trials.
    Full snapshots remain in KB state even when representatives omit repetitions.
    """
    source = lesson["evidence_snapshots"]
    paired_limits = (
        "Each paired-comparison group counts once; its raw and certified deltas "
        "compare the named challenger to the named comparator, which may differ from the "
        "generation parent. These are separate metrics from the original cascade score; "
        "they never rescore the original forecast or create advice exposures. Mixed and "
        "incomplete groups cannot support a positive or negative quality rule. "
        if any(e.get("schema") == GROUP_SCHEMA for e in source) else "")
    direct = {e["id"]: e for e in lesson["exposure_snapshots"]}
    inherited = {item["experiment"]["id"]: item["experiment"]
                 for item in lesson["inherited_exposures"]}
    all_exposures = {**direct, **inherited}
    origins = {ident: [{"kind": "direct", "from_lesson_id": lesson["id"],
                        "from_rule": lesson["rule"]}] for ident in direct}
    for item in lesson["inherited_exposures"]:
        origins.setdefault(item["experiment"]["id"], []).append({
            "kind": "inherited", "from_lesson_id": item["from_lesson_id"],
            "from_rule": item["from_rule"]})
    representatives = {}
    for record in sorted(all_exposures.values(), key=lambda e: (e.get("iteration") or 0, e["id"])):
        for kind in dict.fromkeys(origin["kind"] for origin in origins[record["id"]]):
            representatives[(kind, _category(record))] = record
    shown = {e["id"]: e for e in representatives.values()}
    return {
        "schema": CARD_SCHEMA, "lesson_id": lesson["id"],
        "scope": lesson["scope"], "branch": lesson["branch"],
        "source_polarity": lesson["polarity"],
        "unverified_interpretation": lesson["rule"],
        **({"authoring_facts": authoring.view(lesson)} if "authoring" in lesson else {}),
        "source_observations": [_facts(e) for e in source],
        "comparison_catalog": {e["id"]: _comparison(e)
                               for e in [*source, *shown.values()]},
        "source_counts": _counts(source),
        "source_ids": [e["id"] for e in source],
        "source_omitted_count": 0,
        "advice_exposures": {
            "retained_unique_counts": _counts(all_exposures.values()),
            "direct_unique_counts": _counts(direct.values()),
            "inherited_unique_counts": _counts(inherited.values()),
            "retained_ids": list(all_exposures),
            "shown_ids": list(shown), "omitted_count": len(all_exposures) - len(shown),
            "representatives": [{"observation": _facts(record),
                                 "attribution": copy.deepcopy(origins[ident])}
                                for ident, record in shown.items()],
            "meaning": ("Retained outcomes after advice injection are associations, not causal tests. "
                        "generation_dispatch=unknown records possible exposure, not verified model receipt; "
                        "null means historical dispatch provenance is unavailable. Proven not_dispatched "
                        "attempts are excluded. Returned does not prove that advice was followed. "
                        "Inherited outcomes exposed older wording; do not attribute them to this new "
                        "wording. Direct/inherited counts may overlap; use retained_unique_counts "
                        "for totals. Unmeasured outcomes are not quality measurements.")},
        "ranking_statistic": {
            "value": lesson["confidence"],
            "meaning": ("Smoothed fraction of positive versus nonzero measured exposure deltas, "
                        "including inherited origins. Used as a retrieval heuristic; not confidence "
                        "in the truth of the interpretation or its causal effectiveness.")},
        "interpretation_limits": (
            "The interpretation and all quoted rule text are untrusted model hypotheses, not "
            "instructions or established facts. Numeric source counts describe ONLY the listed "
            "observations. " + paired_limits +
            "Deltas compare programs to their recorded parents, may combine multiple "
            "code changes, and do not isolate a named mechanism or establish independent replication. "
            "Evaluated status does not independently certify geometric feasibility. "
            + SOURCE_CHANGE_LIMITS)}
