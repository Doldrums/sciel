"""Lossless reflection fact references; no evaluation, network or archived edits."""
import copy
import hashlib
import json

import pytest

from sciel.knowledge_base.presentation import (
    REFLECTION_INLINE_SCHEMA,
    REFLECTION_SCHEMA,
    expand_reflection_payload,
    reflection_inline_payload,
    reflection_payload,
)


FACT_FIELDS = [
    "id", "branch", "parent_id", "parent_source_sha256", "parent_score", "score",
    "actual_delta", "status", "source_sha256", "proposed_sha256", "source_repaired",
    "forecast_verdict", "metric", "parent_cases", "evaluation", "generation_dispatch",
    "controlled_change", "operator", "change_excerpt", "change_excerpt_may_be_truncated",
]


def canonical(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@pytest.fixture
def data():
    metric = {"schema": "fixture/1", "units": "dimensionless", "denominator": 2.635}
    records = [
        {"id": "r1", "branch": "p1", "parent_id": "parentA", "parent_score": .4,
         "score": .6, "actual_delta": .2, "status": "evaluated", "metric": metric,
         "parent_cases": {"sum_radii": 1.054},
         "evaluation": {"score": .6, "cases": {"sum_radii": 1.581,
                         "opaque": {"observation_ref": "not-a-reference"}}},
         "free_text": '{"observation_ref":"missing"}'},
        {"id": "r2", "branch": "p1", "parent_id": "parentB", "parent_score": .8,
         "score": .6, "actual_delta": -.2, "status": "evaluated", "metric": metric,
         "parent_cases": {"sum_radii": 2.108},
         "evaluation": {"score": .6, "cases": {"sum_radii": 1.581}}},
        {"id": "r3", "status": "eval_failed", "score": None, "actual_delta": None},
        {"id": "r4", "status": "evaluated", "score": .6, "actual_delta": 0.,
         "metric": None, "parent_cases": None, "evaluation": None},
    ]
    origins = [
        [{"kind": "direct", "from_lesson_id": "l1", "from_rule": "Current rule"},
         {"kind": "inherited", "from_lesson_id": "old", "from_rule": 'Use {"observation_ref":"r2"} literally'}],
        [{"kind": "inherited", "from_lesson_id": "old", "from_rule": "Old rule"}],
        [{"kind": "direct", "from_lesson_id": "l1", "from_rule": "Current rule"}],
        [{"kind": "direct", "from_lesson_id": "l1", "from_rule": "Current rule"}],
    ]
    lessons = [
        {"id": "l1", "rule": "Current rule", "scope": "branch", "branch": "p1",
         "polarity": "positive", "evidence_ids": ["r1"], "retirement_eligible": True,
         "focus_for_consolidation": False,
         "observed_exposures": {
             "positive": 1, "negative": 1, "zero": 1, "unmeasured": 1, "total": 4,
             "shown_ids": ["r1", "r2", "r3", "r4"], "omitted_count": 0,
             "representatives": [{"experiment_id": r["id"], "attribution": origin}
                                 for r, origin in zip(records, origins)]}},
        {"id": "l2", "rule": "A different baseline", "scope": "branch", "branch": "p1",
         "polarity": "negative", "evidence_ids": ["r2"]},
    ]
    return records, lessons


def test_roundtrip_preserves_exact_inline_view_catalog_and_input(data):
    records, lessons = data
    before = copy.deepcopy(data)
    inline = reflection_inline_payload(*data)
    compact = reflection_payload(*data)
    assert inline["schema"] == REFLECTION_INLINE_SCHEMA == "sciel-reflection-presentation/9-inline"
    assert compact["schema"] == REFLECTION_SCHEMA == "sciel-reflection-presentation/9"
    assert compact["observation_fact_fields"] == FACT_FIELDS
    assert compact["expanded_sha256"] == digest(inline)
    assert compact["observation_catalog"] == {r["id"]: r for r in records}
    decoded = expand_reflection_payload(json.loads(canonical(compact)))
    assert decoded == inline
    assert data == before
    decoded["observation_catalog"]["r1"]["metric"]["units"] = "changed"
    decoded["lessons"][0]["source_support"]["observations"][0]["metric"].clear()
    assert data == before
    assert expand_reflection_payload(compact) == inline


def test_references_keep_sources_exposures_and_inherited_origins_distinct(data):
    compact = reflection_payload(*data)
    lesson = compact["lessons"][0]
    assert lesson["source_support"]["observations"] == [{"observation_ref": "r1"}]
    reps = lesson["advice_exposures"]["representatives"]
    assert [r["observation"] for r in reps] == [{"observation_ref": f"r{i}"} for i in range(1, 5)]
    assert [r["also_source_support"] for r in reps] == [True, False, False, False]
    assert [a["kind"] for a in reps[0]["attribution"]] == ["direct", "inherited"]
    expanded = expand_reflection_payload(compact)
    r1 = expanded["lessons"][0]["source_support"]["observations"][0]
    r2 = expanded["lessons"][1]["source_support"]["observations"][0]
    assert (r1["score"], r2["score"]) == (.6, .6)
    assert (r1["parent_score"], r2["parent_score"]) == (.4, .8)
    assert (r1["actual_delta"], r2["actual_delta"]) == (.2, -.2)
    assert r1["parent_cases"] != r2["parent_cases"]


def test_missing_null_legacy_facts_and_opaque_reference_lookalikes(data):
    compact = reflection_payload(*data)
    expanded = expand_reflection_payload(compact)
    assert "metric" not in expanded["observation_catalog"]["r3"]
    assert expanded["observation_catalog"]["r4"]["metric"] is None
    reps = expanded["lessons"][0]["advice_exposures"]["representatives"]
    for r in reps[2:]:
        assert all(r["observation"][key] is None for key in ("metric", "parent_cases", "evaluation"))
    assert reps[2]["observation"]["actual_delta"] is None
    assert reps[3]["observation"]["actual_delta"] == 0.
    assert expanded["observation_catalog"]["r1"]["free_text"] == '{"observation_ref":"missing"}'
    assert reps[0]["observation"]["evaluation"]["cases"]["opaque"] == {"observation_ref": "not-a-reference"}
    assert reps[0]["attribution"][1]["from_rule"] == 'Use {"observation_ref":"r2"} literally'
    unknown = expanded["lessons"][1]["advice_exposures"]
    assert unknown == dict(available=False, retained_unique_counts=None, shown_ids=[],
                           omitted_count=None, representatives=[])


def test_no_record_merging_even_when_all_facts_except_id_equal(data):
    records, lessons = data
    records.append({**copy.deepcopy(records[0]), "id": "r5"})
    lessons[0]["evidence_ids"].append("r5")
    compact = reflection_payload(*data)
    assert set(compact["observation_catalog"]) == {"r1", "r2", "r3", "r4", "r5"}
    refs = compact["lessons"][0]["source_support"]["observations"]
    assert refs == [{"observation_ref": "r1"}, {"observation_ref": "r5"}]
    assert expand_reflection_payload(compact) == reflection_inline_payload(*data)


@pytest.mark.parametrize("kind", ["record", "lesson", "source", "shown"])
def test_duplicate_ids_rejected_even_when_values_identical(data, kind):
    records, lessons = data
    if kind == "record": records.append(copy.deepcopy(records[0]))
    elif kind == "lesson": lessons.append(copy.deepcopy(lessons[0]))
    elif kind == "source": lessons[0]["evidence_ids"].append("r1")
    else:
        info = lessons[0]["observed_exposures"]
        info["shown_ids"].append("r1")
        info["representatives"].append(copy.deepcopy(info["representatives"][0]))
    before = copy.deepcopy(data)
    with pytest.raises(ValueError): reflection_payload(*data)
    assert data == before


@pytest.mark.parametrize("case", [
    "extra_top", "extra_lesson", "extra_source", "extra_exposure", "extra_representative",
    "extra_reference", "unknown_schema", "wrong_fields", "catalog_id", "missing_catalog",
    "source_reference", "source_reordered_membership", "missing_reference", "shown_reference",
    "shown_membership", "source_overlap", "duplicate_lesson", "duplicate_source",
    "duplicate_shown", "invented_available", "wrong_count", "source_omission",
    "inline_instead_of_reference", "digest", "catalog_fact", "attribution_text",
])
def test_tampered_transport_rejected_atomically(data, case):
    compact = reflection_payload(*data)
    lesson = compact["lessons"][0]
    source = lesson["source_support"]
    exposures = lesson["advice_exposures"]
    rep = exposures["representatives"][0]
    if case == "extra_top": compact["extra"] = 1
    elif case == "extra_lesson": lesson["extra"] = 1
    elif case == "extra_source": source["extra"] = 1
    elif case == "extra_exposure": exposures["extra"] = 1
    elif case == "extra_representative": rep["extra"] = 1
    elif case == "extra_reference": source["observations"][0]["extra"] = 1
    elif case == "unknown_schema": compact["schema"] = "future/9"
    elif case == "wrong_fields": compact["observation_fact_fields"].pop()
    elif case == "catalog_id": compact["observation_catalog"]["r2"]["id"] = "r1"
    elif case == "missing_catalog": del compact["observation_catalog"]["r1"]
    elif case == "source_reference": source["observations"][0]["observation_ref"] = "r2"
    elif case == "source_reordered_membership": source["evidence_ids"] = ["r2"]
    elif case == "missing_reference": source["observations"] = []
    elif case == "shown_reference": rep["observation"]["observation_ref"] = "r2"
    elif case == "shown_membership": exposures["shown_ids"] = ["r2", "r1", "r3", "r4"]
    elif case == "source_overlap": rep["also_source_support"] = False
    elif case == "duplicate_lesson": compact["lessons"].append(copy.deepcopy(lesson))
    elif case == "duplicate_source": source["evidence_ids"].append("r1")
    elif case == "duplicate_shown": exposures["shown_ids"][1] = "r1"
    elif case == "invented_available": exposures["available"] = False
    elif case == "wrong_count": exposures["omitted_count"] = 1
    elif case == "source_omission": source["omitted_count"] = 1
    elif case == "inline_instead_of_reference": source["observations"][0] = {"id": "r1"}
    elif case == "digest": compact["expanded_sha256"] = "0" * 64
    elif case == "catalog_fact": compact["observation_catalog"]["r1"]["actual_delta"] = -.2
    elif case == "attribution_text": rep["attribution"][1]["from_rule"] = "Wrong old wording"
    before = copy.deepcopy(compact)
    with pytest.raises(ValueError): expand_reflection_payload(compact)
    assert compact == before


def test_reference_membership_guard_does_not_rely_on_digest(data):
    compact = reflection_payload(*data)
    inline = reflection_inline_payload(*data)
    # A caller can compute hashes, but still cannot redirect a reference while
    # retaining the original source evidence ID.
    compact["lessons"][0]["source_support"]["observations"] = [{"observation_ref": "r2"}]
    inline["lessons"][0]["source_support"]["observations"] = [
        {key: copy.deepcopy(data[0][1].get(key)) for key in FACT_FIELDS}]
    compact["expanded_sha256"] = digest(inline)
    with pytest.raises(ValueError, match="reference differs from membership"):
        expand_reflection_payload(compact)


def test_source_exposure_overlap_expands_without_mutating_input():
    from research_memory_fixtures import overlapping_reflection_history
    original = overlapping_reflection_history()
    records, lessons = original["experiments"], original["lessons"]
    before = copy.deepcopy(original)
    inline = reflection_inline_payload(records, lessons)
    compact = reflection_payload(records, lessons)
    assert expand_reflection_payload(compact) == inline
    assert len(canonical(compact)) < len(canonical(inline))
    lesson, = expand_reflection_payload(compact)["lessons"]
    assert lesson["source_support"]["evidence_ids"] == ["r000002a1", "r000004a1"]
    reps = {r["observation"]["id"]: r for r in lesson["advice_exposures"]["representatives"]}
    assert not reps["r000005a1"]["also_source_support"]
    assert not reps["r000006a1"]["also_source_support"]
    assert reps["r000004a1"]["also_source_support"]
    assert all(r["metric"] is None for r in lesson["source_support"]["observations"])
    assert original == before


def test_empty_view_roundtrip_and_nonfinite_json_rejection(data):
    assert expand_reflection_payload(reflection_payload([], [])) == reflection_inline_payload([], [])
    data[0][0]["score"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        reflection_payload(*data)


def test_outer_measurement_catalog_and_inner_fact_refs_expand_in_defined_order(data):
    from sciel.knowledge_base.catalogs import expand_measurements, pack_measurements

    # Repeated large metric facts make cataloging worthwhile; only explicitly
    # enumerated measurement positions may be factored by the outer codec.
    data[0][0]["metric"]["meaning"] = "Explicit measured units, not causal support. " * 40
    compact = reflection_payload(*data)
    paths = [("observation_catalog", record["id"], "metric") for record in data[0]]
    outer = pack_measurements(compact, paths)
    assert len(outer["bindings"]) == 2
    assert outer["payload"]["lessons"][0]["source_support"]["observations"] == [
        {"observation_ref": "r1"}]
    expanded = expand_reflection_payload(expand_measurements(outer))
    assert canonical(expanded) == canonical(reflection_inline_payload(*data))
    assert expanded["observation_catalog"]["r1"]["evaluation"]["cases"]["opaque"] == {
        "observation_ref": "not-a-reference"}
