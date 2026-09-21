"""Portable actual r2 evidence reproduces the raw/certified sign disagreement."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.hypothesis.prompts import reflection_prompt
from sciel.hypothesis.rendering import decode_evidence, presentation_identity, serialize_reflection
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.comparison_facts import (
    SCHEMA, comparison_facts, bounded_comparison_facts, validate_comparison_facts, identity)
from sciel.knowledge_base.presentation import (
    reflection_payload, expand_reflection_payload, reflection_inline_payload)

FIXTURE = Path(__file__).parents[1] / "fixtures/restart_r2_scope_signs.json"


def actual_records():
    fixture = json.loads(FIXTURE.read_text())
    for record in fixture["records"]:
        assert canonical_sha(record) == fixture["provenance"]["record_sha256"][record["id"]]
    return fixture["records"]


def rehash_card(card):
    card["view_sha256"] = canonical_sha({k: v for k, v in card.items() if k != "view_sha256"})


def test_actual_r2_third_raw_negative_certified_positive_and_native_scope_distinct():
    records = actual_records()
    before = copy.deepcopy(records)
    view = comparison_facts(records)
    group, = view["paired"].values()
    assert group["origin_experiment_id"] == "r000001a1"
    assert group["origin_record_present"] is True
    assert group["scope"] == "paired_shared_seed_group"
    assert group["quality_outcome"] == "mixed"
    assert [(p["raw_physical_sum_direction"], p["exact_certified_sum_direction"])
            for p in group["pairs"]] == [("positive", "positive"),
                                        ("positive", "positive"), ("negative", "positive")]
    third = records[0]["derived"]["pairs"][2]
    assert third["certified_delta"] == {"numerator": 141443671857,
                                         "denominator": 36028797018963968}
    assert third["raw_delta"] == -6.039774023136601e-7
    native = view["ordinary"]["r000001a1"]
    assert native == {"scope": "ordinary_controller_evaluation",
                      "controller_score_direction": "positive", "forecast_position": "above"}
    assert view["rendered_chars"] == len(canonical_json(view))
    assert view["omitted_ids"] == [] and view["max_chars"] is None
    assert records == before


@pytest.mark.parametrize("numerator,expected", [(1, "positive"), (-1, "negative"), (0, "zero")])
def test_certified_direction_uses_exact_ratio_without_float_underflow(numerator, expected):
    records = actual_records()
    card = records[0]
    card["derived"]["pairs"][0]["certified_delta"] = {
        "numerator": numerator, "denominator": 10 ** 400}
    rehash_card(card)
    row = next(iter(comparison_facts(records)["paired"].values()))["pairs"][0]
    assert row["exact_certified_sum_direction"] == expected


@pytest.mark.parametrize("field,value", [("numerator", True), ("denominator", 0),
                                        ("denominator", -1), ("numerator", 1.0)])
def test_malformed_exact_ratio_is_not_reinterpreted(field, value):
    records = actual_records()
    records[0]["derived"]["pairs"][0]["certified_delta"][field] = value
    rehash_card(records[0])
    with pytest.raises(ValueError, match="exact certified ratio"):
        comparison_facts(records)


def test_missing_failed_measurements_are_unknown_and_missing_origin_not_invented():
    card = actual_records()[0]
    pair = card["derived"]["pairs"][0]
    pair.update(comparator_status="unknown", raw_delta=None, certified_delta=None)
    rehash_card(card)
    view = comparison_facts([card])
    group, = view["paired"].values()
    assert group["origin_record_present"] is False and view["ordinary"] == {}
    assert group["pairs"][0]["raw_physical_sum_direction"] == "unknown"
    assert group["pairs"][0]["exact_certified_sum_direction"] == "unknown"


def test_bounded_projection_retains_whole_group_with_origin_or_reports_omission():
    records = actual_records()
    view = bounded_comparison_facts(records, max_chars=6000)
    assert view["omitted_ids"] == [] and view["max_chars"] == 6000
    cap = view["rendered_chars"]
    boundary = bounded_comparison_facts(records, max_chars=cap)
    assert boundary["paired"] and boundary["ordinary"]
    smaller = bounded_comparison_facts(records, max_chars=cap - 1)
    assert smaller is not None and not smaller["paired"]
    assert records[0]["id"] in smaller["omitted_ids"]
    assert smaller["rendered_chars"] <= cap - 1
    assert bounded_comparison_facts(records, max_chars=1) is None
    assert validate_comparison_facts(view, records=records, max_chars=6000) == view
    assert validate_comparison_facts(None, records=records, max_chars=1) is None
    for bad in (True, -1, 1.0):
        with pytest.raises(ValueError):
            bounded_comparison_facts(records, max_chars=bad)


def test_forged_direction_and_scope_link_rejected_even_with_recomputed_outer_hash():
    records = actual_records()
    compact = reflection_payload(records, [])
    inline = reflection_inline_payload(records, [])
    ident = records[0]["id"]
    for change in ("direction", "scope"):
        forged = copy.deepcopy(compact)
        if change == "direction":
            forged["comparison_facts"]["paired"][ident]["pairs"][2]["exact_certified_sum_direction"] = "negative"
        else:
            forged["comparison_facts"]["ordinary"]["r000001a1"]["scope"] = "paired_shared_seed_group"
        fake_inline = copy.deepcopy(inline)
        fake_inline["comparison_facts"] = forged["comparison_facts"]
        forged["expanded_sha256"] = canonical_sha(fake_inline)
        with pytest.raises(ValueError, match="comparison facts differ"):
            expand_reflection_payload(forged)


def test_real_reflection_prompt_includes_exact_projection_and_does_not_rewrite_records():
    records = actual_records()
    original = FIXTURE.read_bytes()
    before = copy.deepcopy(records)
    prompt = reflection_prompt(records, [], structured=True)
    wire = decode_evidence(json.loads(prompt.split("EVIDENCE_JSON:\n", 1)[1]))
    assert wire["schema"] == "sciel-reflection-presentation/9"
    assert wire["comparison_facts"] == comparison_facts(records)
    assert expand_reflection_payload(wire)["comparison_facts"] == comparison_facts(records)
    assert wire["observation_catalog"] == {r["id"]: r for r in records}
    assert "not the same measurement" in prompt
    assert "raw_physical_sum_direction" in prompt and "exact_certified_sum_direction" in prompt
    assert records == before and FIXTURE.read_bytes() == original
    assert serialize_reflection(records, []) == prompt.split("EVIDENCE_JSON:\n", 1)[1].strip()


def test_identity_binds_projection_schema_and_source_bytes():
    from sciel.knowledge_base import comparison_facts as module
    bound = identity()
    assert bound["schema"] == SCHEMA
    assert bound["source_sha256"] == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    assert presentation_identity()["comparison_facts"] == bound
    bound["schema"] = "mutated"
    assert identity()["schema"] == SCHEMA
