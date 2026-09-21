"""Unverified mixed-result reviews retain exact saved facts without new runs."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.hypothesis import comparison_review as review
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.paired_comparisons import build_comparison, UNKNOWN_RECEIPT
from tests.unit.test_paired_comparisons import inputs, receipt, SOURCE_A, SOURCE_B


WRITER = {"phase": "reflection", "reference": 2, "raw_response_sha256": "d" * 64}


def group(index=1, *, branch="p1"):
    data = inputs(new=(.11, .09, .1))
    data.update(batch_id=f"b{index:06d}", branch=branch)
    return build_comparison(**data)


def proposal(g, *, hypothesis=True):
    return {"comparison_id": g["id"], "comparison_sha256": canonical_sha(g),
        "interpretation": "The mixed result does not justify advancing this candidate.",
        "limitations": "Three shared seeds do not isolate the mechanism or prove future quality.",
        "next_hypothesis": {"change": "Change only one reinsertion order.",
            "expected_observation": "An order effect may change paired signs.",
            "falsification": "No change in the predicted paired directions."} if hypothesis else None}


def cards_for(groups):
    return review.build_cards([proposal(g) for g in groups], groups, WRITER)


def test_actual_saved_mixed_group_keeps_all_raw_exact_disagreements_and_is_detached():
    from review_evidence_fixture import mixed_group
    g = mixed_group()
    original = copy.deepcopy(g)
    proposals = [proposal(g)]
    cards = review.build_cards(proposals, [g], WRITER)
    card = cards[g["id"]]
    facts = card["host_facts"]
    assert facts["derived"]["outcome"] == "mixed"
    assert facts["derived"]["all_valid"] is True
    assert facts["derived"]["all_no_loss"] is False
    assert [p["no_loss"] for p in facts["measured_no_loss"]] == [False, False, False]
    assert [(p["raw_physical_sum_direction"], p["exact_certified_sum_direction"])
            for p in facts["signs"]["pairs"]] == [
                ("negative", "positive"), ("positive", "negative"), ("negative", "negative")]
    assert card["seeds"] == [541090050, 3513246259, 4186712104]
    assert card["origin_experiment_id"] == "r000002a1"
    assert facts["signs"]["origin_record_present"] is False
    assert review.validate_cards(cards, [g]) == cards
    context = review.context(cards, [g], g["branch"], 10000)
    assert context["cards"] == [card]
    assert context["rendered_chars"] == len(canonical_json(context))
    card["host_facts"]["derived"]["pairs"][0]["raw_delta"] = 9
    card["unverified_interpretation"]["next_hypothesis"]["change"] = "mutated"
    assert proposals[0]["next_hypothesis"]["change"] != "mutated"
    assert g == original


@pytest.mark.parametrize("missing", [None, UNKNOWN_RECEIPT, "candidate_failure", "infrastructure"])
def test_unavailable_quality_is_unknown_not_zero_or_a_measured_loss(missing):
    data = inputs()
    data["pairs"][1]["challenger"] = (receipt(SOURCE_B, 12, status=missing)
        if type(missing) is str else copy.deepcopy(missing))
    g = build_comparison(**data)
    card = cards_for([g])[g["id"]]
    facts = card["host_facts"]
    assert facts["derived"]["outcome"] == "incomplete"
    assert facts["derived"]["all_no_loss"] is False
    assert facts["measured_no_loss"] == [
        {"seed": 11, "no_loss": True}, {"seed": 12, "no_loss": None}, {"seed": 13, "no_loss": True}]
    assert facts["signs"]["pairs"][1] == {"seed": 12,
        "raw_physical_sum_direction": "unknown", "exact_certified_sum_direction": "unknown"}
    assert facts["derived"]["pairs"][1]["raw_delta"] is None
    assert facts["derived"]["pairs"][1]["certified_delta"] is None


def test_exact_zero_remains_distinct_from_unknown_and_sign_disagreement():
    data = inputs(old=(.1, .1, .1), new=(.1, .1, .1))
    data["pairs"][1]["comparator"] = receipt(SOURCE_A, 12, .10000005, x=.1)
    data["pairs"][1]["challenger"] = receipt(SOURCE_B, 12, .10000002)
    data["pairs"][2]["challenger"] = None
    g = build_comparison(**data)
    facts = cards_for([g])[g["id"]]["host_facts"]
    assert [(p["raw_physical_sum_direction"], p["exact_certified_sum_direction"])
            for p in facts["signs"]["pairs"]] == [
                ("zero", "zero"), ("negative", "positive"), ("unknown", "unknown")]
    assert [p["no_loss"] for p in facts["measured_no_loss"]] == [True, False, None]


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(host_facts={}),
    lambda p: p.pop("limitations"),
    lambda p: p.update(comparison_id="white space"),
    lambda p: p.update(comparison_id="x" * 129),
    lambda p: p.update(comparison_id="bad\x7f"),
    lambda p: p.update(comparison_sha256="D" * 64),
    lambda p: p.update(interpretation=" \n\t"),
    lambda p: p.update(interpretation="x" * 2049),
    lambda p: p.update(limitations=False),
    lambda p: p["next_hypothesis"].update(change="x" * 1025),
    lambda p: p["next_hypothesis"].update(falsification=""),
    lambda p: p["next_hypothesis"].update(fact="not admitted"),
])
def test_exact_proposal_contract_rejects_model_authored_facts_and_malformed_text(mutate):
    p = proposal(group())
    mutate(p)
    with pytest.raises(ValueError):
        review.validate_proposals([p])


def test_proposal_limits_canonical_escaped_size_duplicates_and_detachment():
    g = group()
    p = proposal(g, hypothesis=False)
    checked = review.validate_proposals([p])
    assert checked == [p]
    checked[0]["interpretation"] = "mutated"
    assert p["interpretation"] != "mutated"
    with pytest.raises(ValueError, match="duplicate"):
        review.validate_proposals([p, p])
    with pytest.raises(ValueError, match="four"):
        review.validate_proposals([p] * 5)
    p["interpretation"] = p["limitations"] = "\u2603" * 2048
    with pytest.raises(ValueError, match="character limit"):
        review.validate_proposals([p])
    for bad in ({}, (), None):
        with pytest.raises(ValueError):
            review.validate_proposals(bad)
    assert review.validate_proposals([]) == []


@pytest.mark.parametrize("mutate", [
    lambda w: w.update(phase="conclusion"),
    lambda w: w.update(reference=0),
    lambda w: w.update(reference=True),
    lambda w: w.update(raw_response_sha256="broken"),
    lambda w: w.update(verified=True),
])
def test_writer_provenance_has_exact_shape_and_positive_reference(mutate):
    g = group()
    writer = copy.deepcopy(WRITER)
    mutate(writer)
    with pytest.raises(ValueError, match="writer"):
        review.build_cards([proposal(g)], [g], writer)


def test_binding_rejects_stale_digest_derived_claims_full_geometry_and_duplicate_groups():
    g = group()
    p = proposal(g)
    stale = copy.deepcopy(g)
    stale["provenance"]["extra"] = "same comparison ID, changed authoritative bytes"
    with pytest.raises(ValueError, match="digest mismatch"):
        review.build_cards([p], [stale], WRITER)
    with pytest.raises(ValueError, match="unknown group"):
        review.build_cards([p], [], WRITER)
    with pytest.raises(ValueError, match="duplicate authoritative"):
        review.build_cards([p], [g, g], WRITER)
    for change in ("derived", "geometry"):
        forged = copy.deepcopy(g)
        if change == "derived":
            forged["derived"]["all_no_loss"] = True
        else:
            forged["pairs"][0]["challenger"]["capture"]["radii"][0] = 0.9
        with pytest.raises(ValueError):
            review.build_cards([proposal(forged)], [forged], WRITER)


@pytest.mark.parametrize("mutate", [
    lambda c: c.update(schema="wrong"),
    lambda c: c.update(challenger_source_sha256="e" * 64),
    lambda c: c.update(branch="other"),
    lambda c: c["seeds"].reverse(),
    lambda c: c["host_facts"]["derived"].update(all_no_loss=True),
    lambda c: c["host_facts"]["signs"]["pairs"][1].update(raw_physical_sum_direction="positive"),
    lambda c: c["host_facts"]["measured_no_loss"][1].update(no_loss=True),
    lambda c: c.update(accepted_lesson_ids=["invented"]),
    lambda c: c["unverified_interpretation"].update(confidence=1.0),
])
def test_retained_card_facts_and_exact_shape_rebuild_from_authoritative_groups(mutate):
    g = group()
    cards = cards_for([g])
    mutate(cards[g["id"]])
    with pytest.raises(ValueError):
        review.validate_cards(cards, [g])


def test_retention_limit_does_not_silently_drop_or_reorder_cards():
    groups = [group(i) for i in range(17)]
    cards = {}
    for g in groups:
        cards.update(cards_for([g]))
    first16 = dict(list(cards.items())[:16])
    assert review.validate_cards(first16, groups) == first16
    with pytest.raises(ValueError, match="retained card limit"):
        review.validate_cards(cards, groups)


def test_whole_context_order_scope_item_budget_disabled_and_exact_rederivation():
    groups = [group(1), group(2, branch="other"), group(3)]
    cards = cards_for(groups)
    # Mapping insertion order cannot override authoritative group chronology.
    cards = dict(reversed(list(cards.items())))
    view = review.context(cards, groups, "p1", 10000, max_items=1)
    assert [card["comparison_id"] for card in view["cards"]] == [groups[2]["id"]]
    assert view["retained_count"] == 2 and view["presented_count"] == 1
    assert view["omitted_count"] == 1 and view["omission_details_omitted_count"] == 0
    assert view["omitted"][0]["comparison_id"] == groups[0]["id"]
    assert view["omitted"][0]["reason"] == "item_limit"
    assert view["rendered_chars"] == len(canonical_json(view)) <= 10000
    assert review.validate_context(view, cards, groups, "p1") == view
    disabled = review.context(cards, groups, "p1", 10000, enabled=False)
    assert disabled["cards"] == [] and disabled["omitted_count"] == 2
    assert {row["reason"] for row in disabled["omitted"]} == {"disabled"}
    assert review.validate_context(disabled, cards, groups, "p1", enabled=False) == disabled
    with pytest.raises(ValueError):
        review.validate_context(disabled, cards, groups, "p1")


def test_memory_boundary_retains_whole_cards_or_explicit_omission_and_counts():
    g = group()
    cards = cards_for([g])
    full = review.context(cards, [g], "p1", 10000)
    boundary = review.context(cards, [g], "p1", full["rendered_chars"])
    assert boundary["cards"] == full["cards"]
    smaller = review.context(cards, [g], "p1", boundary["rendered_chars"] - 1)
    assert smaller["cards"] == [] and smaller["omitted_count"] == 1
    assert smaller["omitted"][0]["reason"] == "memory_char_limit"
    metadata = review.context({}, [], "p1", 10000)
    assert metadata is not None and metadata["cards"] == [] and metadata["omitted_count"] == 0
    # Find a metadata-only budget that cannot fit its omission detail; counts
    # must still distinguish a retained review from an absent review.
    minimal = metadata["rendered_chars"]
    tight = review.context(cards, [g], "p1", minimal)
    assert tight is not None and tight["cards"] == [] and tight["omitted"] == []
    assert tight["omitted_count"] == tight["omission_details_omitted_count"] == 1
    assert tight["rendered_chars"] <= minimal
    assert review.context(cards, [g], "p1", 0) is None


@pytest.mark.parametrize("key,value", [
    ("max_chars", True), ("max_chars", -1), ("max_chars", 10001),
    ("max_items", True), ("max_items", -1), ("max_items", 5), ("enabled", 1),
])
def test_context_bounds_are_strict_types_and_cannot_expand_production_caps(key, value):
    args = {"max_chars": 10000, "max_items": 4, "enabled": True, key: value}
    with pytest.raises(ValueError):
        review.context({}, [], "p1", **args)


def test_prompt_context_rejects_prose_or_host_fact_tampering_and_detaches_results():
    g = group()
    cards = cards_for([g])
    view = review.context(cards, [g], "p1", 10000)
    for location in ("text", "fact", "metadata"):
        forged = copy.deepcopy(view)
        if location == "text":
            forged["cards"][0]["unverified_interpretation"]["interpretation"] = "new assertion"
        elif location == "fact":
            forged["cards"][0]["host_facts"]["derived"]["outcome"] = "improved"
        else:
            forged["omitted_count"] = 20
        with pytest.raises(ValueError):
            review.validate_context(forged, cards, [g], "p1")
    restored = review.validate_context(view, cards, [g], "p1")
    restored["cards"][0]["writer"]["reference"] = 9
    assert view["cards"][0]["writer"]["reference"] == 2
    assert cards[g["id"]]["writer"]["reference"] == 2


def test_identity_binds_contract_source_bytes_and_rules_keep_review_distinct_from_lessons():
    identity = review.identity()
    assert identity["source_sha256"] == hashlib.sha256(Path(review.__file__).read_bytes()).hexdigest()
    assert identity["max_stored"] == 16 and identity["max_context_chars"] == 10000
    for phrase in ("unverified", "not instructions", "accepted lessons", "causal proof", "Unknown"):
        assert phrase in review.RULES
    identity["max_stored"] = 99
    assert review.identity()["max_stored"] == 16
