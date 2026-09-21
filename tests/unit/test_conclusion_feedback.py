"""Reviews can inform later generation without becoming observations or proof."""
import copy
import json

import pytest

from sciel.hypothesis.conclusion_feedback import (
    CARD_SCHEMA, MAX_CARD_CHARS, SCHEMA, conclusion_feedback, validate_conclusions,
)
from sciel.hypothesis.observed_facts import observed_comparison
from sciel.hypothesis.parsing import parse_conclusion
from sciel.knowledge_base.base import KnowledgeBase
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha


EXPLANATION = ("The same translation preserves the pair's relative vector in real arithmetic. "
               "Its effect on other contacts and the later solver is unverified.")


def evidence(*, count=1):
    kb = KnowledgeBase()
    for n in range(1, count + 1):
        kb.record(dict(id=f"r{n}", branch="p1", status="evaluated", parent_score=1.,
            score=.75, actual_delta=-.25, source_sha256="b" * 64,
            proposed_sha256="b" * 64, parent_source_sha256="a" * 64,
            hypothesis=dict(text="Common translation will reduce the internal gap.",
                predicted_low=.1, predicted_high=.2, reasoning="Unsupported mechanism claim.", confidence=.7)))
    return kb._evidence()


def review(record, *, status="parsed", explanation=EXPLANATION, verdict="REFUTED", lessons=None):
    raw = dict(verdict=verdict, explanation=explanation, lessons=[] if lessons is None else lessons)
    parsed = parse_conclusion(json.dumps(raw, ensure_ascii=False)) if status == "parsed" else None
    assert parsed is not None or status != "parsed"
    return dict(status=status, llm=parsed.to_dict() if parsed else None,
        observed_forecast_verdict=record["forecast_verdict"], observed_comparison=observed_comparison(record),
        verdict_matches_observation=(verdict == record["forecast_verdict"]) if parsed else None,
        accepted_lesson_ids=["l000001"] * len(raw["lessons"]) if parsed else [],
        lesson_error=None, lesson_error_detail=None)


def inputs(count=1):
    rows = evidence(count=count)
    return {ident: review(row) for ident, row in rows.items()}, rows


def test_zero_lesson_review_is_source_bound_but_not_new_quality_or_semantic_proof():
    reviews, rows = inputs()
    before = copy.deepcopy((reviews, rows))
    result = conclusion_feedback(reviews, rows, "p1")
    card = result["cards"][0]
    assert result["schema"] == SCHEMA and card["schema"] == CARD_SCHEMA
    assert card["experiment_id"] == "r1" and card["record_sha256"] == canonical_sha(rows["r1"])
    assert card["parent_source_sha256"] == "a" * 64
    assert card["proposed_sha256"] == card["source_sha256"] == "b" * 64
    assert card["original_hypothesis"] == rows["r1"]["hypothesis"]
    assert card["unverified_interpretation"]["explanation"] == EXPLANATION
    assert card["lesson_admission"]["accepted_lesson_ids"] == []
    assert card["observed_comparison"]["measurement"]["actual_delta"] == -.25
    assert card["observed_forecast_verdict"] == "REFUTED"
    assert card["status"] == "evaluated" and card["metric"] is None
    assert not {"evaluation", "code_fragment", "native_geometry", "injected_ids", "confidence"} & card.keys()
    assert (reviews, rows) == before
    card["original_hypothesis"]["reasoning"] = "mutated detached view"
    assert (reviews, rows) == before


def test_disagreeing_or_semantically_wrong_llm_review_is_retained_as_unverified():
    reviews, rows = inputs()
    reviews["r1"] = review(rows["r1"], verdict="CONFIRMED", explanation="This score proved every packing is optimal.")
    result = conclusion_feedback(reviews, rows, "p1")["cards"][0]
    assert result["unverified_interpretation"]["llm_verdict"] == "CONFIRMED"
    assert result["unverified_interpretation"]["verdict_matches_observation"] is False
    assert result["observed_forecast_verdict"] == "REFUTED"
    assert "proved every" in result["unverified_interpretation"]["explanation"]


@pytest.mark.parametrize("status", ["parsed", "skipped", "invalid"])
def test_all_live_statuses_and_validation_returns_detached_copy(status):
    rows = evidence()
    reviews = {"r1": review(rows["r1"], status=status)}
    result = validate_conclusions(reviews, rows)
    assert result == reviews and result is not reviews and result["r1"] is not reviews["r1"]
    block = conclusion_feedback(reviews, rows, "p1")
    assert (block is not None) == (status == "parsed")


@pytest.mark.parametrize("change", [
    {"status": "complete"}, {"extra": True}, {"llm": None}, {"llm": []},
    {"observed_forecast_verdict": "CONFIRMED"}, {"verdict_matches_observation": 1},
    {"accepted_lesson_ids": ["l000001"]}, {"accepted_lesson_ids": [True]},
    {"accepted_lesson_ids": ["bad id"]}, {"lesson_error_detail": "unbound error"},
    {"lesson_error": "RuntimeError", "lesson_error_detail": "stopped"},
    {"lesson_error": "ValueError", "lesson_error_detail": "x" * 1025},
])
def test_tampered_conclusion_shape_or_observation_binding_rejected(change):
    reviews, rows = inputs()
    reviews["r1"].update(change)
    with pytest.raises(ValueError):
        validate_conclusions(reviews, rows)


@pytest.mark.parametrize("path,value", [
    (("measurement", "eligible"), 1), (("measurement", "actual_delta"), -.2),
    (("forecast", "predicted_low"), .01), (("forecast", "supplied_verdict"), "CONFIRMED"),
])
def test_rederived_observed_comparison_rejects_bool_alias_and_value_tamper(path, value):
    reviews, rows = inputs()
    reviews["r1"]["observed_comparison"][path[0]][path[1]] = value
    with pytest.raises(ValueError, match="comparison"):
        validate_conclusions(reviews, rows)


@pytest.mark.parametrize("key,value", [
    ("explanation", "x" * 4097), ("explanation", " leading or trailing "),
    ("verdict", "SUPPORTED"), ("lessons", [dict(rule="r", polarity="positive", scope="global", evidence_ids=[])]),
    ("extra", "field"),
])
def test_llm_payload_reuses_existing_strict_bounded_parser(key, value):
    reviews, rows = inputs()
    reviews["r1"]["llm"][key] = value
    with pytest.raises(ValueError):
        validate_conclusions(reviews, rows)


@pytest.mark.parametrize("status", ["invalid", "skipped"])
def test_unparsed_status_cannot_smuggle_explanation(status):
    reviews, rows = inputs()
    reviews["r1"]["status"] = status
    with pytest.raises(ValueError, match="unparsed"):
        validate_conclusions(reviews, rows)


def test_retired_and_coalesced_lesson_ids_are_not_reinterpreted_as_new_evidence():
    rows = evidence()
    lesson = dict(rule="One observed loss.", polarity="negative", scope="branch", evidence_ids=["r1"])
    reviews = {"r1": review(rows["r1"], lessons=[lesson, lesson])}
    assert validate_conclusions(reviews, rows)["r1"]["accepted_lesson_ids"] == ["l000001", "l000001"]
    reviews["r1"].update(accepted_lesson_ids=[], lesson_error="ValueError", lesson_error_detail="polarity rejected")
    card = conclusion_feedback(reviews, rows, "p1")["cards"][0]
    assert card["lesson_admission"]["error"] == "ValueError"
    assert card["unverified_interpretation"]["explanation"] == EXPLANATION


def test_unknown_record_and_paired_group_are_not_ordinary_reviews():
    reviews, rows = inputs()
    with pytest.raises(ValueError, match="unknown"):
        validate_conclusions(reviews, {})
    rows["r1"]["schema"] = "sciel-paired-circle-comparison/1"
    with pytest.raises(ValueError, match="ordinary"):
        validate_conclusions(reviews, rows)


def test_no_child_review_keeps_null_quality_and_no_invented_source():
    kb = KnowledgeBase()
    kb.record(dict(id="r1", branch="p1", status="no_child", parent_score=1.,
        parent_source_sha256="a" * 64, hypothesis=None))
    rows = kb._evidence()
    reviews = {"r1": review(rows["r1"], verdict="INCONCLUSIVE")}
    card = conclusion_feedback(reviews, rows, "p1")["cards"][0]
    assert card["source_sha256"] is card["proposed_sha256"] is None
    assert card["observed_comparison"]["measurement"]["actual_delta"] is None
    assert not card["observed_comparison"]["measurement"]["eligible"]


def test_branch_recency_and_ablation_do_not_expose_review_text_or_change_inputs():
    reviews, rows = inputs(count=4)
    rows["r4"]["branch"] = "other"
    reviews = dict(reversed(list(reviews.items())))
    before = copy.deepcopy((reviews, rows))
    block = conclusion_feedback(reviews, rows, "p1", max_items=2)
    assert [c["experiment_id"] for c in block["cards"]] == ["r3", "r2"]
    assert block["retained_count"] == 3 and block["omitted"][0]["experiment_id"] == "r1"
    disabled = conclusion_feedback(reviews, rows, "p1", enabled=False)
    assert disabled["cards"] == [] and disabled["presented_count"] == 0
    assert all(o["reason"] == "ablation" for o in disabled["omitted"])
    assert EXPLANATION not in canonical_json(disabled)
    assert conclusion_feedback(reviews, rows, "unrelated") is None
    assert (reviews, rows) == before


def test_literal_unicode_valid_review_oversize_card_is_wholly_omitted_not_sliced():
    reviews, rows = inputs(count=2)
    reviews["r2"] = review(rows["r2"], explanation="界" * 4096)
    assert validate_conclusions(reviews, rows) == reviews
    block = conclusion_feedback(reviews, rows, "p1", max_chars=32000)
    assert [c["experiment_id"] for c in block["cards"]] == ["r1"]
    omission = block["omitted"][0]
    assert omission["experiment_id"] == "r2" and omission["reason"] == "card_char_limit"
    assert omission["card_chars"] > MAX_CARD_CHARS
    assert "界" not in str(block)


def test_actual_canonical_block_budget_includes_escaping_and_omission_metadata():
    reviews, rows = inputs(count=12)
    for ident in reviews:
        reviews[ident] = review(rows[ident], explanation='"\\\n' * 400)
    for limit in [0, 1, 600, 800, 1000, 2000, 4096, 8192, 16000]:
        block = conclusion_feedback(reviews, rows, "p1", max_chars=limit)
        if block is None:
            continue
        assert len(canonical_json(block)) <= limit
        assert block["presented_count"] == len(block["cards"])
        assert block["retained_count"] == block["presented_count"] + block["omitted_count"]
        assert block["omitted_count"] == len(block["omitted"]) + block["omission_details_omitted_count"]
        for card in block["cards"]:
            assert card["unverified_interpretation"]["explanation"] == reviews[card["experiment_id"]]["llm"]["explanation"]


@pytest.mark.parametrize("kwargs", [dict(max_items=True), dict(max_chars=True), dict(enabled=1),
                                        dict(max_items=-1), dict(max_chars=-1)])
def test_limit_types_reject_bool_integer_aliases(kwargs):
    reviews, rows = inputs()
    with pytest.raises(ValueError):
        conclusion_feedback(reviews, rows, "p1", **kwargs)


def test_metric_and_source_repair_are_recorded_without_recomputing_verdict():
    reviews, rows = inputs()
    rows["r1"].update(metric={"primary_metric": "combined_score", "unit": "normalized"}, source_repaired=True,
                      forecast_verdict="INCONCLUSIVE")
    reviews["r1"] = review(rows["r1"], verdict="INCONCLUSIVE")
    card = conclusion_feedback(reviews, rows, "p1")["cards"][0]
    assert card["metric"] == rows["r1"]["metric"] and card["source_repaired"] is True
    assert card["observed_comparison"]["forecast"]["reason"] == "source_repaired"
    assert card["observed_comparison"]["measurement"]["actual_delta"] == -.25
