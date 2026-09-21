"""Research memory trusts measured evidence, not generated claims or retries."""
import copy
import hashlib
import json

import pytest

from sciel.knowledge_base import KnowledgeBase
from sciel.hypothesis.rendering import serialize_reflection


def experiment(number, *, branch="seed0", delta=.1, status="evaluated", injected=(), **changes):
    code = f"value = {number}\n"
    digest = hashlib.sha256(code.encode()).hexdigest()
    measured = status == "evaluated"
    result = {"id": f"r{number:06d}a1", "branch": branch, "parent_id": "p1",
        "parent_score": .5, "child_id": f"p{number+1}" if measured else None,
        "source_sha256": digest if measured else None,
        "proposed_sha256": digest if measured else None,
        "hypothesis": {"text": "Use smaller steps", "predicted_low": .05,
                       "predicted_high": .15, "reasoning": "Preserve the good region", "confidence": .7},
        "status": status, "actual_delta": (.5 + delta) - .5 if measured else None,
        "score": .5 + delta if measured else None, "injected_ids": list(injected),
        "code_fragment": code if measured else ""}
    result.update(changes)
    return result


def proposal(*ids, rule="Use smaller steps", polarity="positive", scope="branch"):
    return {"rule": rule, "polarity": polarity, "scope": scope, "evidence_ids": list(ids)}


def lesson(kb, *ids, **kwargs):
    return kb.add_lessons([proposal(*ids, **kwargs)], allowed_evidence_ids=list(ids))[0]


def test_forecast_is_measured_interval_check_not_claimed_causality():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    kb.record(experiment(2, delta=-.1))
    kb.record(experiment(3, source_repaired=True))
    kb.record(experiment(4, proposed_sha256="0" * 64))
    kb.record(experiment(5, hypothesis=None))
    kb.record(experiment(6, status="eval_failed"))
    assert [e["forecast_verdict"] for e in kb.experiments] == [
        "CONFIRMED", "REFUTED", "INCONCLUSIVE", "INCONCLUSIVE", "INCONCLUSIVE", "INCONCLUSIVE"]
    with pytest.raises(ValueError, match="verdict"):
        kb.record(experiment(7, delta=-.1, forecast_verdict="CONFIRMED"))
    assert len(kb.experiments) == 6
    failures = kb.failed_hypotheses("seed0")
    assert any(s.startswith("Unmeasured (eval_failed)") for s in failures)
    assert any(s.startswith("Forecast refuted") for s in failures)


@pytest.mark.parametrize("change", [
    {"score": float("nan")}, {"actual_delta": .4}, {"score": True},
    {"source_sha256": "bad"}, {"code_fragment": "different source"},
    {"status": "timeout", "score": .6, "actual_delta": .1},
    {"unexpected": "field"}, {"injected_ids": ["missing"]},
    {"generation_cost": -1}, {"generation_cost": float("inf")}, {"island": True},
    {"attempt": 0}, {"change_excerpt_may_be_truncated": 1}, {"score": 10 ** 1000},
])
def test_invalid_record_is_atomic(change):
    kb = KnowledgeBase()
    before = kb.state()
    with pytest.raises(ValueError):
        kb.record(experiment(1, **change))
    assert kb.state() == before


def test_controller_metadata_is_retained_without_requiring_a_measured_outcome():
    kb = KnowledgeBase()
    r = experiment(1, status="hypothesis_invalid", hypothesis=None,
        island=2, iteration=3, attempt=1, parent_source_sha256="a" * 64,
        operator="diff", generation_cost=.025, failure="invalid hypothesis",
        change_excerpt="@@ edit @@", change_excerpt_may_be_truncated=True)
    kb.record(r)
    saved = kb.experiments[0]
    assert saved["generation_cost"] == .025 and saved["failure"] == "invalid hypothesis"
    assert saved["actual_delta"] is saved["score"] is None
    assert saved["parent_source_sha256"] == "a" * 64


def test_lesson_evidence_must_be_allowed_measured_and_have_matching_polarity():
    kb = KnowledgeBase()
    for n, delta in [(1, .1), (2, -.1), (3, 0)]:
        kb.record(experiment(n, delta=delta))
    kb.record(experiment(4, status="eval_failed"))
    lesson(kb, "r000001a1")
    lesson(kb, "r000002a1", polarity="negative", rule="Avoid large steps")
    for eid, polarity in [("r000002a1", "positive"), ("r000001a1", "negative"),
                           ("r000003a1", "positive"), ("r000004a1", "negative")]:
        before = kb.state()
        with pytest.raises(ValueError):
            lesson(kb, eid, polarity=polarity)
        assert kb.state() == before
    with pytest.raises(ValueError, match="allowed"):
        kb.add_lessons([proposal("r000001a1")], allowed_evidence_ids=["r000002a1"])


def test_global_transfer_requires_two_branches_and_local_rules_do_not_escape():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    local = lesson(kb, "r000001a1")
    with pytest.raises(ValueError, match="two branches"):
        lesson(kb, "r000001a1", scope="global")
    assert kb.retrieve(branch="foreign", context="steps") == []
    assert kb.fragments_for("foreign") == []
    with pytest.raises(ValueError, match="foreign branch"):
        kb.record(experiment(2, branch="foreign", injected=[local]))
    kb.record(experiment(2, branch="foreign"))
    global_id = lesson(kb, "r000001a1", "r000002a1", scope="global")
    assert [x["id"] for x in kb.retrieve(branch="third", context="steps")] == [global_id]
    assert len(kb.fragments_for("third")) == 2
    with pytest.raises(ValueError, match="exactly one"):
        lesson(kb, "r000001a1", "r000002a1", scope="branch")


@pytest.mark.parametrize("polarity,delta", [("positive", .1), ("negative", -.1)])
def test_source_evidence_and_injected_usefulness_are_separate(polarity, delta):
    kb = KnowledgeBase()
    kb.record(experiment(1, delta=delta))
    lid = lesson(kb, "r000001a1", polarity=polarity)
    assert kb.lessons[0]["source_support_ids"] == ["r000001a1"]
    assert kb.lessons[0]["confidence"] == .5
    for n, status, change in [(2, "eval_failed", None), (3, "no_child", None),
                              (4, "evaluated", .2), (5, "evaluated", -.1), (6, "evaluated", 0)]:
        kb.record(experiment(n, status=status, delta=change, injected=[lid]))
    l = kb.lessons[0]
    assert l["exposure_ids"] == [f"r{n:06d}a1" for n in range(2, 7)]
    assert l["support_ids"] == ["r000004a1"] and l["refute_ids"] == ["r000005a1"]
    assert l["confidence"] == .5
    assert kb.experiments[1]["actual_delta"] is None


def test_dedup_preserves_evidence_without_repeated_credit():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    lid = lesson(kb, "r000001a1")
    assert lesson(kb, "r000001a1", rule="  Use smaller steps  ") == lid
    kb.record(experiment(2, injected=[lid]))
    assert lesson(kb, "r000001a1", "r000002a1") == lid
    l = kb.lessons[0]
    assert l["source_support_ids"] == ["r000001a1", "r000002a1"]
    assert l["support_ids"] == ["r000002a1"]
    assert l["confidence"] == pytest.approx(2 / 3)
    before = kb.state()
    with pytest.raises(ValueError, match="duplicate"):
        kb.record(experiment(2, injected=[lid]))
    assert kb.state() == before


def test_punctuation_only_first_rule_is_rejected_without_poisoning_resume():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    before = kb.state()
    with pytest.raises(ValueError, match="contain words"):
        lesson(kb, "r000001a1", rule="!!!")
    assert kb.state() == before
    restored = KnowledgeBase()
    restored.load_state(json.loads(json.dumps(kb.state())))
    assert restored.state() == before


@pytest.mark.parametrize("first,second", [
    ("Use x < 0 in the branch condition", "Use x > 0 in the branch condition"),
    ("Keep mode = 'A'", "Keep mode = 'a'"),
    ("Match 'a b' exactly", "Match 'a  b' exactly"),
])
def test_dedup_never_merges_semantically_different_code_rules(first, second):
    kb = KnowledgeBase()
    kb.record(experiment(1))
    kb.record(experiment(2))
    first_id = lesson(kb, "r000001a1", rule=first)
    second_id = lesson(kb, "r000002a1", rule=second)
    assert first_id != second_id
    assert [(l["rule"], l["evidence_ids"]) for l in kb.lessons] == [
        (first, ["r000001a1"]), (second, ["r000002a1"])]
    restored = KnowledgeBase()
    restored.load_state(json.loads(json.dumps(kb.state())))
    assert restored.lessons == kb.lessons


@pytest.mark.parametrize("claimed_delta", [5e-13, -5e-13])
def test_tolerance_cannot_fabricate_improvement_or_regression_at_measured_zero(claimed_delta):
    kb = KnowledgeBase()
    before = kb.state()
    with pytest.raises(ValueError, match="measured score difference"):
        kb.record(experiment(1, parent_score=1.0, score=1.0, actual_delta=claimed_delta))
    assert kb.state() == before
    assert kb.fragments == [] and kb.lessons == []
    kb.record(experiment(1, parent_score=1.0, score=1.0, actual_delta=0.0))
    for polarity in ("positive", "negative"):
        with pytest.raises(ValueError, match="matching measured"):
            lesson(kb, "r000001a1", polarity=polarity)
    assert kb.fragments == [] and kb.lessons == []


def test_caps_preserve_bound_evidence_and_duplicate_detection_after_pruning():
    kb = KnowledgeBase(max_experiments=2, max_lessons=2, max_fragments=1)
    kb.record(experiment(1, delta=-.1))
    lid = lesson(kb, "r000001a1", polarity="negative")
    for n in range(2, 6):
        kb.record(experiment(n, injected=[lid]))
    assert len(kb.experiments) == 2 and len(kb.fragments) == 1
    assert kb.lessons[0]["evidence_snapshots"][0]["actual_delta"] == pytest.approx(-.1)
    assert len(kb.lessons[0]["exposure_snapshots"]) == 2
    assert len(kb.state()["seen_experiments"]) == 5
    with pytest.raises(ValueError, match="duplicate"):
        kb.record(experiment(1, delta=.2))
    restored = KnowledgeBase(max_experiments=2, max_lessons=2, max_fragments=1)
    restored.load_state(json.loads(json.dumps(kb.state())))
    assert restored.state() == kb.state()


def test_batch_add_and_reflection_rejections_are_atomic():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    before = kb.state()
    with pytest.raises(ValueError):
        kb.add_lessons([proposal("r000001a1"), proposal("unknown")], allowed_evidence_ids=["r000001a1"])
    assert kb.state() == before
    lid = lesson(kb, "r000001a1")
    before = kb.state()
    for retire in ([lid], ["l999999"]):
        with pytest.raises(ValueError):
            kb.reflect([], retire, allowed_evidence_ids=["r000001a1"])
        assert kb.state() == before


@pytest.mark.parametrize("claimed_branch", ["foreign", None, 0, False, [], {}])
def test_optional_branch_cannot_override_evidence_and_batch_is_atomic(claimed_branch):
    kb = KnowledgeBase()
    kb.record(experiment(1))
    before = kb.state()
    bad = {**proposal("r000001a1", rule="Another rule"), "branch": claimed_branch}
    with pytest.raises(ValueError, match="evidence-derived branch"):
        kb.add_lessons([proposal("r000001a1"), bad], allowed_evidence_ids=["r000001a1"])
    assert kb.state() == before


def test_optional_global_branch_must_be_null_and_still_needs_two_branches():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    kb.record(experiment(2, branch="second"))
    p = {**proposal("r000001a1", "r000002a1", scope="global"), "branch": None}
    lid = kb.add_lessons([p], allowed_evidence_ids=p["evidence_ids"])[0]
    assert kb.lessons[0]["id"] == lid and kb.lessons[0]["branch"] is None
    before = kb.state()
    for branch in ("seed0", "second", "", False):
        with pytest.raises(ValueError, match="evidence-derived branch"):
            kb.reflect([{**p, "branch": branch}], [], allowed_evidence_ids=p["evidence_ids"])
        assert kb.state() == before
    with pytest.raises(ValueError, match="two branches"):
        kb.add_lessons([{**p, "evidence_ids": ["r000001a1"]}], allowed_evidence_ids=["r000001a1"])
    assert kb.state() == before


def test_archived_v4_reflection_shape_applies_only_to_matching_evidence_copy():
    # Portable transcription of the paid V4 response; no artifact dependency or
    # candidate execution. Its redundant branch is an assertion, not authority.
    response = {"lessons": [{"branch": "p1", "evidence_ids": ["r000001a1"],
        "polarity": "negative", "rule": "Adjusting the radius of the central circle resulted in a decrease in the score, refuting the hypothesis that it would lead to tighter packing.",
        "scope": "branch"}], "retire_ids": []}
    source = KnowledgeBase()
    source.record(experiment(1, branch="p1", parent_score=.36423689449571406,
                             score=0.0, actual_delta=-.36423689449571406))
    original = source.state()
    replay = KnowledgeBase()
    replay.load_state(json.loads(json.dumps(original)))
    result = replay.reflect(response["lessons"], response["retire_ids"],
                            allowed_evidence_ids=["r000001a1"])
    assert result == {"lesson_ids": ["l000001"], "retired_ids": []}
    assert replay.lessons[0]["branch"] == "p1"
    assert replay.lessons[0]["evidence_ids"] == ["r000001a1"]
    assert source.state() == original and response["lessons"][0]["branch"] == "p1"
    restored = KnowledgeBase()
    restored.load_state(json.loads(json.dumps(replay.state())))
    assert restored.state() == replay.state()


def test_reflection_keeps_counterevidence_and_does_not_claim_new_rule_was_injected():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    old = lesson(kb, "r000001a1")
    kb.record(experiment(2, delta=-.1, injected=[old]))
    result = kb.reflect([proposal("r000001a1", rule="Use smaller steps with a fallback")],
                        [old], allowed_evidence_ids=["r000001a1"])
    l = kb.lessons[0]
    assert l["id"] != old and result["retired_ids"] == [old]
    assert l["exposure_ids"] == []  # It has not itself been used.
    assert l["inherited_exposures"][0]["from_lesson_id"] == old
    assert l["refute_ids"] == ["r000002a1"] and l["confidence"] == pytest.approx(1/3)
    restored = KnowledgeBase()
    restored.load_state(json.loads(json.dumps(kb.state())))
    assert restored.state() == kb.state()


def test_retrieval_relevance_and_public_data_are_defensive_copies():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    one = lesson(kb, "r000001a1", rule="Use a bounded timeout")
    lesson(kb, "r000001a1", rule="Use smaller optimization steps")
    found = kb.retrieve(branch="seed0", context="bounded timeout")
    assert found[0]["id"] == one
    found[0]["evidence_snapshots"][0]["score"] = 999
    kb.experiments.clear()
    kb.fragments[0]["code"] = "bad"
    assert len(kb.experiments) == 1 and kb.fragments[0]["code"] != "bad"
    assert kb.lessons[0]["evidence_snapshots"][0]["score"] < 1


def test_json_roundtrip_continues_identically():
    original = KnowledgeBase(max_experiments=3)
    original.record(experiment(1))
    lid = lesson(original, "r000001a1")
    original.record(experiment(2, status="no_child", injected=[lid]))
    restored = KnowledgeBase(max_experiments=3)
    restored.load_state(json.loads(json.dumps(original.state(), allow_nan=False)))
    for kb in (original, restored):
        kb.record(experiment(3, injected=[lid]))
        kb.record(experiment(4, branch="second"))
        lesson(kb, "r000003a1", "r000004a1", scope="global")
    assert restored.state() == original.state()
    assert restored.retrieve(branch="second", context="steps") == original.retrieve(branch="second", context="steps")


@pytest.mark.parametrize("field", ["max_lessons", "max_experiments", "max_fragments"])
def test_resume_rejects_different_configured_capacity(field):
    kb = KnowledgeBase()
    kb.record(experiment(1))
    before = kb.state()
    incoming = copy.deepcopy(before)
    incoming["limits"][field] += 1
    with pytest.raises(ValueError, match="configured limits"):
        kb.load_state(incoming)
    assert kb.state() == before


def test_old_lesson_cannot_be_retired_using_only_newer_prompt_evidence():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    old = lesson(kb, "r000001a1")
    kb.record(experiment(2))
    before = kb.state()
    with pytest.raises(ValueError, match="allowed"):
        kb.reflect([proposal("r000001a1", "r000002a1", rule="A consolidated rule")],
                   [old], allowed_evidence_ids=["r000002a1"])
    assert kb.state() == before


def test_reflection_context_reopens_old_evidence_and_preserves_counterobservations():
    kb = KnowledgeBase(max_experiments=2)
    kb.record(experiment(1))
    old = lesson(kb, "r000001a1")
    kb.record(experiment(2, delta=-.1, injected=[old]))
    kb.record(experiment(3))
    kb.record(experiment(4))
    assert {e["id"] for e in kb.experiments} == {"r000003a1", "r000004a1"}
    before = kb.state()
    records, summaries = kb.reflection_context(1)
    by_id = {e["id"]: e for e in records}
    assert set(by_id) == {"r000001a1", "r000002a1", "r000004a1"}
    assert by_id["r000001a1"]["actual_delta"] > 0
    assert by_id["r000002a1"]["actual_delta"] < 0
    assert summaries[0]["observed_exposures"]["negative"] == 1
    assert kb.state() == before
    result = kb.reflect([proposal("r000001a1", rule="Consolidated scoped observation")],
                        [old], allowed_evidence_ids=list(by_id))
    assert result["retired_ids"] == [old]
    assert kb.lessons[0]["refute_ids"] == ["r000002a1"]
    assert kb.lessons[0]["exposure_ids"] == []
    restored = KnowledgeBase(max_experiments=2)
    restored.load_state(json.loads(json.dumps(kb.state())))
    assert restored.state() == kb.state()


def test_reflection_context_bound_never_shows_lesson_without_its_evidence():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    lesson(kb, "r000001a1")
    kb.record(experiment(2))
    records, lessons = kb.reflection_context(1, max_experiments=1)
    assert [r["id"] for r in records] == ["r000001a1"]
    assert len(lessons) == 1 and lessons[0]['evidence_ids'] == ['r000001a1']
    with pytest.raises(ValueError, match="single research observation"):
        kb.reflection_context(1, max_chars=1)


def test_reflection_context_reports_omitted_exposures_without_changing_counts():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    old = lesson(kb, "r000001a1")
    for n, delta in [(2, .1), (3, -.1), (4, .2), (5, -.2)]:
        kb.record(experiment(n, delta=delta, injected=[old]))
    records, summaries = kb.reflection_context(1)
    info = summaries[0]["observed_exposures"]
    assert info["total"] == 4 and info["positive"] == info["negative"] == 2
    assert info["omitted_count"] == 2
    assert set(info["shown_ids"]) == {"r000004a1", "r000005a1"}
    assert set(info["shown_ids"]) <= {r["id"] for r in records}
    records[0]["score"] = 999
    summaries[0]["evidence_ids"].clear()
    assert kb.lessons[0]["evidence_ids"] == ["r000001a1"]


@pytest.mark.parametrize("count,eligible", [(32, True), (33, False)])
def test_reflection_marks_source_sets_exceeding_wire_replacement_limit(count, eligible):
    from sciel.hypothesis import parse_reflection
    kb = KnowledgeBase()
    for number in range(1, count + 1):
        kb.record(experiment(number))
        lesson(kb, f"r{number:06d}a1")
    before = kb.state()
    records, summaries = kb.reflection_context(1)
    summary, = summaries
    assert summary["retirement_eligible"] is eligible
    assert len(summary["evidence_ids"]) == count
    assert set(summary["evidence_ids"]) <= {r["id"] for r in records}
    response = json.dumps({"lessons": [proposal(*summary["evidence_ids"], rule="Replacement")],
                           "retire_ids": [summary["id"]]})
    assert (parse_reflection(response) is not None) is eligible
    assert kb.state() == before


def test_reflection_separates_inherited_and_current_exposure_attribution():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    old = lesson(kb, "r000001a1", rule="Original small-step observation")
    kb.record(experiment(2, delta=-.1, injected=[old]))
    new = kb.reflect([proposal("r000001a1", rule="Consolidated small-step observation")], [old],
                     allowed_evidence_ids=["r000001a1", "r000002a1"])["lesson_ids"][0]
    kb.record(experiment(3, delta=-.2, injected=[new]))
    before = kb.state()
    records, summaries = kb.reflection_context(1)
    summary, = summaries
    info = summary["observed_exposures"]
    assert info["negative"] == info["total"] == 2
    assert set(info["shown_ids"]) == {"r000002a1", "r000003a1"}
    reps = {r["experiment_id"]: r["attribution"] for r in info["representatives"]}
    assert reps["r000002a1"] == [{"kind": "inherited", "from_lesson_id": old,
                                   "from_rule": "Original small-step observation"}]
    assert reps["r000003a1"] == [{"kind": "direct", "from_lesson_id": new,
                                   "from_rule": "Consolidated small-step observation"}]
    assert {r["id"] for r in records} == {"r000001a1", "r000002a1", "r000003a1"}
    # Attribution is part of the whole lesson unit, never a silently sliced rule.
    size = len(serialize_reflection(records, summaries))
    _, bounded = kb.reflection_context(1, max_chars=size - 1)
    assert bounded == [] and kb.state() == before
    restored = KnowledgeBase()
    restored.load_state(json.loads(json.dumps(before)))
    assert restored.reflection_context(1) == (records, summaries)


def test_reflection_deduplicates_experiment_with_multiple_inherited_advice_origins():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    left = lesson(kb, "r000001a1", rule="First observation")
    right = lesson(kb, "r000001a1", rule="Second observation")
    kb.record(experiment(2, delta=-.1, injected=[left, right]))
    kb.reflect([proposal("r000001a1", rule="Combined observation")], [left, right],
               allowed_evidence_ids=["r000001a1", "r000002a1"])
    _, summaries = kb.reflection_context(1)
    info = summaries[0]["observed_exposures"]
    assert info["total"] == info["negative"] == 1
    assert len(info["representatives"]) == 1
    assert {a["from_lesson_id"] for a in info["representatives"][0]["attribution"]} == {left, right}


@pytest.mark.parametrize("damage", [
    lambda s: s.update(schema="other"),
    lambda s: s["experiments"][0].update(score=float("nan")),
    lambda s: s["lessons"][0]["evidence_snapshots"][0].update(actual_delta=.2),
    lambda s: s["lessons"][0].update(confidence=.99),
    lambda s: s["lessons"][0].update(branch="foreign"),
    lambda s: s["lessons"][0].update(evidence_ids=[]),
    lambda s: s["fragments"][0].update(code="tampered"),
    lambda s: s["seen_experiments"].clear(),
    lambda s: s["experiments"].append(copy.deepcopy(s["experiments"][0])),
    lambda s: s.update(next_lesson_id=1),
])
def test_corrupt_resume_is_rejected_atomically(damage):
    kb = KnowledgeBase()
    kb.record(experiment(1))
    lesson(kb, "r000001a1")
    before = kb.state()
    bad = copy.deepcopy(before)
    damage(bad)
    with pytest.raises(ValueError):
        kb.load_state(bad)
    assert kb.state() == before
