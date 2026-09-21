"""A typed reflection view separates source membership from advice exposures."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.hypothesis.rendering import decode_evidence, serialize_reflection
from sciel.knowledge_base.presentation import expand_reflection_payload

from sciel.hypothesis import reflection_prompt
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.presentation import observation_card, reflection_payload as compact_reflection_payload


def memory_fixture():
    from reflection_memory_fixture import memory_records
    path = Path(__file__).with_name('reflection_memory_fixture.py')
    records, lessons = memory_records()
    return path, path.read_bytes(), records, lessons


def reflection_payload(records, lessons):
    return expand_reflection_payload(compact_reflection_payload(records, lessons))


def test_legacy_source_membership_is_inline_and_exposure_only_ids_stay_separate():
    path, raw, records, lessons = memory_fixture()
    before = copy.deepcopy((records, lessons))
    view = reflection_payload(records, lessons)
    lesson, = view["lessons"]
    source = lesson["source_support"]
    assert source["evidence_ids"] == ["r000002a1", "r000004a1"]
    assert [e["id"] for e in source["observations"]] == source["evidence_ids"]
    assert [e["actual_delta"] for e in source["observations"]] == [
        0.5627179948834453, 0.523557102750764]
    assert source["polarity"] == "positive" and source["omitted_count"] == 0
    exposure = lesson["advice_exposures"]
    assert exposure["retained_unique_counts"] == {
        "positive": 1, "negative": 0, "zero": 1, "unmeasured": 2, "total": 4}
    reps = {r["observation"]["id"]: r for r in exposure["representatives"]}
    assert reps["r000005a1"]["observation"]["status"] == "no_child"
    assert reps["r000005a1"]["observation"]["actual_delta"] is None
    assert reps["r000006a1"]["observation"]["actual_delta"] == 0
    assert not reps["r000005a1"]["also_source_support"]
    assert not reps["r000006a1"]["also_source_support"]
    # One observation can be a source AND an exposure, with different advice origin.
    assert reps["r000004a1"]["also_source_support"] is True
    assert reps["r000004a1"]["attribution"][0]["from_lesson_id"] == "l000003"
    assert reps["r000006a1"]["attribution"][0]["from_lesson_id"] == "l000006"
    assert set(view["observation_catalog"]) == {e["id"] for e in records}
    assert "experiments" not in view and "evidence_ids" not in lesson
    assert lesson["unverified_interpretation"] == lessons[0]["rule"]
    assert (records, lessons) == before and path.read_bytes() == raw


def test_prompt_serializes_exact_typed_payload_and_legacy_facts_are_unknown():
    _, _, records, lessons = memory_fixture()
    view = reflection_payload(records, lessons)
    prompt = reflection_prompt(records, lessons)
    assert prompt.split("EVIDENCE_JSON:\n", 1)[1].strip() == serialize_reflection(records, lessons)
    wire = decode_evidence(json.loads(prompt.split("EVIDENCE_JSON:\n", 1)[1]))
    assert wire["schema"] == "sciel-reflection-presentation/9"
    assert wire["lessons"][0]["source_support"]["observations"] == [
        {"observation_ref": ident} for ident in view["lessons"][0]["source_support"]["evidence_ids"]]
    assert expand_reflection_payload(wire) == view
    assert "source_support.evidence_ids" in prompt
    assert "observation_catalog" in prompt and "also_source_support=true" in prompt
    assert "observed_exposures.representatives" not in prompt
    for row in view["lessons"][0]["source_support"]["observations"]:
        assert row["metric"] is row["parent_cases"] is row["evaluation"] is None
    assert all("metric" not in record for record in records), "view must not rewrite legacy records"


def test_metric_and_failed_evaluator_facts_are_copied_without_inventing_a_delta():
    _, _, records, lessons = memory_fixture()
    failure = next(e for e in records if e["id"] == "r000003a1")
    failure.update(metric={"id": "fixture/1", "unit": "normalized ratio", "hash": "a" * 64},
                   parent_cases={"sum_radii": 2.4425261335140847},
                   evaluation={"ok": False, "score": 0., "cases": {"validity": 0.},
                               "reached": "smoke", "confirmed": False, "raw_first_score": 0.})
    before = copy.deepcopy((records, lessons))
    view = reflection_payload(records, lessons)
    row = next(r["observation"] for r in view["lessons"][0]["advice_exposures"]["representatives"]
               if r["observation"]["id"] == failure["id"])
    assert row["score"] is row["actual_delta"] is None
    for field in ("metric", "parent_cases", "evaluation"):
        assert row[field] == failure[field]
    row["evaluation"]["cases"]["validity"] = 99
    view["observation_catalog"][failure["id"]]["parent_cases"].clear()
    assert (records, lessons) == before


def observe(kb, number, delta=.1, injected=()):
    ident = f"r{number:06d}a1"
    source = hashlib.sha256(str(number).encode()).hexdigest()
    score = .5 + delta if delta is not None else None
    kb.record(dict(id=ident, branch="p1", parent_score=.5,
                   score=score, actual_delta=score - .5 if score is not None else None,
                   status="evaluated" if score is not None else "eval_failed",
                   source_sha256=source if score is not None else None,
                   proposed_sha256=source if score is not None else None,
                   injected_ids=list(injected)))
    return ident


def add(kb, rule, ids):
    return kb.add_lessons([dict(rule=rule, polarity="positive", scope="branch", evidence_ids=ids)],
                          allowed_evidence_ids=ids)[0]


@pytest.mark.parametrize("rules", [("First measured interpretation", "Second interpretation"),
                                   ("意味" * 500, "Причина" * 200)])
def test_actual_serialized_boundary_preserves_whole_focus_pair_and_sources(rules):
    kb = KnowledgeBase()
    source = observe(kb, 1)
    pair = [add(kb, text, [source]) for text in rules]
    observe(kb, 2, delta=-.1, injected=pair)
    observe(kb, 3)
    observe(kb, 4)  # Pair turn, with a recent observation outside its evidence.
    before = kb.state()
    records, lessons = kb.reflection_context(1)
    size = len(serialize_reflection(records, lessons))
    assert [s["id"] for s in lessons if s["focus_for_consolidation"]] == pair
    assert kb.reflection_context(1, max_chars=size) == (records, lessons)
    bounded_records, bounded_lessons = kb.reflection_context(1, max_chars=size - 1)
    assert len(serialize_reflection(bounded_records, bounded_lessons)) <= size - 1
    assert [s["id"] for s in bounded_lessons if s["focus_for_consolidation"]] == pair
    assert {r['id'] for r in bounded_records} == {source, 'r000002a1'}
    anchor_size = len(serialize_reflection(bounded_records, bounded_lessons))
    smaller_records, smaller_lessons = kb.reflection_context(1, max_chars=anchor_size - 1)
    assert not any(s['focus_for_consolidation'] for s in smaller_lessons)
    assert len(smaller_lessons) <= 1
    assert len(serialize_reflection(smaller_records, smaller_lessons)) <= anchor_size - 1
    for lesson in reflection_payload(bounded_records, bounded_lessons)["lessons"]:
        assert [r["id"] for r in lesson["source_support"]["observations"]] == [source]
        assert lesson["advice_exposures"]["representatives"][0]["observation"]["actual_delta"] < 0
    assert kb.state() == before
    restored = KnowledgeBase()
    restored.load_state(copy.deepcopy(before))
    assert restored.reflection_context(1) == (records, lessons)


def test_exposure_representative_omissions_remain_explicit_without_dropping_sources():
    kb = KnowledgeBase()
    source = observe(kb, 1)
    lesson = add(kb, "One source, several later failed attempts", [source])
    for number in (2, 3, 4):
        observe(kb, number, delta=None, injected=[lesson])
    observe(kb, 5, delta=0., injected=[lesson])
    before = kb.state()
    records, summaries = kb.reflection_context(1)
    view = reflection_payload(records, summaries)["lessons"][0]
    assert view["source_support"]["evidence_ids"] == [source]
    assert view["source_support"]["omitted_count"] == 0
    exposure = view["advice_exposures"]
    assert exposure["retained_unique_counts"]["total"] == 4
    assert exposure["retained_unique_counts"]["unmeasured"] == 3
    assert exposure["retained_unique_counts"]["zero"] == 1
    assert exposure["shown_ids"] == ["r000004a1", "r000005a1"]
    assert exposure["omitted_count"] == 2
    assert len(kb.lessons[0]["exposure_snapshots"]) == 4 and kb.state() == before


def test_generator_and_reflection_agree_on_metric_facts_and_inherited_origins():
    kb = KnowledgeBase()
    source = observe(kb, 1)
    old = add(kb, "Original wording", [source])
    observe(kb, 2, delta=-.1, injected=[old])
    new = kb.reflect([dict(rule="Replacement wording", polarity="positive", scope="branch",
                           evidence_ids=[source])], [old], allowed_evidence_ids=[source])["lesson_ids"][0]
    observe(kb, 3, delta=None, injected=[new])
    records, summaries = kb.reflection_context(1)
    reflected = reflection_payload(records, summaries)["lessons"][0]
    generated = observation_card(kb.lessons[0])
    assert reflected["source_support"]["observations"] == generated["source_observations"]
    assert reflected["advice_exposures"]["retained_unique_counts"] == generated["advice_exposures"]["retained_unique_counts"]
    assert [{k:v for k,v in r.items() if k != "also_source_support"}
            for r in reflected["advice_exposures"]["representatives"]] == generated["advice_exposures"]["representatives"]
    assert {r["attribution"][0]["from_lesson_id"] for r in reflected["advice_exposures"]["representatives"]} == {old,new}


def test_absent_exposure_summary_is_unknown_not_fabricated_zero():
    records = [dict(id="e1", score=.5, actual_delta=.1, status="evaluated")]
    lessons = [dict(id="l1", rule="Legacy", evidence_ids=["e1"])]
    result = reflection_payload(records, lessons)["lessons"][0]
    assert result["advice_exposures"]["available"] is False
    assert result["advice_exposures"]["retained_unique_counts"] is None
    assert result["retirement_eligible"] is None


@pytest.mark.parametrize("change", ["missing_source", "missing_exposure", "duplicate_record",
                                    "duplicate_lesson", "changed_representative_ids"])
def test_incomplete_or_ambiguous_view_is_rejected_without_mutation(change):
    _, _, records, lessons = memory_fixture()
    if change == "missing_source": records[:] = [r for r in records if r["id"] != "r000002a1"]
    elif change == "missing_exposure": records[:] = [r for r in records if r["id"] != "r000003a1"]
    elif change == "duplicate_record": records.append(copy.deepcopy(records[0]))
    elif change == "duplicate_lesson": lessons.append(copy.deepcopy(lessons[0]))
    else: lessons[0]["observed_exposures"]["representatives"].pop()
    before = copy.deepcopy((records, lessons))
    with pytest.raises(ValueError): reflection_payload(records, lessons)
    assert (records, lessons) == before
