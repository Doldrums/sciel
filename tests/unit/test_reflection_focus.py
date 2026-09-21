"""Pure checks for bounded, rotating and evidence-complete reflection focus."""
import copy
import hashlib
import itertools
import json
from types import SimpleNamespace

import pytest

from sciel.hypothesis.rendering import decode_evidence, serialize_reflection

from sciel.engine.research import ResearchController
from sciel.hypothesis import CONTRACT, parse_reflection, reflection_prompt
from sciel.hypothesis.parsing import MAX_EVIDENCE_IDS, MAX_LESSONS, MAX_RETIRE_IDS
from sciel.knowledge_base import KnowledgeBase


def observe(kb, number, *, delta=.1, injected=()):
    measured = delta is not None
    score = .5 + delta if measured else None
    digest = hashlib.sha256(f"fixture-{number}".encode()).hexdigest()
    ident = f"r{number:06d}a1"
    kb.record({"id": ident, "branch": "p1", "status": "evaluated" if measured else "eval_failed",
               "parent_score": .5, "score": score, "actual_delta": score - .5 if measured else None,
               "source_sha256": digest if measured else None,
               "proposed_sha256": digest if measured else None, "hypothesis": None,
               "injected_ids": list(injected), "code_fragment": ""})
    return ident


def add(kb, rule, *ids):
    return kb.add_lessons([{"rule": rule, "polarity": "positive", "scope": "branch",
                            "evidence_ids": list(ids)}], allowed_evidence_ids=list(ids))[0]


def focused(summaries):
    return [x["id"] for x in summaries if x["focus_for_consolidation"]]


def assert_complete(records, summaries):
    ids = {x["id"] for x in records}
    assert all(set(x["evidence_ids"]) <= ids for x in summaries)
    assert all(set(x["observed_exposures"]["shown_ids"]) <= ids for x in summaries)


def test_focus_pair_is_structural_not_a_semantic_claim_and_preserves_counterevidence():
    kb = KnowledgeBase()
    source = observe(kb, 1)
    left = add(kb, "A smaller step may help", source)
    right = add(kb, "A larger step may help", source)
    observe(kb, 2, delta=-.1, injected=[left, right])
    observe(kb, 3, delta=None, injected=[left])
    observe(kb, 4, injected=[right])
    observe(kb, 5)
    observe(kb, 6)  # Even epoch: pair-anchor turn.
    before = kb.state()
    records, summaries = kb.reflection_context(1)
    assert focused(summaries) == [left, right]
    assert [x["rule"] for x in summaries] == [x["rule"] for x in before["lessons"]]
    assert_complete(records, summaries)
    assert {r["id"] for r in records} == {f"r{i:06d}a1" for i in (1, 2, 3, 4, 6)}
    exposure = {x["id"]: x["observed_exposures"] for x in summaries}
    assert exposure[left]["negative"] == exposure[left]["unmeasured"] == 1
    assert exposure[right]["negative"] == exposure[right]["positive"] == 1
    assert kb.state() == before, "selecting a focus must never mutate or retire knowledge"
    records[0]["score"] = 999
    summaries[0]["observed_exposures"]["representatives"].clear()
    assert kb.state() == before


def test_focus_rotates_all_available_pairs_using_lifetime_attempts_and_resumes_exactly():
    kb = KnowledgeBase(max_experiments=2)
    source = observe(kb, 1)
    ids = [add(kb, rule, source) for rule in ("A", "B", "C")]
    expected = list(itertools.combinations(ids, 2))
    seen = set()
    for number in range(1, 13):
        if number > 1:
            observe(kb, number)
        records, summaries = kb.reflection_context(1)
        pair = tuple(focused(summaries))
        epoch = len(kb.state()["seen_experiments"])
        if epoch % 2 == 0:
            assert pair == expected[(epoch // 2) % len(expected)]
            seen.add(pair)
        else:
            assert pair == ()
            assert summaries[0]["id"] == ids[(epoch // 2) % len(ids)]
        assert_complete(records, summaries)
        restored = KnowledgeBase(max_experiments=2)
        restored.load_state(copy.deepcopy(kb.state()))
        assert restored.reflection_context(1) == (records, summaries)
    assert seen == set(expected)
    assert len(kb.experiments) == 2 and len(kb.state()["seen_experiments"]) == 12


def test_rotation_changes_only_when_the_declared_recent_epoch_changes():
    kb = KnowledgeBase()
    source = observe(kb, 1)
    ids = [add(kb, rule, source) for rule in ("A", "B", "C")]
    assert focused(kb.reflection_context(3)[1]) == ids[:2]
    observe(kb, 2)
    assert focused(kb.reflection_context(3)[1]) == ids[:2]
    observe(kb, 3)
    assert focused(kb.reflection_context(3)[1]) == []  # Singleton turn.
    for number in (4, 5, 6):
        observe(kb, number)
    assert focused(kb.reflection_context(3)[1]) == [ids[0], ids[2]]


def test_default_context_limits_displayed_lessons_to_eight_and_fallback_rotates():
    kb = KnowledgeBase()
    ids = []
    for number in range(1, 13):
        ids.append(add(kb, f"Observation {number}", observe(kb, number)))
    records, summaries = kb.reflection_context(1)
    assert len(summaries) == 8 and focused(summaries) == []
    assert_complete(records, summaries)
    appeared = set()
    for number in range(12, 24):
        if number > 12:
            observe(kb, number)
        records, summaries = kb.reflection_context(1, max_lessons=1)
        assert focused(summaries) == [] and len(summaries) == 1
        appeared.add(summaries[0]["id"])
        assert_complete(records, summaries)
    assert appeared == set(ids), "a fixed retained lesson must not starve behind the first eight"


def test_focus_pair_is_prioritized_over_rotating_regular_lessons():
    kb = KnowledgeBase()
    for number in range(1, 10):
        add(kb, f"Unpaired {number}", observe(kb, number))
    source = observe(kb, 10)
    pair = [add(kb, rule, source) for rule in ("Later A", "Later B")]
    records, summaries = kb.reflection_context(1)
    assert [x["id"] for x in summaries[:2]] == focused(summaries) == pair
    assert len(summaries) == 8
    assert_complete(records, summaries)


@pytest.mark.parametrize("bound", ["one_lesson", "one_character_short", "one_experiment"])
def test_pair_focus_is_never_partially_admitted_when_capacity_cannot_fit_it(bound):
    kb = KnowledgeBase()
    source = observe(kb, 1)
    pair = [add(kb, rule, source) for rule in ("A", "B")]
    observe(kb, 2, delta=-.1, injected=pair)
    observe(kb, 3)
    observe(kb, 4)  # Even epoch; unrelated recent observation is optional.
    before = kb.state()
    full_records, full_summaries = kb.reflection_context(1)
    assert focused(full_summaries) == pair
    if bound == "one_lesson":
        records, summaries = kb.reflection_context(1, max_lessons=1)
        assert len(summaries) == 1
    elif bound == "one_character_short":
        anchor_records = [r for r in full_records if r["id"] in (source, "r000002a1")]
        size = len(serialize_reflection(anchor_records, full_summaries))
        records, summaries = kb.reflection_context(1, max_chars=size - 1)
        assert len(summaries) <= 1
    else:
        records, summaries = kb.reflection_context(1, max_experiments=1)
        assert summaries == []
    assert focused(summaries) == []
    assert_complete(records, summaries)
    assert kb.state() == before


def test_pair_is_eligible_only_when_its_complete_shared_source_set_fits_wire_limit():
    kb = KnowledgeBase(max_experiments=100)
    for number in range(1, 33):
        source = observe(kb, number)
        pair = [add(kb, rule, source) for rule in ("A", "B")]
    records, summaries = kb.reflection_context(1)
    assert focused(summaries) == pair
    assert all(x["retirement_eligible"] and len(x["evidence_ids"]) == MAX_EVIDENCE_IDS for x in summaries)
    source = observe(kb, 33)
    for rule in ("A", "B"):
        add(kb, rule, source)
    records, summaries = kb.reflection_context(1)
    assert len(summaries) == 2 and focused(summaries) == []
    assert all(not x["retirement_eligible"] and len(x["evidence_ids"]) == 33 for x in summaries)
    assert_complete(records, summaries)


def test_focus_matches_evidence_sets_independently_of_id_order():
    kb = KnowledgeBase()
    a, b = observe(kb, 1), observe(kb, 2)
    left = add(kb, "A", a, b)
    right = add(kb, "B", b, a)
    assert focused(kb.reflection_context(1)[1]) == [left, right]


def test_narrow_prompt_request_keeps_terminal_evidence_and_wider_wire_guards():
    kb = KnowledgeBase()
    source = observe(kb, 1)
    for rule in ("A", "B"):
        add(kb, rule, source)
    records, summaries = kb.reflection_context(1)
    prompt = reflection_prompt(records, summaries)
    body = decode_evidence(json.loads(prompt.split("EVIDENCE_JSON:\n", 1)[1]))
    from sciel.knowledge_base.presentation import reflection_payload
    assert body == reflection_payload(records, summaries)
    assert "at most 2 new lessons" in prompt and "at most 2 retire_ids" in prompt
    assert "do not establish semantic equivalence" in prompt
    assert "Leave every other lesson unchanged" in prompt and "consolidation is optional" in prompt
    assert "general causal claims" in prompt
    assert MAX_LESSONS == 8 and MAX_RETIRE_IDS == MAX_EVIDENCE_IDS == 32
    # Two is an instruction for this review, not an undeclared parser policy change.
    proposals = [{"rule": f"Rule {i}", "polarity": "positive", "scope": "branch",
                  "evidence_ids": [source]} for i in range(3)]
    assert parse_reflection(json.dumps({"lessons": proposals, "retire_ids": []})) is not None


@pytest.mark.parametrize("old_contract", ["sciel-research/4", "sciel-research/5", "sciel-research/6", "sciel-research/7", "sciel-research/8"])
def test_controller_refuses_older_checkpoint_before_any_calls_or_state_changes(old_contract):
    assert CONTRACT == "sciel-research/12"
    calls = []
    config = {"enabled": True, "reflection_interval": 10, "max_aux_calls": 3,
              "max_lessons": 50, "max_experiments": 200, "max_fragments": 20,
              "retrieve_k": 3, "model": None}
    llm = SimpleNamespace(models=["fixture"], max_retries=1, generate=lambda *a, **kw: calls.append(a))
    controller = ResearchController(config, llm, SimpleNamespace(research=lambda **kw: calls.append(kw)),
                                    lambda cost: calls.append(cost))
    before = controller.state()
    old = copy.deepcopy(before)
    old["schema"] = old_contract
    with pytest.raises(ValueError, match="contract/configuration mismatch"):
        controller.load_state(old)
    assert controller.state() == before and calls == []
