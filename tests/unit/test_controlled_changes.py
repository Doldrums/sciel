"""Exact controlled-source receipts persist independently of candidate execution."""
import copy
import hashlib
import itertools
import json
from types import SimpleNamespace

import pytest

from sciel.engine.intervention import CircleIntervention
from sciel.engine.research import ResearchController, digest
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_interventions as references
from sciel.hypothesis import circle_restart_scales as scales
from sciel.hypothesis.controlled_changes import validate_controlled_change
from sciel.knowledge_base import KnowledgeBase

from test_intervention_worker import setup, response, plan


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checked(builder=references, parent="initial", target="incumbent"):
    source = builder.source_for(parent)
    built = builder.apply(source, target)
    args = dict(parent_source_sha256=digest(source), proposed_sha256=digest(built["code"]),
        source_sha256=digest(built["code"]), source_repaired=False, parent_code=source,
        proposed_code=built["code"], source_code=built["code"])
    return built, args


TRANSITIONS = tuple((builder, parent, target)
    for builder in (references, scales)
    for parent, target in itertools.permutations(builder.POLICIES, 2))


@pytest.mark.parametrize("builder,parent,target", TRANSITIONS)
def test_all_known_directed_edits_rederive_from_full_sources_and_snapshot_hashes(builder, parent, target):
    built, args = checked(builder, parent, target)
    before = copy.deepcopy(built)
    verified = validate_controlled_change(built["proof"], **args)
    assert verified == built["proof"] and verified is not built["proof"]
    hash_only = {key: value for key, value in args.items() if not key.endswith("_code")}
    assert validate_controlled_change(json.loads(json.dumps(verified)), **hash_only) == verified
    verified["before"]["text"] = "mutated detached receipt"
    assert built == before


@pytest.mark.parametrize("kind", ["extra", "missing", "operation", "parent", "target", "noop",
    "before", "after", "prefix", "suffix", "span", "plan_key", "true_as_one", "int_as_float",
    "tuple_span", "nan", "null"])
def test_proof_rejects_tampering_and_json_numeric_aliases(kind):
    built, args = checked()
    proof = copy.deepcopy(built["proof"])
    if kind == "extra": proof["arbitrary"] = "untrusted"
    elif kind == "missing": del proof["scope"]
    elif kind == "operation": proof["operation"] = "invented/1"
    elif kind == "parent": proof["parent_policy"] = "incumbent"
    elif kind == "target": proof["target_policy"] = "arbitrary"
    elif kind == "noop": proof["target_policy"] = proof["parent_policy"]
    elif kind in ("before", "after"): proof[kind]["text"] += "# forged"
    elif kind == "prefix": proof["unchanged_prefix"]["sha256"] = "0" * 64
    elif kind == "suffix": proof["unchanged_suffix"]["bytes"] -= 1
    elif kind == "span": proof["after"]["byte_span"][1] += 1
    elif kind == "plan_key": proof["plan_key"] += ":different"
    elif kind == "true_as_one": proof["unchanged_outside_assignment"] = 1
    elif kind == "int_as_float": proof["unchanged_prefix"]["bytes"] = float(proof["unchanged_prefix"]["bytes"])
    elif kind == "tuple_span": proof["before"]["byte_span"] = tuple(proof["before"]["byte_span"])
    elif kind == "nan": proof["unchanged_suffix"]["bytes"] = float("nan")
    else: proof = None
    with pytest.raises(ValueError):
        validate_controlled_change(proof, **args)


@pytest.mark.parametrize("field,value", [
    ("parent_source_sha256", "0" * 64), ("proposed_sha256", "0" * 64),
    ("source_sha256", "0" * 64), ("source_repaired", True), ("source_repaired", 0),
    ("parent_code", "# same declared SHA, different actual parent"),
    ("proposed_code", "# forged proposal"), ("source_code", "# postvalidation repair")])
def test_valid_proof_cannot_be_transplanted_onto_other_sources_or_repair(field, value):
    built, args = checked()
    args[field] = value
    with pytest.raises(ValueError):
        validate_controlled_change(built["proof"], **args)


def begin_fixture():
    fixture = setup([])
    ident, _ = fixture.research.begin(fixture.snapshot, 0, "intervention", "Pure receipt fixture")
    built = references.apply(fixture.snapshot.parent_code, "incumbent")
    validation = SimpleNamespace(code=built["code"])
    return fixture, ident, built, validation


@pytest.mark.parametrize("damage", ["proof", "parent", "proposed", "validated"])
def test_proposal_rejects_before_mutating_record_counters_or_journal(damage):
    fixture, ident, built, validation = begin_fixture()
    proof = copy.deepcopy(built["proof"])
    proposed = built["code"]
    if damage == "proof": proof["unchanged_outside_assignment"] = 1
    elif damage == "parent": fixture.research._parents[ident] += "\n"
    elif damage == "proposed": proposed += "\n"
    else: validation.code += "\n"
    before, journal = copy.deepcopy(fixture.research.state()), copy.deepcopy(fixture.log.events)
    with pytest.raises(ValueError):
        fixture.research.proposal(ident, response(), proposed, validation, controlled_change=proof)
    assert fixture.research.state() == before and fixture.log.events == journal


@pytest.mark.parametrize("first_controlled,second_controlled,changed", [
    (True, True, False), (True, True, True), (True, False, False),
    (True, False, True), (False, False, False), (False, True, True)])
def test_one_attempt_cannot_overwrite_committed_source_or_reuse_a_stale_proof(first_controlled, second_controlled, changed):
    fixture, ident, built, validation = begin_fixture()
    fixture.research.proposal(ident, response(), built["code"], validation,
        **({"controlled_change": built["proof"]} if first_controlled else {}))
    before, events = copy.deepcopy(fixture.research.state()), copy.deepcopy(fixture.log.events)
    second = references.apply(fixture.snapshot.parent_code, "alternating") if changed else built
    with pytest.raises(ValueError, match="already committed"):
        fixture.research.proposal(ident, response(plan("alternating")), second["code"],
            SimpleNamespace(code=second["code"]),
            **({"controlled_change": second["proof"]} if second_controlled else {}))
    assert fixture.research.state() == before and fixture.log.events == events
    assert fixture.research.valid_hypotheses == 1


def test_returned_worker_metadata_alone_is_not_a_source_commitment():
    fixture, ident, built, validation = begin_fixture()
    fixture.research.pending[ident].update(generation_dispatch="returned", generation_cost=.03)
    assert fixture.research.proposal(ident, response(), built["code"], validation,
                                     controlled_change=built["proof"])
    assert fixture.research.pending[ident]["controlled_change"] == built["proof"]


@pytest.mark.parametrize("builder,parent,target", [(references, "initial", "incumbent"), (scales, "0.06", "0.03")])
def test_worker_persists_exact_proof_before_evaluation_and_controller_resume(builder, parent, target):
    fixture = setup([response(plan(target, hypothesis=f"{builder.OPERATION}:{target}"))])
    if builder is scales:
        source = scales.source_for(parent)
        fixture.parent = fixture.db.add(source, .8, {"validity": 1.}, 0, "seed")
        fixture.snapshot.parent_id = fixture.parent.id
        fixture.snapshot.parent_code = source
        fixture.snapshot.research_context = fixture.research.context(fixture.db, fixture.parent, 0, 1)
        fixture.worker = CircleIntervention(fixture.research, operation=builder.OPERATION)
    proof = builder.apply(fixture.snapshot.parent_code, target)["proof"]
    def evaluate(source):
        hypothesis_event = [e for e in fixture.log.events if e["phase"] == "hypothesis"][-1]
        assert hypothesis_event["evidence"]["controlled_change"] == proof
        assert next(iter(fixture.research.pending.values()))["controlled_change"] == proof
        assert digest(source) == proof["child_source_sha256"]
        return CascadeOutcome(EvalResult(True, .9, {"validity": 1.}), .9, "full", False)
    result = fixture.worker.run(fixture.snapshot, fixture.model, evaluate)
    assert result.ok
    fixture.research.complete_step()  # Aux cap=0: no additional request, consumes queue.
    state = fixture.research.state()
    assert state["knowledge"]["experiments"][0]["controlled_change"] == proof
    restored = ResearchController(fixture.research.config, fixture.model, fixture.log, lambda _: None)
    restored.load_state(json.loads(json.dumps(state)))
    assert restored.state() == state
    assert len(fixture.model.calls) == 1


@pytest.mark.parametrize("status", ["eval_failed", "no_child", "generation_stopped"])
def test_bound_edit_survives_unmeasured_outcome_without_score_or_quality_lesson(status):
    fixture, ident, built, validation = begin_fixture()
    fixture.research.proposal(ident, response(), built["code"], validation,
                             controlled_change=built["proof"])
    record = fixture.research.resolve(ident, status=status, source=built["code"], failure="fixture failure")
    assert record["controlled_change"] == built["proof"]
    assert record["score"] is record["actual_delta"] is None
    with pytest.raises(ValueError, match="measured quality"):
        fixture.research.knowledge.add_lessons([dict(rule="Not measured", polarity="positive",
            scope="branch", evidence_ids=[ident])], allowed_evidence_ids=[ident])
    assert fixture.research.knowledge.lessons == []


def test_resolve_cannot_attach_other_failure_source_to_a_controlled_proposal():
    fixture, ident, built, validation = begin_fixture()
    fixture.research.proposal(ident, response(), built["code"], validation,
                             controlled_change=built["proof"])
    before = copy.deepcopy(fixture.research.state())
    with pytest.raises(ValueError, match="source bytes"):
        fixture.research.resolve(ident, status="eval_failed", source=built["code"] + "\n")
    assert fixture.research.state() == before


def experiment(ident="e1", *, parent="0.06", target="0.03", delta=.125, injected=()):
    built, args = checked(scales, parent, target)
    return dict(id=ident, branch="p1", parent_id="p1", parent_score=.5,
        parent_source_sha256=args["parent_source_sha256"], source_sha256=args["source_sha256"],
        proposed_sha256=args["proposed_sha256"], source_repaired=False,
        status="evaluated", score=.5 + delta, actual_delta=delta,
        hypothesis=None, injected_ids=list(injected), controlled_change=built["proof"],
        code_fragment=built["code"] if delta > 0 else "", operator="intervention",
        change_excerpt="-uniform(-0.06, 0.06)\n+uniform(-0.03, 0.03)\n")


def test_eviction_preserves_source_and_exposure_proofs_in_lessons_fragments_and_reflection():
    kb = KnowledgeBase(max_experiments=1, max_fragments=1)
    first = experiment()
    kb.record(first)
    lesson, = kb.add_lessons([dict(rule="Local scale observation", polarity="positive", scope="branch",
                                   evidence_ids=["e1"])], allowed_evidence_ids=["e1"])
    second = experiment("e2", parent="0.03", target="0.12", delta=-.125, injected=[lesson])
    kb.record(second)
    third = experiment("e3", parent="0.12", target="0.015")
    kb.record(third)
    assert [row["id"] for row in kb.experiments] == ["e3"]
    state = kb.state()
    retained = state["lessons"][0]
    assert retained["evidence_snapshots"][0]["controlled_change"] == first["controlled_change"]
    assert retained["exposure_snapshots"][0]["controlled_change"] == second["controlled_change"]
    assert state["fragments"][0]["evidence"]["controlled_change"] == third["controlled_change"]
    restored = KnowledgeBase(max_experiments=1, max_fragments=1)
    restored.load_state(json.loads(json.dumps(state)))
    assert restored.state() == state
    records, lessons = restored.reflection_context(1)
    assert {row["id"] for row in records} == {"e1", "e2", "e3"}
    for row in records:
        expected = {"e1": first, "e2": second, "e3": third}[row["id"]]
        assert row["controlled_change"] == expected["controlled_change"]
        assert row["change_excerpt"] == expected["change_excerpt"]
        assert "code_fragment" not in row
    records[0]["controlled_change"]["scope"] = "tampered returned context"
    assert restored.state() == state


@pytest.mark.parametrize("damage", ["proof", "source", "parent", "proposed", "repaired", "null", "unknown"])
def test_kb_insert_revalidates_proof_and_is_atomic(damage):
    kb = KnowledgeBase()
    raw = experiment()
    if damage == "proof": raw["controlled_change"]["unchanged_outside_assignment"] = 1
    elif damage in ("source", "parent", "proposed"):
        raw[{"source": "source_sha256", "parent": "parent_source_sha256", "proposed": "proposed_sha256"}[damage]] = "0" * 64
    elif damage == "repaired": raw["source_repaired"] = True
    elif damage == "null": raw["controlled_change"] = None
    else: raw["controlled_change_typo"] = raw.pop("controlled_change")
    before = kb.state()
    with pytest.raises(ValueError): kb.record(raw)
    assert kb.state() == before


@pytest.mark.parametrize("where", ["experiment", "retained_source", "fragment"])
def test_even_self_consistent_snapshot_digest_cannot_approve_forged_builder_proof(where):
    kb = KnowledgeBase(max_experiments=1)
    kb.record(experiment())
    if where != "fragment":
        kb.add_lessons([dict(rule="Local scale observation", polarity="positive", scope="branch",
                            evidence_ids=["e1"])], allowed_evidence_ids=["e1"])
    if where != "experiment": kb.record(experiment("e2", target="0.12"))
    original = kb.state()
    state = copy.deepcopy(original)
    if where == "experiment": target = state["experiments"][0]
    elif where == "retained_source": target = state["lessons"][0]["evidence_snapshots"][0]
    else: target = state["fragments"][0]["evidence"]
    target["controlled_change"]["unchanged_outside_assignment"] = 1
    # Bool/int dict equality would accept this. A caller also updates the index;
    # verification must re-derive, not rely solely on that untrusted digest.
    snapshot = copy.deepcopy(target)
    if "code_fragment" in snapshot:
        code = snapshot.pop("code_fragment")
        snapshot["code_fragment_sha256"] = digest(code) if code else None
    state["seen_experiments"][target["id"]] = hashlib.sha256(canonical(snapshot).encode()).hexdigest()
    with pytest.raises(ValueError, match="exact builder proof"):
        kb.load_state(state)
    assert kb.state() == original


def test_legacy_record_state_and_digest_remain_byte_exact_and_unknown_fields_reject():
    kb = KnowledgeBase()
    raw = dict(id="legacy_1", branch="p1", status="eval_failed", parent_score=.8,
               source_sha256="a" * 64, proposed_sha256="a" * 64, hypothesis=None)
    kb.record(raw)
    state = kb.state()
    assert "controlled_change" not in state["experiments"][0]
    # Captured before this optional-field implementation, not generated by it.
    assert hashlib.sha256(canonical(state).encode()).hexdigest() == "ca119f9a17b99f866d39239ba13f489a5c444299fdc23d08da739a292ace1477"
    restored = KnowledgeBase()
    restored.load_state(json.loads(json.dumps(state)))
    assert canonical(restored.state()) == canonical(state)
    with pytest.raises(ValueError, match="unknown experiment fields"):
        restored.record({**raw, "id": "legacy_2", "unrecognized": None})


def test_default_proposal_does_not_invent_a_controlled_proof():
    fixture, ident, built, validation = begin_fixture()
    fixture.research.proposal(ident, response(), built["code"], validation)
    assert "controlled_change" not in fixture.research.pending[ident]
    record = fixture.research.resolve(ident, status="evaluated", score=.9, source=built["code"])
    assert "controlled_change" not in record
    assert "controlled_change" not in fixture.research.knowledge.experiments[0]
