"""Migration keeps source ancestry without evaluating or generating any code."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest

from sciel.engine.container_db import ContainerDatabase
from sciel.engine.db import Database
from sciel.engine.research import ResearchController


def database(kind, *, cap=2, islands=3):
    options = dict(num_islands=islands, pop_cap=cap, grid_bins=1)
    if kind == "legacy":
        return Database(**options)
    return ContainerDatabase(**options, container={"kind": "pool", "strict_cap": True,
        "selection": "shared", "k": 4, "max_centres": 6, "seed": 0,
        "temperature_alpha": .02, "split_min_points": 4,
        "vq_threshold": .25, "novelty_threshold": .98})


class NoCalls:
    models = ["local-unused"]
    max_retries = 1

    def generate(self, *args, **kwargs):
        raise AssertionError("this test must not call a model")


class NoLog:
    def research(self, **kwargs):
        raise AssertionError("context retrieval must not create research events")


def controller():
    def no_charge(cost):
        raise AssertionError("context retrieval must not charge a request")

    return ResearchController({"model": None, "max_lessons": 50,
        "max_experiments": 200, "max_fragments": 20, "retrieve_k": 3},
        NoCalls(), NoLog(), no_charge)


def add_family(db):
    root = db.add("radius = 0.1\n", .2, {"validity": 1}, 0, "seed")
    improved = db.add("radius = 0.2\n", .4, {"validity": 1}, 0,
                      "gen", parent_id=root.id)
    return root, improved


def teach(research, root, improved):
    source = hashlib.sha256(improved.code.encode()).hexdigest()
    research.knowledge.record({"id": "r000001a1", "branch": root.id,
        "parent_id": root.id, "parent_score": root.score, "child_id": improved.id,
        "source_sha256": source, "proposed_sha256": source,
        "hypothesis": None, "status": "evaluated", "score": improved.score,
        "actual_delta": improved.score - root.score, "injected_ids": [],
        "code_fragment": improved.code})
    return research.knowledge.add_lessons([{
        "rule": "Increasing the radius was associated with a higher measured score.",
        "polarity": "positive", "scope": "branch", "evidence_ids": ["r000001a1"]}],
        allowed_evidence_ids=["r000001a1"])[0]


@pytest.mark.parametrize("kind", ["legacy", "pool"])
def test_migration_changes_only_ancestry_not_admission_or_retention(kind):
    """The old copy path is an oracle for fitness, order, cap and archive behavior."""
    db = database(kind, cap=3)
    root, improved = add_family(db)
    # Duplicate source bodies still migrate once and still avoid the destination.
    db.add(improved.code, .3, {"validity": 1}, 0, "gen", parent_id=root.id)
    db.add("destination = 1\n", .1, {}, 1, "seed")
    db.add("destination = 2\n", .15, {}, 1, "seed")
    old_copy = database(kind, cap=3)
    old_copy.load_state(json.loads(json.dumps(db.state())))
    start = db._counter
    best = db.pop_best
    source_ids = {p.code: p.id for p in (root, improved)}

    assert db.migrate_unique_elites(0, [1]) == 2
    # Old behavior admitted the same ranked unique programs, with no parent link.
    old_copy.add(improved.code, improved.score, improved.cases, 1, "migrant")
    old_copy.add(root.code, root.score, root.cases, 1, "migrant")
    migrated = [p for p in db._all.values() if int(p.id[1:]) > start]
    assert len(migrated) == 2
    for prog in migrated:
        source = db.get(source_ids[prog.code])
        assert prog.parent_id == source.id
        assert prog.gen == source.gen + 1
        assert prog.origin == "migrant" and prog.island == 1
    assert db.pop_best == best
    # Retention may evict a migrant immediately; all ancestry stays in history.
    without_new_ancestry = copy.deepcopy(db.state())
    for prog in without_new_ancestry["programs"]:
        if int(prog["id"][1:]) > start:
            prog.update(parent_id=None, gen=0)
    assert without_new_ancestry == old_copy.state()


@pytest.mark.parametrize("kind", ["legacy", "pool"])
def test_migration_copies_exact_card_without_creating_research_observations(kind):
    db = database(kind, cap=1)
    root, improved = add_family(db)
    improved.card = '{"family":"grid","mechanism":"Keep x < 0","strategy":"Preserve radii"}'
    research = controller()
    lesson_id = teach(research, root, improved)
    before = copy.deepcopy(research.knowledge.state())
    assert db.migrate_unique_elites(0, [1]) == 1
    migrated = next(iter(db.islands[1].programs.values()))
    assert migrated.code == improved.code and migrated.card == improved.card
    assert migrated.parent_id == improved.id
    context = research.context(db, migrated, 1, 2)
    assert context['branch'] == root.id and context['injected_ids'] == [lesson_id]
    assert research.knowledge.state() == before
    restored = database(kind, cap=1)
    restored.load_state(json.loads(json.dumps(db.state())))
    assert restored.get(migrated.id).card == improved.card


@pytest.mark.parametrize("kind", ["legacy", "pool"])
def test_migrant_descendants_keep_branch_lessons_across_two_moves_and_json_resume(kind):
    db = database(kind, cap=1, islands=4)
    root, improved = add_family(db)
    # The original ancestor can be inactive; its history is still authoritative.
    assert root.id not in db.islands[0].programs
    research = controller()
    lesson_id = teach(research, root, improved)
    original = research.context(db, improved, 0, 1)
    assert original["branch"] == root.id
    assert original["injected_ids"] == [lesson_id]

    assert db.migrate_unique_elites(0, [1]) == 1
    first = next(iter(db.islands[1].programs.values()))
    assert first.parent_id == improved.id
    assert db.migrate_unique_elites(1, [2]) == 1
    second = next(iter(db.islands[2].programs.values()))
    assert second.parent_id == first.id  # retain the whole actual migration path
    child = db.add("radius = 0.25\n", .5, {"validity": 1}, 2,
                   "gen", parent_id=second.id)
    assert second.id not in db.islands[2].programs

    resumed = database(kind, cap=1, islands=4)
    resumed.load_state(json.loads(json.dumps(db.state())))
    restored_research = controller()
    restored_research.knowledge.load_state(json.loads(json.dumps(research.knowledge.state())))
    assert resumed.state() == db.state()
    for current_db, current_research in ((db, research), (resumed, restored_research)):
        context = current_research.context(current_db, current_db.get(child.id), 2, 2)
        assert context["branch"] == root.id
        assert context["parent_id"] == child.id
        assert context["injected_ids"] == original["injected_ids"]
        assert context["lessons"] == original["lessons"]
        assert context["fragments"] == original["fragments"]

    # Identical bytes in an unrelated seed do not establish shared ancestry.
    unrelated = db.add(improved.code, improved.score, improved.cases, 3, "seed")
    foreign = research.context(db, unrelated, 3, 3)
    assert foreign["branch"] == unrelated.id
    assert foreign["injected_ids"] == [] and foreign["fragments"] == []


@pytest.mark.parametrize("kind", ["legacy", "pool"])
@pytest.mark.parametrize("cycle", ["self", "ancestor"])
def test_migration_does_not_hide_corrupt_source_ancestry_from_context_guard(kind, cycle):
    db = database(kind, cap=1)
    root, improved = add_family(db)
    # A corrupted saved ancestry is not legitimized by copying to another island.
    if cycle == "self":
        improved.parent_id = improved.id
    else:
        root.parent_id = improved.id
    restored = database(kind, cap=1)
    restored.load_state(json.loads(json.dumps(db.state())))
    assert restored.migrate_unique_elites(0, [1]) == 1
    migrant = next(iter(restored.islands[1].programs.values()))
    with pytest.raises(ValueError, match="cyclic program ancestry"):
        controller().context(restored, migrant, 1, 1)


@pytest.mark.parametrize("kind", ["legacy", "pool"])
def test_old_parentless_migrant_checkpoint_is_not_repaired_by_source_hash_guess(kind):
    db = database(kind)
    _, improved = add_family(db)
    old = db.add(improved.code, improved.score, improved.cases, 1, "migrant")
    restored = database(kind)
    restored.load_state(json.loads(json.dumps(db.state())))
    assert restored.get(old.id).parent_id is None
    assert controller().context(restored, restored.get(old.id), 1, 1)["branch"] == old.id


@pytest.mark.parametrize("kind", ["legacy", "pool"])
def test_self_migration_or_empty_survivors_does_not_create_ids_or_cycles(kind):
    db = database(kind)
    add_family(db)
    before = copy.deepcopy(db.state())
    assert db.migrate_unique_elites(0, []) == 0
    assert db.migrate_unique_elites(0, [0]) == 0
    assert db.state() == before
