"""Public import of two real observations, using saved bytes and inert clients."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from sciel.engine import external_research as external
from sciel.engine.circle_generation import CircleGeneration
from sciel.engine.research import ResearchController
from sciel.engine.types import Snapshot
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.knowledge_base.base import _snapshot
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.run_log import RunLog

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def actual_history():
    from research_history_fixture import build_history
    return build_history()


class InertModel:
    models = ["anthropic/claude-sonnet-4.5"]
    max_retries = 1

    def __init__(self, model):
        self.models = [model]
        self.calls = []

    def generate(self, *args, **kwargs):
        pytest.fail("foreign history import dispatched a model")


def construct(capsule, final, path):
    cfg = {**final["config"], external.CONFIG_KEY: canonical_sha(capsule), "comparison_reviews": True}
    model = InertModel(cfg['model'])
    log = RunLog(str(path / "events.jsonl"))
    log.run_start(scope="inert actual v2 history test")
    charges = []
    controller = ResearchController(cfg, model, log, charges.append, metric=final["metric"])
    worker = CircleGeneration(controller, execution_profile=builder.EXECUTION_PROFILE)
    old = final["knowledge"]["experiments"][0]
    parent = NS(id="p1", parent_id=None, island=0, code=builder.source_for(),
                score=old["parent_score"], cases=old["parent_cases"], card="")
    return NS(c=controller, worker=worker, model=model, log=log, charges=charges,
              cfg=cfg, parent=parent)


def import_history(fixture, capsule):
    return fixture.worker.import_external_history(capsule,
        expected_sha256=canonical_sha(capsule), parent=fixture.parent)


def rows(log):
    return [json.loads(line) for line in Path(log.path).read_text().splitlines()]


def test_actual_public_import_preserves_two_reviews_three_groups_and_zero_local_activity(actual_history, tmp_path):
    capsule, final = actual_history
    before = copy.deepcopy(capsule)
    f = construct(capsule, final, tmp_path)
    assert import_history(f, capsule) is True
    assert f.c.knowledge.state() == final["knowledge"]
    assert f.c.conclusions == final["conclusions"]
    assert len(f.c.conclusions) == 2 and len(f.c.knowledge.comparisons) == 3
    assert f.c._comparison_reviews == {}
    assert all(getattr(f.c, key) == 0 for key in (
        "attempts", "outcomes", "valid_hypotheses", "aux_calls", "aux_cost",
        "reflections", "reflection_changes", "reflected_through"))
    assert f.worker.calls_started == f.worker.calls_returned == f.worker.evaluations == 0
    assert f.worker.seen_proofs == [] and f.model.calls == f.charges == []
    assert capsule == before
    event = rows(f.log)[1]
    assert event["imported_ids"] == ["r000001a1", "r000002a1"]
    assert len(event["source_journals"]) == 6 and event["attempt_id_floor"] == 2
    assert import_history(f, capsule) is False and len(rows(f.log)) == 2
    f.worker.assert_journal(f.log.path, f.log.anchor())
    restored = construct(capsule, final, tmp_path / "restored")
    restored.c.load_state(f.c.state())
    restored.worker.load_state(f.worker.state())
    restored.worker.assert_journal(f.log.path, f.log.anchor())
    assert restored.c.state() == f.c.state()


def test_actual_next_attempt_starts_at_three_and_keeps_foreign_outcomes_out_of_counters(actual_history, tmp_path):
    capsule, final = actual_history
    f = construct(capsule, final, tmp_path)
    import_history(f, capsule)
    assert external.experiment_number(f.c, 1) == 3
    context = f.c.context(NS(_all={"p1": f.parent}), f.parent, 0, 1)
    snap = Snapshot(island=0, op="circle_generate", model=f.model.models[0], iteration_no=1,
        parent_id="p1", parent_code=f.parent.code, parent_score=f.parent.score,
        task_prompt="oe/circle_packing", research_context=context, kb_lessons=context["lesson_texts"])
    ident, _ = f.c.begin(snap, 0, "circle_generate", f.worker.prepare_prompt(snap),
                         context_scope="circle_restart_body/1")
    assert ident == "r000003a1"
    assert f.c.attempts == 1 and f.c.outcomes == f.c.aux_calls == 0
    assert f.c.conclusions == final["conclusions"] and f.charges == f.model.calls == []
    external.assert_import_journal(f.c, rows(f.log))


@pytest.mark.parametrize("damage", ["parent_id", "parent_source", "nonroot_parent"])
def test_actual_import_binds_to_fresh_default_root_atomically(actual_history, tmp_path, damage):
    capsule, final = actual_history
    f = construct(capsule, final, tmp_path)
    if damage == "parent_id":
        f.parent.id = "p9"
    elif damage == "parent_source":
        f.parent.code = builder.source_for("return best_params.copy()\n")
    else:
        f.parent.parent_id = "p0"
    before = f.c.state()
    with pytest.raises(ValueError, match="parent binding"):
        import_history(f, capsule)
    assert f.c.state() == before and len(rows(f.log)) == 1


def test_evicted_actual_record_matches_only_exact_full_or_lossless_snapshot(actual_history, tmp_path):
    capsule, final = actual_history
    f = construct(capsule, final, tmp_path)
    import_history(f, capsule)
    record = copy.deepcopy(f.c.knowledge.experiments[0])
    snapshot = _snapshot(record)
    # Exercise the retention boundary without recording another synthetic
    # outcome or changing the immutable seen identity of the foreign record.
    f.c.knowledge._experiments = f.c.knowledge._experiments[1:]
    del f.c.conclusions[record["id"]]
    assert external.record_matches(f.c, record)
    assert external.record_matches(f.c, snapshot)
    changed = copy.deepcopy(snapshot)
    changed["actual_delta"] += .1
    assert not external.record_matches(f.c, changed)
    assert not external.record_matches(f.c, {"id": record["id"]})
    external.validate_loaded(external.state(f.c), config=f.cfg, metric=f.c.metric,
        knowledge=f.c.knowledge, conclusions=f.c.conclusions, comparison_reviews={})
    assert external.experiment_number(f.c, 1) == 3
    f.worker.assert_journal(f.log.path, f.log.anchor())


@pytest.mark.parametrize("damage", ["omitted_generation", "omitted_conclusion", "terminal", "group"])
def test_actual_mutated_history_rejected_before_publication_with_recomputed_outer_digest(actual_history, tmp_path, damage):
    original, final = actual_history
    capsule = copy.deepcopy(original)
    if damage == "omitted_generation":
        capsule["operations"] = [op for op in capsule["operations"]
            if op.get("experiment_id") != "r000002a1"]
    elif damage == "omitted_conclusion":
        capsule["operations"] = [op for op in capsule["operations"]
            if not (op["kind"] == "conclusion" and op["journal"] == 4)]
    elif damage == "terminal":
        capsule["terminal"]["knowledge_sha256"] = "0" * 64
    else:
        group = list(capsule["comparison_catalog"].values())[-1]
        group["pairs"][0]["challenger"]["raw_sum"] += .1
    f = construct(capsule, final, tmp_path)
    before = f.c.state()
    with pytest.raises(ValueError):
        import_history(f, capsule)
    assert f.c.state() == before and len(rows(f.log)) == 1
    assert f.model.calls == f.charges == []


def test_actual_missing_or_changed_import_journal_is_not_restorable(actual_history, tmp_path):
    capsule, final = actual_history
    f = construct(capsule, final, tmp_path)
    import_history(f, capsule)
    journal = rows(f.log)
    with pytest.raises(ValueError, match="durable import"):
        external.assert_import_journal(f.c, journal[:1])
    damaged = copy.deepcopy(journal)
    damaged[1]["source_journals"][3]["journal_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="durable import"):
        external.assert_import_journal(f.c, damaged)


@pytest.mark.parametrize("field,value", [("attempt", True), ("island", False)])
def test_actual_attempt_binding_rejects_boolean_numeric_alias(actual_history, tmp_path, field, value):
    original, final = actual_history
    capsule = copy.deepcopy(original)
    journal = [json.loads(line) for line in capsule["journals"][0]["journal"].splitlines()]
    attempt = next(event for event in journal if event.get("phase") == "attempt")
    # Python equality conflates these values; the provenance projection must
    # preserve literal types even when the caller recalculates the outer hash.
    assert attempt["evidence"][field] == value
    assert type(attempt["evidence"][field]) is int
    attempt["evidence"][field] = value
    capsule["journals"][0]["journal"] = "".join(json.dumps(event) + "\n" for event in journal)
    f = construct(capsule, final, tmp_path)
    before = f.c.state()
    with pytest.raises(ValueError, match="attempt binding"):
        import_history(f, capsule)
    assert f.c.state() == before and len(rows(f.log)) == 1
    assert f.model.calls == f.charges == []
