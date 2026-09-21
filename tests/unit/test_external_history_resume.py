"""Synthetic resume-boundary checks after verified import and bounded review aging.

The foreign reducer is replaced by an already-validated detached fixture. The
actual resume checker and auxiliary response parser process the local journal;
this is not evidence of a real model run or an Engine-level capacity campaign.
"""
import json
from types import SimpleNamespace as NS

import pytest

from sciel.engine import external_research_resume as resume
from sciel.hypothesis.auxiliary_schema import identity, response_format
from sciel.hypothesis.comparison_review import MAX_STORED
from sciel.knowledge_base.catalogs import canonical_sha


def aging_fixture(monkeypatch, *, displaced):
    foreign_id = "comparison:foreign_retained_by_lesson_snapshot"
    foreign_card = {"fixture": "already authenticated foreign card"}
    original = NS(lessons=[], _next_lesson_id=1)
    monkeypatch.setattr(resume, "validate", lambda *a, **kw: {
        "knowledge": original, "comparison_reviews": {foreign_id: foreign_card}})
    retained = {foreign_id: canonical_sha(foreign_card)}
    rows = []
    local_count = MAX_STORED if displaced else MAX_STORED - 1
    for offset in range(0, local_count, 4):
        through = offset // 4 + 1
        ids = [f"comparison:local_{index}" for index in range(offset, min(offset + 4, local_count))]
        proposals = [dict(comparison_id=ident, comparison_sha256="a" * 64,
            interpretation="Synthetic comparison interpretation.", limitations="Fixture only.",
            next_hypothesis=None) for ident in ids]
        raw = json.dumps(dict(lessons=[], retire_ids=[], comparison_reviews=proposals))
        for ident in ids:
            retained[ident] = canonical_sha({"fixture": ident})
        # Saved host receipt after this successful reflection. The production
        # card cap is independent of later retention of each source group.
        retained = dict(list(retained.items())[-MAX_STORED:])
        receipt = dict(schema="sciel-external-research-transition/1",
            lessons_sha256=canonical_sha([]), next_lesson_id=1,
            comparison_reviews=dict(retained))
        rows.extend([
            dict(event="research", phase="reflection_context", through=through,
                 lesson_ids=[], comparison_review_source_ids=ids),
            dict(event="research", phase="aux_start", kind="reflection", reference=through,
                 call_index=through, response_format=response_format("reflection", comparison_reviews=True),
                 auxiliary_schema=identity(comparison_reviews=True)),
            dict(event="research", phase="aux_result", kind="reflection", reference=through,
                 call_index=through, response=raw, response_sha256=resume._digest(raw),
                 response_truncated=False, structured_response_usable=True,
                 finish_reason="stop", refusal=None, cost=0.0),
            dict(event="research", phase="reflection", through=through, status="parsed", error=None,
                 applied={"lesson_ids": [], "retired_ids": []}, comparison_review_ids=ids,
                 external_history_transition=receipt),
        ])
    # All local groups have since aged out; the foreign group's evidence remains
    # available. Its old card must not resurrect after a capacity eviction.
    controller = NS(_external_history={"capsule": {}},
        config={"comparison_reviews": True, "comparison_review_version": 1}, metric={},
        knowledge=NS(lessons=[], _next_lesson_id=1), _comparison_reviews={},
        _retained_comparison_reviews=lambda: {}, _review_groups=lambda: [{"id": foreign_id}])
    return controller, rows


def test_capacity_evicted_foreign_review_stays_absent_after_local_groups_age_out(monkeypatch):
    controller, rows = aging_fixture(monkeypatch, displaced=True)
    resume.assert_authoring_journal(controller, rows)


def test_foreign_review_cannot_disappear_if_last_durable_receipt_still_retains_it(monkeypatch):
    controller, rows = aging_fixture(monkeypatch, displaced=False)
    with pytest.raises(ValueError, match="last durable transition"):
        resume.assert_authoring_journal(controller, rows)


def test_missing_transition_receipt_does_not_authorize_empty_memory(monkeypatch):
    controller, rows = aging_fixture(monkeypatch, displaced=True)
    del rows[-1]["external_history_transition"]
    with pytest.raises(ValueError, match="transition receipt missing"):
        resume.assert_authoring_journal(controller, rows)
