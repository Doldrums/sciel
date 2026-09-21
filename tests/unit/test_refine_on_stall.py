"""Exploit-on-stall (audit 2026-08-16): a stalled competitive line is deepened
(diff-biased operator + reformulation directive) instead of re-explored."""

import pytest

from sciel.engine.factory import build_engine
from sciel.engine.llm.mock import MockLLM

import conftest


def _run(tmp_dir, refine_stall, seed=42, budget=30):
    cfg = conftest.mock_config(budget=budget, seed=seed)
    cfg["refine_stall"] = refine_stall
    build_engine(cfg, conftest.TASKS_ROOT, tmp_dir, python=conftest.PYTHON,
                 llm=MockLLM()).run()
    return conftest.read_journal(tmp_dir)


def _refine_events(journal):
    return [e for e in journal
            if e.get("event") == "decision" and e.get("policy") == "refine"]


def _assert_evaluation_references(journal):
    """Check links after read_journal has canonicalized scalar and plural IDs."""
    identifiers = set()
    for row in journal:
        if row.get("policy") == "evaluation" and row.get("action") == "result":
            ident = row["evaluation_id"]
            assert type(ident) is str and ident, "evaluation ID is missing"
            assert ident not in identifiers, "an evaluation ID was reused"
            identifiers.add(ident)

    def visit(value, field=None):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, key)
        elif isinstance(value, list):
            for item in value:
                visit(item, field)
        elif field in {"evaluation_id", "evaluation_ids"} and value is not None:
            assert value in identifiers, "an evaluation reference has no recorded result"

    visit(journal)


def test_refine_fires_on_stall(tmp_path):
    # refine_stall=0 -> any competitive parent triggers the exploit hold
    journal = _run(str(tmp_path / "hot"), refine_stall=0)
    fired = _refine_events(journal)
    assert fired, "refine hold never fired even with refine_stall=0"
    ev = fired[0]
    assert ev["action"] == "intensify" and "parent_score" in ev and "stall" in ev


def test_refine_never_fires_when_threshold_unreachable(tmp_path):
    # a stall threshold above the whole budget can never be reached
    journal = _run(str(tmp_path / "cold"), refine_stall=10_000)
    assert not _refine_events(journal)


def test_refine_run_is_deterministic(tmp_path):
    a = _run(str(tmp_path / "a"), refine_stall=0)
    b = _run(str(tmp_path / "b"), refine_stall=0)
    _assert_evaluation_references(a)
    _assert_evaluation_references(b)
    assert a == b


@pytest.mark.parametrize("mutation", ["duplicate", "scalar_orphan", "plural_orphan"])
def test_reference_validation_keeps_duplicate_and_orphan_checks(mutation):
    row = {"policy": "evaluation", "action": "result", "evaluation_id": "<evaluation_id:1>"}
    journal = [row, {"evaluation_ids": ["<evaluation_id:1>"]}]
    _assert_evaluation_references(journal)
    if mutation == "duplicate":
        journal.append(dict(row))
    elif mutation == "scalar_orphan":
        journal.append({"evaluation_id": "<evaluation_id:2>"})
    else:
        journal[1]["evaluation_ids"].append("<evaluation_id:2>")
    with pytest.raises(AssertionError, match="reused|no recorded result"):
        _assert_evaluation_references(journal)
