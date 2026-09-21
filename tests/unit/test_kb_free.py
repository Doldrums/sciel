"""Free KB: strategy cards minted as lessons, credited by lift, zero extra calls."""

import json

from sciel.components.kb.kb import KnowledgeBase
from sciel.engine.factory import build_engine
from sciel.engine.llm.mock import MockLLM

import conftest


def test_card_lesson_skips_alignment():
    kb = KnowledgeBase()
    # a prose card shares no tokens with code diff, but align=False accepts it
    card = "improved two stage annealing construction"
    assert kb.add_lesson(card, card, diff_text="np.zeros scipy minimize", iteration=1,
                          align=True) is None                 # code-gate rejects prose
    assert kb.add_lesson(card, card, diff_text="", iteration=1, align=False) is not None


def test_cold_start_explore_surfaces_new_lesson():
    kb = KnowledgeBase(retrieve_k=2)
    old = kb.add_lesson("old lesson idea", "old lesson", "", 1, align=False)
    kb.note_exposure(old.id, True, 0.1)                        # old has lift
    new = kb.add_lesson("brand new lesson idea", "brand new", "", 2, align=False)
    top = kb.retrieve(parent_score=0.5)
    ids = {l.id for l in top}
    assert old.id in ids and new.id in ids                    # explore reserves a slot


def _run(tmp_dir, kb_free=False, seed=42, budget=40):
    cfg = conftest.mock_config(budget=budget, seed=seed)
    if kb_free:
        cfg["kb_free"] = True
    build_engine(cfg, conftest.TASKS_ROOT, tmp_dir, python=conftest.PYTHON,
                 llm=MockLLM()).run()
    return conftest.read_journal(tmp_dir)


def test_kb_free_mints_without_extra_cost(tmp_path):
    off = _run(str(tmp_path / "off"))
    on = _run(str(tmp_path / "on"), kb_free=True)
    cost_off = next(e["cost_total"] for e in reversed(off) if e["event"] == "run_end")
    cost_on = next(e["cost_total"] for e in reversed(on) if e["event"] == "run_end")
    # free KB adds no LLM calls -> same total cost as the base run
    assert abs(cost_on - cost_off) < 1e-9
    # and it produced at least one postmortem (a lesson minted from a card)
    assert any(e["event"] == "postmortem" for e in on)


def test_kb_free_deterministic(tmp_path):
    a = _run(str(tmp_path / "a"), kb_free=True)
    b = _run(str(tmp_path / "b"), kb_free=True)
    assert a == b
