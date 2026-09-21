"""C3 knowledge base machinery (§5.10): alignment gate, lift, retrieve, dedup, state."""

import pytest

from sciel.components.kb.kb import KnowledgeBase
from sciel.components.tail import BaseRate, BranchCalibration

DIFF = ("def run_search():\n    return local_search_with_restart_and_perturbation()\n")


def test_aligned_lesson_minted():
    kb = KnowledgeBase()
    ls = kb.add_lesson("Restarting local search escapes local minima",
                       "local search restart perturbation", DIFF, iteration=5)
    assert ls is not None and ls.id == 1
    assert len(kb.lessons) == 1


def test_unaligned_lesson_rejected():
    kb = KnowledgeBase()
    ls = kb.add_lesson("Something happened", "quantum entanglement teleport", DIFF,
                       iteration=5)
    assert ls is None
    assert not kb.lessons


def test_single_token_mechanism_rejected():
    kb = KnowledgeBase()
    ls = kb.add_lesson("Restart helps a lot", "restart", DIFF, 1)
    assert ls is None  # 'restart' is in DIFF, but < min_mechanism_tokens
    assert not kb.lessons


def test_dedup_similar_lesson():
    kb = KnowledgeBase()
    a = kb.add_lesson("Restarting local search escapes local minima",
                      "local search restart", DIFF, 1)
    b = kb.add_lesson("Restarting local search escapes local minima again",
                      "local search restart", DIFF, 2)
    assert a is not None and b is None


def test_lift_against_matched_base_rate():
    kb = KnowledgeBase()
    ls = kb.add_lesson("Local restart helps", "local search restart", DIFF, 1)
    kb.note_exposure(ls.id, outcome=True, baseline=0.2)
    kb.note_exposure(ls.id, outcome=True, baseline=0.2)
    kb.note_exposure(ls.id, outcome=False, baseline=0.2)
    assert ls.exposures == 3 and ls.exposure_events == 2
    assert ls.lift() == pytest.approx(2 / 3 - 0.2)


def test_retrieve_orders_by_lift():
    kb = KnowledgeBase(retrieve_k=1)
    a = kb.add_lesson("Restart local search", "local search restart", DIFF, 1)
    b = kb.add_lesson("Anneal the temperature slowly",
                      "simulated annealing temperature",
                      "simulated annealing temperature schedule", 2)
    kb.note_exposure(b.id, True, 0.1)
    top = kb.retrieve(parent_score=0.5)
    assert len(top) == 1 and top[0].id == b.id


def test_kb_state_roundtrip():
    kb = KnowledgeBase()
    ls = kb.add_lesson("Restart local search", "local search restart", DIFF, 1)
    kb.note_exposure(ls.id, True, 0.3)
    d = kb.state()
    kb2 = KnowledgeBase()
    kb2.load_state(d)
    assert kb2.state() == d


def test_base_rate_predictable():
    br = BaseRate()
    assert br.rate(0, 0) == 0.0
    br.observe(0, 0, True)
    assert br.rate(0, 0) == 1.0
    br.observe(0, 0, False)
    assert br.rate(0, 0) == 0.5


def test_branch_calibration_mean():
    bc = BranchCalibration()
    assert bc.mean("p1") is None
    bc.note("p1", 0.2)
    bc.note("p1", 0.4)
    assert bc.mean("p1") == pytest.approx(0.3)
    d = bc.state()
    bc2 = BranchCalibration()
    bc2.load_state(d)
    assert bc2.mean("p1") == pytest.approx(0.3)
