"""Arch-v3 features: cross operator, strategy cards, no-op directive, spawn plumbing."""

import random

from sciel.engine.operators import OpBandit
from sciel.engine.db import Database
from sciel.engine.prompt.build import compose_directive
from sciel.engine.prompt.parse import extract_card
from sciel.engine.schedule import Schedule


def test_cross_arm_available_when_configured():
    b = OpBandit(arms=["diff", "full_rewrite", "error_repair", "cross"])
    assert "cross" in b._available(fresh_failure=False)
    assert "error_repair" not in b._available(fresh_failure=False)
    op, prop, stat = b.pick(False, random.Random(0))
    assert op in {"diff", "full_rewrite", "cross"} and 0 < prop <= 1


def test_default_arms_unchanged():
    b = OpBandit()
    assert b._available(False) == ["diff", "full_rewrite"]
    assert set(b.state()["events"]) == {"diff", "full_rewrite", "error_repair"}


def test_extract_card_takes_prose_before_fence():
    text = "An improved two-stage approach with annealing.\nMore prose.\n```python\nx=1\n```"
    card = extract_card(text)
    assert card.startswith("An improved two-stage approach")
    assert "```" not in card and "x=1" not in card


def test_extract_card_empty_when_code_first():
    assert extract_card("```python\nx=1\n```") == ""


def test_noop_directive_line():
    d = compose_directive(30.0, 5.0, plateau=False, parent_noop=True)
    assert "SCORE OBSERVATION" in d
    assert "equivalent behavior" in d
    assert "SCORE OBSERVATION" not in compose_directive(30.0, 5.0, plateau=False)


def test_db_add_island_and_card_roundtrip():
    db = Database(num_islands=2)
    p = db.add("x=1", 0.5, {}, 0, origin="gen", card="greedy rings")
    assert db.get(p.id).card == "greedy rings"
    idx = db.add_island()
    assert idx == 2 and len(db.islands) == 3
    state = db.state()
    db2 = Database(num_islands=2)
    db2.load_state(state)                      # spawned island tolerated on load
    assert len(db2.islands) == 3
    assert db2.get(p.id).card == "greedy rings"


def test_schedule_register_island():
    s = Schedule(2, "sh")
    s.register_island(2)
    assert 2 in s.survivors and s.dispatched[2] == 0
