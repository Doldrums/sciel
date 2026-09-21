"""Idea island (arch-v4 #2): last island forces the idea operator; off by default."""

import os

from sciel.engine.factory import build_engine
from sciel.engine.llm.mock import MockLLM

import conftest


def _run(tmp_dir, idea=False, seed=42, budget=24):
    cfg = conftest.mock_config(budget=budget, seed=seed)
    if idea:
        cfg["idea_island"] = True
    engine = build_engine(cfg, conftest.TASKS_ROOT, tmp_dir, python=conftest.PYTHON,
                          llm=MockLLM())
    engine.run()
    return conftest.read_journal(tmp_dir)


def test_idea_template_exists():
    path = os.path.join(os.path.dirname(conftest.__file__), "..", "src", "sciel",
                        "engine", "prompt", "templates", "idea.txt")
    assert os.path.exists(path)


def test_idea_on_by_default(tmp_path):
    # idea island is part of the consolidated default engine
    journal = _run(str(tmp_path / "on_default"))
    ops = {e.get("op_used") for e in journal if e.get("event") == "iteration"}
    assert "idea" in ops


def test_idea_disables_explicitly(tmp_path):
    cfg = conftest.mock_config(budget=24, seed=42)
    cfg["idea_island"] = False
    d = str(tmp_path / "off")
    from sciel.engine.factory import build_engine
    from sciel.engine.llm.mock import MockLLM
    build_engine(cfg, conftest.TASKS_ROOT, d, python=conftest.PYTHON, llm=MockLLM()).run()
    ops = {e.get("op_used") for e in conftest.read_journal(d) if e.get("event") == "iteration"}
    assert "idea" not in ops


def test_idea_island_forces_idea_op(tmp_path):
    journal = _run(str(tmp_path / "on"), idea=True)
    # the last island (num_islands-1) must run the idea operator; others must not
    last = None
    for e in journal:
        if e.get("event") == "run_start":
            last = e["num_islands"] - 1
    idea_islands = {e["island"] for e in journal
                    if e.get("event") == "iteration" and e.get("op_used") == "idea"}
    assert idea_islands, "idea operator never fired on the idea island"
    assert idea_islands == {last}


def test_idea_run_is_deterministic(tmp_path):
    a = _run(str(tmp_path / "a"), idea=True)
    b = _run(str(tmp_path / "b"), idea=True)
    assert a == b
