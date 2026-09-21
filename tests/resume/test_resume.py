"""RNG-resume (§9): restart from checkpoint yields the same decision sequence."""

import json
import os

from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.mock import MockLLM

import conftest


def _decisions(run_dir):
    seq = []
    with open(os.path.join(run_dir, "sciel_events.jsonl")) as f:
        for ln in f:
            e = json.loads(ln)
            if e["event"] != "iteration":
                continue
            seq.append((e["island"], e.get("op_used"), e.get("reason"),
                        e.get("parent_id"), e.get("child_score")))
    return seq


def _engine(cfg, run_dir):
    return build_engine(cfg, conftest.TASKS_ROOT, run_dir, python=conftest.PYTHON,
                        llm=MockLLM())


def test_resume_matches_continuous(tmp_path):
    budget, partial = 20, 8
    cont_dir = str(tmp_path / "cont")
    cont = _engine(conftest.mock_config(budget=budget), cont_dir)
    cont_result = cont.run()
    continuous = _decisions(cont_dir)

    split_dir = str(tmp_path / "split")
    e1 = _engine(conftest.mock_config(budget=budget), split_dir)
    e1._stop_at = partial
    e1.run()
    ckpt = os.path.join(split_dir, "checkpoint.json")

    e2 = _engine(conftest.mock_config(budget=budget), split_dir)
    checkpoint.load(e2, ckpt)
    e2.budget = budget
    e2.run()
    split = _decisions(split_dir)

    assert split == continuous, "resumed decision sequence diverged from continuous"
    assert e2.n_events == cont_result["n_events"], "resume leaked n_events"
    assert abs(e2.cost_total - cont_result["cost_total"]) < 1e-9, "resume leaked cost_total"
    assert e2.B_evals == cont_result["B_evals"]
