"""Structural invariant guards: I-13 currency locality, I-11 backstop."""

import os

from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse

import conftest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src", "sciel")


def test_currency_only_in_reward(  ):
    forbidden = ("d_pop =", "d_isl =", "r_raw =", "improvement_event(self")
    offenders = []
    for dirpath, _dirs, files in os.walk(SRC):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            if os.path.basename(path) == "reward.py":
                continue
            text = open(path).read()
            for pat in forbidden:
                if pat in text:
                    offenders.append((path, pat))
    assert not offenders, f"reward formulas outside reward.py (I-13): {offenders}"


class _AlwaysFailLLM:
    models = ["fail"]

    def choose_model(self, rng):
        return "fail"

    def generate(self, prompt, model, temperature=None):
        code = "def run_search():\n    raise RuntimeError('always fail')\n"
        return LLMResponse(text=f"```python\n{code}```\n", model="fail", cost=0.0,
                           prompt_tokens=1, completion_tokens=1)


def test_backstop_terminates_on_perpetual_failure(tmp_path):
    cfg = conftest.mock_config(budget=5, seed=1)
    cfg["backstop_k"] = 2
    engine = build_engine(cfg, conftest.TASKS_ROOT, str(tmp_path / "bs"),
                          python=conftest.PYTHON, llm=_AlwaysFailLLM())
    summary = engine.run()
    assert summary["reason"] == "backstop"
    assert summary["B_evals"] == 0
    assert engine.iteration_no >= cfg["backstop_k"] * cfg["budget"]
