"""Golden identity (§9а): same seed -> same journal with the mock LLM."""

import os

import pytest

from sciel.engine.factory import build_engine
from sciel.engine.llm.mock import MockLLM

import conftest


def _run(tmp_dir, seed=42, budget=30):
    cfg = conftest.mock_config(budget=budget, seed=seed)
    engine = build_engine(cfg, conftest.TASKS_ROOT, tmp_dir, python=conftest.PYTHON,
                          llm=MockLLM())
    engine.run()
    return conftest.read_journal(tmp_dir)


def test_same_seed_same_journal(tmp_path):
    a = _run(str(tmp_path / "run_a"))
    b = _run(str(tmp_path / "run_b"))
    assert len(a) == len(b)
    for i, (ea, eb) in enumerate(zip(a, b)):
        assert ea == eb, f"divergence at event {i}: {ea.get('event')}"


def test_different_seed_diverges(tmp_path):
    a = _run(str(tmp_path / "run_a"), seed=42)
    b = _run(str(tmp_path / "run_c"), seed=7)
    assert a != b


def test_all_event_types_present(tmp_path):
    a = _run(str(tmp_path / "run_a"))
    types = {e["event"] for e in a}
    assert {"run_start", "iteration", "sh_cut", "checkpoint", "run_end"} <= types


def test_escalation_recovers_from_dup(tmp_path):
    # Pin the relevant transition, rather than hoping a prompt hash eventually
    # draws the duplicate branch of the broad mock distribution.
    from pathlib import Path
    from sciel.engine.llm.client import LLMResponse
    from sciel.eval.cascade import CascadeOutcome
    from sciel.eval.facade import EvalResult
    seed = Path(conftest.TASKS_ROOT, 'test/toy_quadratic/initial_program.py').read_text()
    child = 'def run_search():\n    return (0.25, 0.75)\n'
    class Provider(MockLLM):
        calls = 0
        def generate(self, prompt, model, temperature=None):
            self.calls += 1
            assert self.calls <= 2
            return LLMResponse('```python\n' + (seed if self.calls == 1 else child) + '```',
                               model, .125, 10, 10)
    cfg = conftest.mock_config(budget=1)
    cfg.update(num_islands=1, inflight_cap=1, portfolio_seed=False,
               dynamic_islands=False, idea_island=False, crossover=False,
               milestone_fracs=[], op_fixed='diff', policy={'enabled': False})
    directory, provider, evaluations = str(tmp_path / 'run_a'), Provider(), []
    engine = build_engine(cfg, conftest.TASKS_ROOT, directory,
                          python=conftest.PYTHON, llm=provider)
    def evaluate(code):
        evaluations.append(code)
        score = .1 if code == seed else .2
        return CascadeOutcome(EvalResult(True, score, {}, timing=0.), score, 'full')
    engine._evaluate = evaluate
    engine.run()
    journal = conftest.read_journal(directory)
    escalated_success = [e for e in journal if e["event"] == "iteration"
                         and e.get("child_score") is not None
                         and e.get("meta", {}).get("escalated")]
    assert len(escalated_success) == 1
    assert provider.calls == 2 and evaluations == [seed, child]
    assert engine.B_evals == 1 and engine.cost_total == .25
    assert escalated_success[0]['meta']['dup_attempts'] == 1


@pytest.mark.parametrize("policy", ["stock_70_20_10", "ucb1", "ucb_sw"])
def test_kernel_determinism(policy, tmp_path):
    def run_once(sub):
        cfg = conftest.mock_config(budget=24, seed=42)
        cfg["selection_policy"] = policy
        # kernel arms ablate SELECTION; the default espalier policy needs a
        # tilt-capable kernel (the engine refuses loudly), so switch it off
        cfg["policy"] = {"enabled": False, "plateau_shift": False}
        d = str(tmp_path / sub)
        build_engine(cfg, conftest.TASKS_ROOT, d, python=conftest.PYTHON,
                     llm=MockLLM()).run()
        return conftest.read_journal(d)

    a, b = run_once("a"), run_once("b")
    assert a == b, f"{policy} not deterministic"
    props = [e.get("parent_propensity") for e in a if e["event"] == "iteration"
             and e.get("parent_propensity") is not None]
    assert props and all(0.0 < p <= 1.0 for p in props), f"{policy} propensity range"
