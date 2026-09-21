"""Regressions for confirmation, evaluator contracts and subprocess cleanup."""

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
from unittest.mock import patch

import numpy as np
import pytest

from sciel.eval._runner import _normalize
from sciel.eval.cascade import cascade_evaluate
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from sciel.eval.sandbox import run_stage


ROOT = Path(__file__).resolve().parents[2]


def evaluator(task):
    spec = importlib.util.spec_from_file_location(
        "review_evaluator", ROOT / "tasks" / task / "evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, code):
    path.write_text(textwrap.dedent(code))
    return str(path)


@pytest.mark.parametrize("confirmation", [
    EvalResult(False, 0.0, artifacts={"error": "timed out"}),
    EvalResult(True, .9),
    EvalResult(True, .9, cases={"n_trials": 2.0}, confirmed_sampled=True),
])
def test_confirmation_failure_or_short_sample_never_restores_full(confirmation):
    task = load_task(str(ROOT / "tasks/oe/heilbronn_triangle"))
    with patch("sciel.eval.cascade.run_stage", side_effect=[
            EvalResult(True, .8), EvalResult(True, .99), confirmation]):
        result = cascade_evaluate(task, "unused", record_candidate=True)
    assert not result.result.ok
    assert result.reached == "confirm"
    assert not result.confirmed
    assert result.raw_first_score == .99
    assert result.result.artifacts.get("error")


def test_sufficient_confirmation_replaces_full():
    task = load_task(str(ROOT / "tasks/oe/heilbronn_triangle"))
    confirmation = EvalResult(True, .6, cases={"n_trials": 3.0}, confirmed_sampled=True)
    with patch("sciel.eval.cascade.run_stage", side_effect=[
            EvalResult(True, .8), EvalResult(True, .99), confirmation]):
        result = cascade_evaluate(task, "unused", record_candidate=True)
    assert result.confirmed and result.result.score == .6


@pytest.mark.parametrize("task", [
    "oe/heilbronn_triangle", "oe/circle_packing", "oe/circle_packing_seed",
    "oe/matmul_tensor", "math/beat_the_average", "math/difference_bases",
    "math/edges_vs_triangles", "math/erdos_squarefree",
])
def test_every_worst_of_n_wrapper_keeps_failed_trials(task):
    ev = evaluator(task)
    failed = {"ok": False, "score": 0.0, "cases": {}, "artifacts": {"error": "failed"}}
    good = {"ok": True, "score": .99, "cases": {}, "artifacts": {}}
    with patch.object(ev, "full", side_effect=[failed, good, failed]) as full:
        result = ev.confirm("unused", 3)
    assert full.call_count == 3
    assert not result["ok"]
    assert result["cases"]["n_trials"] == 3


def test_function_minimization_confirmation_requires_every_trial():
    ev = evaluator("oe/function_minimization")
    with patch.object(ev, "_trial", side_effect=[
            RuntimeError("failed"), (-1.704, .678, -1.519, 2, True), RuntimeError("failed")]):
        result = ev.confirm("unused", 3)
    assert not result["ok"]
    assert result["cases"]["reliability_score"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("reported_trials", [None, 1, 2, 3.5])
def test_runner_requires_all_requested_confirm_samples(tmp_path, reported_trials):
    cases = {} if reported_trials is None else {"n_trials": reported_trials}
    path = write(tmp_path / "evaluator.py", f"""
        def confirm(p, n):
            return {{'ok': True, 'score': .8, 'cases': {cases!r}}}
    """)
    result = run_stage(path, "confirm", "unused", timeout=5, n=3)
    assert not result.confirmed_sampled


def test_typed_eval_result_runs_through_the_subprocess(tmp_path):
    path = write(tmp_path / "evaluator.py", """
        from sciel.eval.facade import EvalResult
        def smoke(p):
            return EvalResult(True, .75, cases={'detail': .2},
                              artifacts={'detail': 'kept'}, stderr='note', cost=.1)
    """)
    result = run_stage(path, "smoke", "unused", timeout=5)
    assert result.ok and result.score == .75
    assert result.cases == {"detail": .2}
    assert result.artifacts["detail"] == "kept"
    assert result.stderr == "note" and result.cost == .1


def test_evaluation_child_does_not_inherit_provider_credentials(tmp_path, monkeypatch):
    secret_names = ("OPENAI_API_KEY", "HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "SSH_AUTH_SOCK")
    for name in secret_names:
        monkeypatch.setenv(name, "inert-test-credential")
    monkeypatch.setenv("SCIEL_TEST_RUNTIME_SETTING", "preserved")
    path = write(tmp_path / "evaluator.py", f"""
        import os
        def smoke(p):
            secrets_absent = all(name not in os.environ for name in {secret_names!r})
            runtime_kept = os.environ.get('SCIEL_TEST_RUNTIME_SETTING') == 'preserved'
            return {{'ok': secrets_absent and runtime_kept, 'score': 1.0}}
    """)
    result = run_stage(path, "smoke", "unused", timeout=5)
    assert result.ok, result.stderr
    assert all(os.environ[name] == "inert-test-credential" for name in secret_names)


@pytest.mark.parametrize("value", [None, True, False, np.bool_(True), "0.8", b"0.8", float("nan"), float("inf")])
@pytest.mark.parametrize("key", ["score", "combined_score"])
def test_invalid_scores_have_one_consistent_contract(value, key):
    result = _normalize({key: value})
    assert not result["ok"] and result["score"] == 0.0
    assert result["artifacts"].get("error")


def test_missing_score_is_not_a_successful_zero():
    assert not _normalize({"ok": True, "cases": {}})["ok"]
    assert _normalize({"ok": True, "score": np.float32(.75)})["score"] == .75
    assert _normalize({"ok": True, "cases": {"combined_score": .5}})["score"] == .5


def test_eval_result_subclass_preserves_contract():
    class ExtendedResult(EvalResult):
        pass
    assert _normalize(ExtendedResult(True, .7))["score"] == .7


def running(pid):
    status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                            capture_output=True, text=True, check=False).stdout.strip()
    return bool(status) and not status.startswith("Z")


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
@pytest.mark.parametrize("hang", [False, True])
def test_stage_cleans_children_on_success_and_timeout(tmp_path, hang):
    pidfile = tmp_path / "child.pid"
    path = write(tmp_path / "evaluator.py", f"""
        import subprocess, sys, time
        from pathlib import Path
        def smoke(p):
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            Path({str(pidfile)!r}).write_text(str(child.pid))
            if {hang!r}: time.sleep(20)
            return {{'ok': True, 'score': .5}}
    """)
    pid = None
    try:
        result = run_stage(path, "smoke", "unused", timeout=.5)
        pid = int(pidfile.read_text())
        assert result.ok is not hang
        deadline = time.monotonic() + 2
        while running(pid) and time.monotonic() < deadline:
            time.sleep(.01)
        assert not running(pid), "child survived stage cleanup"
    finally:
        if pid is not None and running(pid):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.skipif(os.name != "posix", reason="trial cleanup uses POSIX process trees")
def test_timed_out_trial_is_reaped_before_return(tmp_path):
    ev = evaluator("oe/function_minimization")
    pidfile = tmp_path / "trial.pid"

    path = write(tmp_path / "candidate.py", f"""
        import os, time
        from pathlib import Path
        def run_search():
            Path({str(pidfile)!r}).write_text(str(os.getpid()))
            time.sleep(20)
    """)

    started = time.monotonic()
    with pytest.raises(TimeoutError):
        ev._run_with_timeout(path, timeout_seconds=1)
    assert time.monotonic() - started < 3
    pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    write(tmp_path / "candidate.py", "def run_search(): return (7, 0)\n")
    assert ev._run_with_timeout(path)[0] == 7


def test_fresh_trials_do_not_replay_numpy_rng_state(tmp_path):
    ev = evaluator("oe/function_minimization")
    path = write(tmp_path / "candidate.py", "import numpy as np\ndef run_search(): return (float(np.random.random()), 0)\n")
    values = [ev._run_with_timeout(path)[0] for _ in range(3)]
    assert len(set(values)) == 3


@pytest.mark.parametrize("candidate_limit", [0, 1, 100000])
def test_objective_budget_belongs_to_evaluator(tmp_path, candidate_limit):
    ev = evaluator("oe/function_minimization")
    path = write(tmp_path / "candidate.py", f"""
        MAX_EVALS = {candidate_limit}
        def _objective(x, y): return 0.0
        def run_search():
            global MAX_EVALS
            MAX_EVALS = {candidate_limit}
            try:
                for _ in range(51): _objective(0, 0)
            except ValueError:
                pass
            return -1.704, .678
    """)
    result = ev.smoke(path)
    assert not result["ok"]
    assert "limit 50" in result["artifacts"]["error"]


def test_objective_query_counts_survive_trial_process(tmp_path):
    ev = evaluator("oe/function_minimization")
    path = write(tmp_path / "candidate.py", """
        MAX_EVALS = 0
        def _objective(x, y): return 0.0
        def run_search():
            for _ in range(3): _objective(0, 0)
            return -1.704, .678
    """)
    result = ev.full(path, num_trials=2)
    assert result["ok"]
    assert result["cases"]["objective_queries"] == 6
    assert result["cases"]["objective_counted_trials"] == 2
    assert result["cases"]["objective_query_budget"] == 50
