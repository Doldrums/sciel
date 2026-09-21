"""Short runtime contracts, also runnable without the checkout's conftest/src path.

    SCIEL_EXPECT_INSTALLED=1 python -I -m pytest tests/integration -q \
        --noconftest --import-mode=importlib -o pythonpath=

All candidate programs are local deterministic fixtures; the resume run uses
MockLLM. No API key, model download, or paid generation is used.
"""

import importlib.resources
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import pytest

import sciel
from sciel.config import validate
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.mock import MockLLM
from sciel.engine.prompt.build import build
from sciel.engine.types import Snapshot
from sciel.eval.registry import load_task
from sciel.eval.sandbox import run_stage


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "tasks"
PROGRAMS = {
    "oe/function_minimization": "def run_search():\n    return (-1.704, 0.678)\n",
    "oe/circle_packing": (
        "import numpy as np\ndef run_packing():\n"
        "    centers = np.array([((i % 6 + .5) / 6, (i // 6 + .5) / 5) for i in range(26)])\n"
        "    radii = np.full(26, .07)\n    return centers, radii, float(radii.sum())\n"),
    "oe/heilbronn_triangle": (
        "import numpy as np\ndef heilbronn_triangle11():\n"
        "    x = np.linspace(.1, .9, 11)\n    return np.column_stack((x, .03 + .15*x*x))\n"),
}


@pytest.fixture(scope="module", autouse=True)
def runtime_preconditions():
    assert os.name == "posix", "SCiEL evaluator isolation currently requires POSIX"
    assert shutil.which("ps"), "function_minimization trial cleanup requires ps"
    if os.environ.get("SCIEL_EXPECT_INSTALLED") == "1":
        location = Path(sciel.__file__).resolve()
        purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
        assert location.is_relative_to(purelib), f"loaded checkout code instead of wheel: {location}"
        assert not location.is_relative_to(ROOT / "src")
        assert not any(p.resolve() == ROOT / "src" for p in map(Path, sys.path) if str(p))


def test_installed_prompt_templates_are_available():
    templates = importlib.resources.files("sciel.engine.prompt") / "templates"
    snapshot = Snapshot(0, "diff", "mock-model", 1, "p1", "def run_search(): return (0., 0.)", .1)
    for operator in ("diff", "full_rewrite", "error_repair", "cross", "idea"):
        assert (templates / f"{operator}.txt").is_file(), f"wheel omitted {operator} template"
        prompt = build(operator, snapshot)
        assert snapshot.parent_code in prompt and "```python" in prompt


@pytest.mark.parametrize("task_name", list(PROGRAMS))
@pytest.mark.parametrize("stage", ["smoke", "full", "confirm"])
def test_primary_task_stages_in_real_subprocess(task_name, stage, tmp_path):
    task = load_task(str(TASKS / task_name))
    program = tmp_path / "candidate.py"
    program.write_text(PROGRAMS[task_name])
    outcome = run_stage(task.evaluator_path, stage, str(program), timeout=15.0,
                        n=2 if stage == "confirm" else None, python=sys.executable)
    assert outcome.ok, (task_name, stage, outcome.stderr, outcome.artifacts)
    assert math.isfinite(outcome.score) and outcome.score > 0
    assert all(math.isfinite(value) for value in outcome.cases.values())
    assert all(key in outcome.cases for key in task.case_metrics if key.startswith("b_"))
    if stage == "confirm":
        assert outcome.confirmed_sampled and outcome.cases["n_trials"] == 2


@pytest.mark.parametrize("task_name", list(PROGRAMS))
def test_primary_task_failure_propagates_from_subprocess(task_name, tmp_path):
    task = load_task(str(TASKS / task_name))
    program = tmp_path / "failure.py"
    program.write_text('raise RuntimeError("runtime contract failure")\n')
    outcome = run_stage(task.evaluator_path, "smoke", str(program), timeout=15.0,
                        python=sys.executable)
    assert not outcome.ok and outcome.score == 0.0
    assert outcome.artifacts.get("error") or outcome.stderr


def _mock_config():
    return validate({
        "task": "test/toy_quadratic", "budget": 4, "num_islands": 1,
        "random_seed": 42, "schedule_policy": "round_robin", "milestone_fracs": [.5],
        "llm": {"mock": True}, "policy": {"enabled": False, "plateau_shift": False},
        "portfolio_seed": False, "idea_island": False, "dynamic_islands": False,
        "embed_grid": False, "embed_niche": False, "three_head": False,
        "confirm_records": False, "sciel_tasks_repo": None, "sciel_tasks_revision": None,
    })


def _decisions(run_dir):
    events = [json.loads(line) for line in (run_dir / "sciel_events.jsonl").read_text().splitlines()]
    return [(e["iteration_no"], e.get("child_id"), e.get("parent_id"), e.get("child_score"),
             e.get("reason"), e.get("op_used")) for e in events if e["event"] == "iteration"]


def _engine(run_dir):
    return build_engine(_mock_config(), str(TASKS), str(run_dir),
                        python=sys.executable, llm=MockLLM())


def _cli(*args):
    return subprocess.run([sys.executable, "-I", "-m", "sciel.cli.main", *args],
                          capture_output=True, text=True, timeout=60)


def test_checkpoint_resume_matches_continuous_mock_search(tmp_path):
    continuous_dir, split_dir = tmp_path / "continuous", tmp_path / "split"
    continuous = _engine(continuous_dir).run()
    split = _engine(split_dir)
    split._stop_at = 2
    assert split.run()["reason"] == "paused"

    # API restore is part of the runtime contract in source and wheel runs alike.
    resumed = _engine(split_dir)
    checkpoint.load(resumed, str(split_dir / "checkpoint.json"))
    summary = resumed.run()
    assert summary["reason"] == continuous["reason"] == "budget"
    assert summary["B_evals"] == continuous["B_evals"] == 4
    assert summary["pop_best"] == continuous["pop_best"]
    assert summary["cost_total"] == continuous["cost_total"] == 0.0
    assert _decisions(split_dir) == _decisions(continuous_dir)


def test_installed_cli_lists_tasks_and_resumes_checkpoint(tmp_path):
    # -I keeps these subprocesses independent of pytest's checkout path injection.
    # This test belongs to wheel validation; normal source-suite invocation also
    # has a package installation, checked here before invoking its entry point.
    probe = _cli("--help")
    if probe.returncode and "No module named 'sciel'" in probe.stderr \
            and os.environ.get("SCIEL_EXPECT_INSTALLED") != "1":
        pytest.skip("CLI isolation requires an installed SCiEL package")
    assert probe.returncode == 0, probe.stderr
    listed = _cli("tasks", "list", "--tasks-root", str(TASKS))
    assert listed.returncode == 0, listed.stderr
    assert all(name in listed.stdout for name in PROGRAMS)
    run_dir = tmp_path / "cli_resume"
    engine = _engine(run_dir)
    engine._stop_at = 2
    assert engine.run()["reason"] == "paused"
    completed = _cli("resume", str(run_dir), "--tasks-root", str(TASKS))
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["reason"] == "budget" and result["B_evals"] == 4
    assert result["cost_total"] == 0.0
