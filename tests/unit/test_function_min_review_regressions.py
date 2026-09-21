"""Function-minimization trial trees and failed confirmation accounting."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
from unittest.mock import patch

import pytest

from sciel.eval.sandbox import run_stage


EVALUATOR = Path(__file__).resolve().parents[2] / "tasks/oe/function_minimization/evaluator.py"


def evaluator():
    spec = importlib.util.spec_from_file_location("function_min_tree_review", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_exists(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def wait_gone(pids):
    deadline = time.monotonic() + 2
    while any(process_exists(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(.01)
    return all(not process_exists(pid) for pid in pids)


def kill_leftovers(pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def descendant_code(pidfile):
    return textwrap.dedent(f"""
        import json, os, subprocess, sys, time
        from pathlib import Path
        leaf = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        Path({str(pidfile)!r}).write_text(json.dumps([os.getpid(), leaf.pid]))
        time.sleep(20)
    """)


def test_all_failed_confirmation_retains_attempt_count():
    ev = evaluator()
    with patch.object(ev, "_trial", side_effect=RuntimeError("failed")) as trial:
        result = ev.confirm("unused", 3)
    assert trial.call_count == 3
    assert not result["ok"]
    assert result["cases"]["n_trials"] == 3
    assert result["cases"]["reliability_score"] == 0


@pytest.mark.skipif(os.name != "posix", reason="trials use POSIX process trees")
@pytest.mark.parametrize("hang", [False, True])
def test_trial_reaps_worker_and_removes_nested_descendants(tmp_path, hang):
    ev = evaluator()
    pidfile, workerfile = tmp_path / "descendants.json", tmp_path / "worker.pid"
    code = descendant_code(pidfile)

    program = tmp_path / "candidate.py"
    program.write_text(textwrap.dedent(f"""
        import os, subprocess, sys, time
        from pathlib import Path
        def run_search():
            Path({str(workerfile)!r}).write_text(str(os.getpid()))
            subprocess.Popen([sys.executable, '-c', {code!r}],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while not Path({str(pidfile)!r}).exists():
                time.sleep(.005)
            if {hang!r}:
                time.sleep(20)
            return 7, 0
    """))

    pids = []
    try:
        if hang:
            with pytest.raises(TimeoutError):
                ev._run_with_timeout(program, timeout_seconds=1)
        else:
            assert ev._run_with_timeout(program, timeout_seconds=2)[0] == 7
        pids = json.loads(pidfile.read_text()) + [int(workerfile.read_text())]
        assert wait_gone(pids), "trial left live or zombie processes before the next trial"
        program.write_text("import os\ndef run_search(): return os.getpgrp(), 0\n")
        assert ev._run_with_timeout(program)[0] == os.getpgrp()
    finally:
        if pidfile.exists():
            pids += json.loads(pidfile.read_text())
        kill_leftovers(pids)


@pytest.mark.skipif(os.name != "posix", reason="outer sandbox uses POSIX killpg")
def test_outer_stage_timeout_still_removes_trial_tree(tmp_path):
    pidfile, workerfile = tmp_path / "descendants.json", tmp_path / "worker.pid"
    program = tmp_path / "candidate.py"
    program.write_text(textwrap.dedent(f"""
        import os, subprocess, sys, time
        from pathlib import Path
        def run_search():
            Path({str(workerfile)!r}).write_text(str(os.getpid()))
            subprocess.Popen([sys.executable, '-c', {descendant_code(pidfile)!r}],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(20)
    """))
    path = tmp_path / "outer_evaluator.py"
    path.write_text(textwrap.dedent(f"""
        import importlib.util, os, subprocess, sys, time
        from pathlib import Path
        spec = importlib.util.spec_from_file_location('inner_eval', {str(EVALUATOR)!r})
        inner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inner)
        def smoke(p):
            inner._run_with_timeout(p, timeout_seconds=20)
            return {{'ok': True, 'score': .5}}
    """))
    pids = []
    try:
        result = run_stage(str(path), "smoke", str(program), timeout=1)
        assert not result.ok and result.artifacts["error_type"] == "TimeoutError"
        pids = json.loads(pidfile.read_text()) + [int(workerfile.read_text())]
        assert wait_gone(pids), "outer timeout lost a trial descendant"
    finally:
        if pidfile.exists():
            pids += json.loads(pidfile.read_text())
        kill_leftovers(pids)
