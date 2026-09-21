"""Path-based trials work from threads and keep nested-process timeouts intact."""
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest

from sciel.eval.sandbox import run_stage
from sciel.eval.facade import EvaluationSetupError

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "tasks/oe/function_minimization/evaluator.py"
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX trial tree cleanup")


def evaluator():
    spec = importlib.util.spec_from_file_location("trial_subprocess_tests", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("through_stage", [False, True], ids=["direct", "stage"])
def test_threaded_caller_executes_candidate_only_in_child(tmp_path, through_stage):
    """A separate test driver bounds a regression that might deadlock at launch."""
    marker = tmp_path / "imports.jsonl"
    candidate = tmp_path / "candidate.py"
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(f"""
        import concurrent.futures, importlib.util, json, os, sys
        from pathlib import Path
        sys.path.insert(0, {str(ROOT / 'src')!r})
        from sciel.eval.sandbox import run_stage
        spec = importlib.util.spec_from_file_location('ev', {str(EVALUATOR)!r})
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        candidate = Path({str(candidate)!r})
        candidate.write_text(
            "import os\\nfrom pathlib import Path\\n"
            + "assert os.getpid() != " + str(os.getpid()) + "\\n"
            + "with Path(" + repr({str(marker)!r}) + ").open('a') as f: f.write(str(os.getpid()) + '\\\\n')\\n"
            + "def run_search(): return (-1.704, .678)\\n")
        def invoke(_):
            if {through_stage!r}:
                r = run_stage({str(EVALUATOR)!r}, 'smoke', str(candidate), timeout=5)
                assert r.ok, (r.stderr, r.artifacts)
                return r.score
            return ev._run_with_timeout(candidate, timeout_seconds=5)[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(invoke, range(2)))
        print(json.dumps({{'pid': os.getpid(), 'results': results}}))
    """))
    result = subprocess.run([sys.executable, "-Werror::DeprecationWarning", str(driver)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    pids = [int(line) for line in marker.read_text().splitlines()]
    assert len(pids) == len(set(pids)) == 2
    assert data["pid"] not in pids
    assert len(data["results"]) == 2
    assert "fork" not in result.stderr.lower()


@pytest.mark.parametrize("where", ["import", "entrypoint"])
def test_import_and_entrypoint_timeout_leave_next_trial_usable(tmp_path, where):
    ev = evaluator()
    marker = tmp_path / "pid"
    path = tmp_path / "candidate.py"
    prefix = ("import os, time\nfrom pathlib import Path\n"
              f"Path({str(marker)!r}).write_text(str(os.getpid()))\n")
    path.write_text(prefix + ("time.sleep(20)\n" if where == "import" else "")
                    + "def run_search():\n    time.sleep(20)\n")
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="trial exceeded"):
        ev._run_with_timeout(path, timeout_seconds=1)
    assert time.monotonic() - started < 3
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)
    path.write_text("def run_search(): return (1, 2)\n")
    assert ev._run_with_timeout(path)[:2] == (1, 2)


@pytest.mark.parametrize("method", [m for m in ("spawn", "forkserver")
                                    if m in multiprocessing.get_all_start_methods()])
def test_candidate_module_is_importable_by_nested_multiprocessing(tmp_path, method):
    # An unusual source filename cannot itself be imported by a spawned child.
    path = tmp_path / "candidate.with-dots.py"
    path.write_text(textwrap.dedent(f"""
        import multiprocessing as mp
        from pathlib import Path
        def worker(queue):
            assert Path(__file__).resolve() == Path({str(path)!r})
            queue.put(7)
        def run_search():
            ctx = mp.get_context({method!r})
            queue = ctx.Queue()
            process = ctx.Process(target=worker, args=(queue,))
            process.start()
            value = queue.get(timeout=2)
            process.join(timeout=2)
            assert process.exitcode == 0
            queue.close()
            queue.join_thread()
            return value, 0
    """))
    assert evaluator()._run_with_timeout(path, timeout_seconds=5)[0] == 7


def test_candidate_can_still_choose_its_rng_seed(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_text("import numpy as np, random\nnp.random.seed(123)\nrandom.seed(456)\n"
                    "def run_search(): return float(np.random.random()), random.random()\n")
    ev = evaluator()
    assert ev._run_with_timeout(path) == ev._run_with_timeout(path)


@pytest.mark.parametrize('mem_bytes', [None, 1024 ** 3], ids=['cpu', 'cpu-and-memory'])
def test_stage_applies_limits_before_evaluator_import_and_keeps_warnings(tmp_path, mem_bytes):
    path = tmp_path / "evaluator.py"
    imported = tmp_path / "evaluator-imported"
    path.write_text(textwrap.dedent("""
        import resource, warnings
        from pathlib import Path
        Path(__file__).with_name('evaluator-imported').write_text('imported')
        cpu = resource.getrlimit(resource.RLIMIT_CPU)
        memory = resource.getrlimit(resource.RLIMIT_AS)
        warnings.warn('stage diagnostic survives success', RuntimeWarning)
        def smoke(p):
            return {'ok': True, 'score': .5,
                    'cases': {'cpu_soft': cpu[0], 'cpu_hard': cpu[1],
                              'memory_soft': memory[0], 'memory_hard': memory[1]}}
    """))
    try:
        result = run_stage(str(path), "smoke", "unused", timeout=3,
                           cpu_seconds=7, mem_bytes=mem_bytes)
    except EvaluationSetupError as error:
        # Darwin can refuse RLIMIT_AS even when getrlimit reports no bound.
        # Refusal must stop before imports, never continue with only a warning.
        assert sys.platform == 'darwin' and mem_bytes == 1024 ** 3
        diagnostic = error.diagnostic
        assert diagnostic['code'] == 'setrlimit_failed'
        assert diagnostic['resource'] == 'mem_bytes'
        assert diagnostic['requested_limits'] == {'cpu_seconds': 7, 'mem_bytes': mem_bytes}
        assert diagnostic['execution'] == diagnostic['resource_usage'] == 'unknown'
        assert diagnostic['reported_wall_seconds'] >= 0
        assert not imported.exists()
        return
    assert result.ok, (result.stderr, result.artifacts)
    assert imported.read_text() == 'imported'
    assert result.cases["cpu_soft"] == 7 and result.cases["cpu_hard"] == 8
    if mem_bytes is not None:
        assert result.cases["memory_soft"] == result.cases["memory_hard"] == mem_bytes
    assert "stage diagnostic survives success" in result.stderr
