"""Real Python import/spawn boundaries using tiny synthetic candidates only.

Never execute historical generated optimizers. Child work is bounded both by
the fixture and the real evaluator sandbox; worker counts are always two.
"""

import base64
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import py_compile
import subprocess
import sys
import textwrap
import time
import types

import pytest

from sciel.candidate_modules import load_candidate
from sciel.eval.native_geometry import parse_artifact, validate_binding
from sciel.eval.sandbox import run_stage


ROOT = Path(__file__).resolve().parents[2]
CIRCLE = ROOT / "tasks/oe/circle_packing/evaluator.py"


@pytest.fixture(autouse=True)
def bounded_numerical_threads(monkeypatch):
    # These are synthetic test subprocesses, not experiment resource settings.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))


def write_program(tmp_path, source, name="candidate.py"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip())
    return path


def evaluator(tmp_path):
    return write_program(tmp_path, """
        from sciel.candidate_modules import load_candidate
        def smoke(program_path):
            return load_candidate(program_path, 'program_smoke').run()
    """, "synthetic_evaluator.py")


def assert_pids_gone(pids):
    def present(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    deadline = time.monotonic() + 2
    while any(present(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(.01)
    assert not [pid for pid in pids if present(pid)], "synthetic pool leaked a process"


def child_unpickle(value):
    # Fresh interpreter imports only stdlib; pickle must discover the candidate
    # namespace on its own, rather than inherit this process's finder/modules.
    return subprocess.run(
        [sys.executable, "-c", "import base64,pickle,sys; "
         "f=pickle.loads(base64.b64decode(sys.stdin.buffer.read())); print(f())"],
        input=base64.b64encode(pickle.dumps(value)), capture_output=True,
        timeout=10, check=False,
    )


@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_real_sandbox_pool_preserves_top_level_dataclass_and_functions(tmp_path, start_method):
    if start_method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"{start_method} is unavailable")
    source = """
        from __future__ import annotations
        from dataclasses import dataclass
        import json, multiprocessing, os
        @dataclass
        class Number:
            value: int
        def square(number):
            return Number(number.value * number.value)
        def run():
            with multiprocessing.get_context('START_METHOD').Pool(2) as pool:
                pids = [p.pid for p in pool._pool]
                values = pool.map(square, [Number(2), Number(3)])
            return {'ok': True, 'score': sum(x.value for x in values),
                    'cases': {'active_children': len(multiprocessing.active_children())},
                    'artifacts': {'pids': json.dumps(pids), 'source_file': __file__,
                                  'module': __name__}}
    """
    program = write_program(tmp_path, source.replace("START_METHOD", start_method))
    result = run_stage(str(evaluator(tmp_path)), "smoke", str(program), timeout=12)
    assert result.ok, result.artifacts
    assert result.score == 13
    assert result.cases["active_children"] == 0
    assert Path(result.artifacts["source_file"]) == program
    assert result.artifacts["module"] != "program_smoke"
    pids = json.loads(result.artifacts["pids"])
    assert len(pids) == len(set(pids)) == 2
    assert_pids_gone(pids)


def test_real_sandbox_scipy_de_workers_two_uses_spawn_and_reaps(tmp_path):
    program = write_program(tmp_path, """
        import multiprocessing
        from scipy.optimize import differential_evolution
        def objective(x):
            return float(x[0]**2 + x[1]**2)
        def run():
            multiprocessing.set_start_method('spawn', force=True)
            result = differential_evolution(
                objective, [(-1, 1), (-1, 1)], workers=2,
                updating='deferred', maxiter=1, popsize=3, seed=7,
                polish=False, tol=0, atol=0)
            return {'ok': True, 'score': float(result.fun),
                    'cases': {'nfev': result.nfev,
                              'active_children': len(multiprocessing.active_children())}}
    """)
    result = run_stage(str(evaluator(tmp_path)), "smoke", str(program), timeout=15)
    assert result.ok, result.artifacts
    assert 0 <= result.score <= 2
    assert result.cases == {"nfev": 12., "active_children": 0.}


def test_nested_local_function_remains_a_candidate_error_and_pool_is_reaped(tmp_path):
    pidfile = tmp_path / "workers.json"
    program = write_program(tmp_path, f"""
        import json, multiprocessing
        from pathlib import Path
        def run():
            def nested_objective(x):
                return x*x
            with multiprocessing.get_context('spawn').Pool(2) as pool:
                Path({str(pidfile)!r}).write_text(json.dumps([p.pid for p in pool._pool]))
                pool.map(nested_objective, [2, 3])
            return {{'ok': True, 'score': 13}}
    """)
    result = run_stage(str(evaluator(tmp_path)), "smoke", str(program), timeout=12)
    assert not result.ok
    assert "local object" in result.stderr
    assert "nested_objective" in result.stderr
    assert "import of module" not in result.stderr
    assert_pids_gone(json.loads(pidfile.read_text()))


def test_circle_native_witness_binds_original_source_under_spawn(tmp_path):
    program = write_program(tmp_path, """
        import multiprocessing
        import numpy as np
        def center(i):
            return (0.1 + 0.16*(i%6), 0.1 + 0.2*(i//6))
        def run_packing():
            with multiprocessing.get_context('spawn').Pool(2) as pool:
                centers = np.asarray(pool.map(center, range(26)), dtype=np.float64)
            radii = np.full(26, 0.01)
            return centers, radii, float(radii.sum())
    """)
    before = program.read_bytes()
    result = run_stage(str(CIRCLE), "smoke", str(program), timeout=12)
    assert result.ok, result.artifacts
    assert result.cases["validity"] == 1
    witness = parse_artifact(result.artifacts["native_geometry"])
    source_hash = hashlib.sha256(before).hexdigest()
    validate_binding(witness, {
        "ok": result.ok, "score": result.score, "cases": result.cases,
        "confirmed": False, "reached": "smoke",
    }, source_hash)
    assert witness["source_sha256"] == source_hash
    assert witness["evaluator_sha256"] == hashlib.sha256(CIRCLE.read_bytes()).hexdigest()
    assert witness["numeric"]["radii"] == [0.01] * 26
    assert program.read_bytes() == before


def test_fresh_interpreter_unpickles_without_preimporting_loader(tmp_path):
    program = write_program(tmp_path, "def value(): return 23\n")
    module = load_candidate(program, "program_smoke")
    child = child_unpickle(module.value)
    assert child.returncode == 0, child.stderr.decode()
    assert child.stdout.strip() == b"23"


def test_arbitrary_source_filename_preserves_ambient_import_paths(tmp_path):
    program = write_program(tmp_path, "def value(): return 23\n", "not a module-ü.txt")
    before = list(sys.path)
    module = load_candidate(program)
    assert sys.path == before
    assert Path(module.__file__) == program
    child = child_unpickle(module.value)
    assert child.returncode == 0, child.stderr.decode()
    assert child.stdout.strip() == b"23"


def test_changed_source_refuses_child_import_of_old_identity(tmp_path):
    program = write_program(tmp_path, "def value(): return 23\n")
    module = load_candidate(program)
    program.write_text("def value(): return 99\n")
    child = child_unpickle(module.value)
    assert child.returncode != 0
    assert b"99" not in child.stdout
    assert any(word in child.stderr.lower() for word in (b"source", b"hash", b"digest"))
    assert load_candidate(program).value() == 99


def test_deleted_source_refuses_child_import(tmp_path):
    program = write_program(tmp_path, "def value(): return 23\n")
    module = load_candidate(program)
    program.unlink()
    child = child_unpickle(module.value)
    assert child.returncode != 0
    assert b"23" not in child.stdout


def test_same_basename_and_labels_do_not_alias_distinct_sources(tmp_path):
    a = write_program(tmp_path, "def value(): return 11\n", "a/program.py")
    b = write_program(tmp_path, "def value(): return 22\n", "b/program.py")
    first = load_candidate(a, "program_smoke")
    second = load_candidate(b, "program_smoke")
    assert first.__name__ != second.__name__
    assert (first.value(), second.value()) == (11, 22)
    assert pickle.loads(pickle.dumps(first.value))() == 11
    assert pickle.loads(pickle.dumps(second.value))() == 22
    assert Path(first.__file__) == a and Path(second.__file__) == b


def test_each_load_is_fresh_without_displacing_earlier_module(tmp_path):
    program = write_program(tmp_path, "values = []\ndef value(): return len(values)\n")
    first = load_candidate(program)
    first.values.append(1)
    second = load_candidate(program)
    assert first.__name__ != second.__name__
    assert first.value() == 1 and second.value() == 0
    assert sys.modules[first.__name__] is first
    assert pickle.loads(pickle.dumps(first.value))() == 1


def test_label_collision_does_not_replace_existing_module(tmp_path, monkeypatch):
    sentinel = types.ModuleType("program_smoke")
    monkeypatch.setitem(sys.modules, "program_smoke", sentinel)
    program = write_program(tmp_path, "def value(): return 7\n")
    module = load_candidate(program, "program_smoke")
    assert module.__name__ != "program_smoke"
    assert sys.modules["program_smoke"] is sentinel
    assert pickle.loads(pickle.dumps(module.value))() == 7


def test_same_size_same_mtime_source_edit_ignores_stale_pyc(tmp_path):
    program = write_program(tmp_path, "def value(): return 11\n")
    stamp = program.stat().st_mtime_ns
    cache = Path(py_compile.compile(str(program), doraise=True))
    old_cache = cache.read_bytes()
    program.write_text("def value(): return 22\n")
    os.utime(program, ns=(stamp, stamp))
    module = load_candidate(program)
    assert module.value() == 22
    assert cache.read_bytes() == old_cache


@pytest.mark.parametrize("source,error", [
    ("partial = True\nraise RuntimeError('synthetic import failure')\n", RuntimeError),
    ("def broken(:\n", SyntaxError),
])
def test_import_failure_rolls_back_partial_module_and_later_load_recovers(tmp_path, source, error):
    program = write_program(tmp_path, source)
    prior = set(sys.modules)
    with pytest.raises(error):
        load_candidate(program)
    leaked = [name for name in set(sys.modules) - prior
              if getattr(sys.modules[name], "__file__", None) == str(program)]
    assert leaked == []
    program.write_text("def value(): return 42\n")
    assert load_candidate(program).value() == 42
