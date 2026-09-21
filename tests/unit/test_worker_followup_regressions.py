"""Import-only graft branches and proven process start-method selection."""

import os
import textwrap

import pytest

from sciel.engine.viability import precheck
from sciel.engine.worker import _graft_entrypoints
from sciel.eval.sandbox import run_stage


@pytest.mark.parametrize("guard", [
    '__name__ != "__main__"',
    '"__main__" != __name__',
    'not (__name__ == "__main__")',
    '__name__ != "__main__" or unknown',
])
def test_inverse_guard_script_else_cannot_supply_imported_entrypoint(guard):
    child = ('def helper(): return 0\n'
             f'if {guard}:\n    pass\nelse:\n    def solve(): return 99\n')
    code, names = _graft_entrypoints(child, "def solve(): return 7\n", ["solve"])
    namespace = {"__name__": "candidate"}
    exec(code, namespace)
    assert names == ["solve"] and namespace["solve"]() == 7


def test_parent_inverse_guard_grafts_only_its_import_branch():
    parent = ('if __name__ != "__main__":\n    def solve(): return 7\n'
              'else:\n    def solve(): return 99\n')
    code, names = _graft_entrypoints("def helper(): return 0\n", parent, ["solve"])
    namespace = {"__name__": "candidate"}
    exec(code, namespace)
    assert names == ["solve"] and namespace["solve"]() == 7


def test_unknown_guard_keeps_both_import_possibilities():
    child = ('flag = False\nif __name__ != "__main__" and flag:\n'
             '    def solve(): return 99\nelse:\n    def solve(): return 7\n')
    assert _graft_entrypoints(child, "def solve(): return 0\n", ["solve"]) == (child, [])


@pytest.mark.parametrize("setter", [
    "mp.set_start_method('fork', force=True)",
    "mp.set_start_method(method='fork', force=True)",
    "select_method('fork', force=True)",
])
def test_unconditional_same_block_fork_accepts_local_process_target(setter):
    code = ("import multiprocessing as mp\n"
            "from multiprocessing import set_start_method as select_method\n"
            "def solve():\n    " + setter + "\n"
            "    q = mp.Queue()\n    return mp.Process(target=lambda: q.put(7))\n")
    assert precheck(code) is None


@pytest.mark.parametrize("process", [
    "mp.get_context('spawn').Process", "mp.get_context('forkserver').Process",
    "mp.get_context(method).Process", "mp.context.SpawnProcess",
])
def test_global_fork_does_not_override_another_context(process):
    code = ("import multiprocessing as mp\ndef solve(method='spawn'):\n"
            "    mp.set_start_method('fork', force=True)\n"
            f"    return {process}(target=lambda: 7)\n")
    assert precheck(code).kind == "resource"


@pytest.mark.parametrize("body", [
    "if flag:\n        mp.set_start_method('fork', force=True)\n"
    "    return mp.Process(target=lambda: 7)",
    "mp.set_start_method('fork', force=True)\n"
    "    if flag:\n        mp.set_start_method('spawn', force=True)\n"
    "    return mp.Process(target=lambda: 7)",
    "mp.set_start_method('fork', force=True)\n"
    "    mp.set_start_method(method, force=True)\n"
    "    return mp.Process(target=lambda: 7)",
    "def unused():\n        mp.set_start_method('fork', force=True)\n"
    "    return mp.Process(target=lambda: 7)",
    "p = mp.Process(target=lambda: 7)\n"
    "    mp.set_start_method('fork', force=True)\n    return p",
    "def switch():\n        mp.set_start_method('spawn', force=True)\n"
    "    mp.set_start_method('fork', force=True)\n"
    "    switch()\n    return mp.Process(target=lambda: 7)",
    "mp.set_start_method('fork', force=True)\n"
    "    p = mp.Process(target=lambda: 7)\n"
    "    mp.set_start_method('spawn', force=True)\n    p.start()\n    return p",
])
def test_conditional_late_other_scope_or_competing_setter_does_not_prove_fork(body):
    code = "import multiprocessing as mp\ndef solve(flag=False, method='spawn'):\n    " + body + "\n"
    assert precheck(code).kind == "resource"


def test_fork_still_pickles_pool_jobs():
    code = ("import multiprocessing as mp\ndef solve():\n"
            "    mp.set_start_method('fork', force=True)\n"
            "    return mp.Pool(2).map(lambda x: x, [1])\n")
    assert precheck(code).kind == "resource"


@pytest.mark.skipif(os.name != "posix", reason="fork is a POSIX start method")
def test_accepted_fork_program_executes_in_isolated_stage(tmp_path):
    code = textwrap.dedent("""
        import multiprocessing as mp
        def solve():
            mp.set_start_method('fork', force=True)
            q = mp.Queue()
            p = mp.Process(target=lambda: q.put(7))
            p.start()
            p.join(2)
            value = q.get(timeout=1)
            q.close()
            return value
    """)
    assert precheck(code) is None
    candidate = tmp_path / "candidate.py"
    candidate.write_text(code)
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(textwrap.dedent("""
        import importlib.util
        def smoke(path):
            spec = importlib.util.spec_from_file_location('candidate', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return {'ok': True, 'score': module.solve()}
    """))
    result = run_stage(str(evaluator), "smoke", str(candidate), timeout=5)
    assert result.ok and result.score == 7
