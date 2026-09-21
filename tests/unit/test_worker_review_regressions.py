"""Executable regressions for worker grafts and conservative pre-eval rejection."""

import ast
from types import SimpleNamespace

import pytest

from sciel.engine.db import normalized_hash
from sciel.engine.types import NoChild, Snapshot
from sciel.engine.viability import precheck
from sciel.engine.worker import _graft_entrypoints, run_worker


def _grafted_namespace(parent, child="def other():\n    return 0\n", name="solve"):
    code, names = _graft_entrypoints(child, parent, [name])
    assert names == [name]
    namespace = {"__name__": "candidate"}
    exec(compile(code, "<grafted>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("wrapper", ["if True:", "try:"])
@pytest.mark.parametrize("definition", [
    "    @lru_cache(None)\n    def solve():\n        return 7\n",
    "    @lru_cache(None)\n    def solve(): return 7\n",
    "    @lru_cache(\n        maxsize=None,\n    )\n    def solve():\n        return 7\n",
])
def test_graft_dedents_guarded_decorated_definitions(wrapper, definition):
    parent = "from functools import lru_cache\n" + wrapper + "\n" + definition
    if wrapper == "try:":
        parent += "except RuntimeError:\n    pass\n"
    ns = _grafted_namespace(parent)
    assert ns["solve"]() == 7
    ns["solve"]()
    assert ns["solve"].cache_info().hits == 1


@pytest.mark.parametrize("literal", [
    '\'\'\'\n        indented literal\nno indentation\n    closing\'\'\'',
    'f\'\'\'\n        value: {2 + 3}\nno indentation\n    closing\'\'\'',
])
def test_dedenting_does_not_change_multiline_literal_contents(literal):
    parent = "if True:\n    def solve():\n        return " + literal + "\n"
    original = {}
    exec(parent, original)
    assert _grafted_namespace(parent)["solve"]() == original["solve"]()


@pytest.mark.parametrize("guard", [
    '__name__ == "__main__"',
    '"__main__" == __name__',
    '__name__ == "__main__" and True',
])
def test_script_only_child_entrypoint_does_not_satisfy_import_contract(guard):
    child = f"def helper(): return 0\nif {guard}:\n    def solve(): return 99\n"
    assert _grafted_namespace("def solve(): return 7\n", child)["solve"]() == 7


def test_parent_script_only_definition_is_never_grafted():
    parent = 'if __name__ == "__main__":\n    def solve(): return 99\n'
    child = "def helper(): return 0\n"
    assert _graft_entrypoints(child, parent, ["solve"]) == (child, [])


def test_main_guard_else_branch_keeps_its_import_time_binding():
    child = ('if __name__ == "__main__":\n    def solve(): return 99\n'
             'else:\n    def solve(): return 7\n')
    assert _graft_entrypoints(child, "def solve(): return 1\n", ["solve"]) == (child, [])


def test_import_capable_or_guard_is_not_mistaken_for_script_only():
    child = ('if __name__ == "__main__" or True:\n    def solve(): return 7\n')
    assert _graft_entrypoints(child, "def solve(): return 1\n", ["solve"]) == (child, [])


@pytest.mark.parametrize("parent,expected", [
    ("N = 7\ndef solve():\n    def local():\n        N = 1\n        return N\n"
     "    return N + local()\n", 8),
    ("N = 7\ndef solve(N=N):\n    return N\n", 7),
    ("N = 7\ndef solve():\n    return N + sum(N for N in range(3))\n", 10),
    ("N = 7\ndef solve():\n    return N + (lambda N: N + 1)(2)\n", 10),
    ("N = 7\ndef solve():\n    class Local:\n        N = 1\n"
     "    return N + Local.N\n", 8),
    ("N = 7\nclass Helper:\n    N = 1\n    def value(self): return N\n"
     "def solve(): return Helper().value()\n", 7),
    ("N = 7\nclass Helper:\n    N = N + 1\ndef solve(): return Helper.N\n", 8),
    ("N = 7\ndef solve():\n    global N\n    N += 1\n    return N\n", 8),
    ("N = 7\ndef solve():\n    def local():\n        global N\n"
     "        N += 1\n        return N\n    return local()\n", 8),
])
def test_graft_preserves_globals_across_lexical_scopes(parent, expected):
    assert _grafted_namespace(parent)["solve"]() == expected


@pytest.mark.parametrize("body", [
    "def solve(N=7):\n    return N\n",
    "def solve():\n    N = 7\n    def local(): return N\n    return local()\n",
    "def solve():\n    N = 6\n    def local():\n        nonlocal N\n"
    "        N += 1\n        return N\n    return local()\n",
    "def solve():\n    return sum(N for N in range(4)) + 1\n",
    "def solve():\n    import math as N\n    return N.isqrt(49)\n",
])
def test_local_bindings_do_not_pull_in_unrelated_parent_initializers(body):
    parent = 'N = (_ for _ in ()).throw(RuntimeError("must not graft N"))\n' + body
    ns = _grafted_namespace(parent)
    assert "N" not in ns
    assert ns["solve"]() == 7


def test_decorator_dependencies_are_resolved_in_enclosing_scope():
    parent = ("from functools import lru_cache\nCACHE = lru_cache(None)\n"
              "@CACHE\ndef solve():\n    CACHE = 7\n    return CACHE\n")
    fn = _grafted_namespace(parent)["solve"]
    assert fn() == 7
    fn()
    assert fn.cache_info().hits == 1


@pytest.mark.parametrize("setup,call", [
    ("import multiprocessing as mp", "mp.Pool(2).map(worker, [1])"),
    ("import multiprocessing.pool as pools", "pools.Pool(2).map(worker, [1])"),
    ("from multiprocessing import Pool as P", "P(2).map(func=worker, iterable=[1])"),
    ("from multiprocessing.pool import Pool as P", "P(2).apply(worker, (1,))"),
    ("import multiprocessing as mp\nP = mp.Pool", "P(2).map(worker, [1])"),
    ("import multiprocessing as mp", "mp.get_context('spawn').Pool(2).map(worker, [1])"),
    ("import multiprocessing as mp", "mp.get_context('fork').Pool(2).map(worker, [1])"),
    ("from multiprocessing import get_context as context\nctx = context('spawn')",
     "ctx.Pool(2).map(worker, [1])"),
    ("from multiprocessing import Process as P", "P(target=worker, args=(1,))"),
    ("import multiprocessing as mp", "mp.Process(None, worker, args=(1,))"),
    ("from concurrent.futures import ProcessPoolExecutor as P", "P(2).submit(worker, 1)"),
    ("import multiprocessing as mp", "mp.get_context(method='spawn').Process(target=worker)"),
])
def test_known_process_receivers_and_constructor_aliases_reject_local_worker(setup, call):
    code = setup + "\ndef solve():\n    def worker(x): return x + 1\n    return " + call + "\n"
    violation = precheck(code)
    assert violation is not None and violation.kind == "resource"


@pytest.mark.parametrize("body", [
    "with mp.Pool(2) as p:\n        return p.map(worker, [1])",
    "p = mp.Pool(2)\n    alias = p\n    return alias.map(worker, [1])",
    "p = mp.Pool(2)\n    submit = p.map\n    callback = worker\n    return submit(callback, [1])",
    "return mp.Pool(2).map(lambda x: x + 1, [1])",
])
def test_pool_context_and_worker_method_aliases_are_not_gate_bypasses(body):
    code = ("import multiprocessing as mp\ndef solve():\n    def worker(x): return x + 1\n    "
            + body + "\n")
    assert precheck(code).kind == "resource"


@pytest.mark.parametrize("body", [
    "return sum(map(lambda x: x*x, range(5)))",
    "def worker(x): return x*x\n    return sum(map(worker, range(5)))",
    "from multiprocessing.pool import ThreadPool\n    return ThreadPool().map(lambda x: x, [1])",
    "from multiprocessing.dummy import Pool\n    return Pool().map(lambda x: x, [1])",
    "from concurrent.futures import ThreadPoolExecutor\n"
    "    return ThreadPoolExecutor().submit(lambda: 1)",
    "class Local:\n        def map(self, fn, xs): return list(map(fn, xs))\n"
    "    return Local().map(lambda x: x, [1])",
])
def test_non_process_maps_remain_eligible_with_multiprocessing_import(body):
    assert precheck("import multiprocessing as mp\ndef solve():\n    " + body + "\n") is None


@pytest.mark.parametrize("code", [
    "import multiprocessing as mp\ndef solve(mp):\n    return mp.Pool().map(lambda x: x, [1])\n",
    "from multiprocessing import Pool\ndef solve(Pool):\n    return Pool().map(lambda x: x, [1])\n",
    "import multiprocessing as mp\nfrom multiprocessing.pool import ThreadPool\n"
    "def solve():\n    mp = SimpleNamespace(Pool=ThreadPool)\n"
    "    return mp.Pool().map(lambda x: x, [1])\n",
    "import multiprocessing as mp\ndef worker(x): return x\n"
    "def unrelated():\n    def worker(x): return x + 1\n    return worker(1)\n"
    "def solve():\n    return mp.Pool().map(worker, [1])\n",
    "import multiprocessing as mp\ndef worker(x): return x\n"
    "def solve():\n    return mp.Pool().map_async(worker, [1], callback=lambda xs: xs)\n",
    "import multiprocessing as mp\ndef solve(pools):\n"
    "    return [mp.map(lambda x: x, [1]) for mp in pools]\n",
    "import multiprocessing as mp\ndef solve():\n    def worker(): return 1\n"
    "    return mp.get_context('fork').Process(target=worker)\n",
])
def test_shadowing_and_parent_process_callbacks_are_not_rejected(code):
    assert ast.parse(code)
    assert precheck(code) is None


def test_pool_binding_in_enclosing_function_is_visible_to_nested_submitter():
    code = ("import multiprocessing as mp\ndef solve():\n"
            "    with mp.Pool() as pool:\n        def submit():\n"
            "            def worker(x): return x\n            return pool.map(worker, [1])\n"
            "        return submit()\n")
    assert precheck(code).kind == "resource"


def test_module_import_below_function_definition_is_resolved():
    code = ("def solve():\n    def worker(x): return x\n"
            "    return mp.Pool().map(worker, [1])\nimport multiprocessing as mp\n")
    assert precheck(code).kind == "resource"


@pytest.mark.parametrize("responses,reason", [
    (["", "", "dup", "dup"], "empty_response"),
    (["def (", "def (", "dup", "dup"], "gate_rejected"),
    (["", "def (", "dup"], "empty_response"),
    (["", "dup", "dup"], "dup_cap"),
])
def test_mixed_exhaustion_reserves_saturation_for_strict_duplicate_lead(responses, reason):
    code = "def solve(): return 7\n"
    snap = Snapshot(0, "diff", "mock", 1, "p", code, 0.5,
                    code_hashes={normalized_hash(code)})
    texts = iter("```python\n" + code + "```" if r == "dup" else r for r in responses)
    llm = SimpleNamespace(generate=lambda *args: SimpleNamespace(text=next(texts), cost=0.1))

    def unexpected_eval(child):
        pytest.fail("duplicates and invalid completions must not reach the evaluator")

    result = run_worker("diff", snap, llm, unexpected_eval, dup_cap=len(responses))
    assert isinstance(result, NoChild) and result.reason == reason
    assert result.cost == pytest.approx(0.1 * len(responses))
