"""Pre-eval gate: statically-knowable failures never reach the sandbox."""

from sciel.engine.worker import _docstring_repair, pre_eval_gate


def test_valid_program_passes():
    assert pre_eval_gate("import numpy as np\ndef run(): return 1\n") is None


def test_syntax_error_rejected():
    reason = pre_eval_gate('An improved algorithm that...\ndef run(): pass\n')
    assert reason is not None and "syntax" in reason


def test_unterminated_string_rejected():
    assert pre_eval_gate('x = "abc\ndef run(): pass\n') is not None


def test_multiprocessing_with_nested_worker_rejected():
    code = (
        "import multiprocessing as mp\n"
        "def run():\n"
        "    def worker(x):\n"
        "        return x * 2\n"
        "    with mp.Pool(2) as p:\n"
        "        return p.map(worker, [1, 2])\n")
    reason = pre_eval_gate(code)
    assert reason is not None and "pickle" in reason


def test_multiprocessing_with_module_worker_passes():
    code = (
        "import multiprocessing as mp\n"
        "def worker(x):\n"
        "    return x * 2\n"
        "def run():\n"
        "    with mp.Pool(2) as p:\n"
        "        return p.map(worker, [1, 2])\n")
    assert pre_eval_gate(code) is None


def test_docstring_repair_fixes_mid_docstring_start():
    broken = 'Improved two-stage circle packing.\n"""\nimport numpy as np\n\ndef run():\n    return 1\n'
    fixed = _docstring_repair(broken)
    assert fixed is not None
    assert pre_eval_gate(fixed) is None


def test_docstring_repair_leaves_normal_code_alone():
    assert _docstring_repair("import numpy as np\ndef run(): pass\n") is None
