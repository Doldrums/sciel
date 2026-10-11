"""Shared loop contracts and pure helpers."""
from __future__ import annotations

import ast
import re


_ENTRY_RE = re.compile(r'hasattr\(\s*[\w.]+\s*,\s*["\'](\w+)["\']\s*\)')

def _required_entrypoints(evaluator_path) -> list:
    """Required entry-point function names, read from the evaluator's own hasattr checks
    -- injected into the prompt so the LLM does not rename/remove them (the dominant
    eval-fail on non-func_min tasks)."""
    if not evaluator_path:
        return []
    try:
        src = open(evaluator_path).read()
    except OSError:
        return []
    return sorted(set(_ENTRY_RE.findall(src)))

def _internal_contract(program_code: str) -> list:
    """Functions defined INSIDE the EVOLVE-BLOCK but called by fixed code OUTSIDE it.
    A fixed wrapper (e.g. run_packing) can call a helper (construct_packing) that lives in
    the evolvable block; a mutation that drops that helper crashes even though the
    evaluator's own entry-point survived. Detected structurally so it is injected into the
    prompt and graft can recover it from the parent. Empty when there is no EVOLVE-BLOCK."""
    S, E = "# EVOLVE-BLOCK-START", "# EVOLVE-BLOCK-END"
    if S not in program_code or E not in program_code:
        return []
    i, j = program_code.index(S), program_code.index(E) + len(E)
    inside, outside = program_code[i:j], program_code[:i] + program_code[j:]
    try:
        in_defs = {n.name for n in ast.walk(ast.parse(inside))
                   if isinstance(n, ast.FunctionDef)}
        out_calls = {n.func.id for n in ast.walk(ast.parse(outside))
                     if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    except SyntaxError:
        return []
    return sorted(in_defs & out_calls)

ENGINE_CONTRACT = "espalier-3.94"

def _evaluator_metadata(task, cfg):
    """Declared ordinary cascade limits, not a claim about candidate isolation."""
    return {
        "backend": "subprocess",
        "scope": "ordinary_evaluator_process",
        "candidate_isolation": "task_defined",
        **{name: getattr(task, name, None) for name in (
            "smoke_timeout", "full_timeout", "confirm_timeout", "per_run_timeout",
            "cpu_seconds", "mem_bytes", "stochastic", "confirm_n", "primary_metric")},
        "confirm_enabled": bool(cfg.get("confirm_records", False) and task.stochastic),
    }

_PARADIGMS = [
    "a DIRECT approach: build the answer explicitly from the structure of the task, "
    "with rules or formulas derived from what the task specifies",
    "a MODEL-AND-OPTIMIZE approach: parameterize the answer and fit or optimize the "
    "parameters directly against the task's objective",
    "a SEARCH approach: generate many varied candidate answers and keep the best one "
    "found within the time budget",
]


def paradigms(task) -> list:
    """Island seeding paradigms declared by the task, or task-neutral ones."""
    return list(getattr(task, "paradigms", None) or _PARADIGMS)

def _paradigm_prompt(task_prompt: str, base_code: str, paradigm: str) -> str:
    return (f"You are writing a Python program for the following task.\n\nTASK:\n"
            f"{task_prompt}\n\nREFERENCE PROGRAM (keep its entry-point function "
            f"names and return interface EXACTLY):\n```python\n{base_code}\n```\n\n"
            f"Write a NEW program using {paradigm}. Output the COMPLETE program in "
            f"a single ```python code block.")

def _preflight_task(cfg, task):
    """Pure task/source checks precede even creation of the run lock file."""
    if cfg.get('repeat_promotion', {}).get('enabled'):
        from .repeat_support import validate_circle_task
        validate_circle_task(task)

def _snapshot_config(cfg: dict, path: str) -> None:
    import yaml
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)

def _read(path: str) -> str:
    with open(path) as f:
        return f.read()

def _changed_region(parent_code: str, child_code: str) -> str:
    import difflib
    p, c = parent_code.splitlines(), child_code.splitlines()
    out = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(None, p, c).get_opcodes():
        if tag in ("replace", "insert"):
            out.extend(c[j1:j2])
    return "\n".join(out)

def _trunc(d: dict, limit: int = 2000) -> dict:
    return {k: (v[:limit] if isinstance(v, str) else v) for k, v in (d or {}).items()}

def _model_names(llm) -> list:
    return list(getattr(llm, "models", ["unknown"]))
