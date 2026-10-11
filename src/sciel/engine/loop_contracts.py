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

ENGINE_CONTRACT = "espalier-3.74"

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
    "a CONSTRUCTIVE/GEOMETRIC approach: an explicit closed-form or pattern-based "
    "construction (grids, rings, symmetry groups, known record structures)",
    "a LOCAL-OPTIMIZER approach: a good initialization plus a strong nonlinear "
    "refinement (e.g. scipy.optimize with explicit constraints, joint optimization "
    "of all variables)",
    "a GLOBAL-STOCHASTIC approach: multi-restart from varied initializations with "
    "basin hopping / simulated annealing style moves, keeping the best under the "
    "time budget",
]

def _paradigm_prompt(task_prompt: str, base_code: str, paradigm: str) -> str:
    return (f"You are writing a Python program for the following task.\n\nTASK:\n"
            f"{task_prompt}\n\nREFERENCE PROGRAM (keep its entry-point function "
            f"names and return interface EXACTLY):\n```python\n{base_code}\n```\n\n"
            f"Write a NEW program using {paradigm}. Output the COMPLETE program in "
            f"a single ```python code block.")

def _preflight_task(cfg, task):
    """Pure task/source checks precede even creation of the run lock file."""
    if any(cfg.get(name, {}).get('enabled') for name in
           ('circle_generation', 'repeat_population', 'repeat_promotion')):
        from .repeat_support import validate_circle_task, circle_initial_source_policy
        validate_circle_task(task, initial_source_policy=circle_initial_source_policy(cfg))
    if cfg.get('circle_intervention', {}).get('enabled'):
        from .intervention import DEFAULT_OPERATION, builder_for
        if task.qualified_name != 'oe/circle_packing':
            raise ValueError('circle_intervention supports only oe/circle_packing')
        builder_for(cfg['circle_intervention'].get('operation', DEFAULT_OPERATION)).policy_of(
            _read(task.initial_program_path))

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
