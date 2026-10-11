"""Static signs that a candidate's result depends on wall-clock time (deadlines, time-sliced search)."""
from __future__ import annotations

import ast

_CLOCKS = {"time", "perf_counter", "monotonic", "process_time", "time_ns", "perf_counter_ns", "monotonic_ns", "now"}


def time_dependent(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    clocks = set(_CLOCKS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in ("time", "datetime"):
            clocks.update(alias.asname or alias.name for alias in node.names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            name = target.attr if isinstance(target, ast.Attribute) else target.id if isinstance(target, ast.Name) else None
            if name in clocks:
                return True
    return False
