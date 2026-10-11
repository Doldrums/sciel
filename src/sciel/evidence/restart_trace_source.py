"""Assemble the historical standalone collector from its maintained modules.

The package imports are unnecessary in the embedded namespace. Definition text,
comments and whitespace remain exact, preserving the historical collector and
scaffold hashes. Assembly neither imports nor executes the generated source.
"""
from __future__ import annotations

import ast
from pathlib import Path


def _definition_tail(filename, name):
    source = Path(__file__).with_name(filename).read_text(encoding='utf-8')
    tree = ast.parse(source)
    index = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    # Prefixes contain only the package docstring/imports, never omitted logic.
    for node in tree.body[:index]:
        if not (isinstance(node, (ast.Import, ast.ImportFrom)) or
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            raise ValueError('unexpected restart collector module prefix')
    return ''.join(source.splitlines(keepends=True)[tree.body[index].lineno - 1:])


def collector_source():
    """Return the exact legacy /2 codec source, with no package dependencies."""
    primitives = Path(__file__).with_name('restart_trace_values.py').read_text(encoding='utf-8')
    return (primitives + _definition_tail('restart_trace_recorder.py', 'RestartTrace')
            + _definition_tail('restart_trace_validation.py', 'validate_trace'))
