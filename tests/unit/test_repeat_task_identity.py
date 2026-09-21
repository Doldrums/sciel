"""Repeat evaluation requires the exact canonical evaluator implementation."""

import pytest

from sciel.engine.repeat_support import validate_circle_task
from sciel.eval.registry import load_task
from sciel.hypothesis import circle_optimizer_traced_body as builder
from test_circle_intervention_engine import task_copy


def test_repeat_gate_rejects_changed_evaluator_before_execution(tmp_path):
    task = task_copy(tmp_path / 'task', initial=builder.source_for())
    validate_circle_task(task, initial_source_policy=builder.OPERATION)
    evaluator = tmp_path / 'task' / 'evaluator.py'
    evaluator.write_text(evaluator.read_text() + '\n# Different evaluator revision.\n')
    changed = load_task(str(tmp_path / 'task'))
    with pytest.raises(ValueError, match='pinned canonical Circle'):
        validate_circle_task(changed, initial_source_policy=builder.OPERATION)
