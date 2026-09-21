"""Saved observations and method exposure fixtures for followup tests."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import pytest
from sciel.engine.observed_changes import ObservedChanges
from sciel.engine.research_cells_context import compact_context
from sciel.hypothesis.measurement import metric_context
from sciel.methods.models import canonical_bytes, content_hash
from sciel.research_tools.followups import (
    FollowupLedger, build_followup, capture_exposures, compact_view,
    validate_exposures, validate_followup,
)
from test_research_cell_context import card


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def seal(value):
    body = {k: v for k, v in value.items() if k != 'sha256'}
    return {**body, 'sha256': content_hash(body)}


def record(number=1, *, tool='c', status='evaluated', dispatch='returned', score=0.0):
    item = card()
    item['source']['tool_id'] = tool * 64
    context = compact_context([item])
    evaluated = status in ('evaluated', 'eval_failed')
    return {'id': f'r{number:06d}a1', 'island': number % 3, 'iteration': number,
        'operator': 'full_rewrite', 'parent_id': 'p1', 'parent_source_sha256': sha('parent'),
        'parent_score': 1.0, 'method_exposures': capture_exposures(context, prompt_sha256=sha('prompt')),
        'generation_dispatch': dispatch,
        'generation_response_sha256': sha('response') if dispatch == 'returned' else None,
        'hypothesis': {'statement': 'A proposed change, not proof of method use.'} if dispatch == 'returned' else None,
        'proposed_sha256': sha('child') if dispatch == 'returned' else None,
        'source_sha256': sha('child') if dispatch == 'returned' else None,
        'source_repaired': False, 'status': status,
        'score': score if status == 'evaluated' else None,
        'actual_delta': score - 1 if status == 'evaluated' else None,
        'evaluation': {'ok': status == 'evaluated', 'score': score, 'cases': {'score': score},
            'reached': 'full', 'raw_first_score': score, 'confirmed': False} if evaluated else None,
        'metric': metric_context(), 'failure': 'Actual failure' if status == 'eval_failed' else None}


def ledger(*, exposure=None, outcome=None):
    return FollowupLedger('f' * 64, authenticate_exposure=exposure or (lambda row: None),
                          authenticate_outcome=outcome)


def observed(source='child', ident='p2', score=0.0):
    history = ObservedChanges()
    return history.observe(SimpleNamespace(id=ident, code=source, parent_id=None),
        {'ok': True, 'score': score, 'cases': {'score': score}, 'reached': 'full', 'confirmed': False})


def output_record(tmp_path, *, disposition='exact_recorded_repeat', status='evaluated'):
    """Saved full-output witnesses only; the fixture evaluator is inert text."""
    from test_output_novelty import observed as observe_output, registry
    current = registry()
    observe_output(current, tmp_path, 'historical-evaluation', sha('historical'), 0)
    current.bind_program('historical-evaluation', 'historical-program')
    evidence = record(score=2.0, status=status)
    if disposition == 'unsupported':
        verdict = current.observe('child-evaluation', None,
            dict(task_hash='a' * 64, source_sha256=evidence['source_sha256'],
                 island_id=evidence['island'], receipt_sha256=sha('receipt'),
                 request_sha256=sha('request')))
    else:
        kwargs = {'output': {'different_full_output': [9]}} if disposition == 'new_output' else {}
        verdict, _, _ = observe_output(current, tmp_path, 'child-evaluation',
            evidence['source_sha256'], evidence['island'], **kwargs)
    assert verdict['status'] == disposition
    evidence['output_novelty'] = verdict
    return evidence
