"""Audited new search from a checkpoint's best source, with a fresh evaluation.

This transfers code, not scores, lessons, budgets, or a stopped research state.
The source checkpoint and journal stay unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .run_lock import RunDirectoryLock

SCHEMA = 'sciel-warm-start/1'


def validate_warm_start(value):
    if not isinstance(value, dict) or set(value) != {'schema', 'source', 'source_sha256', 'parent'}:
        raise ValueError('invalid warm_start fields')
    source = value['source']
    if (value['schema'] != SCHEMA or not isinstance(source, str) or not source.strip()
            or hashlib.sha256(source.encode()).hexdigest() != value['source_sha256']):
        raise ValueError('warm_start source identity differs')
    parent = value['parent']
    if not isinstance(parent, dict) or not all(parent.get(key) for key in (
            'run_id', 'checkpoint_sha256', 'journal_sha256', 'task', 'task_hash', 'program_id')):
        raise ValueError('warm_start parent provenance is missing')
    return value


def from_run(directory, task):
    directory = Path(directory).resolve()
    # Serialize against a running writer while taking the immutable source view.
    lock = RunDirectoryLock(directory)
    try:
        raw = (directory / 'checkpoint.json').read_bytes()
        saved = json.loads(raw)
        provenance = saved.get('provenance') or {}
        if provenance.get('schema') != 'sciel-checkpoint-provenance/1':
            raise ValueError('warm start requires a checkpoint with bound provenance')
        identity = saved.get('identity') or {}
        if identity.get('task') != task:
            raise ValueError('warm start task name differs')
        from .research import assert_no_uncheckpointed_research
        assert_no_uncheckpointed_research(directory / 'sciel_events.jsonl', provenance.get('journal_anchor'))
        journal = (directory / 'sciel_events.jsonl').read_bytes()
        rows = [json.loads(line) for line in journal.splitlines()]
        checkpoint_sha = hashlib.sha256(raw).hexdigest()
        if not any(row.get('event') == 'checkpoint'
                   and row.get('checkpoint_sha256') == checkpoint_sha
                   and row.get('checkpoint_id') == provenance.get('checkpoint_id') for row in rows):
            raise ValueError('warm start checkpoint digest is not recorded in the journal')
        best = saved['db']['pop_best_id']
        program = next(p for p in saved['db']['programs'] if p['id'] == best)
        if type(program['score']) not in (int, float) or not math.isfinite(program['score']):
            raise ValueError('warm start has no finite best score')
        source = program['code']
        value = dict(schema=SCHEMA, source=source,
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            parent=dict(run_id=provenance['run_id'], path=str(directory),
                checkpoint_sha256=checkpoint_sha, journal_sha256=hashlib.sha256(journal).hexdigest(),
                task=task, task_hash=identity['task_hash'], program_id=best,
                previous_score=program['score'], previous_known_cost=saved.get('cost_total'),
                stopped_reason=(saved.get('research') or {}).get('stopped_reason'),
                transfer='source_only_fresh_evaluation'))
        return validate_warm_start(value)
    finally:
        lock.close()
