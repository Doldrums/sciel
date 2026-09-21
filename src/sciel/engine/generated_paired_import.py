"""Explicit, durable import of generation-only paired evidence.

The annex is foreign evidence, not local Engine work. It never creates an
ordinary outcome, lesson, population member, cost, or exposure.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ..knowledge_base.catalogs import canonical_json, canonical_sha

CONFIG_KEY = 'external_component_sha256'
EVENT = 'external_component_import'


def require(value, message):
    if not value:
        raise ValueError('generated paired import: ' + message)


def configured_digest(config):
    value = config.get(CONFIG_KEY)
    require(value is None or (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value)), 'invalid configured digest')
    require(CONFIG_KEY not in config or value is not None, 'explicit digest cannot be null')
    return value


def identity():
    from .external_research_v2 import policy_identity
    return dict(schema='sciel-generated-paired-import/1', policy=policy_identity())


def check(controller, *, require_present=False):
    expected = configured_digest(controller.config)
    require(expected == controller._external_component_digest, 'configured digest changed')
    wanted = identity() if expected is not None else None
    require(controller._external_component_identity == wanted, 'receiving policy changed')
    capsule = controller.knowledge.generated_paired_history
    if capsule is not None:
        require(expected is not None, 'evidence requires explicit configured digest')
        from ..knowledge_base.generated_paired_history import validate_capsule
        validate_capsule(capsule, expected_sha256=expected, metric=controller.metric)
    elif require_present and expected is not None:
        raise ValueError('generated paired import: configured component has not been imported')
    from .generated_paired_review_import import check as check_review
    check_review(controller, require_present=require_present)
    return capsule


def event_fields(controller, capsule):
    from ..knowledge_base.generated_paired_history import proposal_records
    return dict(capsule_sha256=canonical_sha(capsule),
        receiving_root=copy.deepcopy(capsule['root_parent']),
        policy_sha256=canonical_sha(controller._external_component_identity),
        proposal_ids=list(proposal_records(capsule)),
        comparison_ids=[g['id'] for g in capsule['comparisons']],
        new_attempts=0, new_outcomes=0, new_model_calls=0, new_cost=0)


def _event_matches(row, fields):
    # JSON booleans must not alias integer counters (False == 0 in Python).
    return (all(key in row for key in fields)
            and canonical_json({key: row[key] for key in fields}) == canonical_json(fields))


def _anchored_rows(log, anchor):
    require(type(anchor) is dict and type(getattr(log, 'path', None)) is str,
            'durable active journal required')
    data = Path(log.path).read_bytes()[:anchor['byte_offset']]
    require(hashlib.sha256(data).hexdigest() == anchor['prefix_sha256']
            and data.endswith(b'\n'), 'journal anchor differs')
    return [json.loads(line) for line in data.splitlines()]


def import_component(worker, capsule, *, expected_sha256, parent):
    c = worker.research
    old = check(c)
    require(expected_sha256 == c._external_component_digest, 'trusted configured digest differs')
    require(not worker.pair_mode and worker.execution_profile == 'paired_restart_trace/2',
            'single-body traced worker required')
    require(c.config.get('restart_predictions') is True, 'restart predictions must be enabled')
    require(c._external_history_identity is None or c._external_history is not None,
            'import the configured ordinary history first')
    require(parent.parent_id is None and parent.island == 0
            and parent.code == worker.builder.source_for(), 'actual default root required')
    root = dict(id=parent.id, source_sha256=hashlib.sha256(parent.code.encode()).hexdigest())
    from ..knowledge_base.generated_paired_history import validate_capsule
    validated = validate_capsule(capsule, expected_sha256=expected_sha256, metric=c.metric, root=root)
    if old is not None:
        require(canonical_json(old) == canonical_json(validated), 'different component already imported')
        return False
    require(not c.pending and not c.queue and c.stopped_reason is None
        and all(getattr(c, key) == 0 for key in ('attempts', 'outcomes', 'valid_hypotheses',
            'aux_calls', 'aux_cost', 'reflections', 'reflection_changes', 'reflected_through')),
        'component import must precede local research')
    require(not worker.seen_proofs and worker.pending is None and worker.stopped_reason is None
        and all(getattr(worker, key) == 0 for key in ('attempts', 'finished', 'calls_started',
            'calls_returned', 'invalid_responses', 'repeated_proposals', 'evaluations', 'known_cost')),
        'component import must precede local generation')
    # Validate every mutation on a detached KB before committing the event.
    from ..knowledge_base.base import KnowledgeBase
    knowledge = KnowledgeBase(**c.knowledge.state()['limits'])
    knowledge.load_state(c.knowledge.state())
    knowledge.import_generated_paired_history(validated, expected_sha256=expected_sha256,
                                              metric=c.metric, root=root)
    require(callable(getattr(c.log, 'anchor', None)), 'durable journal anchor required')
    rows = _anchored_rows(c.log, c.log.anchor())
    require(not any(r.get('event') == 'research' and r.get('phase') == EVENT for r in rows),
            'component import already in journal; refusing replay')
    fields = event_fields(c, validated)
    c.emit(EVENT, **fields)
    rows = _anchored_rows(c.log, c.log.anchor())
    require(rows[-1].get('event') == 'research' and rows[-1].get('phase') == EVENT
            and _event_matches(rows[-1], fields), 'durable import event differs')
    c.knowledge = knowledge
    return True


def validate_loaded(controller, knowledge, state):
    check(controller)
    expected = controller._external_component_digest
    require(canonical_json(state.get('external_component_identity'))
            == canonical_json(controller._external_component_identity)
            and (expected is not None or 'external_component_identity' not in state),
            'checkpoint policy differs')
    capsule = knowledge.generated_paired_history
    if capsule is not None:
        require(expected is not None, 'checkpoint evidence requires configured digest')
        from ..knowledge_base.generated_paired_history import validate_capsule
        validate_capsule(capsule, expected_sha256=expected, metric=controller.metric)
    elif expected is not None:
        require(controller.knowledge.generated_paired_history is None
            and not any(state.get(k) for k in ('attempts', 'outcomes', 'aux_calls',
                                               'aux_cost', 'reflections', 'reflected_through')),
            'checkpoint removed declared evidence')
    from .generated_paired_review_import import validate_loaded as validate_review
    validate_review(controller, knowledge, state)


def assert_import_journal(controller, rows):
    capsule = check(controller)
    events = [r for r in rows if r.get('event') == 'research' and r.get('phase') == EVENT]
    if capsule is None:
        require(not events, 'journal import without evidence')
        from .generated_paired_review_import import assert_import_journal as assert_review
        assert_review(controller, rows)
        return
    require(len(events) == 1, 'one durable component import required')
    event = events[0]
    require(_event_matches(event, event_fields(controller, capsule)),
            'journal import binding differs')
    index = rows.index(event)
    require(all(i > index for i, r in enumerate(rows) if r.get('event') == 'research'
        and r.get('phase') in ('attempt', 'hypothesis', 'outcome', 'circle_generation')),
        'local research preceded component import')
    from .generated_paired_review_import import assert_import_journal as assert_review
    assert_review(controller, rows)
