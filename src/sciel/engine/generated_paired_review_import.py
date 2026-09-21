"""Durable optional import of one R sidecar over a declared paired component.

The foreign writer is replayed from its recorded response. Import itself owns
no model call, ordinary outcome, lesson, cost or reflection watermark. Host text
decisions remain receiving configuration, never grants in the imported state.
"""
from __future__ import annotations

import copy

from ..knowledge_base.catalogs import canonical_json, canonical_sha

CONFIG_KEY = 'external_component_review_sha256'
EVENT = 'external_component_review_import'
VERSION_KEY = 'generated_paired_review_version'


def require(ok, message):
    if not ok:
        raise ValueError('generated paired review import: ' + message)


def configured_version(config):
    """A separate component-only contract; ordinary reflection stays version 3."""
    value = config.get(VERSION_KEY, 1)
    require(type(value) is int and value in (1, 2), 'generated_paired_review_version must be 1 or 2')
    if VERSION_KEY in config:
        from .generated_paired_import import configured_digest as component_digest
        from .external_research_v2 import _review_text_policy
        require(component_digest(config) is not None
            and config.get('restart_predictions') is True
            and config.get('structured_auxiliary') is True
            and config.get('comparison_reviews') is True
            and type(config.get('comparison_review_version')) is int
            and config['comparison_review_version'] == 3
            and _review_text_policy(config) is not None,
            'explicit generated review version requires a component and reasoned host-gated reviews')
    return value


def _check_version(config, sidecar):
    require(configured_version(config) == configured_version(sidecar['config']),
            'receiving generated review version differs from sidecar')


def configured_digest(config):
    configured_version(config)
    value = config.get(CONFIG_KEY)
    require(value is None or (type(value) is str and len(value) == 64
        and all(c in '0123456789abcdef' for c in value)), 'invalid configured digest')
    require(CONFIG_KEY not in config or value is not None, 'explicit digest cannot be null')
    if value is None:
        return None
    from .generated_paired_import import configured_digest as component_digest
    from .external_research_v2 import _review_text_policy
    require(component_digest(config) is not None, 'review requires declared source component')
    require(config.get('restart_predictions') is True
        and type(config.get('comparison_review_version')) is int
        and config['comparison_review_version'] == 3,
        'review requires explicit reasoned version 3 and restart predictions')
    require(_review_text_policy(config) is not None,
        'review requires an explicit host text eligibility policy')
    return value


def identity(config=None):
    from .external_research_v2 import policy_identity
    version = configured_version(config or {})
    result = dict(schema='sciel-generated-paired-review-import/1', policy=policy_identity())
    if version == 2:
        result.update(schema='sciel-generated-paired-review-import/2', generated_paired_review_version=2)
    return result


def check(controller, *, require_present=False):
    """Validate this extension only; base-import helpers call it, never vice versa."""
    expected = configured_digest(controller.config)
    require(expected == getattr(controller, '_external_component_review_digest', None),
            'configured digest changed')
    wanted = identity(controller.config) if expected is not None else None
    require(getattr(controller, '_external_component_review_identity', None) == wanted,
            'receiving policy changed')
    if expected is not None:
        controller._review_text_options()
    sidecar = controller.knowledge.generated_paired_review
    if sidecar is not None:
        require(expected is not None, 'review requires explicit configured digest')
        component = controller.knowledge.generated_paired_history
        require(component is not None, 'review requires imported source component')
        from .generated_paired_reflection import validate_sidecar
        validate_sidecar(sidecar, component=component, expected_sha256=expected)
        _check_version(controller.config, sidecar)
    elif require_present and expected is not None:
        raise ValueError('generated paired review import: configured review has not been imported')
    return sidecar


def event_fields(controller, sidecar):
    component = controller.knowledge.generated_paired_history
    require(component is not None, 'review requires imported source component')
    return dict(sidecar_sha256=canonical_sha(sidecar),
        source_component_sha256=canonical_sha(component),
        receiving_root=copy.deepcopy(component['root_parent']),
        policy_sha256=canonical_sha(controller._external_component_review_identity),
        raw_response_sha256=_raw_sha(sidecar),
        review_cards_sha256=canonical_sha(sidecar['review_cards']),
        comparison_ids=list(sidecar['review_cards']),
        new_attempts=0, new_outcomes=0, new_model_calls=0, new_aux_calls=0,
        new_reflections=0, new_cost=0)


def _raw_sha(sidecar):
    from .external_reflection import _digest
    return _digest(sidecar['response']['text'])


def import_review(worker, sidecar, *, expected_sha256):
    from .generated_paired_import import (check as check_component, _anchored_rows,
        _event_matches, assert_import_journal as assert_component_journal)
    controller = worker.research
    component = check_component(controller)
    require(component is not None, 'import the declared source component first')
    old = check(controller)
    require(expected_sha256 == controller._external_component_review_digest,
            'trusted configured digest differs')
    require(not worker.pair_mode and worker.execution_profile == 'paired_restart_trace/2',
            'single-body traced worker required')
    from .generated_paired_reflection import validate_sidecar
    validated = validate_sidecar(sidecar, component=component, expected_sha256=expected_sha256)
    _check_version(controller.config, validated)
    if old is not None:
        require(canonical_json(old) == canonical_json(validated), 'different review already imported')
        return False
    require(not controller.pending and not controller.queue and controller.stopped_reason is None
        and all(getattr(controller, key) == 0 for key in ('attempts', 'outcomes',
            'valid_hypotheses', 'aux_calls', 'aux_cost', 'reflections', 'reflection_changes', 'reflected_through')),
        'review import must precede local research')
    require(not worker.seen_proofs and worker.pending is None and worker.stopped_reason is None
        and all(getattr(worker, key) == 0 for key in ('attempts', 'finished', 'calls_started',
            'calls_returned', 'invalid_responses', 'repeated_proposals', 'evaluations', 'known_cost')),
        'review import must precede local generation')
    from ..knowledge_base.base import KnowledgeBase
    knowledge = KnowledgeBase(**controller.knowledge.state()['limits'])
    knowledge.load_state(controller.knowledge.state())
    knowledge.import_generated_paired_review(validated, expected_sha256=expected_sha256)
    require(callable(getattr(controller.log, 'anchor', None)), 'durable journal anchor required')
    rows = _anchored_rows(controller.log, controller.log.anchor())
    assert_component_journal(controller, rows)
    require(not any(row.get('event') == 'research' and row.get('phase') == EVENT for row in rows),
            'review import already in journal; refusing replay')
    fields = event_fields(controller, validated)
    controller.emit(EVENT, **fields)
    rows = _anchored_rows(controller.log, controller.log.anchor())
    require(rows[-1].get('event') == 'research' and rows[-1].get('phase') == EVENT
        and _event_matches(rows[-1], fields), 'durable import event differs')
    controller.knowledge = knowledge
    return True


def validate_loaded(controller, knowledge, state):
    check(controller)
    expected = getattr(controller, '_external_component_review_digest', None)
    expected_identity = getattr(controller, '_external_component_review_identity', None)
    require(canonical_json(state.get('external_component_review_identity'))
        == canonical_json(expected_identity)
        and (expected is not None or 'external_component_review_identity' not in state),
        'checkpoint policy differs')
    sidecar = knowledge.generated_paired_review
    if sidecar is not None:
        require(expected is not None and knowledge.generated_paired_history is not None,
                'checkpoint review requires declared component and review digests')
        from .generated_paired_reflection import validate_sidecar
        validate_sidecar(sidecar, component=knowledge.generated_paired_history, expected_sha256=expected)
        _check_version(controller.config, sidecar)
    elif expected is not None:
        require(controller.knowledge.generated_paired_review is None
            and not state.get('pending') and not state.get('queue')
            and not any(state.get(key) for key in ('attempts', 'outcomes', 'aux_calls',
                'aux_cost', 'reflections', 'reflected_through')),
            'checkpoint removed declared review')


def assert_import_journal(controller, rows):
    sidecar = check(controller)
    events = [row for row in rows if row.get('event') == 'research' and row.get('phase') == EVENT]
    if sidecar is None:
        require(not events, 'journal review import without retained review')
        return
    require(len(events) == 1, 'one durable review import required')
    from .generated_paired_import import EVENT as COMPONENT_EVENT, _event_matches
    bases = [row for row in rows if row.get('event') == 'research' and row.get('phase') == COMPONENT_EVENT]
    require(len(bases) == 1 and rows.index(bases[0]) < rows.index(events[0]),
            'review import must follow its durable component import')
    require(_event_matches(events[0], event_fields(controller, sidecar)), 'journal import binding differs')
    index = rows.index(events[0])
    local = {'attempt', 'hypothesis', 'outcome', 'circle_generation', 'aux_start',
             'aux_result', 'aux_skipped', 'aux_stopped', 'conclusion', 'reflection', 'reflection_context'}
    require(all(i > index for i, row in enumerate(rows) if row.get('event') == 'research'
        and row.get('phase') in local), 'local research preceded review import')
