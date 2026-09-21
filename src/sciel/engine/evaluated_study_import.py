"""Durable, explicitly authorized import of one detached evaluated study.

The portable evidence and its authority record are retained together, but only
an independently configured authority digest grants admission to a controller.
No native outcome, lesson credit, exposure or cost is invented by this import.
"""
from __future__ import annotations

import copy
import hashlib

from ..knowledge_base.catalogs import canonical_json, canonical_sha
from .generated_paired_import import _anchored_rows, _event_matches

CONFIG_KEY = 'external_evaluated_study_sha256'
AUTHORITY_KEY = 'external_evaluated_study_authority_sha256'
EVENT = 'external_evaluated_study_import'

# Local controller/worker events, including the historical auxiliary alias.
# External import anchors and run metadata are deliberately not local activity.
_LOCAL_RESEARCH_PHASES = frozenset({
    'attempt', 'structured_response', 'structured_pair', 'structured_hypothesis',
    'proposal_group', 'hypothesis', 'outcome', 'circle_generation', 'idea', 'intervention',
    'aux_request', 'aux_start', 'aux_result', 'aux_skipped', 'aux_stopped',
    'conclusion_rejected', 'conclusion', 'reflection_context', 'reflection',
    'initializer_witness_presentation', 'restart_observation_presentation',
    'initializer_witness_unavailable', 'initializer_witness_intent',
    'initializer_witness_result', 'initializer_witness_registered', 'initializer_witness_stopped',
})


def _is_local_research(row):
    return row.get('event') == 'research' and row.get('phase') in _LOCAL_RESEARCH_PHASES


def require(value, message):
    if not value:
        raise ValueError('evaluated study import: ' + message)


def configured_digests(config):
    values = tuple(config.get(key) for key in (CONFIG_KEY, AUTHORITY_KEY))
    for key, value in zip((CONFIG_KEY, AUTHORITY_KEY), values):
        require(key not in config or (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value)), 'invalid configured digest')
    require((CONFIG_KEY in config) == (AUTHORITY_KEY in config),
            'study and independent authority digests must be configured together')
    return values


def identity():
    from ..knowledge_base.evaluated_detached_study import identity as adapter_identity
    return dict(schema='sciel-evaluated-study-import/1', adapter=adapter_identity())


def validate_bundle(bundle, config, *, metric, root=None):
    expected, authority_sha = configured_digests(config)
    require(type(bundle) is dict and set(bundle) == {'capsule', 'author_dependencies', 'author_authority'},
            'invalid portable study bundle')
    require(expected is not None and canonical_sha(bundle['author_authority']) == authority_sha,
            'independent configured author authority differs')
    from ..knowledge_base.evaluated_detached_study import validate_capsule
    validate_capsule(bundle['capsule'], expected_sha256=expected, metric=metric, root=root,
        author_dependencies=bundle['author_dependencies'], author_authority=bundle['author_authority'])
    return copy.deepcopy(bundle)


def check(controller, *, require_present=False):
    digests = configured_digests(controller.config)
    require(digests == controller._external_evaluated_study_digests, 'configured digests changed')
    wanted = identity() if digests[0] is not None else None
    require(canonical_json(wanted) == canonical_json(controller._external_evaluated_study_identity),
            'receiving policy changed')
    bundle = controller.knowledge.evaluated_detached_study
    if bundle is not None:
        validate_bundle(bundle, controller.config, metric=controller.metric)
    elif require_present and digests[0] is not None:
        raise ValueError('evaluated study import: configured study has not been imported')
    return bundle


def event_fields(controller, bundle):
    capsule = bundle['capsule']
    return dict(capsule_sha256=canonical_sha(capsule),
        author_dependencies_sha256=canonical_sha(bundle['author_dependencies']),
        author_authority_sha256=canonical_sha(bundle['author_authority']),
        receiving_root=copy.deepcopy(capsule['receiving_root']),
        policy_sha256=canonical_sha(controller._external_evaluated_study_identity),
        author=copy.deepcopy(capsule['author']), comparison_id=capsule['comparison']['id'],
        new_attempts=0, new_outcomes=0, new_model_calls=0, new_cost=0,
        new_lessons=0, new_exposures=0)


def import_study(worker, capsule, dependencies, authority, *, expected_sha256, parent):
    c = worker.research
    old = check(c)
    require(expected_sha256 == c._external_evaluated_study_digests[0], 'trusted configured digest differs')
    require(not worker.pair_mode and worker.execution_profile == 'paired_restart_trace/2'
            and c.config.get('restart_predictions') is True, 'single-body traced worker required')
    require(c._external_history_identity is None or c._external_history is not None,
            'import the configured ordinary history first')
    require(parent.parent_id is None and parent.island == 0
            and parent.code == worker.builder.source_for(), 'actual default root required')
    root = dict(id=parent.id, source_sha256=hashlib.sha256(parent.code.encode()).hexdigest())
    bundle = validate_bundle(dict(capsule=capsule, author_dependencies=dependencies,
        author_authority=authority), c.config, metric=c.metric, root=root)
    if old is not None:
        require(canonical_json(old) == canonical_json(bundle), 'different study already imported')
        return False
    require(not c.pending and not c.queue and c.stopped_reason is None
        and all(getattr(c, key) == 0 for key in ('attempts', 'outcomes', 'valid_hypotheses',
            'aux_calls', 'aux_cost', 'reflections', 'reflection_changes', 'reflected_through')),
        'study import must precede local research')
    require(not worker.seen_proofs and worker.pending is None and worker.stopped_reason is None
        and all(getattr(worker, key) == 0 for key in ('attempts', 'finished', 'calls_started',
            'calls_returned', 'invalid_responses', 'repeated_proposals', 'evaluations', 'known_cost')),
        'study import must precede local generation')
    from ..knowledge_base.base import KnowledgeBase
    knowledge = KnowledgeBase(**c.knowledge.state()['limits'])
    knowledge.load_state(c.knowledge.state())
    knowledge.import_evaluated_detached_study(capsule, dependencies, authority,
        expected_sha256=expected_sha256, metric=c.metric, root=root)
    require(callable(getattr(c.log, 'anchor', None)), 'durable journal anchor required')
    rows = _anchored_rows(c.log, c.log.anchor())
    require(not any(r.get('event') == 'research' and r.get('phase') == EVENT for r in rows),
            'study import already in journal; refusing replay')
    require(not any(_is_local_research(row) for row in rows),
            'local research preceded study import')
    fields = event_fields(c, bundle)
    c.emit(EVENT, **fields)
    rows = _anchored_rows(c.log, c.log.anchor())
    require(rows[-1].get('event') == 'research' and rows[-1].get('phase') == EVENT
            and _event_matches(rows[-1], fields), 'durable import event differs')
    c.knowledge = knowledge
    return True


def validate_loaded(controller, knowledge, state):
    check(controller)
    expected = controller._external_evaluated_study_digests[0]
    require(canonical_json(state.get('external_evaluated_study_identity'))
            == canonical_json(controller._external_evaluated_study_identity)
            and (expected is not None or 'external_evaluated_study_identity' not in state),
            'checkpoint policy differs')
    bundle = knowledge.evaluated_detached_study
    if bundle is not None:
        validate_bundle(bundle, controller.config, metric=controller.metric)
    elif expected is not None:
        require(controller.knowledge.evaluated_detached_study is None
            and not any(state.get(k) for k in ('attempts', 'outcomes', 'aux_calls',
                'aux_cost', 'reflections', 'reflected_through')), 'checkpoint removed declared study')


def assert_import_journal(controller, rows):
    bundle = check(controller)
    events = [r for r in rows if r.get('event') == 'research' and r.get('phase') == EVENT]
    if bundle is None:
        require(not events, 'journal import without evidence')
        return
    require(len(events) == 1, 'one durable study import required')
    event = events[0]
    require(_event_matches(event, event_fields(controller, bundle)), 'journal import binding differs')
    index = rows.index(event)
    require(all(i > index for i, r in enumerate(rows) if _is_local_research(r)),
        'local research preceded study import')
