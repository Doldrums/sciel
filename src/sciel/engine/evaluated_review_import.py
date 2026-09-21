"""Retain one raw-bound study review under independent receiving authority.

Import is not text acceptance, authoring, local research progress or execution.
The original reflection configuration travels as data and is separately pinned
by the receiving controller before raw request/result reconstruction.
"""
from __future__ import annotations
import copy
from ..knowledge_base.catalogs import canonical_json, canonical_sha
from .evaluated_study_import import _is_local_research

CONFIG_KEY = 'external_evaluated_review_sha256'
REFLECTION_CONFIG_KEY = 'source_reflection_config_sha256'
EVENT = 'external_evaluated_review_import'


def require(ok, message):
    if not ok:
        raise ValueError('evaluated review import: ' + message)


def configured_digests(config):
    values = tuple(config.get(k) for k in (CONFIG_KEY, REFLECTION_CONFIG_KEY))
    for key, value in zip((CONFIG_KEY, REFLECTION_CONFIG_KEY), values):
        require(key not in config or (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value)), 'invalid configured digest')
    require((CONFIG_KEY in config) == (REFLECTION_CONFIG_KEY in config),
            'review and independent reflection config pins required together')
    if values[0] is not None:
        from .evaluated_study_import import configured_digests as study_digests
        from .external_research_v2 import _review_text_policy
        require(study_digests(config)[0] is not None, 'review requires declared source study')
        require(config.get('restart_predictions') is True and _review_text_policy(config) is not None,
                'review requires restart facts and explicit host text policy')
    return values


def identity():
    from .external_research_v2 import policy_identity
    return dict(schema='sciel-evaluated-review-import/1', policy=policy_identity())


def validate_retained(value, bundle, config):
    expected, reflection_sha = configured_digests(config)
    require(type(value) is dict and set(value) == {'sidecar', 'reflection_config'}, 'invalid retained review')
    require(expected is not None and canonical_sha(value['reflection_config']) == reflection_sha,
            'independent reflection configuration differs')
    from .evaluated_study_reflection import validate_sidecar
    checked = validate_sidecar(value['sidecar'], bundle=bundle, expected_sha256=expected,
                               config=value['reflection_config'])
    return dict(sidecar=checked, reflection_config=copy.deepcopy(value['reflection_config']))


def check(controller, *, require_present=False):
    digests = configured_digests(controller.config)
    require(digests == controller._external_evaluated_review_digests, 'configured digests changed')
    wanted = identity() if digests[0] is not None else None
    require(canonical_json(wanted) == canonical_json(controller._external_evaluated_review_identity),
            'receiving policy changed')
    if digests[0] is not None:
        controller._review_text_options()
    retained = controller.knowledge.evaluated_study_review
    if retained is not None:
        from .evaluated_study_import import check as check_study
        bundle = check_study(controller, require_present=True)
        require(bundle is not None, 'review requires imported source study')
        validate_retained(retained, bundle, controller.config)
    elif require_present and digests[0] is not None:
        raise ValueError('evaluated review import: configured review has not been imported')
    return retained


def event_fields(controller, retained):
    from .external_reflection import _digest
    sidecar = retained['sidecar']; bundle = controller.knowledge.evaluated_detached_study
    require(bundle is not None, 'review requires imported source study')
    return dict(sidecar_sha256=canonical_sha(sidecar),
        reflection_config_sha256=canonical_sha(retained['reflection_config']),
        source_study_sha256=canonical_sha(bundle['capsule']),
        author_authority_sha256=canonical_sha(bundle['author_authority']),
        receiving_root=copy.deepcopy(bundle['capsule']['receiving_root']),
        policy_sha256=canonical_sha(controller._external_evaluated_review_identity),
        raw_response_sha256=_digest(sidecar['response']['text']),
        review_card_sha256=canonical_sha(sidecar['review_card']),
        new_attempts=0, new_outcomes=0, new_model_calls=0, new_aux_calls=0,
        new_reflections=0, new_cost=0, new_lessons=0, new_exposures=0)


def import_review(worker, sidecar, reflection_config, *, expected_sha256):
    from .evaluated_study_import import check as check_study, assert_import_journal as assert_study_journal
    from .generated_paired_import import _anchored_rows, _event_matches
    c = worker.research
    bundle = check_study(c, require_present=True)
    require(bundle is not None, 'import the declared source study first')
    old = check(c)
    require(expected_sha256 == c._external_evaluated_review_digests[0], 'trusted configured digest differs')
    require(not worker.pair_mode and worker.execution_profile == 'paired_restart_trace/2',
            'single-body traced worker required')
    validated = validate_retained(dict(sidecar=sidecar, reflection_config=reflection_config), bundle, c.config)
    if old is not None:
        require(canonical_json(old) == canonical_json(validated), 'different review already imported')
        return False
    require(not c.pending and not c.queue and c.stopped_reason is None
        and all(getattr(c, k) == 0 for k in ('attempts', 'outcomes', 'valid_hypotheses', 'aux_calls',
            'aux_cost', 'reflections', 'reflection_changes', 'reflected_through')),
        'review import must precede local research')
    require(not worker.seen_proofs and worker.pending is None and worker.stopped_reason is None
        and all(getattr(worker, k) == 0 for k in ('attempts', 'finished', 'calls_started',
            'calls_returned', 'invalid_responses', 'repeated_proposals', 'evaluations', 'known_cost')),
        'review import must precede local generation')
    from ..knowledge_base.base import KnowledgeBase
    knowledge = KnowledgeBase(**c.knowledge.state()['limits'])
    knowledge.load_state(c.knowledge.state())
    knowledge.import_evaluated_study_review(validated['sidecar'], validated['reflection_config'],
        expected_sha256=expected_sha256)
    require(callable(getattr(c.log, 'anchor', None)), 'durable journal anchor required')
    rows = _anchored_rows(c.log, c.log.anchor())
    assert_study_journal(c, rows)
    require(not any(r.get('event') == 'research' and r.get('phase') == EVENT for r in rows),
            'review import already in journal; refusing replay')
    require(not any(_is_local_research(row) for row in rows),
            'local research preceded review import')
    fields = event_fields(c, validated)
    c.emit(EVENT, **fields)
    rows = _anchored_rows(c.log, c.log.anchor())
    require(rows[-1].get('event') == 'research' and rows[-1].get('phase') == EVENT
        and _event_matches(rows[-1], fields), 'durable import event differs')
    c.knowledge = knowledge
    return True


def validate_loaded(controller, knowledge, state):
    check(controller)
    expected = controller._external_evaluated_review_digests[0]
    require(canonical_json(state.get('external_evaluated_review_identity'))
        == canonical_json(controller._external_evaluated_review_identity)
        and (expected is not None or 'external_evaluated_review_identity' not in state),
        'checkpoint policy differs')
    retained = knowledge.evaluated_study_review
    if retained is not None:
        require(expected is not None and knowledge.evaluated_detached_study is not None,
                'checkpoint review requires declared source study and review')
        validate_retained(retained, knowledge.evaluated_detached_study, controller.config)
    elif expected is not None:
        require(controller.knowledge.evaluated_study_review is None
            and not state.get('pending') and not state.get('queue')
            and not any(state.get(k) for k in ('attempts', 'outcomes', 'aux_calls', 'aux_cost',
                                             'reflections', 'reflected_through')),
            'checkpoint removed declared review')


def assert_import_journal(controller, rows):
    from .generated_paired_import import _event_matches
    from .evaluated_study_import import EVENT as STUDY_EVENT
    retained = check(controller)
    events = [r for r in rows if r.get('event') == 'research' and r.get('phase') == EVENT]
    if retained is None:
        require(not events, 'journal review import without retained review')
        return
    require(len(events) == 1, 'one durable review import required')
    bases = [r for r in rows if r.get('event') == 'research' and r.get('phase') == STUDY_EVENT]
    require(len(bases) == 1 and rows.index(bases[0]) < rows.index(events[0]),
            'review import must follow its durable study import')
    require(_event_matches(events[0], event_fields(controller, retained)), 'journal import binding differs')
    index = rows.index(events[0])
    require(all(i > index for i,r in enumerate(rows) if _is_local_research(r)),
            'local research preceded review import')
