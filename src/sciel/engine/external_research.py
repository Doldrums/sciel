"""Explicit, bounded import of completed Circle history into a fresh run.

The imported journals remain foreign provenance. No old request, observation,
lesson admission, cost or advice exposure is replayed in the new component.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

from ..hypothesis.circle_source_family import builder_for_profile
from ..hypothesis.conclusion_feedback import validate_conclusions
from ..hypothesis.parsing import decode_conclusion, parse_reflection
from ..knowledge_base.base import KnowledgeBase, _experiment
from ..knowledge_base.catalogs import canonical_json, canonical_sha

SCHEMA = 'sciel-external-research-history/1'
REGISTRY_SCHEMA = 'sciel-external-research-registry/1'
SCHEMA_V2 = 'sciel-external-research-history/2'
REGISTRY_SCHEMA_V2 = 'sciel-external-research-registry/2'
CONFIG_KEY = 'external_history_sha256'
MAX_BYTES = 8 * 1024**2
MAX_DEPTH = 64
EVENT = 'external_observation_import'
PROFILE = 'paired_restart_trace/2'


def _require(value, message):
    if not value:
        raise ValueError('external research: ' + message)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _bounded(value):
    """Reject oversized/nonliteral trees before canonical serialization."""
    remaining = MAX_BYTES
    active = set()

    def visit(item, depth):
        nonlocal remaining
        _require(depth < MAX_DEPTH, 'capsule depth exceeded')
        remaining -= 1
        _require(remaining >= 0, 'capsule exceeds 8 MiB')
        if type(item) is str:
            _require(len(item) <= remaining, 'capsule exceeds 8 MiB')
            remaining -= len(item.encode('utf-8'))
        elif item is None or type(item) is bool:
            return
        elif type(item) is int:
            _require(item.bit_length() <= 8192, 'integer out of bounds')
        elif type(item) is float:
            _require(math.isfinite(item), 'nonfinite value')
        else:
            _require(type(item) in (dict, list), 'literal JSON required')
            _require(id(item) not in active, 'cyclic capsule')
            active.add(id(item))
            if type(item) is dict:
                for key, child in item.items():
                    _require(type(key) is str, 'string keys required')
                    visit(key, depth + 1)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
            active.remove(id(item))
        _require(remaining >= 0, 'capsule exceeds 8 MiB')

    visit(value, 0)
    encoded = canonical_json(value)
    _require(len(encoded.encode('utf-8')) <= MAX_BYTES, 'capsule exceeds 8 MiB')
    return encoded


def configured_digest(config):
    value = config.get(CONFIG_KEY)
    _require(value is None or _sha(value), 'invalid configured capsule digest')
    return value


def identity():
    from .external_research_v2 import policy_identity
    return {'schema': SCHEMA, 'registry_schema': REGISTRY_SCHEMA, 'max_bytes': MAX_BYTES,
            'v2_policy': policy_identity(),
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _identity_check(controller):
    _require(getattr(controller, '_external_history_identity', None) == identity(),
             'external history policy changed or was not enabled')


def _journal(text):
    _require(type(text) is str and text.endswith('\n'), 'complete raw journal required')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate journal key')
            result[key] = value
        return result

    try:
        rows = [json.loads(line, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
                for line in text.splitlines()]
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('external research: invalid journal') from exc
    _require(rows and all(type(row) is dict for row in rows), 'journal records required')
    first = rows[0]
    _require(first.get('event') == 'run_start' and first.get('resumed') is False,
             'original non-resumed journal required')
    _require(type(first.get('run_id')) is str and bool(first['run_id'])
             and type(first.get('segment_id')) is str and bool(first['segment_id']), 'journal identity absent')
    _require(all(type(row.get('event_seq')) is int and row['event_seq'] == index
                 and row.get('run_id') == first['run_id'] and row.get('segment_id') == first['segment_id']
                 for index, row in enumerate(rows, 1)), 'journal sequence or identity differs')
    return rows


def _one(rows, **fields):
    matched = [row for row in rows if all(row.get(key) == value for key, value in fields.items())]
    _require(len(matched) == 1, 'one exact source event required')
    return matched[0]


def _knowledge(value, config, metric):
    kb = KnowledgeBase(max_lessons=config['max_lessons'], max_experiments=config['max_experiments'],
                       max_fragments=config['max_fragments'])
    kb.load_state(value)
    _require(kb.state() == value, 'knowledge must be exact canonical state')
    _require(not kb.lessons, 'nonempty imported lessons are not supported')
    _require(len(kb.experiments) == 1 and len(kb.comparisons) == 2,
             'only one ordinary observation and two groups are supported')
    record = kb.experiments[0]
    _require(record['metric'] == metric and record['operator'] == 'circle_generate'
             and record['status'] == 'evaluated' and not record['source_repaired']
             and record['generation_dispatch'] == 'returned', 'ordinary scope or metric differs')
    _require(not record['injected_ids'], 'historical advice exposures are unsupported')
    from .research import ResearchController
    from types import SimpleNamespace
    for group in kb.comparisons:
        ResearchController._check_comparison_binding(SimpleNamespace(metric=metric), group, kb)
    _require(kb.comparisons[-1]['origin_experiment_id'] == record['id']
             and kb.comparisons[-1]['derived']['outcome'] == 'mixed', 'original paired origin/outcome differs')
    return kb, record


def _validate(capsule, *, expected_sha256, config, metric):
    from .external_research_v2 import _review_text_policy
    if _review_text_policy(config) is not None:
        _require(type(capsule) is dict and capsule.get('schema') == SCHEMA_V2,
                 'review text policy requires explicit v2 history; legacy import is unsupported')
    encoded = _bounded(capsule)
    _require(_sha(expected_sha256) and configured_digest(config) == expected_sha256
             and _digest(encoded) == expected_sha256, 'capsule differs from declared digest')
    if type(capsule) is dict and capsule.get('schema') == SCHEMA_V2:
        from .external_research_v2 import validate
        result = validate(capsule, config=config, metric=metric)
        registry = {'schema': REGISTRY_SCHEMA_V2, 'capsule_sha256': expected_sha256,
                    'capsule': copy.deepcopy(capsule), 'attempt_id_floor': result['floor'],
                    'imported_ids': result['imported_ids'], 'generation': result['generation'],
                    'comparison_reviews': result['comparison_reviews']}
        return registry, result['knowledge'], result['conclusions']
    _require(type(capsule) is dict and set(capsule) == {'schema', 'source', 'conclusion', 'reflection'}
             and capsule['schema'] == SCHEMA, 'capsule schema differs')
    blocks = [capsule[name] for name in ('source', 'conclusion', 'reflection')]
    for block in blocks:
        _require(type(block) is dict and set(block) == {'manifest_sha256', 'state', 'journal'}
                 and _sha(block['manifest_sha256']) and type(block['state']) is dict, 'source block differs')
    source, conclusion, reflection = [block['state'] for block in blocks]
    source_rows, conclusion_rows, reflection_rows = [_journal(block['journal']) for block in blocks]
    kb, record = _knowledge(reflection['knowledge'], config, metric)
    _require(source['knowledge'] == conclusion['knowledge'] == reflection['knowledge'],
             'historical knowledge changed between measured source and reviews')
    _require(all(block['metric'] == metric and not block['pending'] and not block['queue']
                 and all(type(block.get(key)) is int and block[key] == 1
                         for key in ('attempts', 'outcomes', 'valid_hypotheses'))
                 and block['lessons'] == 0 for block in (source, conclusion, reflection)),
             'historical state accounting differs')
    _require(conclusion['aux_calls'] == 1 and conclusion['reflected_through'] == 0
             and conclusion['reflections'] == 0 and conclusion['stopped_reason'] is None
             and reflection['aux_calls'] == 2 and reflection['reflected_through'] == 1
             and reflection['reflections'] == 1 and reflection['reflection_changes'] == 0
             and reflection['stopped_reason'] is None, 'completed C/R states required')
    reviews = validate_conclusions(reflection['conclusions'], {record['id']: record}, batch_only_lessons=True)
    _require(reviews == conclusion['conclusions'] and set(reviews) == {record['id']}, 'accepted review changed')
    review = reviews[record['id']]
    _require(review['status'] == 'parsed' and review['verdict_matches_observation'] is True
             and review.get('lesson_admission', {}).get('status') == 'suppressed', 'bound matching review required')
    ident = record['id']
    request = _one(source_rows, event='research', phase='circle_generation', action='request_start', experiment_id=ident)
    result = _one(source_rows, event='research', phase='circle_generation', action='request_result', experiment_id=ident)
    commit = _one(source_rows, event='research', phase='circle_generation', action='source_binding', experiment_id=ident)
    hypothesis = _one(source_rows, event='research', phase='hypothesis', experiment_id=ident)
    outcome = _one(source_rows, event='research', phase='outcome', experiment_id=ident)
    _require([row.get('experiment_id') for row in source_rows if row.get('phase') == 'outcome'] == [ident]
             and [row.get('experiment_id') for row in source_rows if row.get('action') == 'request_result'] == [ident],
             'original journal contains other generation outcomes')
    # resolve emits before KB canonicalization adds optional child_id/error=None.
    # Validate that precise legacy omission, then use the public KB's validator;
    # do not drop arbitrary fields or rewrite stored historical knowledge.
    ordinary = outcome['evidence']
    omitted = set(record) - set(ordinary)
    _require(omitted <= {'child_id', 'error'} and not set(ordinary) - set(record)
             and all(record[key] is None for key in omitted)
             and request['event_seq'] < result['event_seq'] < commit['event_seq'] < hypothesis['event_seq'] < outcome['event_seq']
             and _experiment(ordinary) == record, 'original source/outcome sequence differs')
    _require(all(hypothesis['evidence'].get(key) == record.get(key) for key in
                 ('hypothesis', 'restart_prediction_binding', 'controlled_change', 'source_sha256',
                  'proposed_sha256', 'parent_source_sha256')), 'pre-evaluation hypothesis commitment differs')
    raw = result['response'];proof = commit['proof'];builder = builder_for_profile(PROFILE)
    _require(type(raw) is str and result['response_sha256'] == _digest(raw)
             and result['response_truncated'] is False and result['finish_reason'] == 'stop'
             and result.get('refusal') in (None, '') and result['cost'] == record['generation_cost'],
             'complete original generation response required')
    expected_proof = builder.apply(builder.source_for(proof['parent_body']), proof['target_body'])['proof']
    _require(proof == expected_proof and proof['parent_body'] == builder.DEFAULT_BODY
             and commit['source_origin'] == 'deterministic_body_compiler'
             and commit['semantic_or_causal_claim'] is False, 'supported original source proof required')
    from ..hypothesis.circle_restart_body import parse_hypothesis_response
    parsed = parse_hypothesis_response(raw)
    binding = record['restart_prediction_binding']
    _require(parsed is not None and parsed.to_dict() == record['hypothesis']
             and json.loads(raw)['body'] == proof['target_body'] == record['controlled_change']['target_body']
             and binding['response_sha256'] == _digest(raw)
             and binding['spec_sha256'] == canonical_sha(parsed.restart_predictions)
             and proof['child_source_sha256'] == binding['source_sha256'] == record['source_sha256']
             and record['proposed_sha256'] == binding['proposed_sha256'] == record['source_sha256']
             and proof['parent_source_sha256'] == record['parent_source_sha256']
             and request['parent_source_sha256'] == record['parent_source_sha256'], 'original forecast/source binding differs')
    c_start = _one(conclusion_rows, event='research', phase='aux_start', kind='conclusion', reference=ident)
    c_result = _one(conclusion_rows, event='research', phase='aux_result', kind='conclusion', reference=ident)
    c_event = _one(conclusion_rows, event='research', phase='conclusion', experiment_id=ident)
    _require(c_start['event_seq'] < c_result['event_seq'] < c_event['event_seq']
             and c_start['call_index'] == c_result['call_index'] == 1
             and c_result.get('structured_response_usable') is True and c_result['response_truncated'] is False
             and c_result.get('finish_reason') == 'stop' and c_result.get('refusal') in (None, '')
             and c_result['response_sha256'] == _digest(c_result['response']), 'complete accepted C response required')
    decoded = decode_conclusion(c_result['response'], require_mechanism=True)['conclusion']
    _require(decoded is not None and decoded.to_dict() == review['llm']
             and {key: c_event.get(key) for key in review} == review, 'accepted C differs from actual reply/journal')
    r_start = _one(reflection_rows, event='research', phase='aux_start', kind='reflection', reference=1)
    r_result = _one(reflection_rows, event='research', phase='aux_result', kind='reflection', reference=1)
    r_event = _one(reflection_rows, event='research', phase='reflection', through=1)
    _require(r_start['event_seq'] < r_result['event_seq'] < r_event['event_seq']
             and r_start['call_index'] == r_result['call_index'] == 2
             and r_result.get('structured_response_usable') is True and r_result['response_truncated'] is False
             and r_result.get('finish_reason') == 'stop' and r_result.get('refusal') in (None, '')
             and r_result['response_sha256'] == _digest(r_result['response'])
             and r_event['status'] == 'parsed' and r_event['error'] is None
             and r_event['applied'] == {'lesson_ids': [], 'retired_ids': []}
             and r_event['changed'] is False, 'successful empty reflection required')
    reflected = parse_reflection(r_result['response'])
    _require(reflected is not None and reflected.to_dict() == {'lessons': [], 'retire_ids': []},
             'nonempty imported reflection is not supported')
    _require([row.get('kind') for row in reflection_rows if row.get('phase') in ('aux_start', 'aux_result')]
             == ['reflection', 'reflection'], 'reflection component contains other requests')
    _require(conclusion['aux_cost'] == c_result['cost']
             and reflection['aux_cost'] == c_result['cost'] + r_result['cost'], 'historical auxiliary cost provenance differs')
    ids = list(kb.state()['seen_experiments'])
    _require(ids == [ident] and re.fullmatch(r'r[0-9]+a[0-9]+', ident) is not None, 'one canonical imported experiment ID required')
    floor = int(ident[1:ident.index('a')])
    _require(0 < floor < 2**31, 'imported experiment floor out of bounds')
    generation = {ident: {'record_sha256': canonical_sha(record), 'raw_result': copy.deepcopy(result),
                         'proof': copy.deepcopy(proof)}}
    registry = {'schema': REGISTRY_SCHEMA, 'capsule_sha256': expected_sha256,
                'capsule': copy.deepcopy(capsule), 'attempt_id_floor': floor,
                'imported_ids': ids, 'generation': generation}
    return registry, kb, reviews


def state(controller):
    if getattr(controller, '_external_history', None) is not None:
        _identity_check(controller)
    return copy.deepcopy(getattr(controller, '_external_history', None))


def validate_loaded(raw, *, config, metric, knowledge, conclusions, comparison_reviews=None):
    fields = {'schema', 'capsule_sha256', 'capsule', 'attempt_id_floor', 'imported_ids', 'generation'}
    if type(raw) is dict and raw.get('schema') == REGISTRY_SCHEMA_V2:
        fields.add('comparison_reviews')
    _require(type(raw) is dict and set(raw) == fields, 'registry shape differs')
    expected, old, reviews = _validate(raw['capsule'], expected_sha256=raw['capsule_sha256'], config=config, metric=metric)
    _require(raw == expected, 'registry derivation differs')
    original = old.state(); current = knowledge.state()
    for ident, digest in original['seen_experiments'].items():
        _require(current['seen_experiments'].get(ident) == digest, 'imported observation identity disappeared or changed')
    old_records = {record['id']: record for record in old.experiments}
    for record in knowledge.experiments:
        if record['id'] in old_records:
            _require(record == old_records[record['id']], 'retained imported observation changed')
            _require(conclusions.get(record['id']) == reviews.get(record['id']), 'retained imported review missing or changed')
    old_evidence = old._evidence()
    for ident, record in knowledge._evidence().items():
        if ident in old_evidence:
            _require(record == old_evidence[ident], 'retained imported evidence snapshot changed')
    before_groups = original.get('paired_comparisons', {'seen': {}, 'records': []})
    after_groups = current.get('paired_comparisons', {})
    for ident, digest in before_groups['seen'].items():
        _require(after_groups.get('seen', {}).get(ident) == digest, 'imported comparison identity disappeared or changed')
    old_groups = {group['id']: group for group in old.comparisons}
    for group in knowledge.comparisons:
        if group['id'] in old_groups:
            _require(group == old_groups[group['id']], 'retained imported comparison changed')
    for ident, review in conclusions.items():
        if ident in reviews:
            _require(review == reviews[ident], 'retained imported review changed')
    if raw['schema'] == REGISTRY_SCHEMA_V2:
        _require(type(comparison_reviews) is dict, 'comparison reviews required for v2 restore')
        # Exact foreign cards keep foreign writer references. A replacement is
        # local and must pass local accounting and the anchored journal check.
        foreign_writers = {canonical_sha(card['writer']) for card in raw['comparison_reviews'].values()}
        for ident, card in comparison_reviews.items():
            if canonical_sha(card['writer']) in foreign_writers:
                _require(card == raw['comparison_reviews'].get(ident), 'foreign comparison review changed')
    return copy.deepcopy(expected)


def experiment_number(controller, next_attempt):
    expected = configured_digest(controller.config)
    registry = getattr(controller, '_external_history', None)
    _require(type(next_attempt) is int and 0 < next_attempt < 2**31, 'invalid local experiment number')
    if expected is None:
        _require(registry is None, 'undeclared import registry')
        return next_attempt
    _identity_check(controller)
    _require(registry is not None and registry['capsule_sha256'] == expected, 'declared history has not been imported')
    return registry['attempt_id_floor'] + next_attempt


def generation_records(controller):
    registry = getattr(controller, '_external_history', None)
    if registry is not None:
        _identity_check(controller)
        validate_loaded(registry, config=controller.config, metric=controller.metric,
                        knowledge=controller.knowledge, conclusions=controller.conclusions,
                        comparison_reviews=controller._comparison_reviews)
    return copy.deepcopy(registry['generation']) if registry is not None else {}


def record_matches(controller, record):
    """A retained full record or its lossless evidence snapshot must be exact."""
    from ..knowledge_base.base import _snapshot
    registry = getattr(controller, '_external_history', None)
    if registry is None or record.get('id') not in registry['imported_ids']:
        return False
    if registry['schema'] == REGISTRY_SCHEMA_V2:
        old = registry['generation'][record['id']]['record']
    else:
        old, = registry['capsule']['reflection']['state']['knowledge']['experiments']
    return record == (old if 'code_fragment' in record else _snapshot(old))


def import_history(worker, capsule, *, expected_sha256, parent):
    c = worker.research
    _identity_check(c)
    _require(not worker.pair_mode and worker.execution_profile == PROFILE, 'single-body traced worker required')
    _require(c.config.get('restart_predictions') is True and c.config.get('batch_only_lessons') is True
             and c.config.get('structured_auxiliary') is True, 'explicit structured research policy required')
    registry, knowledge, reviews = _validate(capsule, expected_sha256=expected_sha256, config=c.config, metric=c.metric)
    if registry['schema'] == REGISTRY_SCHEMA_V2:
        root = capsule['root_parent']
        _require(parent.parent_id is None and parent.island == 0 and parent.id == root['id']
                 and parent.code == worker.builder.source_for()
                 and _digest(parent.code) == root['source_sha256'], 'actual fresh default parent binding differs')
    else:
        record = knowledge.experiments[0]
        _require(parent.parent_id is None and parent.island == 0 and parent.id == record['branch'] == record['parent_id']
                 and parent.code == worker.builder.source_for()
                 and _digest(parent.code) == record['parent_source_sha256']
                 == knowledge.comparisons[-1]['comparator_source_sha256'], 'actual fresh default parent binding differs')
    if getattr(c, '_external_history', None) is not None:
        _require(c._external_history == registry, 'different history already imported')
        return False
    empty = KnowledgeBase(max_lessons=c.config['max_lessons'], max_experiments=c.config['max_experiments'],
                          max_fragments=c.config['max_fragments']).state()
    _require(c.knowledge.state() == empty
             and not c.conclusions and not c._comparison_reviews and not c.pending and not c.queue and c.stopped_reason is None
             and all(getattr(c, key) == 0 for key in ('attempts', 'outcomes', 'valid_hypotheses',
                 'aux_calls', 'aux_cost', 'reflections', 'reflection_changes', 'reflected_through')),
             'fresh research controller required')
    _require(not worker.seen_proofs and worker.pending is None and worker.stopped_reason is None
             and all(getattr(worker, key) == 0 for key in ('attempts', 'finished', 'calls_started',
                 'calls_returned', 'invalid_responses', 'repeated_proposals', 'evaluations', 'known_cost')),
             'fresh generation worker required')
    c.emit(EVENT, **_import_event(registry))
    # Everything above is validated before the durable import, and the remaining
    # assignments cannot invoke an external callback or replay an observation.
    c.knowledge, c.conclusions, c._external_history = knowledge, reviews, registry
    c._comparison_reviews = copy.deepcopy(registry.get('comparison_reviews', {}))
    return True


def _import_event(registry):
    capsule = registry['capsule']
    provenance = ({'source_journals': [
        {'manifest_sha256': block['manifest_sha256'], 'journal_sha256': _digest(block['journal'])}
        for block in capsule['journals']]}
        if registry['schema'] == REGISTRY_SCHEMA_V2 else
        {'source_journal_sha256': _digest(capsule['source']['journal'])})
    return {'capsule_sha256': registry['capsule_sha256'], 'imported_ids': registry['imported_ids'],
            'attempt_id_floor': registry['attempt_id_floor'], **provenance,
            'new_attempts': 0, 'new_outcomes': 0, 'new_model_calls': 0}


def assert_import_journal(controller, rows):
    registry = getattr(controller, '_external_history', None)
    imports = [row for row in rows if row.get('event') == 'research' and row.get('phase') == EVENT]
    if registry is None:
        _require(not imports, 'import event without registry')
        return
    validate_loaded(registry, config=controller.config, metric=controller.metric,
                    knowledge=controller.knowledge, conclusions=controller.conclusions,
                    comparison_reviews=controller._comparison_reviews)
    _require(len(imports) == 1, 'one durable import event required')
    event = imports[0]
    expected = _import_event(registry)
    _require(all(event.get(key) == value for key, value in expected.items()), 'durable import event differs')
    import_index = rows.index(event)
    local = [(index, row) for index, row in enumerate(rows)
             if row.get('event') == 'research' and row.get('phase') in
             ('attempt', 'hypothesis', 'outcome', 'circle_generation')]
    _require(all(index > import_index
                 and row.get('experiment_id') not in registry['imported_ids'] for index, row in local),
             'imported/local journal identity collision or order differs')
    if registry['schema'] == REGISTRY_SCHEMA_V2:
        from .external_research_resume import assert_authoring_journal
        assert_authoring_journal(controller, rows[import_index + 1:])
