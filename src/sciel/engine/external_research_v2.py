"""Pure reconstruction of a bounded, explicitly declared foreign research history.

The journal and geometry checks authenticate relationships within the declared
capsule, not external execution or the truth of model prose. Reconstruction uses
detached knowledge objects only: no controller, model, evaluator or event sink is
called, and no old counters or costs become counters or costs of a fresh run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

from ..hypothesis.circle_source_family import builder_for_profile
from ..hypothesis.conclusion_feedback import validate_conclusions
from ..hypothesis.parsing import decode_conclusion, parse_reflection
from ..hypothesis.rendering import serialize_reflection
from ..knowledge_base.base import KnowledgeBase, _experiment, _number
from ..knowledge_base.catalogs import canonical_json, canonical_sha
from ..knowledge_base.paired_comparisons import SCHEMA as PAIRED_SCHEMA, validate_comparison

SCHEMA = 'sciel-external-research-history/2'
POLICY_SCHEMA = 'sciel-external-research-reducer/2'
PROFILE = 'paired_restart_trace/2'
MAX_JOURNALS = 32
MAX_OPERATIONS = 256
MAX_GROUPS = 64
_CONCLUSION_FIELDS = {'status', 'llm', 'observed_forecast_verdict', 'observed_comparison',
    'verdict_matches_observation', 'accepted_lesson_ids', 'lesson_error', 'lesson_error_detail'}


def _require(value, message):
    if not value:
        raise ValueError('external research v2: ' + message)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _review_text_policy(config):
    """Validate the host-only receiving policy at standalone entry points."""
    if 'comparison_review_text_policy' not in config:
        return None
    from ..hypothesis.review_eligibility import validate_policy
    policy = validate_policy(config['comparison_review_text_policy'])
    _require(config.get('comparison_reviews') is True
             and config.get('structured_auxiliary') is True
             and type(config.get('comparison_review_text')) is bool
             and config.get('batch_only_lessons') is True,
             'review text policy requires structured reviews, explicit text mode and batch-only lessons')
    return policy


def _reflection_effects_policy(context, config):
    """Historical request markers cannot be reinterpreted by a receiver."""
    marked = 'reflection_effects_policy' in context
    if marked:
        _require(context['reflection_effects_policy'] == 'review_only/1',
                 'unsupported historical reflection effects policy')
        _require('comparison_review_text_policy' in config,
                 'review-only reflection history requires receiving review text policy')
    return marked


def policy_identity():
    """Pin this validation policy; these are not claimed historical runtimes."""
    root = Path(__file__).resolve().parents[1]
    # Admission and prompt rendering have transitive pure dependencies. Pin the
    # installed Python tree instead of silently omitting a newly imported helper.
    # This deliberately makes unrelated production edits require a fresh pure
    # reconstruction; it never authorizes replay of historical paid requests.
    paths = tuple(sorted(p.relative_to(root).as_posix() for p in root.rglob('*.py') if p.is_file()))
    return dict(schema=POLICY_SCHEMA, profile=PROFILE, max_journals=MAX_JOURNALS,
        max_operations=MAX_OPERATIONS, max_groups=MAX_GROUPS,
        ordinary_review_retirement='after_generation_resolution_and_step_end',
        source_scope='installed_sciel_python_tree',
        sources={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths})


def _one(rows, **fields):
    result = [row for row in rows if all(row.get(k) == v for k, v in fields.items())]
    _require(len(result) == 1, 'one exact journal event required')
    return result[0]


def _attempt_delivery(attempt, request, record, kb):
    """Prove credited historical advice was current and present on the wire."""
    from .research import ResearchController
    pending = attempt.get('evidence')
    fields = (
        'id', 'branch', 'island', 'iteration', 'parent_id', 'parent_score', 'parent_source_sha256',
        'injected_ids', 'attempt', 'operator', 'metric', 'parent_cases')
    _require(type(pending) is dict and set(fields) <= set(pending), 'pre-generation attempt fields missing')
    declared = {key: pending[key] for key in fields}
    # The producer accepts integer scores; KB normalizes them to float. Keep
    # that explicit normalization while rejecting bool/int aliases elsewhere.
    declared['parent_score'] = _number(declared['parent_score'], 'parent_score', nullable=True)
    _require(canonical_json(declared) == canonical_json({key: record[key] for key in fields}),
             'pre-generation attempt binding differs')
    _require(pending.get('status') == 'pending' and pending.get('generation_dispatch') == 'unknown',
             'pre-generation attempt was not pending')
    prompt = request.get('prompt')
    _require(type(prompt) is str and request.get('prompt_sha256') == attempt.get('prompt_sha256') == _digest(prompt),
             'original generation prompt digest differs')
    lessons = {lesson['id']: lesson for lesson in kb.lessons}
    ids = record['injected_ids']
    _require(all(ident in lessons and (lessons[ident]['scope'] == 'global'
             or lessons[ident]['branch'] == record['branch']) for ident in ids),
             'historical exposure requires a currently available lesson')
    texts = [ResearchController.lesson_text(lessons[ident]) for ident in ids]
    presentation = attempt.get('lesson_presentation')
    _require(type(presentation) is dict and presentation.get('schema') == 'sciel-memory-presentation/5'
             and presentation.get('presented') == [dict(id=ident, sha256=_digest(text), chars=len(text))
                                                   for ident, text in zip(ids, texts)]
             and presentation.get('rendered_chars') == sum(map(len, texts))
             and presentation.get('max_card_chars') == 16000
             and presentation.get('max_memory_chars') == 32000,
             'historical lesson presentation differs from current literal cards')
    _require(all(len(text) <= 16000 for text in texts) and sum(map(len, texts)) <= 32000
             and prompt.endswith('LESSONS (unverified interpretations):\n' + '\n'.join(texts)),
             'whole historical lesson delivery differs from generation prompt')


def _generation(rows, ident, *, metric, root_parent, known_sources, known_parent_ids, kb):
    _require(type(ident) is str and re.fullmatch(r'r[0-9]+a[0-9]+', ident) is not None,
             'canonical generation ID required')
    attempt = _one(rows, event='research', phase='attempt', experiment_id=ident)
    request = _one(rows, event='research', phase='circle_generation', action='request_start', experiment_id=ident)
    result = _one(rows, event='research', phase='circle_generation', action='request_result', experiment_id=ident)
    commit = _one(rows, event='research', phase='circle_generation', action='source_binding', experiment_id=ident)
    hypothesis = _one(rows, event='research', phase='hypothesis', experiment_id=ident)
    outcome = _one(rows, event='research', phase='outcome', experiment_id=ident)
    _require(attempt['event_seq'] < request['event_seq'] < result['event_seq'] < commit['event_seq']
             < hypothesis['event_seq'] < outcome['event_seq'], 'generation event order differs')
    record = _experiment(outcome['evidence'])
    omitted = set(record) - set(outcome['evidence'])
    _require(omitted <= {'child_id', 'error'} and not set(outcome['evidence']) - set(record)
             and all(record[k] is None for k in omitted), 'noncanonical ordinary source evidence')
    _require(record['id'] == ident and record['metric'] == metric
             and record['operator'] == 'circle_generate'
             and record['status'] in ('evaluated', 'eval_failed')
             and not record['source_repaired'] and record['generation_dispatch'] == 'returned',
             'unsupported ordinary outcome, operator or metric')
    _require(record['branch'] == root_parent['id'], 'generation branch differs from declared root')
    _attempt_delivery(attempt, request, record, kb)
    hypothesis_fields = (
        'hypothesis', 'restart_prediction_binding', 'controlled_change', 'source_sha256',
        'proposed_sha256', 'parent_source_sha256')
    _require(type(hypothesis.get('evidence')) is dict and set(hypothesis_fields) <= set(hypothesis['evidence'])
             and canonical_json({key: hypothesis['evidence'][key] for key in hypothesis_fields})
             == canonical_json({key: record[key] for key in hypothesis_fields}),
             'pre-evaluation hypothesis commitment differs')
    raw, proof = result.get('response'), commit.get('proof')
    cost = _returned_usd_subtotal(result)
    _require(type(raw) is str and result.get('response_sha256') == _digest(raw)
             and result.get('response_truncated') is False and result.get('finish_reason') == 'stop'
             and result.get('refusal') in (None, '')
             and type(cost) in (int, float) and math.isfinite(cost) and cost >= 0
             and cost == record['generation_cost'],
             'complete original generation response required')
    builder = builder_for_profile(PROFILE)
    from ..hypothesis.restart_predictions import json_schema
    wire = builder.response_format()
    wire['json_schema']['schema']['properties']['hypothesis']['required'].append('restart_predictions')
    wire['json_schema']['schema']['properties']['hypothesis']['properties']['restart_predictions'] = json_schema()
    wire['json_schema']['name'] += '_restart_predictions_v1'
    _require(canonical_json(request.get('response_format')) == canonical_json(wire),
             'original generation response schema differs')
    _require(type(proof) is dict and proof.get('parent_source_sha256') in known_sources,
             'generation parent source has no preceding authenticated origin')
    parent_source = known_sources[proof['parent_source_sha256']]
    expected = builder.apply(parent_source, proof['target_body'])['proof']
    _require(proof == expected and commit.get('source_origin') == 'deterministic_body_compiler'
             and commit.get('semantic_or_causal_claim') is False, 'generation source proof differs')
    from ..hypothesis.circle_restart_body import parse_hypothesis_response
    parsed = parse_hypothesis_response(raw)
    binding = record.get('restart_prediction_binding')
    _require(parsed is not None and type(binding) is dict
             and parsed.to_dict() == record['hypothesis']
             and json.loads(raw)['body'] == proof['target_body'] == record['controlled_change']['target_body']
             and binding['response_sha256'] == _digest(raw)
             and binding['spec_sha256'] == canonical_sha(parsed.restart_predictions)
             and proof['child_source_sha256'] == binding['source_sha256'] == record['source_sha256']
             and record['proposed_sha256'] == binding['proposed_sha256'] == record['source_sha256']
             and proof['parent_source_sha256'] == record['parent_source_sha256']
             == request['parent_source_sha256'], 'generation forecast/source binding differs')
    _require(record['parent_id'] in known_parent_ids.get(record['parent_source_sha256'], set()),
             'historical parent ID has no preceding authenticated child identity')
    _require(record['source_sha256'] not in known_sources, 'repeated generation source')
    from .db import normalized_hash
    _require(normalized_hash(builder.source_for(proof['target_body']))
             not in {normalized_hash(source) for source in known_sources.values()},
             'normalized historical generation repeats a consumed source')
    return dict(record_sha256=canonical_sha(record), record=record,
                raw_result=copy.deepcopy(result), proof=copy.deepcopy(proof)), attempt['event_seq'], outcome['event_seq']


def _returned_usd_subtotal(result):
    """Accept explicit unpriced completion without inventing a dollar charge."""
    from .llm.client import response_accounting
    try:
        return response_accounting(SimpleNamespace(
            cost=result.get('cost'), cost_provenance=result.get('cost_provenance'),
            finish_reason=result.get('finish_reason'), response_error=result.get('response_error'))
        )['known_usd']
    except ValueError:
        return None


def _auxiliary(rows, event, kind, reference):
    """Find the request that actually produced this particular terminal event."""
    before = [r for r in rows if r['event_seq'] < event['event_seq']]
    results = [r for r in before if r.get('event') == 'research' and r.get('phase') == 'aux_result'
               and r.get('kind') == kind and r.get('reference') == reference]
    _require(bool(results), 'auxiliary terminal event has no result')
    result = results[-1]
    index = result.get('call_index')
    _require(type(index) is int and index > 0, 'invalid auxiliary call index')
    start = _one(rows, event='research', phase='aux_start', kind=kind,
                 reference=reference, call_index=index)
    _require(start['event_seq'] < result['event_seq'] < event['event_seq']
             and len([r for r in rows if r.get('phase') == 'aux_result' and r.get('call_index') == index]) == 1,
             'auxiliary request/result identity differs')
    _require(not any(r.get('phase') in ('aux_stopped', kind)
                     and result['event_seq'] < r['event_seq'] < event['event_seq'] for r in rows),
             'auxiliary result already terminated')
    raw = result.get('response')
    cost = _returned_usd_subtotal(result)
    _require(type(raw) is str and result.get('response_sha256') == _digest(raw)
             and result.get('response_truncated') is False
             and result.get('structured_response_usable') is True
             and result.get('finish_reason') == 'stop' and result.get('refusal') in (None, '')
             and type(cost) in (int, float) and math.isfinite(cost) and cost >= 0,
             'complete usable auxiliary response required')
    envelope = start.get('response_format')
    _require(type(envelope) is dict and envelope.get('type') == 'json_schema'
             and type(envelope.get('json_schema')) is dict, 'original auxiliary schema required')
    wire = envelope['json_schema']
    expected_names = {'sciel_' + kind + '_v1', 'sciel_' + kind + '_v2'}
    if kind == 'reflection':
        expected_names.update(('sciel_reflection_v3', 'sciel_reflection_v4'))
    _require(wire.get('name') in expected_names and wire.get('strict') is True
             and type(wire.get('schema')) is dict, 'unsupported historical auxiliary schema')
    shape = wire['schema']
    fields = shape.get('properties')
    _require(type(fields) is dict and shape.get('type') == 'object'
             and shape.get('additionalProperties') is False
             and type(shape.get('required')) is list and set(shape['required']) == set(fields),
             'original auxiliary schema shape differs')
    auxiliary_identity = start.get('auxiliary_schema')
    _require(type(auxiliary_identity) is dict
             and auxiliary_identity.get('response_format_sha256', {}).get(kind) == canonical_sha(envelope),
             'original auxiliary schema digest differs')
    from ..hypothesis.auxiliary_schema import response_format
    supported = {canonical_sha(response_format(kind, comparison_reviews=mode)) for mode in (False, True)}
    if kind == 'reflection':
        supported.update(canonical_sha(response_format(kind, comparison_reviews=True,
            comparison_review_version=version)) for version in (2, 3))
    if kind == 'conclusion':
        # Exact archived C/1 and pre-conclusion-source C/2 envelopes. The
        # latter differs from today's C/2 only in the allowed source-field enum.
        # A changed old wire stays unsupported even with a recalculated SHA.
        supported.update(('347b7f61c33b473b13a6dadbf6f73c201600b0583b5d19fffb24830ce6cdb94f',
                          '7275dd52cbf86be666d945fad37e29e89092c8cb784b226fdfeb036f9f945eee'))
    _require(canonical_sha(envelope) in supported, 'unsupported complete historical auxiliary envelope')
    if kind == 'conclusion':
        _require(set(fields) == {'verdict', 'explanation', 'lessons', 'mechanism_review'},
                 'unsupported conclusion wire fields')
        typed = wire['name'] == 'sciel_conclusion_v2'
        properties = fields['mechanism_review'].get('items', {}).get('properties', {})
        _require(('evidence_kind' in properties) == typed, 'conclusion wire type differs')
        return start, result, typed
    review_mode = wire['name'] in ('sciel_reflection_v2', 'sciel_reflection_v3', 'sciel_reflection_v4')
    _require(set(fields) == ({'lessons', 'retire_ids', 'comparison_reviews'} if review_mode
                            else {'lessons', 'retire_ids'}), 'unsupported reflection wire fields')
    return start, result, review_mode


def _prune(kb, conclusions, reviews, *, step_end=False):
    ordinary_ids = {r['id'] for r in kb.experiments}
    groups = {ident for ident, value in kb._evidence().items() if value.get('schema') == PAIRED_SCHEMA}
    if step_end:
        for ident in list(conclusions):
            if ident not in ordinary_ids:
                del conclusions[ident]
    for ident in list(reviews):
        if ident not in groups:
            del reviews[ident]


def _validate(capsule, *, config, metric):
    """Return authenticated detached objects; caller owns digest and publication."""
    _review_text_policy(config)
    from . import external_research as legacy
    legacy._bounded(capsule)
    _require(type(capsule) is dict and set(capsule) == {'schema', 'profile', 'metric', 'root_parent',
        'policy', 'journals', 'comparison_catalog', 'operations', 'terminal'} and capsule['schema'] == SCHEMA,
        'capsule schema differs')
    _require(capsule['profile'] == PROFILE and capsule['metric'] == metric
             and capsule['policy'] == policy_identity(), 'profile, metric or reducer policy differs')
    root = capsule['root_parent']
    builder = builder_for_profile(PROFILE)
    _require(type(root) is dict and set(root) == {'id', 'source_sha256'}
             and type(root['id']) is str and 0 < len(root['id']) <= 128
             and not any(c.isspace() or ord(c) < 32 for c in root['id'])
             and root['source_sha256'] == _digest(builder.source_for()), 'declared root parent differs')
    blocks = capsule['journals']
    _require(type(blocks) is list and 0 < len(blocks) <= MAX_JOURNALS, 'journal capacity differs')
    journals, identities = [], set()
    for block in blocks:
        _require(type(block) is dict and set(block) == {'manifest_sha256', 'journal'}
                 and _sha(block['manifest_sha256']), 'journal source block differs')
        rows = legacy._journal(block['journal'])
        for row in rows:
            if row.get('event') == 'research' and row.get('phase') == 'reflection_context':
                _reflection_effects_policy(row, config)
        identity = rows[0]['run_id'], rows[0]['segment_id']
        _require(identity not in identities, 'duplicate journal identity')
        identities.add(identity)
        journals.append(rows)
    operations = capsule['operations']
    catalog = capsule['comparison_catalog']
    _require(type(operations) is list and 0 < len(operations) <= MAX_OPERATIONS,
             'operation capacity differs')
    _require(type(catalog) is dict and len(catalog) <= MAX_GROUPS, 'comparison catalog capacity differs')
    _require(all(type(group) is dict and group.get('id') == ident for ident, group in catalog.items()),
             'comparison catalog identity differs')
    terminal = capsule['terminal']
    _require(type(terminal) is dict and set(terminal) == {'knowledge_sha256', 'conclusions_sha256',
        'comparison_reviews_sha256'} and all(_sha(v) for v in terminal.values()), 'terminal digests differ')
    kb = KnowledgeBase(max_lessons=config['max_lessons'], max_experiments=config['max_experiments'],
                       max_fragments=config['max_fragments'])
    conclusions, reviews, generations, consumed_groups = {}, {}, {}, set()
    known_sources = {root['source_sha256']: builder.source_for()}
    known_parent_ids = {root['source_sha256']: {root['id']}}
    ends, consumed_events = {}, set()
    for operation in operations:
        _require(type(operation) is dict, 'operation must be an object')
        kind = operation.get('kind')
        if kind == 'step_end':
            # complete_step has no dedicated journal event. This explicit
            # declared boundary is constrained by later reflection context and
            # terminal hashes; it is not proof of an external function call.
            _require(set(operation) == {'kind'}, 'step boundary fields differ')
            _prune(kb, conclusions, reviews, step_end=True)
            continue
        if kind == 'comparison':
            _require(set(operation) == {'kind', 'comparison_id'}, 'comparison operation fields differ')
            ident = operation['comparison_id']
            _require(type(ident) is str and ident in catalog and ident not in consumed_groups,
                     'unknown or duplicate comparison operation')
            group = validate_comparison(catalog[ident])
            from .research import ResearchController
            ResearchController._check_comparison_binding(SimpleNamespace(metric=metric), group, kb)
            if group['origin_experiment_id'] is not None:
                _require(group['origin_experiment_id'] in generations, 'comparison origin has no source provenance')
            kb.record_comparison(group)
            consumed_groups.add(ident)
            _prune(kb, conclusions, reviews)
            continue
        expected_fields = ({'kind', 'journal', 'experiment_id'} if kind == 'generation'
                           else {'kind', 'journal', 'event_seq', 'interval'} if kind == 'reflection'
                           else {'kind', 'journal', 'event_seq'} if kind == 'conclusion' else None)
        _require(expected_fields is not None and set(operation) == expected_fields, 'operation fields differ')
        index = operation['journal']
        _require(type(index) is int and 0 <= index < len(journals), 'operation journal index differs')
        rows = journals[index]
        if kind == 'generation':
            ident = operation['experiment_id']
            _require(type(ident) is str and ident not in generations, 'duplicate generation ID')
            generation, begin, end = _generation(rows, ident, metric=metric, root_parent=root,
                known_sources=known_sources, known_parent_ids=known_parent_ids, kb=kb)
            _require(begin > ends.get(index, 0), 'generation operation is out of journal order')
            kb.record(generation['record'])
            generations[ident] = generation
            known_sources[generation['record']['source_sha256']] = builder.source_for(generation['proof']['target_body'])
            child_id = generation['record']['child_id']
            known_parent_ids[generation['record']['source_sha256']] = {child_id} if child_id is not None else set()
        else:
            end = operation['event_seq']
            _require(type(end) is int and 1 <= end <= len(rows), 'invalid endpoint event sequence')
            event = rows[end - 1]
            _require(event.get('event') == 'research' and event.get('phase') == kind,
                     'operation endpoint kind differs')
            # Check the endpoint before detached admission, then check the complete
            # request span returned by the dedicated validator as well.
            _require(end > ends.get(index, 0), 'auxiliary operation is out of journal order')
            if kind == 'conclusion':
                begin = _conclusion(kb, conclusions, rows, event, config=config)
            else:
                begin = _reflection(kb, reviews, rows, event, interval=operation['interval'], config=config,
                                    conclusions=conclusions)
            _require(begin > ends.get(index, 0), 'auxiliary operation overlaps previous operation')
        _require((index, end) not in consumed_events, 'duplicate operation endpoint')
        consumed_events.add((index, end))
        ends[index] = end
        # ResearchController.resolve retires reviews immediately after recording
        # an outcome, before that step can conclude or reflect. Reconstruct the
        # same boundary even when an old lesson still retains the observation.
        _prune(kb, conclusions, reviews, step_end=kind in ('generation', 'reflection'))
    _require(consumed_groups == set(catalog), 'unconsumed comparison catalog entries')
    for index, rows in enumerate(journals):
        for event in rows:
            if event.get('event') != 'research':
                continue
            phase = event.get('phase')
            required = (phase == 'outcome' or
                (phase == 'conclusion' and event.get('status') == 'parsed') or
                (phase == 'reflection' and event.get('status') == 'parsed' and event.get('error') is None))
            _require(not required or (index, event['event_seq']) in consumed_events,
                     'successful observation or auxiliary transition omitted')
    _require(bool(generations), 'at least one authenticated generation required')
    # The v2 terminal describes a completed history, after complete_step has
    # pruned ordinary reviews. Intermediate C-only returns require step_end in
    # the declared operation stream when their pruning can affect a later R.
    _prune(kb, conclusions, reviews, step_end=True)
    state = kb.state()
    _require(terminal == dict(knowledge_sha256=canonical_sha(state),
        conclusions_sha256=canonical_sha(conclusions), comparison_reviews_sha256=canonical_sha(reviews)),
        'terminal reconstructed history differs')
    # Run the public loader once more on the exact derived state. This also
    # checks cross-lesson authoring catalogs and retained snapshot identities.
    kb.load_state(state)
    imported_ids = list(state['seen_experiments'])
    floor = max(int(ident[1:ident.index('a')]) for ident in imported_ids)
    _require(0 < floor < 2**31, 'historical experiment floor out of bounds')
    return dict(knowledge=kb, conclusions=copy.deepcopy(conclusions), comparison_reviews=copy.deepcopy(reviews),
                generation=copy.deepcopy(generations), imported_ids=imported_ids, floor=floor,
                root_parent=copy.deepcopy(root))


def validate(capsule, *, config, metric):
    """Normalize malformed-input errors without publishing partial detached state."""
    try:
        return _validate(capsule, config=config, metric=metric)
    except (KeyError, TypeError, IndexError, OverflowError, AttributeError) as exc:
        raise ValueError('external research v2: malformed capsule or unsupported journal fields') from exc

from .external_research_reflection import (_conclusion, _reflection, _reflection_version)
