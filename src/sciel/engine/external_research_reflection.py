"""Pure conclusion and reflection reconstruction for external research histories."""
from __future__ import annotations

import copy
from ..hypothesis.conclusion_feedback import validate_conclusions
from ..hypothesis.parsing import decode_conclusion, parse_reflection
from ..hypothesis.rendering import serialize_reflection
from ..knowledge_base.catalogs import canonical_json
from ..knowledge_base.paired_comparisons import SCHEMA as PAIRED_SCHEMA


def _reflection_version(start):
    """The immutable request envelope determines historical proposal semantics."""
    return {'sciel_reflection_v1': 1, 'sciel_reflection_v2': 1,
            'sciel_reflection_v3': 2, 'sciel_reflection_v4': 3}[
                start['response_format']['json_schema']['name']]

def _conclusion(kb, conclusions, rows, event, *, config):
    from .external_research_v2 import (
        _CONCLUSION_FIELDS,
        _auxiliary,
        _digest,
        _require,
    )
    ident = event.get('experiment_id')
    evidence = kb._evidence()
    _require(ident in evidence and evidence[ident].get('schema') != PAIRED_SCHEMA,
             'conclusion source evidence unavailable')
    start, result, typed = _auxiliary(rows, event, 'conclusion', ident)
    decoded = decode_conclusion(result['response'], require_mechanism=True, require_typed=typed)['conclusion']
    _require(decoded is not None, 'selected conclusion did not parse')
    fields = _CONCLUSION_FIELDS | (set(event) & {'lesson_admission', 'source_evidence'})
    review = {k: copy.deepcopy(event[k]) for k in fields}
    _require(review['status'] == 'parsed' and review['llm'] == decoded.to_dict(),
             'accepted conclusion differs from raw reply')
    validate_conclusions({ident: review}, {ident: evidence[ident]},
                         batch_only_lessons=config.get('batch_only_lessons', False))
    if 'lesson_admission' not in review:
        accepted, error, detail = [], None, None
        try:
            accepted = kb.add_lessons(decoded.lessons, allowed_evidence_ids=[ident], writer_context={
                'phase': 'conclusion', 'reference': ident, 'raw_response_sha256': _digest(result['response'])})
        except (ValueError, TypeError, KeyError) as exc:
            error, detail = type(exc).__name__, str(exc)[:1024]
        _require(review['accepted_lesson_ids'] == accepted and review['lesson_error'] == error
                 and review['lesson_error_detail'] == detail, 'conclusion lesson admission differs')
    conclusions[ident] = review
    return start['event_seq']

def _reflection(kb, reviews, rows, event, *, interval, config, conclusions=None):
    from .external_research_v2 import (
        MAX_OPERATIONS,
        _auxiliary,
        _digest,
        _reflection_effects_policy,
        _require,
    )
    through = event.get('through')
    _require(type(through) is int and 0 < through <= len(kb._seen), 'invalid historical reflection watermark')
    _require(type(interval) is int and 0 < interval <= MAX_OPERATIONS, 'invalid reflection interval')
    start, result, review_mode = _auxiliary(rows, event, 'reflection', through)
    review_version = _reflection_version(start)
    _require(not review_mode or config.get('comparison_reviews', False),
             'comparison review history requires enabled receiving policy')
    receiving_version = config.get('comparison_review_version', 1)
    _require(type(receiving_version) is int and receiving_version in (1, 2, 3),
             'unsupported receiving comparison review version')
    _require(receiving_version >= review_version,
             'grounded review history requires receiving version ' + str(review_version))
    contexts = [r for r in rows if r.get('event') == 'research' and r.get('phase') == 'reflection_context'
                and r.get('through') == through and r['event_seq'] < start['event_seq']]
    _require(bool(contexts), 'reflection context event missing')
    context = contexts[-1]
    _require('generated_paired_history' not in context and 'generated_paired_reviews' not in context
             and 'evaluated_detached_study' not in context and 'evaluated_study_review' not in context,
             'v2 history cannot replay an annex-aware reflection')
    review_only_request = _reflection_effects_policy(context, config)
    from .reflection_context import ALLOCATION_POLICY
    if 'allocation_policy' in context or 'selection_epoch' in context:
        _require(context.get('allocation_policy') == ALLOCATION_POLICY
                 and type(context.get('selection_epoch')) is int
                 and 0 <= context['selection_epoch'] < through,
                 'unsupported reflection allocation policy or epoch')
        selection_epoch = context['selection_epoch']
        records, lessons = kb.reflection_context(interval, selection_epoch=selection_epoch)
    else:
        selection_epoch = None
        records, lessons = kb.legacy_reflection_context(interval)
    _require(context.get('experiment_ids') == [r['id'] for r in records]
             and context.get('lesson_ids') == [l['id'] for l in lessons]
             and context.get('omitted_lesson_count') == len(kb.lessons) - len(lessons)
             and context.get('context_sha256') == _digest(serialize_reflection(records, lessons)),
             'reflection context differs from causal knowledge')
    parsed = parse_reflection(result['response'], comparison_reviews=review_mode,
        comparison_review_version=review_version)
    _require(parsed is not None and event.get('status') == 'parsed' and event.get('error') is None
             and event.get('error_detail') is None, 'successful selected reflection required')
    if 'comparison_review_text_policy' in config:
        from ..hypothesis.review_eligibility import require_review_only
        require_review_only(parsed)
    _require(set(parsed.retire_ids) <= {l['id'] for l in lessons if l.get('retirement_eligible') is True},
             'reflection retirement outside eligible presented lessons')
    writer = {'phase': 'reflection', 'reference': through, 'raw_response_sha256': _digest(result['response'])}
    updated = copy.deepcopy(reviews)
    additions = {}
    all_groups = []
    if review_mode:
        from ..hypothesis.comparison_review import build_cards, MAX_STORED
        retained = kb._evidence()
        all_groups = [retained[r['id']] for r in records if r['id'] in retained
                      and retained[r['id']].get('schema') == PAIRED_SCHEMA]
        groups = all_groups[-4:]
        _require(context.get('comparison_review_source_ids') == [g['id'] for g in groups]
                 and context.get('comparison_review_omitted_ids') == [g['id'] for g in all_groups[:-4]],
                 'comparison review selected source differs')
        additions = build_cards(parsed.comparison_reviews, groups, writer, version=review_version,
            records=kb._evidence(), seen=kb._seen)
        for key, card in additions.items():
            updated.pop(key, None)
            updated[key] = card
        updated = dict(list(updated.items())[-MAX_STORED:])
    if conclusions is not None:
        from ..hypothesis.conclusion_feedback import (reflection_review_context, REFLECTION_REVIEW_MAX_CHARS,
            REFLECTION_REVIEW_RULES, REFLECTION_REVIEW_PREFIX)
        from ..hypothesis.rendering import MAX_CONTEXT_CHARS
        from ..hypothesis.prompts import comparison_review_source_block
        ordinary = {key: r for key, r in kb._evidence().items() if r.get('schema') != PAIRED_SCHEMA}
        ordinary.update((r['id'], r) for r in kb.experiments)
        selected_ids = [r['id'] for r in records if r['id'] in ordinary]
        source_chars = len(comparison_review_source_block(all_groups[-4:], version=review_version,
            records=kb._evidence(), seen=kb._seen)) if review_mode else 0
        if review_only_request:
            from .reflection_context import REVIEW_ONLY_INSTRUCTION
            source_chars += len(REVIEW_ONLY_INSTRUCTION)
        allowance = max(0, min(REFLECTION_REVIEW_MAX_CHARS, MAX_CONTEXT_CHARS
            - len(serialize_reflection(records, lessons)) - len(REFLECTION_REVIEW_RULES)
            - len(REFLECTION_REVIEW_PREFIX) - 1 - source_chars))
        ordinary_reviews = reflection_review_context(conclusions, ordinary, selected_ids, max_chars=allowance,
            batch_only_lessons=config.get('batch_only_lessons', False))
        expected_review_metadata = {
            'review_context_sha256': _digest(canonical_json(ordinary_reviews)) if ordinary_reviews is not None else None,
            'review_selected_experiment_ids': selected_ids, 'review_context_max_chars': allowance,
            'review_context_omission': 'no_selected_ordinary' if not selected_ids else
                                      'memory_char_limit' if ordinary_reviews is None else None,
            'review_presented_count': ordinary_reviews['presented_count'] if ordinary_reviews else 0,
            'review_omitted_count': ordinary_reviews['omitted_count'] if ordinary_reviews else len(selected_ids)}
        _require(all(context.get(key) == value for key, value in expected_review_metadata.items()),
                 'reflection ordinary review context differs')
        # The typed historical envelope determines its prompt mode. A saved
        # result/context must also bind the complete request that produced it;
        # an otherwise consistent capsule cannot substitute a different SHA.
        from .reflection_context import prepare_reflection_context
        historical_config = {**config, 'structured_auxiliary': True, 'comparison_reviews': review_mode,
                             'comparison_review_version': review_version}
        # The receiving run may append a G-only annex after this old history.
        # It was not present on the historical wire and must not rewrite it.
        historical_config.pop('external_component_sha256', None)
        historical_config.pop('external_component_review_sha256', None)
        historical_config.pop('external_evaluated_study_sha256', None)
        historical_config.pop('external_evaluated_study_authority_sha256', None)
        historical_config.pop('external_evaluated_review_sha256', None)
        historical_config.pop('source_reflection_config_sha256', None)
        if not review_only_request:
            historical_config.pop('comparison_review_text_policy', None)
        preparation = prepare_reflection_context(kb, conclusions,
            config=historical_config,
            interval=interval, through=through, selection_epoch=selection_epoch)
        _require(start.get('prompt_sha256') == _digest(preparation['prompt']),
                 'reflection complete prompt digest differs')
    before_lessons = kb.lessons
    applied = kb.reflect(parsed.lessons, parsed.retire_ids,
                         allowed_evidence_ids=[r['id'] for r in records], writer_context=writer)
    _require(event.get('applied') == applied and event.get('changed') is (before_lessons != kb.lessons),
             'reflection admitted lesson transition differs')
    if review_mode:
        _require(event.get('comparison_review_ids') == list(additions)
                 and event.get('comparison_reviews_changed') is (updated != reviews),
                 'reflection comparison review transition differs')
    else:
        _require('comparison_review_ids' not in event and 'comparison_reviews_changed' not in event,
                 'legacy reflection contains review mutations')
    reviews.clear()
    reviews.update(updated)
    return context['event_seq']
