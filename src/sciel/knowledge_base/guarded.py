"""One source-diverse candidate, protected by whole-card/fact/lexical invariants.

The helper is stateless and has no knowledge-base mutation or exposure-credit
side effect. Only its final admitted lessons may be credited by the caller. A
passing guard preserves specified facts and lexical counts, not model quality,
causal validity or all interpretations in discarded positive-only cards.
"""
from __future__ import annotations

import copy
import hashlib
import re

from .catalogs import canonical_json, canonical_sha
from .presentation import (MAX_CARD_CHARS, MAX_MEMORY_CHARS, observation_card,
                           decision_card, DECISION_CARD_SCHEMA)
from .retrieval import SourceDiverseSelection, selection_identity


def _text_sha(text):
    return hashlib.sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest()


def guarded_admission(ranked, context, retrieve_k, render,
                      max_card_chars=MAX_CARD_CHARS, max_memory_chars=MAX_MEMORY_CHARS,
                      *, compact_decision=False):
    """Return exact final admission plus explicitly provisional decision diagnostics.

    ``ranked`` is the already eligible, ordered output of KB.retrieve. ``render``
    is the current pure whole-card renderer: render(lesson) -> final prompt text
    for observation_card(lesson), or decision_card(lesson) in compact mode.
    Its bytes are cached lazily once per lesson. Compact coverage counts only
    the representative source rows shown, not omitted facts behind references.
    The fact projection is obtained directly, never by interpreting arbitrary
    text or recursively treating reference-shaped task data as instructions.
    Injection-off is caller-owned and must bypass this helper.
    """
    if not isinstance(context, str) or not callable(render):
        raise ValueError('guarded admission requires context text and a whole-card renderer')
    if type(compact_decision) is not bool:
        raise ValueError('compact_decision must be boolean')
    for name, limit in (('retrieve_k', retrieve_k), ('max_card_chars', max_card_chars),
                        ('max_memory_chars', max_memory_chars)):
        if type(limit) is not int or limit < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    lessons = copy.deepcopy(list(ranked))
    selector = SourceDiverseSelection(lessons, 0.5)  # Also validates unique IDs/support sets.
    identity = selection_identity({'lesson_selection': {'mode': 'source_diverse_guarded'},
                                   'compact_lesson_memory': compact_decision})
    if compact_decision:
        identity['decision_presentation'] = DECISION_CARD_SCHEMA
        identity['source_fact_scope'] = 'shown_decision_source_rows_only'
    tokens = set(re.findall(r'\w+', context.casefold()))
    limits = {'retrieve_k': retrieve_k, 'max_card_chars': max_card_chars,
              'max_memory_chars': max_memory_chars}
    input_sha = canonical_sha({'ranked': lessons, 'context': context, 'limits': limits,
                               'selection_identity': identity})
    cache = {}

    def materialize(lesson):
        ident = lesson['id']
        if ident not in cache:
            card = decision_card(lesson) if compact_decision else observation_card(lesson)
            if compact_decision:
                source_ids = [e['id'] for e in lesson['evidence_snapshots']]
                matches = (source_ids == lesson['source_support_ids'] and
                           set(card['source_ids']) <= set(source_ids))
            else:
                matches = card['source_ids'] == lesson['source_support_ids']
            if not matches:
                raise ValueError('guarded card source facts differ from ranked support IDs')
            text = render(copy.deepcopy(lesson))
            if not isinstance(text, str):
                raise ValueError('whole-card renderer must return text')
            cache[ident] = {'card': card, 'text': text, 'chars': len(text),
                            'sha256': _text_sha(text),
                            'lexical': len(tokens & set(re.findall(r'\w+', lesson['rule'].casefold())))}
        return cache[ident]

    def empty():
        return {'lessons': [], 'texts': [], 'presented': [], 'omitted': [], 'used': 0}

    def admit(result, lesson):
        value = materialize(lesson)
        reason = ('card_char_limit' if value['chars'] > max_card_chars else
                  'memory_char_limit' if result['used'] + value['chars'] > max_memory_chars else None)
        if reason:
            result['omitted'].append({'id': lesson['id'], 'reason': reason, 'chars': value['chars']})
            return reason
        result['lessons'].append(lesson)
        result['texts'].append(value['text'])
        result['presented'].append({'id': lesson['id'], 'sha256': value['sha256'], 'chars': value['chars']})
        result['used'] += value['chars']
        return None

    baseline = empty()
    for lesson in lessons:
        if len(baseline['lessons']) >= retrieve_k:
            break
        admit(baseline, lesson)

    candidate = empty()
    while selector.has_remaining and len(candidate['lessons']) < retrieve_k:
        lesson, decision = selector.next()
        reason = admit(candidate, lesson)
        selector.record(decision, admitted=reason is None, reason=reason)

    def fact_set(result):
        return {canonical_json(fact) for lesson in result['lessons']
                for fact in cache[lesson['id']]['card']['source_observations']}

    def source_ids(result):
        return sorted({fact['id'] for lesson in result['lessons']
                       for fact in cache[lesson['id']]['card']['source_observations']})

    def lexical(result):
        return sorted((cache[lesson['id']]['lexical'] for lesson in result['lessons']), reverse=True)

    def view(result, facts):
        return {'ids': [lesson['id'] for lesson in result['lessons']],
                'cards': copy.deepcopy(result['presented']), 'omitted': copy.deepcopy(result['omitted']),
                'rendered_chars': result['used'], 'lexical_vector': lexical(result),
                'source_ids': source_ids(result), 'source_id_count': len(source_ids(result)),
                'source_fact_count': len(facts), 'source_fact_set_sha256': canonical_sha(sorted(facts)),
                'source_fact_sha256': sorted(_text_sha(fact) for fact in facts)}

    protected = []
    for lesson in baseline['lessons']:
        value = cache[lesson['id']]
        card = value['card']
        counts = card['source_counts']
        categories = ('negative', 'zero', 'unmeasured', 'mixed', 'incomplete') if compact_decision else (
            'negative', 'zero', 'unmeasured')
        nonpositive_source = any(counts.get(k, 0) for k in categories)
        any_exposure = card['advice_exposures']['retained_unique_counts']['total'] > 0
        if nonpositive_source or any_exposure:
            protected.append({'id': lesson['id'], 'sha256': value['sha256'],
                              'nonpositive_or_unmeasured_source': nonpositive_source,
                              'any_retained_advice_exposure': any_exposure})

    baseline_facts, candidate_facts = fact_set(baseline), fact_set(candidate)
    blex, dlex = lexical(baseline), lexical(candidate)
    candidate_pairs = {(card['id'], card['sha256']) for card in candidate['presented']}
    checks = {
        'card_count': len(candidate['lessons']) == len(baseline['lessons']),
        'lexical_relevance': len(dlex) == len(blex) and all(d >= b for d, b in zip(dlex, blex)),
        'whole_card_anchors': all((item['id'], item['sha256']) in candidate_pairs for item in protected),
        'source_fact_coverage': baseline_facts <= candidate_facts,
        'strict_source_id_gain': len(source_ids(candidate)) > len(source_ids(baseline)),
    }
    failed = [name for name, passed in checks.items() if not passed]
    final = candidate if not failed else baseline
    candidate_metadata = selector.metadata()
    metadata = {
        **identity, 'ranked_input_sha256': input_sha, 'context_sha256': _text_sha(context), 'limits': limits,
        'baseline': view(baseline, baseline_facts), 'candidate': view(candidate, candidate_facts),
        'final': view(final, candidate_facts if not failed else baseline_facts),
        'protected_cards': protected, 'checks': checks, 'failed_checks': failed,
        'selected': 'diverse' if not failed else 'baseline',
        'candidate_decisions': candidate_metadata.pop('decisions'),
        'candidate_selection': candidate_metadata,
        'candidate_decisions_are_exposure_credit': False,
        'rendered_once_ids': list(cache),
        'limitations': ('Guards preserve specified shown facts, anchored whole cards and a lexical proxy. '
                        'They do not establish semantic equivalence, causal benefit, model adherence or '
                        'zero quality regression; replaced fresh positive-only interpretations can disappear.'),
    }
    return {**copy.deepcopy(final), 'selection': metadata}
