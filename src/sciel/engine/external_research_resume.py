"""Authenticate retained text writers against foreign or local admission events.

The foreign reducer owns the initial chronology. This check binds subsequent
writer text and review replacements to the fresh run's anchored journal; it is
not a claim that model interpretations are true or causally established.
"""
from __future__ import annotations

import copy

from ..hypothesis.comparison_review import build_cards
from ..hypothesis.parsing import decode_conclusion, parse_reflection
from ..knowledge_base.catalogs import canonical_sha
from .external_research_v2 import (
    _auxiliary, _digest, _require, _reflection_version, _review_text_policy,
    _reflection_effects_policy, validate,
)


def transition_binding(controller):
    """Small host receipt after every operation that can mutate lesson state."""
    return {'schema': 'sciel-external-research-transition/1',
            'lessons_sha256': canonical_sha(controller.knowledge.lessons),
            'next_lesson_id': controller.knowledge._next_lesson_id,
            'comparison_reviews': {ident: canonical_sha(card)
                                   for ident, card in controller._retained_comparison_reviews().items()}}


def _owners(lessons):
    for lesson in lessons:
        data = lesson.get('authoring')
        _require(type(data) is dict and data.get('prior_writers_unknown') is False,
                 'retained lesson has no authenticated writer')
        yield lesson['id'], data, data['catalog']
        for node in data['catalog'].values():
            if node['kind'] == 'retired_lesson':
                _require(node['prior_writers_unknown'] is False, 'retired lesson writer is unknown')
                yield node['lesson_id'], node, data['catalog']


def assert_authoring_journal(controller, rows):
    _review_text_policy(controller.config)
    for row in rows:
        if row.get('event') == 'research' and row.get('phase') == 'reflection_context':
            _reflection_effects_policy(row, controller.config)
    registry = controller._external_history
    original = validate(registry['capsule'], config=controller.config, metric=controller.metric)
    foreign = {}
    for lesson_id, owner, catalog in _owners(original['knowledge'].lessons):
        for ref in owner['writer_refs']:
            foreign[(lesson_id, ref)] = catalog[ref]

    # A resumed segment resets event_seq. Physical anchored order is authoritative
    # here; keep original rows intact and normalize only detached lookup copies.
    local = [dict(copy.deepcopy(row), event_seq=index) for index, row in enumerate(rows, 1)]
    transitions = [row for row in local if row.get('event') == 'research'
                   and row.get('phase') in ('outcome', 'conclusion', 'reflection')]
    expected_transition = {
        'schema': 'sciel-external-research-transition/1',
        'lessons_sha256': canonical_sha(original['knowledge'].lessons),
        'next_lesson_id': original['knowledge']._next_lesson_id,
        'comparison_reviews': {ident: canonical_sha(card)
                               for ident, card in original['comparison_reviews'].items()}}
    for event in transitions:
        saved = event.get('external_history_transition')
        _require(type(saved) is dict and set(saved) == set(expected_transition)
                 and saved.get('schema') == expected_transition['schema']
                 and type(saved.get('comparison_reviews')) is dict,
                 'local lesson transition receipt missing')
        expected_transition = copy.deepcopy(saved)
    retained_groups = {group['id'] for group in controller._review_groups()}
    expected_transition['comparison_reviews'] = {
        ident: digest for ident, digest in expected_transition['comparison_reviews'].items()
        if ident in retained_groups}
    _require(transition_binding(controller) == expected_transition,
             'current lessons or reviews differ from the last durable transition')
    admitted = {}
    local_reviews = {}
    groups = {group['id']: group for group in controller._review_groups()}
    for event in local:
        if event.get('event') != 'research' or event.get('status') != 'parsed':
            continue
        phase = event.get('phase')
        if phase == 'conclusion':
            if event.get('lesson_error') is not None or 'lesson_admission' in event:
                continue
            reference = event.get('experiment_id')
            _, receipt, typed = _auxiliary(local, event, phase, reference)
            parsed = decode_conclusion(receipt['response'], require_mechanism=True,
                                       require_typed=typed)['conclusion']
            _require(parsed is not None and event.get('llm') == parsed.to_dict(),
                     'local conclusion differs from admitted response')
            lesson_ids = event.get('accepted_lesson_ids')
        elif phase == 'reflection' and event.get('error') is None:
            reference = event.get('through')
            start, receipt, review_mode = _auxiliary(local, event, phase, reference)
            review_version = _reflection_version(start)
            # The request contract remains part of history even when its reply
            # was empty or its authored card has since been evicted. Checking
            # only the current cards lets a newer local R bypass a downgrade.
            _require(not review_mode or controller.config.get('comparison_reviews') is True,
                     'comparison review history requires enabled receiving policy')
            receiving_version = controller.config.get('comparison_review_version', 1)
            _require(type(receiving_version) is int and receiving_version in (1, 2, 3),
                     'unsupported receiving comparison review version')
            _require(receiving_version >= review_version,
                     'grounded review history requires receiving version ' + str(review_version))
            parsed = parse_reflection(receipt['response'], comparison_reviews=review_mode,
                comparison_review_version=review_version)
            _require(parsed is not None and type(event.get('applied')) is dict,
                     'local reflection admission is unavailable')
            if 'comparison_review_text_policy' in controller.config:
                from ..hypothesis.review_eligibility import require_review_only
                require_review_only(parsed)
            lesson_ids = event['applied'].get('lesson_ids')
            contexts = [row for row in local if row.get('phase') == 'reflection_context'
                        and row.get('through') == reference and row['event_seq'] < start['event_seq']]
            _require(bool(contexts), 'local reflection context is unavailable')
            context = contexts[-1]
            _require(set(parsed.retire_ids) <= set(context.get('lesson_ids', []))
                     and event['applied'].get('retired_ids') == sorted(parsed.retire_ids),
                     'local retirement admission differs')
            if review_mode:
                selected = context.get('comparison_review_source_ids', [])
                _require([p['comparison_id'] for p in parsed.comparison_reviews]
                         == event.get('comparison_review_ids'), 'local comparison review admission differs')
                for proposal in parsed.comparison_reviews:
                    ident = proposal['comparison_id']
                    _require(ident in selected, 'local review source was not presented')
                    # A pruned group cannot keep a card alive. Only current cards
                    # need reconstruction, using their authoritative full group.
                    if ident in groups:
                        retained_card = controller._comparison_reviews.get(ident)
                        writer = {'phase': phase, 'reference': reference,
                                  'raw_response_sha256': _digest(receipt['response'])}
                        # Only the final retained author must reproduce this
                        # card. An older overwritten/evicted author may have
                        # owned a different origin snapshot, now legitimately
                        # absent; rebuilding it from today's card is invalid.
                        if retained_card is None or retained_card.get('writer') != writer:
                            continue
                        # A retained card owns its normalized origin snapshot.
                        # Recheck the lifetime digest before using a now-aged
                        # origin to reconstruct the exact original admission.
                        records = controller.knowledge._evidence()
                        origin = retained_card.get('origin_snapshot') if review_version >= 2 else None
                        if review_version >= 2:
                            records = {}  # Preserve admission-time unavailability.
                        if origin is not None:
                            _require(controller.knowledge._seen.get(origin.get('id')) == canonical_sha(origin),
                                     'local review origin differs from lifetime observation')
                            records[origin['id']] = origin
                        cards = build_cards([proposal], [groups[ident]], writer, version=review_version,
                            records=records, seen=controller.knowledge._seen)
                        local_reviews.update(cards)
        else:
            continue
        _require(type(lesson_ids) is list, 'local lesson admission IDs missing')
        writer = (phase, reference, _digest(receipt['response']))
        for index, proposal in enumerate(parsed.lessons):
            key = (*writer, index)
            _require(key not in admitted, 'ambiguous local writer identity')
            admitted[key] = (proposal['rule'], set(lesson_ids))


    for lesson_id, owner, catalog in _owners(controller.knowledge.lessons):
        for ref in owner['writer_refs']:
            node = catalog[ref]
            if foreign.get((lesson_id, ref)) == node:
                continue
            key = (node['phase'], node['reference'], node['raw_response_sha256'], node['proposal_index'])
            proposal = admitted.get(key)
            _require(proposal is not None and proposal[0] == node['raw_rule'] and lesson_id in proposal[1],
                     'retained lesson writer has no local or foreign admission')
    for ident, card in controller._comparison_reviews.items():
        if card == original['comparison_reviews'].get(ident):
            continue
        _require(card == local_reviews.get(ident), 'comparison review has no local admitted response')
