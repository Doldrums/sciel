"""Bounded auxiliary cell observations; never canonical task measurements."""
from copy import deepcopy
import hashlib
import json

from ..research_tools.replay_feedback import validate_feedback
from ..research_tools.decision_projection import validate_decision

SCHEMA = 'sciel-research-cells-context/1'
CARD_SCHEMA = 'sciel-research-check/3'
MAX_CHARS = 12000
MAX_CARDS = 32
MAX_RESERVED_CHARS = 10000
RESERVATION_FRAMING_CHARS = 512


def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _identifier(value, *, optional=False):
    return ((optional and value is None) or
            (type(value) is str and 0 < len(value) <= 256
             and not any(ord(character) < 32 for character in value)))


def _card(value, *, detach=True):
    fields = {'schema', 'cell_id', 'revision_id', 'question', 'source', 'observations',
              'interpretation', 'next_check', 'limitations', 'replay_comparison'}
    if type(value) is not dict:
        return None
    from ..research_tools.comparison_assessment import CARD_SCHEMA as ASSESSED_CARD_SCHEMA, validate as validate_comparison
    if value.get('schema') == ASSESSED_CARD_SCHEMA:
        fields.add('comparison_assessment')
        if (type(value.get('source')) is not dict
                or not validate_comparison(value.get('comparison_assessment'), value['source'].get('receipt_id'))):
            return None
        if 'decision' in value:
            fields.add('decision')
            if not validate_decision(value['decision']):
                return None
    elif value.get('schema') == 'sciel-research-check/4':
        fields.add('decision')
        if not validate_decision(value.get('decision')):
            return None
    elif value.get('schema') != CARD_SCHEMA:
        return None
    if set(value) != fields:
        return None
    if not all(_identifier(value[key]) for key in ('cell_id', 'revision_id')):
        return None
    if not all(type(value[key]) is str for key in ('question', 'interpretation', 'next_check')):
        return None
    if (type(value['limitations']) is not list or len(value['limitations']) > 32
            or any(type(item) is not str for item in value['limitations'])):
        return None
    source = value['source']
    if (type(source) is not dict or set(source) != {'tool_id', 'invocation_id', 'receipt_id', 'candidate_id',
                'target_kind','target_source_sha256','ancestry_parent_id','research_targets_sha256',
                'population_best_id','population_best_source_sha256'}
            or not all(_identifier(item, optional=True) for item in source.values())):
        return None
    if (source['target_kind'] not in ('program','failed_candidate')
            or (source['target_kind'] == 'failed_candidate' and source['candidate_id'] is not None)
            or (source['target_kind'] == 'program' and source['candidate_id'] is None)):
        return None
    if (source['ancestry_parent_id'] is None
            or (source['population_best_id'] is None) != (source['population_best_source_sha256'] is None)):
        return None
    for key in ('target_source_sha256','research_targets_sha256') + (
            ('population_best_source_sha256',) if source['population_best_id'] is not None else ()):
        if (type(source[key]) is not str or len(source[key]) != 64
                or any(c not in '0123456789abcdef' for c in source[key])):
            return None
    comparison = value['replay_comparison']
    if comparison is not None and (not validate_feedback(comparison)
            or comparison['tool_id'] != source['tool_id']
            or comparison['replay_receipt_id'] != source['receipt_id']):
        return None
    observation = value['observations']
    if (type(observation) is not dict or set(observation) != {'status', 'output_excerpt', 'resources'}
            or type(observation['status']) is not str or type(observation['output_excerpt']) is not str
            or type(observation['resources']) is not dict):
        return None
    try:
        _json(value)
    except (TypeError, ValueError, RecursionError):
        return None
    return deepcopy(value) if detach else value


def compact_context(value, *, max_chars=MAX_CHARS):
    """Drop malformed/oversize cards with an explicit count, without aborting G."""
    omitted, reasons = 0, {}
    if type(value) is dict and value.get('schema') == SCHEMA:
        prior = value.get('omitted_count', 0)
        omitted = prior if type(prior) is int and prior >= 0 else 0
        prior_reasons = value.get('omission_reasons', {})
        if type(prior_reasons) is dict:
            reasons = {key: n for key, n in prior_reasons.items()
                       if key in ('invalid_card', 'character_limit', 'card_limit', 'invalid_collection',
                                  'shared_memory_limit', 'research_context_limit', 'unmeasured_interpretation',
                                  'prior_evidence_limit', 'cell_limit')
                       and type(n) is int and n >= 0}
        value = value.get('cards')
    if type(value) is not list:
        return {'schema': SCHEMA, 'cards': [], 'omitted_count': omitted + 1,
                'omission_reasons': {**reasons, 'invalid_collection': reasons.get('invalid_collection', 0) + 1}}
    cards = []
    for raw in value:
        card = _card(raw)
        reason = ('invalid_card' if card is None else 'card_limit' if len(cards) >= MAX_CARDS else None)
        if reason is None:
            trial = {'schema': SCHEMA, 'cards': cards + [card], 'omitted_count': omitted,
                     'omission_reasons': reasons}
            if len(_json(trial)) > min(MAX_CHARS, max_chars):
                reason = 'character_limit'
        if reason:
            omitted += 1
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            cards.append(card)
    result = {'schema': SCHEMA, 'cards': cards, 'omitted_count': omitted, 'omission_reasons': reasons}
    # Omissions themselves consume space; enforce the actual final envelope.
    while result['cards'] and len(_json(result)) > min(MAX_CHARS, max_chars):
        omit_last(result, 'character_limit')
    return result


def card_identity(card):
    return card['cell_id'], card['source']['receipt_id'] or 'terminal_decision'


def merge_context(previous, current, *, max_chars=MAX_CHARS):
    old = compact_context(previous) if previous is not None else compact_context([])
    new = compact_context(current)
    newest = {}
    for card in list(reversed(new['cards'])) + old['cards']:
        # A new interpretation cannot replace a different, already measured
        # receipt. Repeated delivery of one receipt is still one exposure.
        key = card_identity(card)
        if key not in newest:
            newest[key] = card
    # Current episode cards receive memory priority. Separate immutable receipt
    # identities and the latest decision coexist without inventing an invocation.
    result = compact_context(list(newest.values()), max_chars=max_chars)
    for prior in (old, new):
        result['omitted_count'] += prior['omitted_count']
        for key, count in prior['omission_reasons'].items():
            result['omission_reasons'][key] = result['omission_reasons'].get(key, 0) + count
    while result['cards'] and len(_json(result)) > min(MAX_CHARS, max_chars):
        omit_last(result, 'character_limit')
    return result


def context_metadata(context):
    return {'schema': SCHEMA, 'cell_ids': [card['cell_id'] for card in context['cards']],
            'rendered_chars': len(_json(context)),
            'card_sha256': [hashlib.sha256(_json(card).encode()).hexdigest() for card in context['cards']],
            'receipt_ids': [card['source']['receipt_id'] for card in context['cards']
                            if card['source']['receipt_id'] is not None],
            'omitted_count': context['omitted_count'], 'omission_reasons': context['omission_reasons']}


def omit_last(context, reason):
    context['cards'].pop()
    context['omitted_count'] += 1
    context['omission_reasons'][reason] = context['omission_reasons'].get(reason, 0) + 1
