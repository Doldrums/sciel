"""Bounded terminal model interpretations, separate from measured observations.

The immutable proposal contains complete text and evidence references. Excerpts
retain exact head/tail text, disclose omissions, and can be authenticated without
executing a model, tool or evaluator.
"""
from copy import deepcopy
import hashlib
import json

from ..methods.models import require_hash

SCHEMA = 'sciel-research-decision/1'
SCOPE = 'Model interpretation of the recorded evidence; not canonical task validity, score or causal benefit.'
TEXT_FIELDS = ('question', 'interpretation', 'next_decision')
NO_MEASUREMENT_QUESTION = 'No new tool invocation; see the terminal decision question.'
_MIDDLE = '\n[Excerpt: middle omitted; complete text is in the saved proposal.]\n'
_OMITTED = '\n[Entire text omitted; complete text is in the saved proposal.]\n'


def proposal_decision(proposal):
    """Read the accepted finish wire fields; never synthesize a conclusion."""
    raw = proposal['response']['text'].strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    action = json.loads(raw)
    if (type(action) is not dict or action.get('action') != 'finish'
            or action.get('conclusion') not in ('supported', 'refuted', 'inconclusive')):
        raise ValueError('terminal decision requires a saved finish proposal')
    values = {'question': action.get('question'), 'interpretation': action.get('reason', ''),
              'next_decision': action.get('next_decision')}
    if any(type(value) is not str for value in values.values()):
        raise ValueError('terminal decision text must be saved strings')
    if not values['question'].strip() or not values['next_decision'].strip():
        raise ValueError('terminal decision lacks question or next decision')
    return {'conclusion': action['conclusion'], **values}


def _excerpt(text, kept):
    if len(text) <= kept:
        return text
    if kept == 0:
        return _OMITTED
    head = kept // 2
    tail = kept - head
    return text[:head] + _MIDDLE + (text[-tail:] if tail else '')


def project_decision(proposal, proposal_id, *, text_limits=(600, 2000, 600), max_receipts=16):
    require_hash(proposal_id, 'decision proposal')
    require_hash(proposal['action_id'], 'decision action')
    raw = proposal_decision(proposal)
    receipts = proposal['evidence_receipt_ids']
    if (type(receipts) is not list or len(receipts) != len(set(receipts))
            or type(max_receipts) is not int or not 0 <= max_receipts <= 16):
        raise ValueError('invalid terminal evidence references')
    for ident in receipts:
        require_hash(ident, 'decision evidence receipt')
    result = {'schema': SCHEMA, 'proposal_id': proposal_id, 'action_id': proposal['action_id'],
        'conclusion': raw['conclusion'], 'scope': SCOPE,
        'receipt_ids': receipts[-max_receipts:] if max_receipts else [],
        'receipts_omitted': max(0, len(receipts) - max_receipts),
        'original_chars': {}, 'omitted_chars': {}, 'original_text_sha256': {}}
    for key, limit in zip(TEXT_FIELDS, text_limits, strict=True):
        if type(limit) is not int or limit < 0:
            raise ValueError('invalid terminal text budget')
        text = raw[key]
        result[key] = _excerpt(text, limit)
        result['original_chars'][key] = len(text)
        result['omitted_chars'][key] = max(0, len(text) - limit)
        result['original_text_sha256'][key] = hashlib.sha256(text.encode()).hexdigest()
    return result


def validate_decision(value):
    fields = {'schema', 'proposal_id', 'action_id', 'conclusion', 'scope', 'receipt_ids',
              'receipts_omitted', 'original_chars', 'omitted_chars', 'original_text_sha256', *TEXT_FIELDS}
    if (type(value) is not dict or set(value) != fields or value['schema'] != SCHEMA
            or value['scope'] != SCOPE or value['conclusion'] not in ('supported', 'refuted', 'inconclusive')):
        return False
    try:
        for key in ('proposal_id', 'action_id'):
            require_hash(value[key], key)
        if (type(value['receipt_ids']) is not list or len(value['receipt_ids']) > 16
                or len(set(value['receipt_ids'])) != len(value['receipt_ids'])
                or type(value['receipts_omitted']) is not int or value['receipts_omitted'] < 0):
            return False
        for ident in value['receipt_ids']:
            require_hash(ident, 'decision receipt')
        for mapping in ('original_chars', 'omitted_chars', 'original_text_sha256'):
            if type(value[mapping]) is not dict or set(value[mapping]) != set(TEXT_FIELDS):
                return False
        for key in TEXT_FIELDS:
            total, omitted = value['original_chars'][key], value['omitted_chars'][key]
            if (type(value[key]) is not str or type(total) is not int or type(omitted) is not int
                    or not 0 <= omitted <= total):
                return False
            require_hash(value['original_text_sha256'][key], 'decision text')
            marker_size = len(_OMITTED if omitted == total else _MIDDLE) if omitted else 0
            if len(value[key]) != total - omitted + marker_size:
                return False
    except (TypeError, ValueError):
        return False
    return True


def authenticate_projection(value, proposal, proposal_id):
    """Reject rewritten, stale or fabricated conclusions and omission metadata."""
    if not validate_decision(value):
        raise ValueError('invalid terminal decision projection')
    limits = tuple(value['original_chars'][key] - value['omitted_chars'][key] for key in TEXT_FIELDS)
    expected = project_decision(proposal, proposal_id, text_limits=limits,
                                max_receipts=len(value['receipt_ids']))
    if value != expected:
        raise ValueError('terminal decision differs from its immutable proposal')
    return deepcopy(value)
