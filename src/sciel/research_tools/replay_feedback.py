"""Bounded, evidence-bound replay feedback; never changes release policy.

Receipt arguments are authenticated invocation payloads. The projection is
separate from the release proof so old policy checks retain their exact shape.
"""
from copy import deepcopy
import json
import re

from ..methods.models import canonical_bytes, content_hash
from .result_contract import assess_receipt, replay_evidence

SCHEMA = 'sciel-replay-feedback/1'
PINNED = ('method_id', 'input_sha256', 'source_sha256', 'runtime_sha256', 'limits')
REASONS = ('observations_match', 'independent_execution_passed', 'observations_differ',
           'reference_not_successful', 'replay_not_successful', 'binding_mismatch')
FIELDS = {'schema', 'tool_id', 'policy', 'status', 'matched', 'measurement_reproduced',
          'reason', 'reference_receipt_id', 'replay_receipt_id',
          'reference_observations_sha256', 'replay_observations_sha256',
          'binding_mismatches', 'changed_paths', 'changed_paths_omitted'}
_MISSING = object()


def _summary(value):
    if value is _MISSING:
        return '<absent>'
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
    if len(encoded) <= 64:
        return encoded
    kind = type(value).__name__
    size = len(value) if isinstance(value, (dict, list, str)) else len(encoded)
    return f'{kind}[{size}] sha256:{content_hash(value)[:32]}'


def _changes(left, right, path='', depth=0):
    # Lists are atomic: thousands of changed array elements must not expand
    # feedback. Long keys/deep objects are represented at their nearest exact
    # parent JSON pointer, with bounded summaries of the whole subtree.
    if canonical_bytes(left) == canonical_bytes(right):
        return
    if type(left) is dict and type(right) is dict and depth < 8:
        keys = sorted(set(left) | set(right))
        children = [(key, path + '/' + key.replace('~', '~0').replace('/', '~1')) for key in keys]
        if all(len(pointer) <= 128 for _, pointer in children):
            for key, pointer in children:
                if key in left and key in right:
                    yield from _changes(left[key], right[key], pointer, depth + 1)
                else:
                    yield {'path': pointer, 'reference': _summary(left.get(key, _MISSING)),
                           'replay': _summary(right.get(key, _MISSING))}
            return
    yield {'path': path, 'reference': _summary(left), 'replay': _summary(right)}


def build_feedback(reference, replay, *, tool_id, reference_receipt_id, replay_receipt_id):
    """Compare exact saved invocation payloads, without invoking their code."""
    policy = reference['replay_policy']
    if replay['replay_policy'] != policy:
        raise ValueError('replay feedback cannot change the immutable policy')
    before, after = reference['receipt'], replay['receipt']
    if before.get('method_id') != tool_id or after.get('method_id') != tool_id:
        raise ValueError('replay feedback tool differs from its receipts')
    proof = replay_evidence(before, after, policy)
    a, b = assess_receipt(before), assess_receipt(after)
    mismatches = [key for key in PINNED if before.get(key) is None or before.get(key) != after.get(key)]
    if a['status'] != 'success':
        reason = 'reference_not_successful'
    elif b['status'] != 'success':
        reason = 'replay_not_successful'
    elif mismatches:
        reason = 'binding_mismatch'
    elif policy == 'independent_execution':
        reason = 'independent_execution_passed'
    else:
        reason = 'observations_match' if proof['matched'] else 'observations_differ'
    changes, omitted = [], 0
    if a['observations_sha256'] is not None and b['observations_sha256'] is not None:
        for row in _changes(before['output']['observations'], after['output']['observations']):
            if len(changes) < 8:
                changes.append(row)
            else:
                omitted += 1
    result = dict(schema=SCHEMA, tool_id=tool_id, **proof,
        status=('reproduced' if proof['measurement_reproduced'] else 'independently_executed')
            if proof['matched'] else 'not_reproduced', reason=reason,
        reference_receipt_id=reference_receipt_id, replay_receipt_id=replay_receipt_id,
        binding_mismatches=mismatches, changed_paths=changes, changed_paths_omitted=omitted)
    if not validate_feedback(result):
        raise ValueError('invalid replay feedback evidence')
    return result


def validate_feedback(value):
    """Validate presentation shape/consistency; receipt readers bind the facts."""
    def is_hash(item):
        return type(item) is str and re.fullmatch('[0-9a-f]{64}', item) is not None
    if type(value) is not dict or set(value) != FIELDS or value.get('schema') != SCHEMA:
        return False
    if any(not is_hash(value[k]) for k in ('tool_id', 'reference_receipt_id', 'replay_receipt_id')):
        return False
    if any(value[k] is not None and not is_hash(value[k]) for k in
           ('reference_observations_sha256', 'replay_observations_sha256')):
        return False
    if (value['policy'] not in ('exact_observations', 'independent_execution')
            or type(value['matched']) is not bool or type(value['measurement_reproduced']) is not bool
            or value['reason'] not in REASONS):
        return False
    expected_status = ('reproduced' if value['measurement_reproduced'] else 'independently_executed') if value['matched'] else 'not_reproduced'
    if (value['status'] != expected_status
            or value['measurement_reproduced'] != (value['matched'] and value['policy'] == 'exact_observations')
            or value['matched'] != (value['reason'] in ('observations_match', 'independent_execution_passed'))
            or value['reason'] == 'observations_match' and value['policy'] != 'exact_observations'
            or value['reason'] == 'independent_execution_passed' and value['policy'] != 'independent_execution'):
        return False
    fields = value['binding_mismatches']
    rows = value['changed_paths']
    if (type(fields) is not list or any(type(k) is not str or k not in PINNED for k in fields)
            or len(fields) != len(set(fields)) or value['matched'] and fields
            or type(rows) is not list or len(rows) > 8
            or type(value['changed_paths_omitted']) is not int or value['changed_paths_omitted'] < 0):
        return False
    before, after = value['reference_observations_sha256'], value['replay_observations_sha256']
    if (value['matched'] and (before is None or after is None)
            or value['reason'] == 'binding_mismatch' and not fields
            or value['reason'] == 'observations_differ' and
                (value['policy'] != 'exact_observations' or fields or before is None or after is None or before == after)):
        return False
    for row in rows:
        if (type(row) is not dict or set(row) != {'path', 'reference', 'replay'}
                or type(row['path']) is not str or len(row['path']) > 128
                or row['path'] and not row['path'].startswith('/')
                or any(type(row[k]) is not str or len(row[k]) > 80 for k in ('reference', 'replay'))):
            return False
    if value['reason'] == 'observations_match' and (rows or value['changed_paths_omitted']
            or value['reference_observations_sha256'] is None
            or value['reference_observations_sha256'] != value['replay_observations_sha256']):
        return False
    return True


def shorten_feedback(value, count):
    """Drop only whole change summaries, explicitly retaining the omitted count."""
    result = deepcopy(value)
    removed = max(0, len(result['changed_paths']) - count)
    result['changed_paths'] = result['changed_paths'][:count]
    result['changed_paths_omitted'] += removed
    return result


def is_projection(actual, expected):
    if not validate_feedback(actual):
        return False
    return actual == shorten_feedback(expected, len(actual['changed_paths']))
