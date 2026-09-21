"""Bounded host-side motion facts from captured bytes, never new executions.

Matching describes coordinates under one explicit floating-point rule. It is
neither global assignment nor physical identity, a basin test, or causal credit.
Geometry validity and incumbent admission remain separate observations.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
import math
import struct

from . import restart_optimizer_trace as codec
from . import restart_trace as legacy

SCHEMA = 'sciel-restart-motion/1'
MAX_BYTES = 8192
COLUMNS = ['ordinal', 'restart_index', 'effective_locations_changed', 'effective_reason',
           'result_center_max', 'result_center_reason', 'match_status', 'match_reason',
           'match_center_max', 'match_radius_max']
POLICY = {
    'schema': 'sciel-restart-motion-policy/1',
    'vectors': 'Captured finite real shape78; decode to binary64; integer values not exactly representable are unavailable. No array coercion or execution.',
    'effective_locations_changed': 'Exact numeric center-pair multiset best versus effective, including multiplicity; signed zeros are equal.',
    'result_center_max': 'Maximum indexwise math.hypot center displacement from effective to optimizer_result.',
    'matching': 'Each result center selects its unique nearest best center by math.hypot; computed ties or a non-bijection are ambiguous. Not global assignment or physical identity.',
    'matched_errors': 'Maximum matched center distance and absolute result-radius minus best-radius at the matched center.',
    'scope': 'Retained calls only. No tolerance threshold, feasibility, basin, causal credit, lesson support or quality claim. Nonfinite arithmetic and unavailable snapshots stay unknown.',
}
_FIELDS = {'schema', 'trace_schema', 'trace_sha256', 'identity', 'prefix',
           'policy', 'columns', 'cases', 'motion_sha256'}


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _hash(value):
    return hashlib.sha256(_wire(value).encode('ascii')).hexdigest()


def _require(condition, reason):
    if not condition:
        raise ValueError('restart motion: ' + reason)


def _copy(value):
    try:
        # Bound depth, types and structure before serializing arbitrary transport.
        value = legacy._rt_clone(value)
        raw = _wire(value)
        _require(len(raw.encode('ascii')) <= MAX_BYTES, 'payload limit')
        return json.loads(raw)
    except (TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError('restart motion: invalid JSON') from exc


def _decode(trace, descriptor, label, cache):
    if descriptor is None:
        return None, label + ':legacy_not_recorded'
    if descriptor['ref'] is None:
        return None, label + ':' + descriptor['reason']
    ref = descriptor['ref']
    if ref not in cache:
        vector = trace['vectors'][ref]
        reason, values = None, None
        if vector['shape'] != [78]:
            reason = 'shape_not_78'
        elif vector['dtype'][1] not in 'fiu':
            reason = 'non_real_dtype'
        else:
            raw, spec = legacy._rt_vector(vector)
            decoded = tuple(item[0] for item in struct.iter_unpack(spec[0], raw))
            values = tuple(float(item) for item in decoded)
            if vector['dtype'][1] in 'iu' and any(int(converted) != original
                                                for original, converted in zip(decoded, values)):
                values, reason = None, 'integer_not_exact_binary64'
            elif not all(math.isfinite(x) for x in values):
                values, reason = None, 'nonfinite_vector'
        cache[ref] = values, reason
    values, reason = cache[ref]
    return values, None if reason is None else label + ':' + reason


def _centers(values):
    return list(zip(values[:52:2], values[1:52:2]))


def _distance(a, b):
    result = math.hypot(a[0] - b[0], a[1] - b[1])
    if not math.isfinite(result):
        raise ArithmeticError('nonfinite_distance')
    return result


def _match(best, result):
    before, after = _centers(best), _centers(result)
    indices, distances = [], []
    try:
        for center in after:
            row = [_distance(center, old) for old in before]
            distance = min(row)
            matches = [j for j, candidate in enumerate(row) if candidate == distance]
            if len(matches) != 1:
                return 'ambiguous', 'nearest_tie', None, None
            indices.append(matches[0]); distances.append(distance)
        if len(set(indices)) != 26:
            return 'ambiguous', 'non_bijective_nearest', None, None
        radius_error = max(abs(result[52+i] - best[52+j]) for i, j in enumerate(indices))
        if not math.isfinite(radius_error):
            return 'unavailable', 'nonfinite_radius_difference', None, None
        return 'matched', None, max(distances), radius_error
    except ArithmeticError:
        return 'unavailable', 'nonfinite_distance', None, None


def build_motion(trace, **bindings):
    """Validate a /1-/3 trace and derive whole retained-prefix motion facts.

    ``bindings`` are the codec's source_sha256, seed, runtime_sha256,
    scaffold_sha256 and invocation_id expectations. Receipt import must supply
    its external identities; a internally valid trace is not authenticated.
    """
    trace = codec.validate_trace(trace, **bindings)
    value = dict(schema=SCHEMA, trace_schema=trace['schema'], trace_sha256=trace['trace_sha256'],
        identity=copy.deepcopy(trace['identity']), prefix=copy.deepcopy(trace['prefix']),
        policy=copy.deepcopy(POLICY), columns=list(COLUMNS), cases=[])
    cache = {}
    for call in trace['calls']:
        best, best_error = _decode(trace, call['inputs']['best'], 'best', cache)
        effective, effective_error = _decode(trace, call['effective']['vector'], 'effective', cache)
        result, result_error = _decode(trace, call.get('optimizer_result'), 'result', cache)
        changed, changed_error = None, best_error or effective_error
        if changed_error is None:
            changed = Counter(_centers(best)) != Counter(_centers(effective))
        movement, movement_error = None, effective_error or result_error
        if movement_error is None:
            try:
                movement = max(_distance(a, b) for a, b in zip(_centers(effective), _centers(result)))
            except ArithmeticError:
                movement_error = 'nonfinite_distance'
        match_error = best_error or result_error
        match = (('unavailable', match_error, None, None) if match_error else _match(best, result))
        value['cases'].append([call['ordinal'], call['restart_index'], changed, changed_error,
                              movement, movement_error, *match])
    value['motion_sha256'] = _hash(value)
    return _validate_shape(value)


def _reason(value):
    return type(value) is str and 0 < len(value) <= 96


def _magnitude(value):
    return type(value) is float and math.isfinite(value) and value >= 0.


def _validate_shape(value):
    """Transport consistency only; use validate_motion for receipt-bound facts."""
    value = _copy(value)
    _require(type(value) is dict and set(value) == _FIELDS and value['schema'] == SCHEMA,
             'schema fields')
    _require(value['motion_sha256'] == _hash({k: v for k, v in value.items() if k != 'motion_sha256'}), 'digest')
    _require(value['trace_schema'] in (*codec.LEGACY_TRACE_SCHEMAS, codec.TRACE_SCHEMA)
             and legacy._rt_hex(value['trace_sha256']), 'trace binding')
    legacy._rt_identity(value['identity'])
    _require(_wire(value['policy']) == _wire(POLICY) and value['columns'] == COLUMNS, 'policy/layout')
    rows, prefix = value['cases'], value['prefix']
    _require(type(rows) is list and len(rows) <= 16, 'case bound')
    legacy._rt_keys(prefix, ('limit', 'captured_count', 'omitted_count', 'first_omitted_ordinal', 'omission_reason'), 'motion prefix')
    _require(type(prefix['limit']) is int and prefix['limit'] == 16
        and type(prefix['captured_count']) is int and prefix['captured_count'] == len(rows)
        and type(prefix['omitted_count']) is int and 0 <= prefix['omitted_count']
        and prefix['omitted_count'] + len(rows) <= 2500,
        'prefix counts')
    if prefix['omitted_count']:
        _require(type(prefix['first_omitted_ordinal']) is int and prefix['first_omitted_ordinal'] == len(rows)
            and prefix['omission_reason'] in ('byte_limit', 'prefix_limit'), 'prefix omission')
        _require(prefix['omission_reason'] != 'prefix_limit' or len(rows) == 16, 'prefix limit omission')
    else:
        _require(prefix['first_omitted_ordinal'] is None and prefix['omission_reason'] is None, 'unexpected omission')
    for ordinal, row in enumerate(rows):
        _require(type(row) is list and len(row) == len(COLUMNS)
            and type(row[0]) is int and row[0] == ordinal
            and type(row[1]) is int and row[1] == ordinal*4, 'case order')
        _, _, changed, changed_reason, movement, movement_reason, status, reason, center, radius = row
        _require((type(changed) is bool and changed_reason is None)
            or (changed is None and _reason(changed_reason)), 'effective multiset availability')
        _require((_magnitude(movement) and movement_reason is None)
            or (movement is None and _reason(movement_reason)), 'movement availability')
        if status == 'matched':
            _require(reason is None and _magnitude(center) and _magnitude(radius), 'matched errors')
        else:
            _require(status in ('ambiguous', 'unavailable') and _reason(reason)
                and center is None and radius is None, 'unknown matching')
            _require(status != 'ambiguous' or reason in ('nearest_tie', 'non_bijective_nearest'), 'ambiguity reason')
    return value


def validate_motion(value, *, trace, **bindings):
    """Recompute against the external trace; self-rehashed claims do not suffice."""
    expected = build_motion(trace, **bindings)
    _require(_wire(_validate_shape(value)) == _wire(expected), 'projection differs from trace')
    return expected


def motion_cases(value):
    """Detached row expansion; hash/shape integrity is not authentication."""
    value = _validate_shape(value)
    return [dict(zip(COLUMNS, row)) for row in value['cases']]
