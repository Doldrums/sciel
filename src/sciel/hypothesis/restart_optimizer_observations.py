"""Source-bound optimizer outcomes and independently checked result geometry.

This host-side projection never executes a candidate or certifies its result.
The original /1 observation is reused internally without rewriting a receipt.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from . import restart_observations as base
from . import restart_optimizer_trace as codec
from ..eval import packing_geometry as geometry

SCHEMA = 'sciel-restart-observation/2'
STATE_COLUMNS = ['success', 'success_reason', 'status', 'status_reason',
                 'result_tolerance_valid', 'result_geometry_reason']
CASE_COLUMNS = base.CASE_COLUMNS + ['optimizer_state_ref']
SCOPE = (base.SCOPE + ' Optimizer success is the observed truth value used by the scaffold, '
         'not a feasibility or admission certificate. Integer status is recorded independently. '
         'Result validity is recomputed from captured bytes using the declared repeat-geometry '
         'policy; it is not an exact certificate or the native incumbent decision.')


def geometry_policy():
    return dict(schema='sciel-result-geometry-policy/1', tolerance=geometry.TOLERANCE,
        validator_sha256=hashlib.sha256(Path(geometry.__file__).read_bytes()).hexdigest(),
        rule='26 circles; finite real shape78; nonnegative radii; wall and hypot-pair slack >= -tolerance',
        scope='Host-derived float validity only. Native NumPy rounding and exact Fraction certification are separate.')


def _seal(value):
    value = copy.deepcopy(value)
    value['observation_sha256'] = base._hash({k: v for k, v in value.items() if k != 'observation_sha256'})
    return base._copy(value, base.MAX_OBSERVATION_BYTES)


def _base_projection(value):
    """Internal shape-check projection; never publish it as a legacy observation."""
    result = copy.deepcopy(value)
    result.update(schema=base.SCHEMA, trace_schema=codec.LEGACY_RESULT_SCHEMA, scope=base.SCOPE)
    result.pop('optimizer_state_catalog')
    result.pop('result_geometry_policy')
    result['columns'] = base._columns()
    result['cases'] = [row[:-1] for row in result['cases']]
    return _seal(result)


def _result_validity(trace, call):
    descriptor = call['optimizer_result']
    if descriptor['ref'] is None:
        return None, descriptor['reason']
    vector = trace['vectors'][descriptor['ref']]
    if vector['shape'] != [78]:
        return False, 'result_shape_not_78'
    if vector['dtype'][1] not in 'fiu':
        return False, 'result_dtype_not_real'
    values = list(base._values(vector))
    result = geometry.validate_packing(dict(
        centers=[values[2*i:2*i+2] for i in range(26)], radii=values[52:]))
    return result['valid'], result['error']


def build_observation(receipt, *, source_sha256, seed, adapter_identity, scaffold_sha256=None):
    original = base._copy(receipt)
    adapter_identity = base._copy(adapter_identity)
    base._require(type(original) is dict and type(adapter_identity) is dict and
        type(adapter_identity.get('runtime')) is dict, 'optimizer receipt/adapter shape')
    diagnostic = original.get('restart_trace')
    base._require(type(diagnostic) is dict and diagnostic.get('status') == 'captured',
                  'optimizer observation requires a captured trace')
    # Validate the new envelope before applying any internal legacy projection.
    trace = codec.validate_trace(diagnostic.get('trace'), source_sha256=source_sha256,
        seed=seed, runtime_sha256=base._hash(adapter_identity['runtime']), scaffold_sha256=scaffold_sha256)
    base._require(trace['schema'] == codec.TRACE_SCHEMA, 'optimizer trace schema')
    projected = copy.deepcopy(original)
    projected['restart_trace']['trace'] = codec.legacy_projection(trace)
    result = base.build_observation(projected, source_sha256=source_sha256, seed=seed,
        adapter_identity=adapter_identity, scaffold_sha256=scaffold_sha256)
    result.update(schema=SCHEMA, receipt_sha256=base._hash(original), trace_schema=trace['schema'],
        trace_sha256=trace['trace_sha256'], scope=SCOPE,
        optimizer_state_catalog=[], result_geometry_policy=geometry_policy())
    result['columns'].update(cases=CASE_COLUMNS, optimizer_state=STATE_COLUMNS)
    if result['status'] == 'available':
        for row, call in zip(result['cases'], trace['calls'], strict=True):
            outcome = call['optimizer_outcome']
            valid, reason = _result_validity(trace, call)
            state = [outcome['success']['value'], outcome['success']['reason'],
                     outcome['status']['value'], outcome['status']['reason'], valid, reason]
            if state not in result['optimizer_state_catalog']:
                result['optimizer_state_catalog'].append(state)
            row.append(result['optimizer_state_catalog'].index(state))
    return validate_shape(_seal(result))


def validate_shape(value):
    """Presentation integrity only; full import still reconstructs from receipt."""
    value = base._copy(value, base.MAX_OBSERVATION_BYTES)
    base._require(type(value) is dict and set(value) == base._FIELDS | {
        'optimizer_state_catalog', 'result_geometry_policy'} and value['schema'] == SCHEMA,
        'optimizer observation fields')
    base._require(value['trace_schema'] == codec.TRACE_SCHEMA and value['scope'] == SCOPE,
        'optimizer observation trace/scope')
    base._require(value['observation_sha256'] == base._hash({
        k: v for k, v in value.items() if k != 'observation_sha256'}), 'optimizer observation digest')
    columns = base._columns(); columns.update(cases=CASE_COLUMNS, optimizer_state=STATE_COLUMNS)
    base._require(value['columns'] == columns and value['result_geometry_policy'] == geometry_policy(),
        'optimizer observation policy/layout')
    catalog = value['optimizer_state_catalog']
    base._require(type(catalog) is list and len(catalog) <= 16 and
        all(type(row) is list and len(row) == len(STATE_COLUMNS) for row in catalog), 'optimizer state catalog')
    base._require(len({base._canonical(row) for row in catalog}) == len(catalog), 'duplicate optimizer state')
    for state in catalog:
        success, success_reason, status, status_reason, valid, reason = state
        base._require((type(success) is bool and success_reason is None) or
            (success is None and success_reason in ('not_returned', 'not_observed')), 'optimizer success observation')
        base._require((type(status) is int and -(2**63) <= status < 2**63 and status_reason is None) or
            (status is None and status_reason in ('not_returned', 'not_observed', 'missing',
                                                  'unsupported_type', 'read_error')), 'optimizer status observation')
        base._require(valid is None or type(valid) is bool, 'result validity type')
        base._require((valid is True and reason is None) or
            (valid is not True and type(reason) is str and 0 < len(reason) <= 300), 'result validity reason')
    base._require(type(value['cases']) is list and all(type(row) is list and len(row) == len(CASE_COLUMNS)
        and type(row[-1]) is int and 0 <= row[-1] < len(catalog) for row in value['cases']), 'optimizer case reference')
    old = _base_projection(value)
    base._compact(old)
    seen = set()
    for row, case in zip(value['cases'], base.observation_cases(old), strict=True):
        if row[-1] not in seen:
            base._require(row[-1] == len(seen), 'optimizer state first-use order')
            seen.add(row[-1])
        success, sr, status, tr, valid, reason = catalog[row[-1]]
        if not case['optimizer_returned']:
            base._require(success is None and sr == 'not_returned' and status is None and tr == 'not_returned'
                and valid is None and reason == 'not_applicable', 'outcome without optimizer return')
        else:
            base._require(sr != 'not_returned' and tr != 'not_returned', 'returned outcome marked not returned')
            if success is None:
                base._require(sr == tr == 'not_observed' and status is None and case['terminal'] != 'completed',
                    'unobserved outcome must be an interrupted return')
            else:
                base._require(tr != 'not_observed', 'observed success with unobserved status')
        descriptor = case['snapshots']['optimizer_result']
        if descriptor['state'] != 'available':
            base._require(valid is None and reason == descriptor['reason'], 'unknown result geometry')
        else:
            base._require(type(valid) is bool, 'available result validity missing')
            if descriptor['shape'] != [78] or descriptor['finite'] is False or descriptor['dtype'][1] not in 'fiu':
                base._require(valid is False, 'invalid result metadata marked feasible')
    base._require(len(seen) == len(catalog), 'unused optimizer state')
    base._require(value['status'] != 'unavailable' or not catalog, 'unavailable observation has optimizer outcomes')
    return value


def observation_cases(value):
    value = validate_shape(value)
    cases = base.observation_cases(_base_projection(value))
    for case, row in zip(cases, value['cases'], strict=True):
        case['optimizer_state_ref'] = row[-1]
        case.update(zip(STATE_COLUMNS, value['optimizer_state_catalog'][row[-1]]))
    return cases
