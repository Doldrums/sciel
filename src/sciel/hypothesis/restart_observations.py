"""Compact observations derived from durable restart receipts, never new evidence.

No candidate, helper, solver or process is invoked. Arrays remain in the receipt.
Import validation reconstructs facts from that receipt; a rehashed projection is
only presentation transport and cannot authenticate or replace its source.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct

from . import restart_trace as codec

SCHEMA = 'sciel-restart-observation/1'
VIEW_SCHEMA = 'sciel-restart-observation-view/1'
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_OBSERVATION_BYTES = 65536
PHASES = ['start', 'helper_finished', 'optimizer_entered', 'optimizer_returned', 'terminal']
INPUTS = ['best', 'initial', 'noise', 'ordinary']
SNAPSHOTS = INPUTS + ['post_' + key for key in INPUTS] + ['raw_return', 'effective', 'optimizer_result']
STATE_COLUMNS = ['raw_return_kind', 'effective_origin', 'helper_finished', 'error', 'terminal',
    'helper_accepted', 'optimizer_entered', 'optimizer_returned', 'input_unchanged']
CASE_COLUMNS = ['ordinal', 'restart_index', 'state_ref', 'phase_quality_refs', 'snapshot_refs',
    'result_values_equal_effective', 'result_values_equal_best']
SCOPE = ('Same-invocation diagnostic observations, not new trials or lesson evidence IDs. '
    'Whole-run origin counts are separate from the first scheduled-call prefix and its omissions. '
    'Acceptance, optimizer return and incumbent gain are distinct; interval gains do not assign causal credit. '
    'An optimizer-result snapshot is recorded before incumbent admission and may be invalid geometry. '
    'Missing/legacy snapshots remain unknown, never unchanged. No polarity, quality superiority, '
    'generalization or knowledge benefit follows from this trace. Hashes bind data, not authenticity.')
_FIELDS = {'schema', 'receipt_sha256', 'bindings', 'receipt_status', 'trace_schema', 'trace_sha256',
    'sidecar_sha256', 'status', 'reason', 'run', 'prefix', 'columns', 'descriptor_catalog', 'phase_catalog', 'case_state_catalog', 'cases',
    'quality_join', 'scope', 'observation_sha256'}
_BINDINGS = {'source_sha256', 'seed', 'adapter_identity_sha256', 'runtime_sha256', 'scaffold_sha256', 'invocation_id'}


def _require(condition, message):
    if not condition:
        raise ValueError('restart observation: ' + message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _copy(value, maximum=MAX_RECEIPT_BYTES):
    def literal(item, depth=0):
        _require(depth <= 40, 'JSON depth limit')
        if item is None or type(item) in (bool, str, int):
            return
        if type(item) is float:
            _require(math.isfinite(item), 'nonfinite JSON')
        elif type(item) is list:
            for entry in item: literal(entry, depth + 1)
        elif type(item) is dict:
            _require(all(type(key) is str for key in item), 'JSON keys')
            for entry in item.values(): literal(entry, depth + 1)
        else:
            raise ValueError('restart observation: literal JSON required')
    try:
        literal(value)
        raw = _canonical(value)
        _require(len(raw.encode('ascii')) <= maximum, 'JSON byte limit')
        return json.loads(raw)
    except (TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError('restart observation: invalid JSON') from exc


def _hash(value):
    return hashlib.sha256(_canonical(value).encode('ascii')).hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _columns():
    return {'cases': list(CASE_COLUMNS), 'case_state': list(STATE_COLUMNS), 'phases': list(PHASES),
            'snapshots': list(SNAPSHOTS), 'input_unchanged': list(INPUTS)}


def _bindings(value):
    _require(type(value) is dict and set(value) == _BINDINGS, 'binding fields')
    _require(all(_sha(value[key]) for key in ('source_sha256', 'adapter_identity_sha256', 'runtime_sha256')) and
        type(value['seed']) is int and 0 <= value['seed'] < 2**32 and
        (value['scaffold_sha256'] is None or _sha(value['scaffold_sha256'])) and
        (value['invocation_id'] is None or (type(value['invocation_id']) is str and
         re.fullmatch('[0-9a-f]{32}', value['invocation_id']) is not None)), 'binding values')


def _values(vector):
    raw, spec = codec._rt_vector(vector)
    return tuple(value[0] for value in struct.iter_unpack(spec[0], raw))


def _descriptor(trace, descriptor, *, legacy=False, post=False):
    result = {'state': 'unavailable', 'reason': None, 'dtype': None, 'shape': None, 'finite': None}
    if legacy:
        return {**result, 'reason': 'legacy_not_recorded'}
    ref, reason = descriptor['ref'], descriptor['reason']
    if ref is None:
        return {**result, 'state': 'not_applicable' if reason == 'not_applicable' else 'unavailable', 'reason': reason}
    if post and ref not in trace['vectors']:
        return {**result, 'state': 'hash_only', 'reason': 'post_values_not_retained'}
    vector = trace['vectors'][ref]
    return {**result, 'state': 'available', 'dtype': vector['dtype'], 'shape': list(vector['shape']),
            'finite': all(math.isfinite(value) for value in _values(vector))}


def _comparable(trace, descriptor):
    if descriptor is None or descriptor['ref'] is None:
        return None
    vector = trace['vectors'][descriptor['ref']]
    if vector['shape'] != [78] or vector['dtype'][1] not in 'fiu':
        return None
    values = _values(vector)
    return values if all(math.isfinite(value) for value in values) else None


def _quality_join(receipt, trace):
    if receipt['status'] != 'valid':
        return None
    raw = receipt.get('raw_sum')
    _require(type(raw) in (int, float) and math.isfinite(raw) and raw >= 0, 'valid receipt raw sum')
    total = trace['final_quality']['sum']
    bound = 0.0 if raw == 0.0 else 32 * math.ulp(raw)
    return {'raw_sum': raw, 'trace_sum': total, 'maximum_roundoff': bound,
            'within_roundoff': abs(total-raw) <= bound, 'terminal_status_compatible': trace['status'] != 'error'}


def build_observation(receipt, *, source_sha256, seed, adapter_identity, scaffold_sha256=None):
    """Derive a compact diagnostic from one receipt and its external draw binding.

    The caller retains/validates the full numeric receipt as ordinary paired
    evidence; this function does not replace geometry or certificate validation.
    Historical trace /1 is readable without claiming a current runtime match.
    """
    if (type(receipt) is dict and type(receipt.get('restart_trace')) is dict and
            type(receipt['restart_trace'].get('trace')) is dict and
            receipt['restart_trace']['trace'].get('schema') == 'sciel-restart-trace/3'):
        from . import restart_optimizer_observations
        return restart_optimizer_observations.build_observation(receipt, source_sha256=source_sha256,
            seed=seed, adapter_identity=adapter_identity, scaffold_sha256=scaffold_sha256)
    _require(_sha(source_sha256) and type(seed) is int and 0 <= seed < 2**32, 'expected source/seed')
    _require(scaffold_sha256 is None or _sha(scaffold_sha256), 'expected scaffold')
    receipt, adapter = _copy(receipt), _copy(adapter_identity)
    _require(type(receipt) is dict and receipt.get('schema') in ('sciel-repeat-draw/1', 'sciel-repeat-coordinator-failure/1'), 'receipt schema')
    _require(type(adapter) is dict and _sha(adapter.get('identity_sha256')) and
        adapter['identity_sha256'] == _hash({k: v for k, v in adapter.items() if k != 'identity_sha256'}) and
        type(adapter.get('runtime')) is dict, 'adapter identity')
    _require(receipt.get('source_sha256') == source_sha256 and type(receipt.get('seed')) is int and
        receipt['seed'] == seed and _canonical(receipt.get('adapter_identity')) == _canonical(adapter), 'receipt external binding differs')
    _require(receipt.get('status') in ('valid', 'candidate_failure', 'infrastructure'), 'receipt status')
    bindings = dict(source_sha256=source_sha256, seed=seed, adapter_identity_sha256=adapter['identity_sha256'],
        runtime_sha256=_hash(adapter['runtime']), scaffold_sha256=scaffold_sha256, invocation_id=None)
    value = dict(schema=SCHEMA, receipt_sha256=_hash(receipt), bindings=bindings, receipt_status=receipt['status'],
        trace_schema=None, trace_sha256=None, sidecar_sha256=None, status='unavailable', reason='trace_absent',
        run=None, prefix=None, columns=_columns(), descriptor_catalog=[], phase_catalog=[], case_state_catalog=[], cases=[], quality_join=None, scope=SCOPE)

    def finish():
        value['observation_sha256'] = _hash(value)
        return _copy(value, MAX_OBSERVATION_BYTES)

    if 'restart_trace' not in receipt:
        return finish()
    diagnostic = receipt['restart_trace']
    _require(type(diagnostic) is dict and diagnostic.get('status') in ('captured', 'missing', 'invalid', 'error'), 'trace diagnostic status')
    sidecar = diagnostic.get('sidecar_sha256')
    _require(sidecar is None or _sha(sidecar), 'sidecar digest')
    value['sidecar_sha256'] = sidecar
    if diagnostic['status'] != 'captured':
        _require(diagnostic.get('trace') is None, 'unavailable diagnostic carries a trace')
        reason = diagnostic.get('reason')
        _require(type(reason) is str and 0 < len(reason) <= 256, 'unavailable diagnostic reason')
        value['reason'] = diagnostic['status'] + ':' + reason
        return finish()
    _require(sidecar is not None and diagnostic.get('reason') is None, 'captured sidecar metadata')
    trace = codec.validate_trace(diagnostic.get('trace'), source_sha256=source_sha256, seed=seed,
        runtime_sha256=bindings['runtime_sha256'], scaffold_sha256=scaffold_sha256)
    value.update(trace_schema=trace['schema'], trace_sha256=trace['trace_sha256'])
    bindings.update(scaffold_sha256=trace['identity']['scaffold_sha256'], invocation_id=trace['identity']['invocation_id'])
    join = _quality_join(receipt, trace)
    value['quality_join'] = join
    if join is not None and not (join['within_roundoff'] and join['terminal_status_compatible']):
        value['reason'] = 'trace_result_mismatch'
        return finish()
    value.update(status='available', reason=None, prefix=trace['prefix'], run={
        'status': trace['status'], 'total_starts': trace['total_starts'], 'total_scheduled_calls': trace['total_scheduled_calls'],
        'initial_quality': trace['initial_quality'], 'final_quality': trace['final_quality'], 'origins': trace['aggregates']})
    catalog = value['descriptor_catalog']
    def reference(descriptor):
        if descriptor not in catalog: catalog.append(descriptor)
        return catalog.index(descriptor)
    def phase_reference(quality):
        if quality is None: return None
        pair = [quality['sum'], quality['improvements']]
        encoded = _canonical(pair)
        existing = [_canonical(p) for p in value['phase_catalog']]
        if encoded not in existing:
            value['phase_catalog'].append(pair)
            return len(existing)
        return existing.index(encoded)
    for call in trace['calls']:
        legacy = 'optimizer_result' not in call
        result_descriptor = call.get('optimizer_result')
        descriptors = [_descriptor(trace, call['inputs'][key]) for key in INPUTS]
        descriptors += [_descriptor(trace, call['post_inputs'][key], post=True) for key in INPUTS]
        descriptors += [_descriptor(trace, call['raw_return']['vector']), _descriptor(trace, call['effective']['vector']),
                        _descriptor(trace, result_descriptor, legacy=legacy)]
        phases = call['phases']
        result_values = _comparable(trace, result_descriptor)
        def equals(descriptor):
            other = _comparable(trace, descriptor)
            return None if result_values is None or other is None else result_values == other
        unchanged = [None if call['inputs'][key]['ref'] is None or call['post_inputs'][key]['ref'] is None
                     else call['inputs'][key]['ref'] == call['post_inputs'][key]['ref'] for key in INPUTS]
        case = dict(ordinal=call['ordinal'], restart_index=call['restart_index'], raw_return_kind=call['raw_return']['kind'],
            effective_origin=call['effective']['origin'], helper_finished=call['helper_finished'], error=call['error'],
            terminal=call['terminal'], helper_accepted=(call['effective']['origin'] == 'accepted_return') if call['helper_finished'] else None,
            optimizer_entered=phases['optimizer_entered'] is not None, optimizer_returned=phases['optimizer_returned'] is not None,
            phase_quality_refs=[phase_reference(phases[p]) for p in PHASES], input_unchanged=unchanged,
            snapshot_refs=[reference(d) for d in descriptors], result_values_equal_effective=equals(call['effective']['vector']),
            result_values_equal_best=equals(call['inputs']['best']))
        state = [case[k] for k in STATE_COLUMNS]
        if state not in value['case_state_catalog']: value['case_state_catalog'].append(state)
        case['state_ref'] = value['case_state_catalog'].index(state)
        value['cases'].append([case[k] for k in CASE_COLUMNS])
    return finish()


def validate_observation(value, *, receipt, source_sha256, seed, adapter_identity, scaffold_sha256=None):
    """Evidence import needs its original receipt, not a rehashed summary alone."""
    if type(value) is dict and value.get('schema') == 'sciel-restart-observation/3':
        from .restart_motion_observations import validate_observation as validate_motion_observation
        return validate_motion_observation(value, receipt=receipt, source_sha256=source_sha256,
            seed=seed, adapter_identity=adapter_identity, scaffold_sha256=scaffold_sha256)
    expected = build_observation(receipt, source_sha256=source_sha256, seed=seed,
        adapter_identity=adapter_identity, scaffold_sha256=scaffold_sha256)
    _require(_canonical(_copy(value, MAX_OBSERVATION_BYTES)) == _canonical(expected), 'projection differs from durable receipt')
    return expected


def _compact(value):
    if type(value) is dict and value.get('schema') == 'sciel-restart-observation/3':
        from .restart_motion_observations import validate_shape
        return validate_shape(value)
    if type(value) is dict and value.get('schema') == 'sciel-restart-observation/2':
        from .restart_optimizer_observations import validate_shape
        return validate_shape(value)
    value = _copy(value, MAX_OBSERVATION_BYTES)
    _require(type(value) is dict and set(value) == _FIELDS and value['schema'] == SCHEMA and value['scope'] == SCOPE, 'observation fields')
    _require(_sha(value['observation_sha256']) and value['observation_sha256'] == _hash({k: v for k, v in value.items() if k != 'observation_sha256'}), 'observation digest')
    _require(value['columns'] == _columns() and value['status'] in ('available', 'unavailable'), 'observation layout')
    _bindings(value['bindings'])
    _require(type(value['cases']) is list and len(value['cases']) <= 16 and type(value['descriptor_catalog']) is list and
             len(value['descriptor_catalog']) <= 176 and type(value['phase_catalog']) is list and len(value['phase_catalog']) <= 80 and
             type(value['case_state_catalog']) is list and len(value['case_state_catalog']) <= 16, 'case/catalog limits')
    _require(all(type(state) is list and len(state) == len(STATE_COLUMNS) for state in value['case_state_catalog']), 'case state layout')
    for phase in value['phase_catalog']:
        _require(type(phase) is list and len(phase) == 2 and type(phase[0]) is float and math.isfinite(phase[0]) and
                 type(phase[1]) is int and phase[1] >= 0, 'phase quality')
    for descriptor in value['descriptor_catalog']:
        _require(type(descriptor) is dict and set(descriptor) == {'state', 'reason', 'dtype', 'shape', 'finite'}, 'descriptor fields')
        state = descriptor['state']
        _require(type(state) is str and state in ('available', 'unavailable', 'hash_only', 'not_applicable'), 'descriptor state')
        if state == 'available':
            _require(descriptor['reason'] is None and codec._rt_dtype(descriptor['dtype']) is not None and
                type(descriptor['shape']) is list and len(descriptor['shape']) <= 4 and
                all(type(n) is int and 0 <= n <= 32768 for n in descriptor['shape']) and type(descriptor['finite']) is bool, 'available descriptor')
        else:
            reasons = {'unavailable': ('array_unavailable', 'unsupported_dtype', 'unsupported_shape', 'oversized', 'legacy_not_recorded'),
                       'hash_only': ('post_values_not_retained',), 'not_applicable': ('not_applicable',)}
            _require(descriptor['reason'] in reasons[state] and all(descriptor[k] is None for k in ('dtype', 'shape', 'finite')), 'unavailable descriptor')
    for index, row in enumerate(value['cases']):
        _require(type(row) is list and len(row) == len(CASE_COLUMNS), 'case row layout')
        case = dict(zip(CASE_COLUMNS, row))
        _require(type(case['state_ref']) is int and 0 <= case['state_ref'] < len(value['case_state_catalog']), 'case state reference')
        case.update(zip(STATE_COLUMNS, value['case_state_catalog'][case['state_ref']]))
        refs, phase_refs = case['snapshot_refs'], case['phase_quality_refs']
        _require(type(refs) is list and len(refs) == len(SNAPSHOTS) and
            all(type(ref) is int and 0 <= ref < len(value['descriptor_catalog']) for ref in refs), 'descriptor reference')
        _require(type(phase_refs) is list and len(phase_refs) == len(PHASES) and
            all(ref is None or (type(ref) is int and 0 <= ref < len(value['phase_catalog'])) for ref in phase_refs), 'phase reference')
        phases = [None if ref is None else value['phase_catalog'][ref] for ref in phase_refs]
        _require(type(case['ordinal']) is int and case['ordinal'] == index and type(case['restart_index']) is int and
            case['restart_index'] == 4*index and phases[0] is not None and phases[-1] is not None, 'case order')
        _require(all(type(case[k]) is bool for k in ('helper_finished', 'optimizer_entered', 'optimizer_returned')) and
            case['helper_finished'] == (phases[1] is not None) and case['optimizer_entered'] == (phases[2] is not None) and
            case['optimizer_returned'] == (phases[3] is not None) and
            (not case['optimizer_returned'] or case['optimizer_entered']), 'phase flags')
        _require(type(case['input_unchanged']) is list and len(case['input_unchanged']) == 4 and
            all(x is None or type(x) is bool for x in case['input_unchanged']), 'mutation flags')
        _require(case['terminal'] in ('completed','deadline','error','incomplete') and
            (case['terminal'] != 'completed' or case['optimizer_returned']), 'terminal phase')
        if not case['helper_finished']:
            _require(all(case[k] is None for k in ('raw_return_kind','effective_origin','helper_accepted','error')) and
                not case['optimizer_entered'], 'unfinished helper')
        elif case['error'] is not None:
            error = case['error']
            _require(type(error) is dict and set(error) == {'stage','code'} and error['stage'] in ('helper','normalization') and
                type(error['code']) is str and re.fullmatch('[A-Za-z_][A-Za-z_0-9]{0,95}',error['code']) is not None and
                case['helper_accepted'] is False and case['effective_origin'] is None and not case['optimizer_entered'] and
                ((error['stage'] == 'helper' and case['raw_return_kind'] == 'exception') or
                 (error['stage'] == 'normalization' and case['raw_return_kind'] in ('none','value'))), 'failed helper')
        else:
            _require((case['raw_return_kind'],case['effective_origin'],case['helper_accepted']) in
                (('none','saved_ordinary_fallback',False),('value','accepted_return',True)) and
                type(case['helper_accepted']) is bool, 'helper origin/acceptance')
        result = value['descriptor_catalog'][refs[-1]]
        if value['trace_schema'] == codec.LEGACY_TRACE_SCHEMA:
            _require(result['reason'] == 'legacy_not_recorded', 'legacy result snapshot')
        else:
            _require((result['state'] == 'not_applicable') == (not case['optimizer_returned']) and
                result['reason'] != 'legacy_not_recorded', 'result/return phase')
        comparable = result['state'] == 'available' and result['shape'] == [78] and result['finite'] is True and result['dtype'][1] in 'fiu'
        for name, snapshot in (('result_values_equal_effective','effective'), ('result_values_equal_best','best')):
            other = value['descriptor_catalog'][refs[SNAPSHOTS.index(snapshot)]]
            other_comparable = other['state'] == 'available' and other['shape'] == [78] and other['finite'] is True and other['dtype'][1] in 'fiu'
            _require(case[name] is None or (type(case[name]) is bool and comparable and other_comparable), 'unknown result equality')
    if value['status'] == 'unavailable':
        _require(value['run'] is None and value['prefix'] is None and not value['cases'] and not value['descriptor_catalog'] and not value['phase_catalog'] and not value['case_state_catalog'] and
                 type(value['reason']) is str and 0 < len(value['reason']) <= 300, 'unavailable observation')
    else:
        _require(value['reason'] is None and type(value['run']) is dict and type(value['prefix']) is dict and
                 value['prefix'].get('captured_count') == len(value['cases']), 'available observation')
    return value


def observation_cases(value):
    """Detached lossless row expansion; snapshots are metadata, never array bytes."""
    if type(value) is dict and value.get('schema') == 'sciel-restart-observation/3':
        from .restart_motion_observations import observation_cases as motion_cases
        return motion_cases(value)
    if type(value) is dict and value.get('schema') == 'sciel-restart-observation/2':
        from .restart_optimizer_observations import observation_cases as optimizer_cases
        return optimizer_cases(value)
    value = _compact(value)
    result = []
    for row in value['cases']:
        case = dict(zip(CASE_COLUMNS, row))
        case.update(zip(STATE_COLUMNS, value['case_state_catalog'][case['state_ref']]))
        phases = [None if ref is None else value['phase_catalog'][ref] for ref in case['phase_quality_refs']]
        case.update(phase_sums=[None if p is None else p[0] for p in phases],
            phase_improvements=[None if p is None else p[1] for p in phases],
            snapshots={name: value['descriptor_catalog'][ref] for name, ref in zip(SNAPSHOTS, case['snapshot_refs'])})
        result.append(case)
    return _copy(result, MAX_OBSERVATION_BYTES)


def observation_view(value, max_chars=8192):
    """Whole bounded card, or explicit omission; no partial prefix presentation."""
    _require(type(max_chars) is int and max_chars > 0, 'view budget')
    value = _compact(value)
    view = dict(schema=VIEW_SCHEMA, observation_sha256=value['observation_sha256'], bindings=value['bindings'],
        observation_status=value['status'], included=True, omission_reason=None, observation=value)
    def seal():
        result = dict(view)
        result['view_sha256'] = _hash(result)
        return _copy(result)
    result = seal()
    if len(_canonical(result)) > max_chars:
        view.update(included=False, omission_reason='card_char_limit', observation=None)
        result = seal()
    _require(len(_canonical(result)) <= max_chars, 'omission metadata exceeds view budget')
    return result


def validate_observation_view(value):
    """Validate presentation shape/integrity only; never import it as evidence."""
    value = _copy(value)
    _require(type(value) is dict and set(value) == {'schema', 'observation_sha256', 'bindings', 'observation_status',
        'included', 'omission_reason', 'observation', 'view_sha256'} and value['schema'] == VIEW_SCHEMA, 'view fields')
    _require(type(value['included']) is bool and value['observation_status'] in ('available', 'unavailable') and
        _sha(value['observation_sha256']) and type(value['bindings']) is dict and set(value['bindings']) == _BINDINGS, 'view metadata')
    _bindings(value['bindings'])
    _require(_sha(value['view_sha256']) and value['view_sha256'] == _hash({k: v for k, v in value.items() if k != 'view_sha256'}), 'view digest')
    if value['included']:
        observation = _compact(value['observation'])
        _require(value['omission_reason'] is None and observation['observation_sha256'] == value['observation_sha256'] and
            _canonical(observation['bindings']) == _canonical(value['bindings']) and observation['status'] == value['observation_status'], 'view observation binding')
    else:
        _require(value['observation'] is None and value['omission_reason'] == 'card_char_limit', 'whole omission')
    return value
