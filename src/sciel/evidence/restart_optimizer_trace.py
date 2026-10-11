"""Add bounded optimizer outcome metadata without rewriting historical codecs.

The legacy /2 collector remains the owner of arrays, phases and aggregates.
Schema /3 adds only the already-used success boolean and a separately observed
integer status code for retained scheduled calls. Neither asserts feasibility,
incumbent admission, causal credit or quality. Legacy /1 and /2 read unchanged.
"""
from __future__ import annotations

import numbers as _numbers

from . import restart_trace as _legacy

TRACE_SCHEMA = 'sciel-restart-trace/3'
LEGACY_RESULT_SCHEMA = _legacy.TRACE_SCHEMA
LEGACY_TRACE_SCHEMAS = (_legacy.LEGACY_TRACE_SCHEMA, _legacy.TRACE_SCHEMA)
MAX_TRACE_BYTES = _legacy.MAX_TRACE_BYTES
PREFIX_LIMIT = _legacy.PREFIX_LIMIT
MAX_STARTS = _legacy.MAX_STARTS
STATUS_MIN, STATUS_MAX = -(2**63), 2**63 - 1
_MISSING = object()


def _missing(reason):
    return {'success': {'value': None, 'reason': reason},
            'status': {'value': None, 'reason': reason}}


def _status(result):
    # Called inside the safe recorder, only for a retained scheduled call.
    # Do not stringify arbitrary result objects or infer a code from success.
    try:
        value = getattr(result, 'status', _MISSING)
    except Exception:
        return {'value': None, 'reason': 'read_error'}
    if value is _MISSING:
        return {'value': None, 'reason': 'missing'}
    if isinstance(value, bool) or not isinstance(value, _numbers.Integral):
        return {'value': None, 'reason': 'unsupported_type'}
    try:
        integer = int(value)
    except Exception:
        return {'value': None, 'reason': 'read_error'}
    if not STATUS_MIN <= integer <= STATUS_MAX:
        return {'value': None, 'reason': 'unsupported_type'}
    return {'value': integer, 'reason': None}


def _check_outcome(value, call):
    _legacy._rt_keys(value, ('success', 'status'), 'optimizer outcome')
    for descriptor in value.values():
        _legacy._rt_keys(descriptor, ('value', 'reason'), 'optimizer outcome descriptor')
    success, status = value['success'], value['status']
    returned = call['phases']['optimizer_returned'] is not None
    if not returned:
        _legacy._rt_require(value == _missing('not_returned'), 'outcome without optimizer return')
        return
    if success['value'] is None:
        _legacy._rt_require(value == _missing('not_observed') and call['terminal'] != 'completed',
                            'unobserved optimizer outcome')
        return
    _legacy._rt_require(type(success['value']) is bool and success['reason'] is None,
                        'optimizer success must be the observed boolean')
    if status['value'] is None:
        _legacy._rt_require(status['reason'] in ('missing', 'unsupported_type', 'read_error'),
                            'unavailable optimizer status')
    else:
        _legacy._rt_require(type(status['value']) is int and STATUS_MIN <= status['value'] <= STATUS_MAX
                            and status['reason'] is None, 'optimizer status integer')


def _seal(value):
    value.pop('trace_sha256', None)
    value['trace_sha256'] = _legacy._rt_digest(value)
    return value


def validate_trace(value, **bindings):
    """Validate /3 additions and every legacy invariant; never upgrade old data."""
    _legacy._rt_require(type(value) is dict, 'trace must be an object')
    if value.get('schema') != TRACE_SCHEMA:
        return _legacy.validate_trace(value, **bindings)
    value = _legacy._rt_clone(value)
    _legacy._rt_require(len(_legacy._rt_json_text(value).encode('ascii')) <= MAX_TRACE_BYTES,
                        'payload byte cap')
    expected = value.get('trace_sha256')
    bare = {key: item for key, item in value.items() if key != 'trace_sha256'}
    _legacy._rt_require(_legacy._rt_hex(expected) and _legacy._rt_digest(bare) == expected, 'trace hash')
    legacy = _legacy._rt_clone(value)
    _legacy._rt_require(type(legacy.get('calls')) is list and len(legacy['calls']) <= PREFIX_LIMIT,
                        'optimizer call prefix')
    outcomes = []
    for call in legacy['calls']:
        _legacy._rt_require(type(call) is dict and 'optimizer_outcome' in call, 'optimizer outcome required')
        outcomes.append(call.pop('optimizer_outcome'))
    legacy['schema'] = _legacy.TRACE_SCHEMA
    checked = _legacy.validate_trace(_seal(legacy), **bindings)
    for outcome, call in zip(outcomes, checked['calls']):
        _check_outcome(outcome, call)
    return value


def parse_trace(text, **bindings):
    _legacy._rt_require(type(text) in (str, bytes), 'JSON input type')
    try:
        raw = text.encode('utf-8') if type(text) is str else text
        _legacy._rt_require(len(raw) <= MAX_TRACE_BYTES, 'payload byte cap')
        def unique(items):
            value = {}
            for key, item in items:
                _legacy._rt_require(key not in value, 'duplicate JSON key')
                value[key] = item
            return value
        value = _legacy._rt_json.loads(raw, object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (UnicodeError, RecursionError, _legacy._rt_json.JSONDecodeError) as exc:
        raise ValueError('restart trace: invalid JSON') from exc
    return validate_trace(value, **bindings)


def encode_trace(value):
    return _legacy._rt_json_text(validate_trace(value))


def legacy_projection(value):
    """Validated /3 -> /2 arithmetic view, never evidence of legacy execution.

    Original /1 or /2 input is returned detached and unchanged. Callers must
    retain the original receipt/trace hashes when presenting derived facts.
    """
    value = validate_trace(value)
    if value['schema'] != TRACE_SCHEMA:
        return value
    value['schema'] = _legacy.TRACE_SCHEMA
    for call in value['calls']:
        call.pop('optimizer_outcome')
    return _legacy.validate_trace(_seal(value))


class RestartTrace(_legacy.RestartTrace):
    """Keep old phase/capture mechanics, with separately bounded /3 metadata."""

    def __init__(self, identity):
        super().__init__(identity)
        self._outcomes = {}

    def _project(self, value):
        result = _legacy._rt_clone(value)
        result['schema'] = TRACE_SCHEMA
        result.pop('trace_sha256', None)
        for call in result['calls']:
            default = 'not_returned' if call['phases']['optimizer_returned'] is None else 'not_observed'
            call['optimizer_outcome'] = _legacy._rt_clone(self._outcomes.get(call['ordinal'], _missing(default)))
        return result

    def _trim(self):
        # Account for complete /3 records during collection, not after an
        # oversized diagnostic has already been presented as valid evidence.
        if len(_legacy._rt_json_text(self._project(self._wire())).encode('ascii')) <= MAX_TRACE_BYTES - 2048:
            return
        record = self._active['record'] if self._active is not None else None
        _legacy._rt_require(record is not None and self.calls and self.calls[-1] is record,
                            'fixed metadata exceeds byte cap')
        self.calls.pop()
        self._outcomes.pop(record['ordinal'], None)
        self._active['record'] = None
        self._byte_stop = True
        used = {ref for item in self.calls for ref in _legacy._rt_record_refs(item)}
        self.vectors = {key: value for key, value in self.vectors.items() if key in used}

    def optimizer_outcome(self, success, result):
        """Record the existing bool conversion; read status once within prefix."""
        self._live()
        active = self._active
        _legacy._rt_require(active is not None and active['returned'], 'outcome before optimizer return')
        record = active['record']
        if record is None:
            return  # Ordinary/omitted calls must not access result.status.
        ordinal = record['ordinal']
        _legacy._rt_require(ordinal not in self._outcomes, 'duplicate optimizer outcome')
        _legacy._rt_require(type(success) is bool, 'optimizer success must be the existing boolean')
        self._outcomes[ordinal] = {'success': {'value': success, 'reason': None}, 'status': _status(result)}
        self._trim()

    def finish(self, status, *, incumbent_sum, improvements):
        # super validates its exact /2 model, including array/phase joins; the
        # inherited class's globals stay isolated from this module's /3 names.
        legacy = super().finish(status, incumbent_sum=incumbent_sum, improvements=improvements)
        return validate_trace(_seal(self._project(legacy)))
