"""Restart-observation wire schemas and exact field encodings."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from ..hypothesis import restart_observations as observations


SCHEMA = 'sciel-restart-observation-context/5'

PREVIOUS_FACTORED_SCHEMA = 'sciel-restart-observation-context/4'

PREVIOUS_SCHEMA = 'sciel-restart-observation-context/3'

LEGACY_SCHEMA = 'sciel-restart-observation-context/2'

EXPANDED_SCHEMA = 'sciel-restart-observation-context-expanded/1'

PREFIX = 'RESTART_OBSERVATIONS_JSON:\n'

RULES = (
    'RESTART REFERENCES: receipt_refs use columns.receipt_refs; group/pair/role indices select '
    'groups, group.seeds and columns.roles. Cards/omissions select receipt rows. For /4 and /5, '
    'observation_refs select observation_catalog [field,value] entries; merge their distinct '
    'fields to restore each exact whole observation and original hash. Fields named in '
    'columns.field_values use those value columns; run quality pairs use columns.quality, '
    'and run.origins uses columns.origin_rows and columns.origin_counters. For /3, merge '
    'observation_common[observation_common_ref] into the disjoint observation remainder to restore '
    'its exact whole object and original hash. Each observation retains scope. Its columns label rows: '
    'state_ref selects case_state_catalog; '
    'phase_quality_refs select phase_catalog [sum,improvements] in columns.phases order; '
    'in observation/3, phase_refs_ref and snapshot_refs_ref first select phase_refs_catalog '
    'and snapshot_refs_catalog; motion_state_ref selects motion_state_catalog. '
    'snapshot_refs select descriptor_catalog in columns.snapshots order; shared descriptor refs '
    'mean equal metadata, not equal array values. These same-invocation '
    'diagnostics add no lesson support. Interval gains are not causal. Optimizer result.x '
    'precedes incumbent admission and may be infeasible. Missing, invalid, legacy and omitted '
    'facts are unknown, not zero or unchanged. Numeric equality implies no semantic equivalence.\n'
)

_ROLES = ('comparator', 'challenger')

_REF_FIELDS = {'group_id', 'role', 'seed', 'source_sha256', 'receipt_sha256', 'receipt_status'}

_COLUMNS = {'receipt_refs': ['group_ref', 'pair_ref', 'role_ref', 'receipt_sha256', 'receipt_status'],
            'roles': list(_ROLES), 'omissions': ['receipt_ref', 'reason']}

_FIELD_COLUMNS = {
    'bindings': ['source_sha256', 'seed', 'adapter_identity_sha256', 'runtime_sha256',
                 'scaffold_sha256', 'invocation_id'],
    'prefix': ['limit', 'captured_count', 'omitted_count', 'first_omitted_ordinal', 'omission_reason'],
    'quality_join': ['raw_sum', 'trace_sum', 'maximum_roundoff', 'within_roundoff', 'terminal_status_compatible'],
    'run': ['status', 'total_starts', 'total_scheduled_calls', 'initial_quality', 'final_quality', 'origins'],
}

_QUALITY_COLUMNS = ['sum', 'improvements']

_ORIGINS = ['accepted_return', 'helper_error', 'ordinary', 'saved_ordinary_fallback', 'scheduled_unresolved']

_ORIGIN_COLUMNS = ['starts', 'helper_calls', 'optimizer_entered', 'optimizer_returned',
    'improvements', 'gain', 'completed', 'deadline', 'error', 'incomplete']

_COLUMNS4 = {**_COLUMNS, 'field_values': _FIELD_COLUMNS, 'quality': _QUALITY_COLUMNS,
             'origin_rows': _ORIGINS, 'origin_counters': _ORIGIN_COLUMNS}

_COMMON_FIELDS = {'schema', 'scope', 'columns', 'result_geometry_policy'}

_WIRE_FIELDS = {'schema', 'columns', 'groups', 'receipt_refs', 'cards', 'omissions',
                'selected_count', 'presented_count', 'omitted_count', 'max_items', 'max_chars',
                'rendered_chars', 'context_sha256'}

MAX_CARD_CHARS = 8192

_MAX_VIEW_CHARS = observations.MAX_OBSERVATION_BYTES * 2

_ADMISSION_POLICY = {
    'schema': 'sciel-restart-observation-admission/1',
    'wire_schema': SCHEMA,
    'card_chars': MAX_CARD_CHARS,
    'card_basis': 'Canonical {card,observation_catalog} with the actual packed card '
                  'and all its referenced field entries, including entries shared with other cards.',
    'context_basis': 'Complete canonical wire, including group/receipt/column metadata; '
                     'the caller character limit is unchanged.',
    'selection': 'Input group, seed, comparator then challenger order; whole cards only; '
                 'no fact clipping, outcome ranking or replacement facts.',
    'legacy': '/2, /3 and /4 retain the original whole-view character guard.',
}

def _require(condition, reason):
    if not condition:
        raise ValueError('restart context: ' + reason)

def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)

def _digest(value):
    return hashlib.sha256(_wire(value).encode()).hexdigest()

def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None

def _text(value, maximum=128):
    return type(value) is str and 0 < len(value) <= maximum and bool(value.strip())

def _seal(value):
    value = copy.deepcopy(value)
    value['rendered_chars'] = 0
    value['context_sha256'] = '0' * 64
    while value['rendered_chars'] != len(_wire(value)):
        value['rendered_chars'] = len(_wire(value))
    value['context_sha256'] = _digest({k: v for k, v in value.items() if k != 'context_sha256'})
    return value

def _common_fields(schema):
    _require(schema in ('sciel-restart-observation/1', 'sciel-restart-observation/2'), 'common observation schema')
    return (_COMMON_FIELDS - {'result_geometry_policy'} if schema == 'sciel-restart-observation/1'
            else _COMMON_FIELDS)

def _record_values(value, columns):
    _require(type(value) is dict and set(value) == set(columns), 'encoded record fields')
    return [copy.deepcopy(value[key]) for key in columns]

def _record_from_values(value, columns):
    _require(type(value) is list and len(value) == len(columns), 'encoded record width')
    return dict(zip(columns, copy.deepcopy(value)))

def _encode_field(field, value):
    if field not in _FIELD_COLUMNS or value is None:
        return copy.deepcopy(value)
    value = copy.deepcopy(value)
    if field == 'run':
        for key in ('initial_quality', 'final_quality'):
            value[key] = _record_values(value[key], _QUALITY_COLUMNS)
        _require(type(value['origins']) is dict and set(value['origins']) == set(_ORIGINS), 'origin fields')
        value['origins'] = [_record_values(value['origins'][key], _ORIGIN_COLUMNS) for key in _ORIGINS]
    return _record_values(value, _FIELD_COLUMNS[field])

def _decode_field(field, value):
    if field not in _FIELD_COLUMNS or value is None:
        return copy.deepcopy(value)
    value = _record_from_values(value, _FIELD_COLUMNS[field])
    if field == 'run':
        for key in ('initial_quality', 'final_quality'):
            value[key] = _record_from_values(value[key], _QUALITY_COLUMNS)
        rows = value['origins']
        _require(type(rows) is list and len(rows) == len(_ORIGINS), 'origin row count')
        value['origins'] = {name: _record_from_values(row, _ORIGIN_COLUMNS)
                            for name, row in zip(_ORIGINS, rows, strict=True)}
    return value

def _check_wire(value, schema, extra=()):
    _require(type(value) is dict and set(value) == _WIRE_FIELDS | set(extra)
        and value['schema'] == schema and value['columns'] == (
            _COLUMNS4 if schema in (SCHEMA, PREVIOUS_FACTORED_SCHEMA) else _COLUMNS),
        'wire schema fields')
    _require(value['context_sha256'] == _digest({k: v for k, v in value.items() if k != 'context_sha256'}), 'wire digest')
    _require(type(value['rendered_chars']) is int and type(value['max_chars']) is int
        and value['rendered_chars'] == len(_wire(value)) <= value['max_chars'], 'wire rendered limit')
    _require(type(value['groups']) is list and type(value['receipt_refs']) is list
        and type(value['cards']) is list and type(value['omissions']) is list, 'wire lists')
