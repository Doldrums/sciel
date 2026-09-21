"""Whole source-bound motion facts with exact legacy observation reconstruction.

This host projection executes no candidate. Receipt validation rederives facts;
the shape validator and hashes provide transport integrity, not authenticity.
"""
from __future__ import annotations

import copy

from . import restart_observations as base
from . import restart_optimizer_observations as optimizer
from . import restart_motion as motion

SCHEMA = 'sciel-restart-observation/3'
MOTION_STATE_COLUMNS = ['effective_locations_changed', 'effective_reason',
    'result_center_reason', 'match_status', 'match_reason']
MOTION_VALUE_COLUMNS = ['result_center_max', 'match_center_max', 'match_radius_max']
SCOPE = ('Same-invocation retained-prefix diagnostics, not trials or lesson support. '
    'Origins distinguish helper returns from ordinary fallback. Success, result validity '
    'and incumbent admission differ. Missing/ambiguous facts stay unknown. '
    'Motion and interval gains establish no causal, basin, quality or knowledge claim. '
    'base_schema/base_observation_sha256 bind the exactly reconstructible older observation.')
MOTION_POLICY = {
    'schema': motion.POLICY['schema'],
    'vectors': 'Finite real shape78 decoded to binary64; lossy integers unavailable.',
    'effective_locations_changed': 'Exact numeric best/effective center-pair multiset; signed zeros equal.',
    'result_center_max': 'Max indexwise hypot distance effective to result.',
    'matching': 'Each result center has a unique nearest best center by hypot; require bijection. '
                'Ties/non-bijection ambiguous; not optimal assignment or physical identity.',
    'matched_errors': 'Max matched center distance and absolute radius-at-location difference.',
    'scope': 'Descriptive even for invalid geometry; no threshold, feasibility, admission or causal claim.',
}
_EXTRA = {'base_schema', 'base_observation_sha256', 'phase_refs_catalog',
    'snapshot_refs_catalog', 'motion_state_catalog', 'motion_schema', 'motion_policy',
    'motion_policy_sha256', 'motion_sha256'}
_CATALOGS = {'phase_refs_ref': 'phase_refs_catalog',
    'snapshot_refs_ref': 'snapshot_refs_catalog', 'motion_state_ref': 'motion_state_catalog'}


def _require(condition, reason):
    base._require(condition, 'motion ' + reason)


def _base_columns(schema):
    _require(schema in (base.SCHEMA, optimizer.SCHEMA), 'base schema')
    columns = base._columns()
    if schema == optimizer.SCHEMA:
        columns.update(cases=list(optimizer.CASE_COLUMNS), optimizer_state=list(optimizer.STATE_COLUMNS))
    return columns


def _columns(schema):
    columns = _base_columns(schema)
    columns['cases'] = [dict(phase_quality_refs='phase_refs_ref', snapshot_refs='snapshot_refs_ref').get(k, k)
                        for k in columns['cases']] + ['motion_state_ref'] + MOTION_VALUE_COLUMNS
    columns['motion_state'] = list(MOTION_STATE_COLUMNS)
    return columns


def _seal(value):
    value = copy.deepcopy(value)
    value['observation_sha256'] = base._hash({k: v for k, v in value.items() if k != 'observation_sha256'})
    return base._copy(value, base.MAX_OBSERVATION_BYTES)


def _reference(catalog, row):
    encoded = base._canonical(row)
    for index, previous in enumerate(catalog):
        if base._canonical(previous) == encoded:
            return index
    catalog.append(copy.deepcopy(row))
    return len(catalog) - 1


def build_observation(receipt, *, source_sha256, seed, adapter_identity, scaffold_sha256=None):
    """Build the new presentation without changing the legacy builder default."""
    old = base.build_observation(receipt, source_sha256=source_sha256, seed=seed,
        adapter_identity=adapter_identity, scaffold_sha256=scaffold_sha256)
    value = copy.deepcopy(old)
    value.update(schema=SCHEMA, scope=SCOPE, base_schema=old['schema'],
        base_observation_sha256=old['observation_sha256'], columns=_columns(old['schema']),
        phase_refs_catalog=[], snapshot_refs_catalog=[], motion_state_catalog=[], cases=[],
        motion_schema=motion.SCHEMA, motion_policy=copy.deepcopy(MOTION_POLICY),
        motion_policy_sha256=base._hash(motion.POLICY), motion_sha256=None)
    if old['status'] == 'available':
        bindings = {key: old['bindings'][key] for key in
                    ('source_sha256', 'seed', 'runtime_sha256', 'scaffold_sha256', 'invocation_id')}
        movement = motion.build_motion(receipt['restart_trace']['trace'], **bindings)
        _require(movement['trace_sha256'] == old['trace_sha256'] and
            base._canonical(movement['prefix']) == base._canonical(old['prefix']), 'trace/prefix join')
        value['motion_sha256'] = movement['motion_sha256']
        for row, moved in zip(old['cases'], motion.motion_cases(movement), strict=True):
            case = dict(zip(old['columns']['cases'], row))
            _require(case['ordinal'] == moved['ordinal'] and case['restart_index'] == moved['restart_index'], 'case join')
            case['phase_refs_ref'] = _reference(value['phase_refs_catalog'], case.pop('phase_quality_refs'))
            case['snapshot_refs_ref'] = _reference(value['snapshot_refs_catalog'], case.pop('snapshot_refs'))
            case['motion_state_ref'] = _reference(value['motion_state_catalog'],
                [moved[k] for k in MOTION_STATE_COLUMNS])
            case.update((key, moved[key]) for key in MOTION_VALUE_COLUMNS)
            value['cases'].append([case[key] for key in value['columns']['cases']])
    return validate_shape(_seal(value))


def _rows(value):
    """Check exact canonical catalog references before dereferencing."""
    for name in _CATALOGS.values():
        catalog = value[name]
        _require(type(catalog) is list and len(catalog) <= 16 and
            all(type(row) is list for row in catalog), 'catalog shape')
        _require(len({base._canonical(row) for row in catalog}) == len(catalog), 'duplicate catalog row')
    _require(all(len(row) == len(MOTION_STATE_COLUMNS) for row in value['motion_state_catalog']), 'state columns')
    _require(type(value['cases']) is list and len(value['cases']) <= 16, 'case bound')
    seen = {key: set() for key in _CATALOGS}
    rows = []
    for raw in value['cases']:
        _require(type(raw) is list and len(raw) == len(value['columns']['cases']), 'case columns')
        case = dict(zip(value['columns']['cases'], raw))
        for field, catalog in _CATALOGS.items():
            ref = case[field]
            _require(type(ref) is int and 0 <= ref < len(value[catalog]), 'catalog reference')
            if ref not in seen[field]:
                _require(ref == len(seen[field]), 'catalog first-use order')
                seen[field].add(ref)
        rows.append(case)
    _require(all(len(seen[k]) == len(value[name]) for k, name in _CATALOGS.items()), 'unused catalog row')
    return rows


def _restore_base(value, rows):
    old = {k: copy.deepcopy(v) for k, v in value.items() if k not in _EXTRA}
    old.update(schema=value['base_schema'],
        scope=base.SCOPE if value['base_schema'] == base.SCHEMA else optimizer.SCOPE,
        columns=_base_columns(value['base_schema']), observation_sha256=value['base_observation_sha256'], cases=[])
    for case in rows:
        restored = dict(case, phase_quality_refs=value['phase_refs_catalog'][case['phase_refs_ref']],
                        snapshot_refs=value['snapshot_refs_catalog'][case['snapshot_refs_ref']])
        old['cases'].append([copy.deepcopy(restored[k]) for k in old['columns']['cases']])
    return base._compact(old)


def _restore_motion(value, rows):
    restored = dict(schema=motion.SCHEMA, trace_schema=value['trace_schema'], trace_sha256=value['trace_sha256'],
        identity={k: value['bindings'][k] for k in
                  ('source_sha256', 'seed', 'runtime_sha256', 'scaffold_sha256', 'invocation_id')},
        prefix=copy.deepcopy(value['prefix']), policy=copy.deepcopy(motion.POLICY),
        columns=list(motion.COLUMNS), cases=[], motion_sha256=value['motion_sha256'])
    for row in rows:
        case = dict(row)
        case.update(zip(MOTION_STATE_COLUMNS, value['motion_state_catalog'][row['motion_state_ref']]))
        restored['cases'].append([case[k] for k in motion.COLUMNS])
    return restored


def _comparable(descriptor):
    return (descriptor['state'] == 'available' and descriptor['finite'] is True and
            descriptor['shape'] == [78] and descriptor['dtype'][1] in 'fiu')


def validate_shape(value):
    """Detached transport validation; receipt rederivation is still required."""
    try:
        value = base._copy(value, base.MAX_OBSERVATION_BYTES)
        _require(type(value) is dict and value.get('schema') == SCHEMA, 'schema')
        columns = _columns(value.get('base_schema'))
        fields = base._FIELDS | _EXTRA
        if value['base_schema'] == optimizer.SCHEMA:
            fields |= {'optimizer_state_catalog', 'result_geometry_policy'}
        _require(set(value) == fields and value['scope'] == SCOPE, 'fields/scope')
        _require(base._canonical(value['columns']) == base._canonical(columns), 'columns')
        _require(value['motion_schema'] == motion.SCHEMA and
            base._canonical(value['motion_policy']) == base._canonical(MOTION_POLICY) and
            value['motion_policy_sha256'] == base._hash(motion.POLICY), 'policy')
        _require(base._sha(value['base_observation_sha256']) and base._sha(value['observation_sha256']) and
            value['observation_sha256'] == base._hash({k: v for k, v in value.items() if k != 'observation_sha256'}), 'digest')
        rows = _rows(value)
        old = _restore_base(value, rows)
        if value['status'] == 'unavailable':
            _require(value['motion_sha256'] is None and not rows, 'unavailable motion')
        else:
            moved = motion.motion_cases(_restore_motion(value, rows))
            for case, facts in zip(base.observation_cases(old), moved, strict=True):
                descriptors = case['snapshots']
                if facts['effective_locations_changed'] is not None:
                    _require(_comparable(descriptors['best']) and _comparable(descriptors['effective']), 'effective metadata')
                if facts['result_center_max'] is not None:
                    _require(_comparable(descriptors['effective']) and _comparable(descriptors['optimizer_result']), 'result metadata')
                if facts['match_status'] in ('matched', 'ambiguous'):
                    _require(_comparable(descriptors['best']) and _comparable(descriptors['optimizer_result']), 'matching metadata')
                if case['result_values_equal_effective'] is True:
                    _require(facts['result_center_max'] in (None, 0.), 'equal effective motion')
                if case['result_values_equal_best'] is True and facts['match_status'] == 'matched':
                    _require(facts['match_center_max'] == facts['match_radius_max'] == 0., 'equal best motion')
        return value
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('restart observation: malformed motion projection') from exc


def base_observation(value):
    """Restore exactly the original /1 or /2 object, including its old digest."""
    value = validate_shape(value)
    return _restore_base(value, _rows(value))


def observation_cases(value):
    """Expand the old case facts plus the eight original quantitative motion fields."""
    value = validate_shape(value)
    rows = _rows(value)
    cases = base.observation_cases(_restore_base(value, rows))
    if cases:
        moved = motion.motion_cases(_restore_motion(value, rows))
        for case, facts in zip(cases, moved, strict=True):
            case.update((key, val) for key, val in facts.items() if key not in ('ordinal', 'restart_index'))
    return cases


def validate_observation(value, *, receipt, source_sha256, seed, adapter_identity, scaffold_sha256=None):
    expected = build_observation(receipt, source_sha256=source_sha256, seed=seed,
        adapter_identity=adapter_identity, scaffold_sha256=scaffold_sha256)
    _require(base._canonical(validate_shape(value)) == base._canonical(expected), 'projection differs from durable receipt')
    return expected
