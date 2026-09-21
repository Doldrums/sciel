"""Compact observations of saved paired geometry; never execute a candidate.

These are comparisons on three recorded seeds, not semantic equivalence,
speed measurements or additional evidence for quality/admission. The measured
comparator need not be the parent that generated the challenger.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import re

from ..hypothesis.change_audit import build_change_audit
from ..knowledge_base.paired_comparisons import validate_comparison

SCHEMA = 'sciel-observed-paired/1'
MAX_BYTES = 16384
MAX_UNITS = 8
GEOMETRY_REPRESENTATION = 'validated_capture_binary64_json'
LIMITATION = ('Three recorded paired seeds only. Exact saved numeric geometry, not original array dtypes, '
              'permutation-invariant geometry, semantic equivalence, speed, causal credit or quality admission. '
              'The parent role here is the measured comparator, not necessarily the generative parent.')
_STATUSES = {'same_geometry_on_paired_seeds', 'same_scores_on_paired_seeds',
             'different_geometry_on_paired_seeds', 'incomplete'}
_DRAW_STATUSES = {'valid', 'candidate_failure', 'infrastructure', 'unknown', 'not_dispatched'}
_PAIR_FIELDS = {'seed', 'parent_status', 'child_status', 'parent_raw_sum', 'child_raw_sum',
                'parent_certified_sum', 'child_certified_sum', 'scores_equal', 'geometry_equal',
                'parent_geometry_sha256', 'child_geometry_sha256'}
_FIELDS = {'schema', 'id', 'group_id', 'parent_source_sha256', 'child_source_sha256', 'parent_role',
           'scope', 'status', 'group_sha256', 'comparison_identity_sha256', 'pairs',
           'geometry_representation', 'original_array_dtypes', 'limitation', 'edited_units',
           'edited_units_available', 'edited_units_omitted', 'record_sha256'}


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _sha(value):
    return hashlib.sha256(_wire(value).encode('ascii')).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError('observed paired: ' + message)


def _digest(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _exact(value):
    if (type(value) is not dict or set(value) != {'numerator', 'denominator'}
            or type(value['numerator']) is not int or type(value['denominator']) is not int
            or value['numerator'] < 0 or value['denominator'] <= 0):
        return False
    reduced = Fraction(value['numerator'], value['denominator'])
    return reduced.numerator == value['numerator'] and reduced.denominator == value['denominator']


def _status(rows):
    if any(row['parent_status'] != 'valid' or row['child_status'] != 'valid' for row in rows):
        return 'incomplete'
    if all(row['geometry_equal'] is True for row in rows):
        return 'same_geometry_on_paired_seeds'
    if all(row['scores_equal'] is True for row in rows):
        return 'same_scores_on_paired_seeds'
    return 'different_geometry_on_paired_seeds'


def validate_record(record):
    """Validate a detached checkpoint projection, not its external provenance.

    Recomputed hashes provide integrity, not authentication. Importing a new
    observation requires paired_record and the original validated group.
    """
    try:
        encoded = _wire(record)
        _require(len(encoded) <= MAX_BYTES, 'record exceeds size limit')
        value = json.loads(encoded)
        _require(type(value) is dict and set(value) == _FIELDS, 'record fields')
        _require(value['schema'] == SCHEMA and value['scope'] == 'paired'
                 and value['parent_role'] == 'comparator', 'schema or scope')
        _require(type(value['id']) is str and re.fullmatch(r'comparison:[0-9a-f]{64}', value['id'])
                 and value['group_id'] == value['id'], 'group identity')
        for key in ('parent_source_sha256', 'child_source_sha256', 'group_sha256',
                    'comparison_identity_sha256', 'record_sha256'):
            _require(_digest(value[key]), 'invalid ' + key)
        _require(value['parent_source_sha256'] != value['child_source_sha256'], 'same source')
        _require(value['geometry_representation'] == GEOMETRY_REPRESENTATION
                 and value['original_array_dtypes'] == 'not_recorded'
                 and value['limitation'] == LIMITATION, 'geometry policy')
        rows = value['pairs']
        _require(type(rows) is list and len(rows) == 3, 'three paired seeds required')
        seeds = set()
        for row in rows:
            _require(type(row) is dict and set(row) == _PAIR_FIELDS, 'pair fields')
            seed = row['seed']
            _require(type(seed) is int and 0 <= seed < 2**32 and seed not in seeds, 'paired seed')
            seeds.add(seed)
            for role in ('parent', 'child'):
                status = row[role + '_status']
                _require(type(status) is str and status in _DRAW_STATUSES, 'draw status')
                raw, exact, geometry = [row[role + key] for key in
                    ('_raw_sum', '_certified_sum', '_geometry_sha256')]
                if status == 'valid':
                    _require(_finite(raw) and _exact(exact) and _digest(geometry), 'valid draw facts')
                else:
                    _require(raw is None and exact is None and geometry is None, 'failed draw carries values')
            valid = row['parent_status'] == row['child_status'] == 'valid'
            scores = (row['parent_raw_sum'] == row['child_raw_sum']
                      and row['parent_certified_sum'] == row['child_certified_sum']) if valid else None
            geometry = row['parent_geometry_sha256'] == row['child_geometry_sha256'] if valid else None
            _require(row['scores_equal'] is scores and row['geometry_equal'] is geometry, 'pair comparisons')
            _require(not geometry or scores is True, 'same geometry with different derived sums')
        _require(type(value['status']) is str and value['status'] in _STATUSES
                 and value['status'] == _status(rows), 'derived status')
        units = value['edited_units']
        _require(type(units) is list and len(units) <= MAX_UNITS
                 and all(type(unit) is str and 0 < len(unit) <= 160 for unit in units)
                 and len(set(units)) == len(units), 'edited units')
        _require(type(value['edited_units_available']) is bool
                 and type(value['edited_units_omitted']) is int and value['edited_units_omitted'] >= 0,
                 'edited unit coverage')
        _require(value['edited_units_available'] or (not units and value['edited_units_omitted'] == 0),
                 'unavailable edited units')
        _require(value['record_sha256'] == _sha({key: item for key, item in value.items()
                    if key != 'record_sha256'}), 'record digest differs')
        return value
    except (KeyError, TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError('observed paired: malformed record') from exc


def _geometry(receipt):
    if receipt is None or receipt['status'] != 'valid':
        return None
    capture = receipt['capture']
    # Capture is re-derived by validate_comparison. Its original numpy dtypes
    # are not retained. Preserve ordering and signed zero in the saved values.
    return _sha({key: capture[key] for key in ('centers', 'radii')})


def paired_record(group):
    """Project one validated, complete or incomplete paired comparison."""
    group = validate_comparison(group)
    rows = []
    for pair, derived in zip(group['pairs'], group['derived']['pairs'], strict=True):
        parent, child = pair['comparator'], pair['challenger']
        a, b = _geometry(parent), _geometry(child)
        valid = derived['comparator_status'] == derived['challenger_status'] == 'valid'
        rows.append(dict(seed=pair['seed'], parent_status=derived['comparator_status'],
            child_status=derived['challenger_status'], parent_raw_sum=derived['comparator_raw'],
            child_raw_sum=derived['challenger_raw'], parent_certified_sum=derived['comparator_certified'],
            child_certified_sum=derived['challenger_certified'],
            scores_equal=(derived['raw_delta'] == 0 and derived['certified_delta']['numerator'] == 0)
                         if valid else None,
            geometry_equal=(a == b) if valid else None, parent_geometry_sha256=a, child_geometry_sha256=b))
    units, available, omitted = [], False, 0
    if group.get('source_pair') is not None:
        audit = build_change_audit(group['source_pair']['comparator'], group['source_pair']['challenger'])
        available = audit['comparison_available']
        if available:
            edited = audit['direct_edited_unit_ids']
            units = [unit for unit in edited if len(unit) <= 160][:MAX_UNITS]
            omitted = len(edited) - len(units)
    identity = {key: group[key] for key in ('schema', 'comparator_source_sha256',
        'challenger_source_sha256', 'adapter_identity', 'limits', 'task_metric')}
    identity['seeds'] = [row['seed'] for row in rows]
    value = dict(schema=SCHEMA, id=group['id'], group_id=group['id'],
        parent_source_sha256=group['comparator_source_sha256'],
        child_source_sha256=group['challenger_source_sha256'], parent_role='comparator', scope='paired',
        status=_status(rows), group_sha256=_sha(group), comparison_identity_sha256=_sha(identity),
        pairs=rows, geometry_representation=GEOMETRY_REPRESENTATION, original_array_dtypes='not_recorded',
        limitation=LIMITATION, edited_units=units, edited_units_available=available, edited_units_omitted=omitted)
    value['record_sha256'] = _sha(value)
    return validate_record(value)
