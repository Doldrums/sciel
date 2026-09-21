"""Exact measurement fields, source catalogs and versioned projection schemas."""

import hashlib
import json


PROJECTION_SCHEMA = 'sciel-tool-result-projection/1'

COMPARISON_PROJECTION_SCHEMA = 'sciel-tool-result-projection/2'

ROW_PROJECTION_SCHEMA = 'sciel-tool-result-projection/3'

SPARSE_PROJECTION_SCHEMA = 'sciel-tool-result-projection/4'

PROJECTION_SCHEMAS = frozenset((PROJECTION_SCHEMA, COMPARISON_PROJECTION_SCHEMA,
                              ROW_PROJECTION_SCHEMA, SPARSE_PROJECTION_SCHEMA))

_ROW_SCOPE = ('Exact whole tool-reported observation rows in source path/index order; not verified scientific claims. '
    'Counts cover eligible arrays not already delivered whole in fields. Array and row hashes bind the receipt. '
    'Omitted rows remain there; atomic field counts still count '
    'each supplementary array as an omitted field.')

_COMPARISON_PATH = '/output/observations/comparison_results'

_CORE = ('arm_id', 'seed', 'cpu_seconds', 'best_result', 'conditions')

_COMPARISON_SCOPE = ('Exact tool-reported rows, grouped by reported seed; not verified adherence or benefit. '
    'Omitted values remain in the original receipt. Generic field counts treat the array as one omitted field.')

_MAX_OMITTED_PATHS = 16

_STATE_CONTAINERS = frozenset(('continuation_state', 'saved_state'))

_COMPACT_SCALAR_CHARS = 256

def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)

_OMITTED = _json({'schema': PROJECTION_SCHEMA, 'omitted': True})

MIN_CONTEXT_CHARS = len(_OMITTED)

def _fields(item):
    """Iteratively enumerate exact atomic fields; never truncate a value."""
    pending = [((), item)]
    while pending:
        parts, value = pending.pop()
        if type(value) is dict and value:
            pending.extend((parts + (key,), child) for key, child in value.items())
            continue
        path = ''.join('/' + key.replace('~', '~0').replace('/', '~1') for key in parts)
        observation = parts[:2] == ('output', 'observations')
        diagnostic = parts[:2] == ('output', 'diagnostics') or parts[:1] == ('diagnostic',)
        field = {'path': path, 'value': value}
        yield (0 if observation else 2 if diagnostic else 1,
               len(_json(field)), path, observation, field)

def projection_fields(value):
    """Decode exact atomic fields; /4 tree dictionaries are only path containers.

    Lists (including numerical vectors) and empty containers remain whole values.
    A nonempty mapping is never advertised as a complete original subtree.
    """
    if value.get('schema') != SPARSE_PROJECTION_SCHEMA:
        return value.get('fields', [])
    tree = value.get('fields')
    if type(tree) is not dict or not tree:
        raise ValueError('Invalid sparse measurement tree')
    return [row[4] for row in sorted(_fields(tree), key=lambda row: row[2])]

def projection_values(value):
    """Exact visible values, including immutable paired cores and whole rows."""
    if value.get('schema') not in PROJECTION_SCHEMAS:
        return [('', value)]
    values = [(field['path'], field['value']) for field in projection_fields(value)]
    for array in value.get('observation_rows', {}).get('arrays', []):
        values.extend((array['path'] + '/' + str(row['index']), row['value'])
                      for row in array['rows'])
    comparison = value.get('comparison_projection') or {}
    for group in comparison.get('groups', []):
        for row in group['rows']:
            values.extend((row['path'] + '/' + key, item)
                          for key, item in row['values'].items())
    return values

def preserves_measurements(previous, proposed):
    """Typed exact coverage; fuller containers may cover earlier atomic values."""
    known, pending = {}, projection_values(proposed)
    while pending:
        path, value = pending.pop()
        known[path] = _json(value)
        if type(value) is dict:
            pending.extend((path + '/' + key.replace('~', '~0').replace('/', '~1'), item)
                           for key, item in value.items())
        elif type(value) is list:
            pending.extend((path + '/' + str(index), item) for index, item in enumerate(value))
    return all(known.get(path) == _json(value) for path, value in projection_values(previous))

def _hash(value):
    return hashlib.sha256(_json(value).encode('ascii')).hexdigest()

def _comparison_catalog(item):
    output = item.get('output')
    observations = output.get('observations') if type(output) is dict else None
    rows = observations.get('comparison_results') if type(observations) is dict else None
    if type(rows) is not list or not rows:
        return None
    by_seed = {}
    for index, row in enumerate(rows):
        if type(row) is dict and type(row.get('seed')) is int:
            by_seed.setdefault(row['seed'], []).append((index, row))
    groups = []
    for seed, members in sorted(by_seed.items()):
        # Never keep only the cheap arm of one reported seed. A malformed or
        # oversized member makes its whole seed group unavailable at this budget.
        if any(not all(key in row for key in _CORE) or type(row['arm_id']) is not str
               for _, row in members):
            continue
        projected = []
        for index, row in members:
            omitted = {key: {'sha256': _hash(value), 'json_chars': len(_json(value))}
                       for key, value in sorted(row.items()) if key not in _CORE}
            projected.append({'index': index, 'path': _COMPARISON_PATH + '/' + str(index),
                'original_sha256': _hash(row),
                'values': {key: row[key] for key in _CORE}, 'omitted_values': omitted})
        groups.append({'seed': seed, 'rows': projected})
    return rows, groups

def _comparison_metadata(rows, groups):
    shown = sum(len(group['rows']) for group in groups)
    return {'path': _COMPARISON_PATH, 'original_sha256': _hash(rows),
        'total_rows': len(rows), 'shown_rows': shown, 'omitted_rows': len(rows) - shown,
        'groups': groups, 'scope': _COMPARISON_SCOPE}

def _state_path(path):
    return any(part in _STATE_CONTAINERS for part in path.split('/')[3:])

def _row_catalog(item):
    """Only arrays of objects are records; numerical/mixed arrays stay atomic.

    Do not descend inside a record or reinterpret comparison_results: a record
    must remain whole and standard paired comparisons keep their seed groups.
    """
    arrays = []
    for _, _, path, observation, field in sorted(_fields(item), key=lambda row: row[2]):
        rows = field['value']
        if (not observation or path == _COMPARISON_PATH or _state_path(path) or type(rows) is not list
                or not rows or any(type(row) is not dict for row in rows)):
            continue
        arrays.append({'path': path, 'original_sha256': _hash(rows), 'total_rows': len(rows),
            'rows': [{'index': index, 'path': path + '/' + str(index),
                      'original_sha256': _hash(row), 'value': row}
                     for index, row in enumerate(rows)]})
    return arrays

def _row_metadata(catalog, selected):
    arrays = []
    for array in catalog:
        rows = selected.get(array['path'], [])
        if rows:
            arrays.append({**array, 'rows': rows, 'shown_rows': len(rows),
                           'omitted_rows': array['total_rows'] - len(rows)})
    total = sum(array['total_rows'] for array in catalog)
    shown = sum(len(array['rows']) for array in arrays)
    return {'total_arrays': len(catalog), 'shown_arrays': len(arrays),
        'omitted_arrays': len(catalog) - len(arrays), 'total_rows': total,
        'shown_rows': shown, 'omitted_rows': total - shown,
        'arrays': arrays, 'scope': _ROW_SCOPE}
