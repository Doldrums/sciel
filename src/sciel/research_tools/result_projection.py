"""Fit saved tool measurements to a context budget without changing values."""

import hashlib
import json
from .projection_data import (
    COMPARISON_PROJECTION_SCHEMA,
    MIN_CONTEXT_CHARS,
    PROJECTION_SCHEMA,
    PROJECTION_SCHEMAS,
    ROW_PROJECTION_SCHEMA,
    SPARSE_PROJECTION_SCHEMA,
    _COMPACT_SCALAR_CHARS,
    _COMPARISON_PATH,
    _COMPARISON_SCOPE,
    _CORE,
    _MAX_OMITTED_PATHS,
    _OMITTED,
    _ROW_SCOPE,
    _STATE_CONTAINERS,
    _comparison_catalog,
    _comparison_metadata,
    _fields,
    _hash,
    _json,
    _row_catalog,
    _row_metadata,
    _state_path,
    preserves_measurements,
    projection_fields,
    projection_values,
)
from .projection_validation import (
    verify_projection,
)


def _render(value, sparse=False):
    if not sparse or not value.get('fields'):
        return _json(value)
    tree = {}
    for field in value['fields']:
        parts = [part.replace('~1', '/').replace('~0', '~')
                 for part in field['path'].split('/')[1:]]
        cursor = tree
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = field['value']
    return _json({**value, 'schema': SPARSE_PROJECTION_SCHEMA, 'fields': tree})

def _legacy_context(item, limit, *, sparse=False):
    """Return valid ASCII JSON of at most ``limit`` characters.

The full wrapper is returned unchanged semantically when it fits. Otherwise
small observation fields have priority, including fields in nested dictionaries;
metadata follows, then diagnostics. Large arrays and strings are omitted whole.
No selected value is interpreted, rounded, reworded, or mutated.

Below the exported minimum an explicit omission marker cannot fit, so callers
must omit the enclosing card rather than cut this string. At small supported
limits a versioned whole-result omission marker is returned. More detailed
omission counts and paths are included only when their own JSON fits.
"""
    if type(limit) is not int or limit < MIN_CONTEXT_CHARS:
        raise ValueError('result context limit must be an integer >= ' + str(MIN_CONTEXT_CHARS))
    if type(item) is not dict:
        raise ValueError('result context must be a JSON object')
    original = _json(item)
    if len(original) <= limit:
        return original

    original_sha256 = hashlib.sha256(original.encode('ascii')).hexdigest()
    candidates = sorted(_fields(item), key=lambda row: row[:3])
    total = len(candidates)
    observation_total = sum(row[3] for row in candidates)
    observations = item.get('output', {}).get('observations') if type(item.get('output')) is dict else None
    selected = []
    selected_paths = set()
    shown_observations = 0

    def render(paths=()):
        if type(observations) is not dict:
            state = 'not_present'
        elif not observations:
            state = 'originally_empty'
        elif not shown_observations:
            state = 'all_omitted'
        elif shown_observations == observation_total:
            state = 'complete'
        else:
            state = 'partial'
        omitted = total - len(selected)
        return _render({'schema': PROJECTION_SCHEMA,
            'original_sha256': original_sha256,
            'fields': selected, 'observations_state': state,
            'omitted_fields': omitted,
            'omitted_observation_fields': observation_total - shown_observations,
            'omitted_paths': list(paths), 'omitted_paths_omitted': omitted - len(paths)}, sparse)

    if len(render()) > limit:
        marker = {'schema': PROJECTION_SCHEMA, 'omitted': True}
        for key, value in (('omitted_fields', total),
                           ('omitted_observation_fields', observation_total)):
            proposed = {**marker, key: value}
            if len(_json(proposed)) <= limit:
                marker = proposed
        return _json(marker)

    for _, size, path, observation, field in candidates:
        if size > limit:
            continue
        selected.append(field)
        shown_observations += observation
        if len(render()) <= limit:
            selected_paths.add(path)
        else:
            selected.pop()
            shown_observations -= observation

    # Path names are explanatory overhead; they cannot evict selected data.
    omitted_paths = sorted((row[2] for row in candidates if row[2] not in selected_paths),
                           key=lambda path: (len(_json(path)), path))
    paths = []
    for path in omitted_paths[:_MAX_OMITTED_PATHS]:
        if len(render(paths + [path])) <= limit:
            paths.append(path)
        else:
            break
    return render(paths)

def _paired_context(item, limit, *, sparse=False):
    """Exact bounded /1 projection, with /2 paired rows before bulky saved state.

    /2 never modifies the saved output. Each group contains every original row
    for one reported seed, including duplicate arm IDs if the tool reported them.
    It preserves all five core values wholesale, even invalid/null values. This
    proves delivery, not completeness of the declared experiment. Oversized or
    malformed groups are omitted as a whole. The original array remains an
    omitted atomic field; supplementary row groups cannot claim full output.
    """
    legacy = _legacy_context(item, limit, sparse=sparse)
    if len(_json(item)) <= limit:
        return legacy
    legacy_value = json.loads(legacy)
    if any(field.get('path') == _COMPARISON_PATH for field in projection_fields(legacy_value)):
        return legacy  # Already retains the complete array, including saved state.
    catalog = _comparison_catalog(item)
    if catalog is None:
        return legacy
    rows, groups = catalog
    if not groups:
        return legacy
    candidates = sorted(_fields(item), key=lambda row: row[:3])
    observation_total = sum(row[3] for row in candidates)
    selected, selected_groups, selected_paths = [], [], set()

    def render(paths=(_COMPARISON_PATH,)):
        observed = sum(path.startswith('/output/observations/') for path in selected_paths)
        omitted = len(candidates) - len(selected)
        return _render({'schema': COMPARISON_PROJECTION_SCHEMA, 'original_sha256': _hash(item),
            'fields': selected, 'observations_state': 'partial',
            'omitted_fields': omitted, 'omitted_observation_fields': observation_total - observed,
            'omitted_paths': list(paths), 'omitted_paths_omitted': omitted - len(paths),
            'comparison_projection': _comparison_metadata(rows, selected_groups)}, sparse)

    # Keep whole matched-seed groups before optional audits, geometry, prose,
    # metadata and diagnostics. Original row indices retain the source ordering.
    for group in groups:
        selected_groups.append(group)
        if len(render()) > limit:
            selected_groups.pop()
    if not selected_groups:
        return legacy
    for _, size, path, observation, field in candidates:
        if path == _COMPARISON_PATH or size > limit:
            continue
        selected.append(field)
        selected_paths.add(path)
        if len(render()) > limit:
            selected.pop()
            selected_paths.remove(path)
    omitted = sorted((row[2] for row in candidates if row[2] not in selected_paths),
                     key=lambda path: (len(_json(path)), path))
    paths = [_COMPARISON_PATH]
    for path in (path for path in omitted if path != _COMPARISON_PATH):
        if len(paths) >= _MAX_OMITTED_PATHS:
            break
        if len(render(paths + [path])) > limit:
            break
        paths.append(path)
    return render(paths)

def _context(item, limit, *, sparse=False):
    """Retain exact complete results, or bounded authenticated observations.

    Paired seed groups take precedence. Other observation arrays of dictionary
    records receive one whole source-order row per array, then compact scalar
    summaries and more rows, before optional state/source. No sorting by
    reported results, rounding, or inferred semantics.
    Historical /1 and /2 formats remain unchanged when no generic row is needed
    or can fit; all arrays still count as atomic fields in omission accounting.
    """
    previous = _paired_context(item, limit, sparse=sparse)  # validates input and legacy budget
    if len(_json(item)) <= limit:
        return previous
    catalog = _row_catalog(item)
    if not catalog:
        return previous
    previous_value = json.loads(previous)
    complete = {field['path'] for field in projection_fields(previous_value)}
    if all(array['path'] in complete for array in catalog):
        return previous
    candidates = sorted(_fields(item), key=lambda row: row[:3])
    total_observations = sum(row[3] for row in candidates)
    paired = _comparison_catalog(item)
    protected = {array['path'] for array in catalog if array['path'] in complete}
    # Increasing a card's allowance must not replace already delivered compact
    # measurements with row-index metadata. Small vectors are atomic summaries
    # too; protecting them is based on structure/size, never their values.
    protected.update(path for _, size, path, observation, _ in candidates
                     if path in complete and observation and not _state_path(path)
                     and size <= _COMPACT_SCALAR_CHARS)
    if _COMPARISON_PATH in complete:
        protected.add(_COMPARISON_PATH)
    # Supplementary metadata covers only arrays not already delivered whole.
    catalog = [array for array in catalog if array['path'] not in protected]
    selected = [row[4] for row in candidates if row[2] in protected]
    selected_paths, rows = set(protected), {}
    prior_comparison = previous_value.get('comparison_projection')
    groups = list(prior_comparison['groups']) if prior_comparison else []
    array_paths = {array['path'] for array in catalog}
    reserved_paths = ([_COMPARISON_PATH] if groups else []) + [array['path'] for array in catalog]
    reserved_paths = reserved_paths[:_MAX_OMITTED_PATHS]

    def render(paths=None):
        paths = reserved_paths if paths is None else paths
        observed = sum(row[3] for row in candidates if row[2] in selected_paths)
        omitted = len(candidates) - len(selected)
        return _render({'schema': ROW_PROJECTION_SCHEMA, 'original_sha256': _hash(item),
            'fields': selected, 'observations_state': 'partial',
            'omitted_fields': omitted, 'omitted_observation_fields': total_observations - observed,
            'omitted_paths': list(paths), 'omitted_paths_omitted': omitted - len(paths),
            'comparison_projection': _comparison_metadata(paired[0], groups) if groups else None,
            'observation_rows': _row_metadata(catalog, rows)}, sparse)

    # /3 overhead may itself be too expensive: never trade an already
    # delivered paired group or complete array for unrelated generic rows.
    if len(render()) > limit:
        return previous

    def add_row(array, row):
        kept = rows.setdefault(array['path'], [])
        if any(existing['index'] == row['index'] for existing in kept):
            return False
        kept.append(row)
        if len(render()) <= limit:
            return True
        kept.pop()
        return False

    # One whole row per array precedes small exact summaries. This prevents one
    # long row array from consuming all space before other observed records.
    for array in catalog:
        for row in array['rows']:
            if add_row(array, row):
                break
    if not any(rows.values()):
        return previous

    def add_field(candidate):
        _, size, path, _, field = candidate
        if (path in selected_paths or path in array_paths
                or path == _COMPARISON_PATH and groups or size > limit):
            return
        selected.append(field)
        selected_paths.add(path)
        if len(render()) > limit:
            selected.pop()
            selected_paths.remove(path)

    # Preserve compact atomic observations (including short summary vectors)
    # before additional rows. Bulky vectors, echoed source and continuation
    # state cannot occupy this reservation. No value's sign or score is used.
    for candidate in candidates:
        _, size, path, observation, field = candidate
        if observation and not _state_path(path) and size <= _COMPACT_SCALAR_CHARS:
            add_field(candidate)
    for array in catalog:
        for row in array['rows']:
            add_row(array, row)
    for candidate in candidates:
        add_field(candidate)
    omitted = sorted((row[2] for row in candidates if row[2] not in selected_paths),
                     key=lambda path: (len(_json(path)), path))
    paths = list(reserved_paths)
    for path in omitted:
        if path in paths:
            continue
        if len(paths) >= _MAX_OMITTED_PATHS or len(render(paths + [path])) > limit:
            break
        paths.append(path)
    return render(paths)

def success_context(item, limit):
    """Prefer lossless path compression only when it adds authentic coverage.

    The same outcome-agnostic selection runs with two encodings. /4 cannot erase
    any formerly visible atomic value, complete array, paired core or whole row.
    Full-fitting wrappers and historical projections remain unchanged otherwise.
    """
    previous = _context(item, limit)
    if len(_json(item)) <= limit:
        return previous
    proposed = _context(item, limit, sparse=True)
    old, new = json.loads(previous), json.loads(proposed)
    if (new.get('schema') == SPARSE_PROJECTION_SCHEMA
            and preserves_measurements(old, new)
            and not preserves_measurements(new, old)):
        return proposed
    return previous
