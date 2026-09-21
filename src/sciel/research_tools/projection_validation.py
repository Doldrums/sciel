"""Authenticate projected measurement values and omission accounting."""

from .projection_data import (
    COMPARISON_PROJECTION_SCHEMA,
    PROJECTION_SCHEMA,
    PROJECTION_SCHEMAS,
    ROW_PROJECTION_SCHEMA,
    SPARSE_PROJECTION_SCHEMA,
    _COMPARISON_PATH,
    _comparison_catalog,
    _comparison_metadata,
    _fields,
    _hash,
    _json,
    _row_catalog,
    _row_metadata,
    projection_fields,
)


def verify_projection(value, original):
    """Authenticate full output, /1 fields, /2 pairs, /3 rows and /4 sparse views.

    Pure saved-JSON verification, reusable by observers without Engine imports.
    A successful check authenticates provenance and exact values, not claims.
    """
    if _json(value) == _json(original):
        return
    if type(value) is not dict:
        raise ValueError('Invalid measurement excerpt')
    if value.get('schema') == SPARSE_PROJECTION_SCHEMA:
        # Expansion preserves exact leaf types and escaped JSON-pointer paths.
        # The existing verifier then authenticates every leaf, omission count,
        # complete paired group and whole row against the immutable receipt.
        fields = projection_fields(value)
        schema = (ROW_PROJECTION_SCHEMA if 'observation_rows' in value
                  else COMPARISON_PROJECTION_SCHEMA if 'comparison_projection' in value
                  else PROJECTION_SCHEMA)
        value = {**value, 'schema': schema, 'fields': fields}
    candidates = list(_fields(original))
    fields = {row[2]: (row[4]['value'], row[3]) for row in candidates}
    total_observations = sum(flag for _, flag in fields.values())
    if value.get('omitted') is True:
        if (set(value) - {'schema', 'omitted', 'omitted_fields', 'omitted_observation_fields'}
                or value.get('schema') not in (None, PROJECTION_SCHEMA)
                or type(value.get('omitted_fields', len(fields))) is not int
                or value.get('omitted_fields', len(fields)) != len(fields)
                or type(value.get('omitted_observation_fields', total_observations)) is not int
                or value.get('omitted_observation_fields', total_observations) != total_observations):
            raise ValueError('Omitted measurement fields differ')
        return
    schema = value.get('schema')
    if type(schema) is not str:
        raise ValueError('Invalid measurement projection schema')
    expected_keys = {'schema', 'original_sha256', 'fields', 'observations_state',
        'omitted_fields', 'omitted_observation_fields', 'omitted_paths', 'omitted_paths_omitted'}
    if schema in (COMPARISON_PROJECTION_SCHEMA, ROW_PROJECTION_SCHEMA):
        expected_keys.add('comparison_projection')
    if schema == ROW_PROJECTION_SCHEMA:
        expected_keys.add('observation_rows')
    if (schema not in PROJECTION_SCHEMAS or set(value) != expected_keys
            or value.get('original_sha256') != _hash(original)
            or type(value.get('fields')) is not list):
        raise ValueError('Measurement projection identity differs from receipt')
    shown = set()
    for field in value['fields']:
        if type(field) is not dict or set(field) != {'path', 'value'} or type(field['path']) is not str:
            raise ValueError('Invalid measurement projection field')
        path = field['path']
        if path in shown or path not in fields or _json(field['value']) != _json(fields[path][0]):
            raise ValueError('Measurement projection changes a saved value')
        shown.add(path)
    observed = sum(fields[path][1] for path in shown)
    output = original.get('output')
    observations = output.get('observations') if type(output) is dict else None
    state = ('not_present' if type(observations) is not dict else 'originally_empty' if not observations
        else 'all_omitted' if not observed else 'complete' if observed == total_observations else 'partial')
    if schema == COMPARISON_PROJECTION_SCHEMA or (schema == ROW_PROJECTION_SCHEMA
            and value['comparison_projection'] is not None):
        catalog = _comparison_catalog(original)
        comp = value['comparison_projection']
        if (catalog is None or type(comp) is not dict or type(comp.get('groups')) is not list
                or not comp['groups'] or _COMPARISON_PATH in shown):
            raise ValueError('Invalid paired measurement projection')
        rows, available = catalog
        by_seed = {group['seed']: group for group in available}
        prior = None
        for group in comp['groups']:
            if (type(group) is not dict or type(group.get('seed')) is not int
                    or prior is not None and group['seed'] <= prior
                    or group['seed'] not in by_seed
                    or _json(group) != _json(by_seed[group['seed']])):
                raise ValueError('Paired measurement group differs from receipt')
            prior = group['seed']
        if _json(comp) != _json(_comparison_metadata(rows, comp['groups'])):
            raise ValueError('Paired measurement coverage differs from receipt')
        state = 'partial'  # Core rows are not the complete saved observations.
    if schema == ROW_PROJECTION_SCHEMA:
        catalog = [array for array in _row_catalog(original) if array['path'] not in shown]
        actual = value['observation_rows']
        if type(actual) is not dict or type(actual.get('arrays')) is not list or not actual['arrays']:
            raise ValueError('Invalid whole-row measurement projection')
        available = {array['path']: array for array in catalog}
        selected = {}
        prior_path = None
        for array in actual['arrays']:
            path = array.get('path') if type(array) is dict else None
            if (type(path) is not str or path not in available or path in shown
                    or prior_path is not None and path <= prior_path
                    or type(array.get('rows')) is not list or not array['rows']):
                raise ValueError('Invalid observation array identity')
            prior_path = path
            prior_index = -1
            for row in array['rows']:
                index = row.get('index') if type(row) is dict else None
                if (type(index) is not int or index <= prior_index
                        or index >= available[path]['total_rows']
                        or _json(row) != _json(available[path]['rows'][index])):
                    raise ValueError('Observation row differs from receipt')
                prior_index = index
            selected[path] = array['rows']
        if _json(actual) != _json(_row_metadata(catalog, selected)):
            raise ValueError('Observation row coverage differs from receipt')
        state = 'partial'
    omitted = set(fields) - shown
    paths = value.get('omitted_paths')
    counts = {'omitted_fields': len(omitted), 'omitted_observation_fields': total_observations-observed,
              'omitted_paths_omitted': len(omitted)-len(paths) if type(paths) is list else None}
    if (value.get('observations_state') != state or type(paths) is not list
            or any(type(path) is not str for path in paths) or len(paths) != len(set(paths))
            or not set(paths) <= omitted
            or any(type(value.get(key)) is not int or value[key] != count for key, count in counts.items())):
        raise ValueError('Measurement projection accounting differs from receipt')
