"""Lossless measurement dictionaries at explicitly trusted JSON paths.

References are authorized only by external bindings, never inferred from field
names inside opaque task data. JSON types and canonical numeric representations
are preserved. This is a data codec, not a claim about model comprehension.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import re


SCHEMA = 'sciel-measurement-catalog/1'
_MISSING = object()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _clone(value, active=None):
    kind = type(value)
    if value is None or kind in (str, bool, int):
        return value
    if kind is float:
        _require(math.isfinite(value), 'nonfinite JSON number')
        return value
    _require(kind in (dict, list), 'unsupported JSON value type')
    active = set() if active is None else active
    ident = id(value)
    _require(ident not in active, 'cyclic JSON value')
    active.add(ident)
    try:
        if kind is list:
            return [_clone(item, active) for item in value]
        _require(all(type(key) is str for key in value), 'JSON object keys must be strings')
        return {key: _clone(item, active) for key, item in value.items()}
    finally:
        active.remove(ident)


def _dump(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(',', ':'))


def canonical_json(value):
    """Strict deterministic JSON; preserve int/float and the sign of float zero."""
    try:
        return _dump(_clone(value))
    except RecursionError:
        raise ValueError('JSON nesting exceeds codec capacity') from None


def canonical_sha(value):
    return hashlib.sha256(canonical_json(value).encode('ascii')).hexdigest()


def _hash(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _paths(paths):
    _require(type(paths) in (list, tuple), 'paths must be a sequence')
    normalized = []
    for path in paths:
        _require(type(path) in (list, tuple), 'each path must be a sequence')
        _require(all(type(part) is str or (type(part) is int and part >= 0)
                     for part in path), 'invalid path component')
        normalized.append(tuple(path))
    _require(len(normalized) == len(set(normalized)), 'duplicate binding path')
    for index, path in enumerate(normalized):
        for other in normalized[index + 1:]:
            size = min(len(path), len(other))
            _require(path[:size] != other[:size], 'overlapping binding paths')
    return sorted(normalized, key=lambda path: _dump(list(path)))


def _get(value, path, *, optional):
    current = value
    for part in path:
        if current is None and optional:
            return _MISSING
        if type(current) is dict:
            _require(type(part) is str, 'object path requires a string key')
            if part not in current:
                _require(optional, 'binding path is missing')
                return _MISSING
            current = current[part]
        elif type(current) is list:
            _require(type(part) is int, 'array path requires an integer index')
            if part >= len(current):
                _require(optional, 'binding path is missing')
                return _MISSING
            current = current[part]
        else:
            raise ValueError('binding path traverses a non-container value')
    return current


def _set(value, path, item):
    if not path:
        return item
    parent = _get(value, path[:-1], optional=False)
    parent[path[-1]] = item
    return value


def pack_measurements(value, paths=()):
    """Factor repeated dicts only at supplied paths when canonical bytes shrink.

    Missing/null paths stay missing/null. A non-null scalar or list at a marked
    target is an error. Overlaps are rejected even when a path would be missing.
    A stable greedy SHA order considers full envelope/catalog/binding overhead,
    relative to the same envelope without references. Callers separately decide
    whether using an envelope is preferable to their original inline wire form.
    No objects in ``value`` or ``paths`` are mutated or retained by reference.
    """
    try:
        return _pack(value, paths)
    except RecursionError:
        raise ValueError('JSON nesting exceeds codec capacity') from None


def _pack(value, paths):
    original = _clone(value)
    normalized = _paths(paths)
    envelope = {'schema': SCHEMA, 'original_sha256': canonical_sha(original),
                'payload': _clone(original), 'catalog': {}, 'bindings': []}
    groups = defaultdict(list)
    values = {}
    for path in normalized:
        item = _get(original, path, optional=True)
        if item is _MISSING or item is None:
            continue
        _require(type(item) is dict, 'marked measurement must be an object or null')
        digest = canonical_sha(item)
        if digest in values:
            _require(_dump(item) == _dump(values[digest]), 'catalog hash collision')
        values[digest] = item
        groups[digest].append(path)
    for digest in sorted(groups):
        if len(groups[digest]) < 2:
            continue
        trial = _clone(envelope)
        trial['catalog'][digest] = _clone(values[digest])
        for path in groups[digest]:
            trial['payload'] = _set(trial['payload'], path, {'catalog_ref': digest})
            trial['bindings'].append({'path': list(path), 'sha256': digest})
        trial['bindings'].sort(key=lambda binding: _dump(binding['path']))
        if len(_dump(trial)) < len(_dump(envelope)):
            envelope = trial
    return envelope


def expand_measurements(envelope):
    """Validate the complete catalog and restore exact JSON; never skip a ref."""
    try:
        return _expand(envelope)
    except RecursionError:
        raise ValueError('JSON nesting exceeds codec capacity') from None


def _expand(envelope):
    saved = _clone(envelope)
    _require(type(saved) is dict and set(saved) == {
        'schema', 'original_sha256', 'payload', 'catalog', 'bindings'},
        'invalid measurement envelope schema')
    _require(saved['schema'] == SCHEMA and _hash(saved['original_sha256']),
             'invalid measurement envelope identity')
    catalog, bindings = saved['catalog'], saved['bindings']
    _require(type(catalog) is dict and type(bindings) is list,
             'invalid catalog or bindings type')
    for digest, item in catalog.items():
        _require(_hash(digest) and type(item) is dict and canonical_sha(item) == digest,
                 'catalog value/hash mismatch')
    for binding in bindings:
        _require(type(binding) is dict and set(binding) == {'path', 'sha256'}
                 and type(binding['path']) is list and _hash(binding['sha256']),
                 'invalid measurement binding')
    _paths([binding['path'] for binding in bindings])
    used = {binding['sha256'] for binding in bindings}
    _require(used == set(catalog), 'dangling or unused catalog entry')
    payload = saved['payload']
    # Check every placeholder before changing any part of the decoded payload.
    for binding in bindings:
        actual = _get(payload, binding['path'], optional=False)
        _require(type(actual) is dict and set(actual) == {'catalog_ref'}
                 and actual['catalog_ref'] == binding['sha256'], 'bound reference differs')
    for binding in bindings:
        payload = _set(payload, binding['path'], _clone(catalog[binding['sha256']]))
    _require(canonical_sha(payload) == saved['original_sha256'],
             'expanded measurement SHA mismatch')
    return payload
