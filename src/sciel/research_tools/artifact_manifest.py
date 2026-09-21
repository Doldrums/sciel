"""Bounded authoring metadata for exact saved artifacts; never execute data.

The projection describes syntax and structure, not validity or observations.
Pointers address unchanged tool inputs; decoded structure is relative to the
JSON value inside a string, not additional fields in that input.
"""
from __future__ import annotations

import hashlib
import json
import math

SCHEMA = 'sciel-artifact-manifest/1'
MAX_PARSE_BYTES = 16384
MAX_PARSE_DEPTH = 16
MAX_STRUCTURE_DEPTH = 4
MAX_STRUCTURE_NODES = 64
MAX_SHAPE_NODES = 512


def _json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      sort_keys=True, separators=(',', ':'))


def _type(value):
    return {dict: 'object', list: 'array', str: 'string', bool: 'boolean',
            int: 'integer', float: 'number', type(None): 'null'}[type(value)]


def _token(key):
    return key.replace('~', '~0').replace('/', '~1')


def _shape(value):
    remaining = MAX_SHAPE_NODES

    def visit(item):
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise OverflowError
        if type(item) is not list:
            return []
        if not item:
            return [0]
        first = visit(item[0])
        for child in item[1:]:
            if visit(child) != first:
                raise ValueError
        return [len(item)] + first

    try:
        return {'shape': visit(value), 'shape_status': 'verified'}
    except OverflowError:
        return {'shape_status': 'omitted_node_limit'}
    except (ValueError, RecursionError):
        return {'shape_status': 'ragged_or_mixed'}


def _structure(value):
    remaining = MAX_STRUCTURE_NODES

    def visit(item, depth):
        nonlocal remaining
        remaining -= 1
        result = {'type': _type(item)}
        if type(item) is list:
            result.update(length=len(item), **_shape(item))
        elif type(item) is dict:
            result.update(fields={}, fields_omitted=len(item))
            if depth >= MAX_STRUCTURE_DEPTH:
                result['omission_reason'] = 'depth_limit'
            else:
                for key in sorted(item):
                    if remaining <= 0:
                        result['omission_reason'] = 'node_limit'
                        break
                    result['fields'][key] = visit(item[key], depth + 1)
                    result['fields_omitted'] -= 1
        return result

    return visit(value, 0)


def _strict_json(text):
    # A lexical depth cap avoids handing arbitrarily nested strings to the
    # decoder. It ignores braces inside strings; json.loads checks the syntax.
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '[{':
            depth += 1
            if depth > MAX_PARSE_DEPTH:
                raise OverflowError
        elif char in ']}':
            depth -= 1

    def reject(_):
        raise ValueError('non-finite JSON')

    def number(text):
        value = float(text)
        if not math.isfinite(value):
            reject(text)
        return value

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('ambiguous duplicate JSON key')
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject, parse_float=number,
                      object_pairs_hook=pairs)


def _entry(key, value, pointer):
    result = {'key': key, 'pointer': pointer + '/' + _token(key),
              'outer_type': _type(value)}
    data = (value.encode('utf-8', errors='surrogatepass') if type(value) is str
            else _json(value).encode('ascii'))
    result.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                  hash_encoding='utf8_string' if type(value) is str else 'canonical_json_ascii')
    if type(value) is str:
        if len(data) > MAX_PARSE_BYTES:
            result['json_syntax'] = 'omitted_byte_limit'
            return result
        try:
            decoded = _strict_json(value)
        except OverflowError:
            result['json_syntax'] = 'omitted_depth_limit'
            return result
        except (ValueError, RecursionError):
            result['json_syntax'] = 'not_strict_json'
            return result
        result.update(json_syntax='valid', structure_scope='json_decoded_value')
    else:
        decoded = value
        result.update(json_syntax='not_a_string', structure_scope='outer_value')
    result['structure'] = _structure(decoded)
    if type(decoded) is dict and type(decoded.get('schema')) is str:
        schema = decoded['schema']
        if len(_json(schema)) <= 160:
            result['declared_schema'] = schema
        else:
            result['declared_schema_omitted'] = True
    return result


def artifact_manifest(saved, pointer, budget=0, *, inline_reference=True):
    """Return whole metadata within budget, or the mandatory omission header.

    Budget zero deliberately returns only the small header. The caller reserves
    this header with its required input references before sharing spare prompt
    space equally between roles. Every missing entry/detail is explicit.
    """
    available = saved is not None and saved['status'] == 'available'
    artifacts = saved['record']['result']['artifacts'] if available else {}
    reference = saved.get('reference') if saved else None
    result = {'entries': [], 'entries_omitted': len(artifacts)}
    if inline_reference:
        result.update(schema=SCHEMA, status='available' if available else 'unavailable',
            record_sha256=reference['sha256'] if reference else None, payload_pointer=pointer)
    # Inside the frozen target view, the enclosing role already supplies the
    # exact saved-record SHA, availability, and access_paths.saved_artifacts.
    # Do not spend a small prompt budget repeating those same references.
    if not available:
        return result
    for key in sorted(artifacts):
        # First admit a complete identity/type header for each artifact, then
        # divide remaining space fairly for structure; one huge field cannot
        # spend the budget intended for every other field.
        brief = {'key': key, 'outer_type': _type(artifacts[key]), 'details_omitted': True}
        trial = {**result, 'entries': result['entries'] + [brief],
                 'entries_omitted': result['entries_omitted'] - 1}
        if len(_json(trial)) <= budget:
            result = trial
    if not result['entries']:
        return result
    extra = max(0, budget - len(_json(result))) // len(result['entries'])
    for index, brief in enumerate(result['entries']):
        full = _entry(brief['key'], artifacts[brief['key']], pointer)
        allowance = len(_json(brief)) + extra
        structure = full.get('structure')
        if len(_json(full)) > allowance and structure and structure['type']=='object':
            # Preserve exact root names when detailed nested types do not fit.
            # This is a labeled shallower view, never a partial scalar value.
            full['structure'] = {'type':'object','keys':list(structure['fields']),
                'fields_omitted':structure['fields_omitted'],'nested_details_omitted':True}
        if len(_json(full)) > allowance:
            # Keep key/type and syntax information before optional deep shape.
            full.pop('structure', None)
            full['structure_omitted'] = True
        if len(_json(full)) <= allowance:
            result['entries'][index] = full
    return result
