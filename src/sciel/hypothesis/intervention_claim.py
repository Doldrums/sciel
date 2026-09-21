"""Audit-only, pre-evaluation intervention declarations for controlled probes.

This wire is separate from HYPOTHESIS_JSON and is not an engine admission gate.
It validates bounded structure and exact parent-unit references, never semantic
truth, completeness of an edit, speedup or a causal explanation. The caller binds
the parent catalog to source bytes and audits the final repaired/grafted source.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re


SCHEMA = 'sciel-intervention-claim/1'
MARKER = 'INTERVENTION_JSON'
MAX_RESPONSE_CHARS = 262144
MAX_BLOCK_CHARS = 8192
MAX_CHANGED_UNITS = 16
MAX_CATALOG_UNITS = 256
MAX_UNIT_ID_CHARS = 256
MAX_MECHANISM_CHARS = 600
MAX_LIMITATIONS = 3
MAX_FACT_CHARS = 240
CLAIM_TYPES = ('quality', 'speed', 'search_effort')
_FIELDS = {'schema', 'parent_unit_ids', 'changed_units', 'claim_type', 'mechanism',
           'speed_used_for_search', 'search_effort_link', 'limitations'}
_MARKERS = re.compile(r'^[ \t]*([A-Z][A-Z0-9_]*_JSON)[ \t]*:', re.MULTILINE)
_FENCE = re.compile(r'^[ \t]*(?:`{3,}|~{3,})', re.MULTILINE)


class InterventionClaimError(ValueError):
    def __init__(self, code, path, message):
        self.diagnostic = dict(code=code, path=path, message=message)
        super().__init__(message)


def _require(condition, code, path, message):
    if not condition:
        raise InterventionClaimError(code, path, message)


def _text(value, limit):
    return (type(value) is str and 0 < len(value) <= limit and bool(value.strip())
            and not any(ord(char) < 32 and not char.isspace() for char in value))


def _unit_id(value):
    if not _text(value, MAX_UNIT_ID_CHARS) or any(char.isspace() for char in value):
        return False
    if value == '<module>':
        return True
    for part in value.split('.'):
        name, separator, occurrence = part.partition('#')
        if not name.isidentifier():
            return False
        if separator and (not occurrence.isascii() or not occurrence.isdigit()
                          or occurrence.startswith('0') or int(occurrence) < 2):
            return False
    return True


def _catalog(parent_units):
    _require(type(parent_units) is list and 0 < len(parent_units) <= MAX_CATALOG_UNITS,
             'invalid_parent_catalog', 'parent_units', 'A bounded nonempty parent-unit catalog is required.')
    result = {}
    for index, item in enumerate(parent_units):
        _require(type(item) is dict and _unit_id(item.get('id'))
                 and item.get('kind') in ('module', 'function', 'async_function', 'class'),
                 'invalid_parent_catalog', f'parent_units[{index}]', 'Invalid parent-unit ID or kind.')
        ident = item['id']
        _require(ident not in result, 'invalid_parent_catalog', f'parent_units[{index}].id',
                 'Parent-unit IDs must be unique.')
        result[ident] = item
    return result


def validate(value, parent_units):
    """Return a detached declaration, or raise a structured validation error.

    A modify preserves its qualified ID; express a rename explicitly as remove
    plus add. Added IDs are proposed labels, not evidence that such a unit exists.
    """
    catalog = _catalog(parent_units)
    _require(type(value) is dict and set(value) == _FIELDS,
             'invalid_fields', '$', 'The declaration must contain exactly the documented fields.')
    _require(value['schema'] == SCHEMA, 'invalid_schema', 'schema', 'Unsupported intervention schema.')
    parents = value['parent_unit_ids']
    _require(type(parents) is list and len(parents) <= MAX_CHANGED_UNITS
             and all(type(ident) is str and ident in catalog for ident in parents),
             'unknown_parent_unit', 'parent_unit_ids', 'Use only exact IDs from the parent catalog.')
    _require(len(set(parents)) == len(parents), 'duplicate_parent_unit', 'parent_unit_ids',
             'Parent-unit IDs must not repeat.')
    changes = value['changed_units']
    _require(type(changes) is list and 1 <= len(changes) <= MAX_CHANGED_UNITS,
             'invalid_changed_units', 'changed_units', 'Declare between one and sixteen changed units.')
    referenced, proposed, targets = [], [], []
    for index, change in enumerate(changes):
        path = f'changed_units[{index}]'
        _require(type(change) is dict and set(change) == {'parent_unit_id', 'proposed_name', 'change'},
                 'invalid_change_fields', path, 'Each unit requires parent_unit_id, proposed_name and change.')
        action, parent, name = change['change'], change['parent_unit_id'], change['proposed_name']
        _require(action in ('modify', 'add', 'remove'), 'invalid_change_kind', path+'.change',
                 'Change must be modify, add or remove.')
        if action == 'add':
            _require(parent is None and _unit_id(name) and name not in catalog,
                     'invalid_added_unit', path, 'An addition has null parent and a new qualified unit ID.')
        else:
            _require(type(parent) is str and parent in catalog,
                     'unknown_parent_unit', path+'.parent_unit_id', 'Use an exact parent-catalog ID.')
            referenced.append(parent)
            _require(name == parent if action == 'modify' else name is None,
                     'invalid_proposed_name', path+'.proposed_name',
                     'Modify retains the ID; remove uses null. Represent a rename as remove plus add.')
        if name is not None:
            proposed.append(name)
        targets.append(name if action == 'add' else parent)
    _require(len(set(targets)) == len(targets) and len(set(proposed)) == len(proposed),
             'duplicate_changed_unit', 'changed_units', 'A unit may be declared only once.')
    _require(set(parents) == set(referenced), 'parent_set_mismatch', 'parent_unit_ids',
             'Parent IDs must exactly match units declared modified or removed.')
    _require(value['claim_type'] in CLAIM_TYPES, 'invalid_claim_type', 'claim_type',
             'Claim type must be quality, speed or search_effort.')
    _require(_text(value['mechanism'], MAX_MECHANISM_CHARS), 'invalid_mechanism', 'mechanism',
             'State the proposed mechanism in one to 600 characters.')
    speed_search = value['speed_used_for_search']
    _require(type(speed_search) is bool and (not speed_search or value['claim_type'] == 'speed'),
             'invalid_speed_flag', 'speed_used_for_search',
             'The explicit speed-to-search flag is boolean and only applies to a speed claim.')
    link = value['search_effort_link']
    needs_link = speed_search or value['claim_type'] == 'search_effort'
    if needs_link:
        _require(type(link) is dict and set(link) == {'resource_saved', 'additional_work', 'budget_condition'}
                 and all(_text(text, MAX_FACT_CHARS) for text in link.values()),
                 'invalid_search_effort_link', 'search_effort_link',
                 'Specify resource_saved, additional_work and budget_condition, each one to 240 characters.')
    else:
        _require(link is None, 'unexpected_search_effort_link', 'search_effort_link',
                 'Use null when no search-effort increase is declared.')
    limitations = value['limitations']
    _require(type(limitations) is list and 1 <= len(limitations) <= MAX_LIMITATIONS
             and all(_text(text, MAX_FACT_CHARS) for text in limitations),
             'invalid_limitations', 'limitations', 'Provide one to three limitations, each one to 240 characters.')
    return deepcopy(value)


def declared_targets(claim):
    """Exact IDs for the structural audit; call only on a validated claim."""
    return [item['proposed_name'] if item['change'] == 'add' else item['parent_unit_id']
            for item in claim['changed_units']]


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('nonfinite JSON constant')


def extract(response, parent_units):
    """Parse the auxiliary block and leave ordinary hypothesis/code bytes intact.

    Even an invalid auxiliary declaration is removed from the ordinary parser's
    input, so a probe can still evaluate valid code (intention to treat). Markers
    inside or after the first code fence are never removed. The raw response must
    be retained independently by the caller. Invalid claims return targets=None,
    which must not be confused with an empty declared intervention.
    """
    result = dict(claim=None, error=None, response_without_intervention=response,
                  block_sha256=None, declared_targets=None)
    if type(response) is not str:
        result['error'] = dict(code='response_not_text', path='$', message='Expected response text.')
        return result
    if len(response) > MAX_RESPONSE_CHARS:
        result['error'] = dict(code='response_too_long', path='$', message='Response exceeds the bounded parser limit.')
        return result
    fence = _FENCE.search(response)
    prefix_end = fence.start() if fence else len(response)
    markers = list(_MARKERS.finditer(response, 0, prefix_end))
    spans = [(marker.start(), markers[index+1].start() if index+1 < len(markers) else prefix_end,
              marker.end()) for index, marker in enumerate(markers) if marker.group(1) == MARKER]
    clean = response
    for begin, end, _ in reversed(spans):
        clean = clean[:begin] + clean[end:]
    result['response_without_intervention'] = clean
    if len(spans) != 1:
        result['error'] = dict(code='missing_block' if not spans else 'multiple_blocks', path='$',
                              message='Exactly one INTERVENTION_JSON block is required before candidate code.')
        return result
    begin, end, content = spans[0]
    block = response[content:end].strip()
    result['block_sha256'] = hashlib.sha256(response[begin:end].encode('utf-8', errors='surrogatepass')).hexdigest()
    if len(block) > MAX_BLOCK_CHARS:
        result['error'] = dict(code='block_too_long', path='$', message='Intervention block exceeds 8192 characters.')
        return result
    try:
        value = json.loads(block, object_pairs_hook=_unique, parse_constant=_constant)
    except (ValueError, RecursionError):
        result['error'] = dict(code='invalid_json', path='$', message='Expected one strict, unfenced JSON object.')
        return result
    try:
        result['claim'] = validate(value, parent_units)
    except InterventionClaimError as exc:
        result['error'] = exc.diagnostic
        return result
    result['declared_targets'] = declared_targets(result['claim'])
    return result


parse = extract


def instruction(parent_units):
    """One shared declaration instruction for every diagnostic treatment arm."""
    catalog = _catalog(parent_units)
    units = [dict(id=ident, kind=item['kind']) for ident, item in catalog.items()]
    encoded = json.dumps(units, ensure_ascii=False, separators=(',', ':'))
    _require(len(encoded) <= 32768, 'catalog_too_long', 'parent_units', 'Parent-unit instruction exceeds its limit.')
    return (
        'AUDIT-ONLY INTERVENTION DECLARATION. Before the unchanged HYPOTHESIS_JSON block and Python code, '
        'emit INTERVENTION_JSON: followed by one unfenced JSON object. Do not merge its fields into HYPOTHESIS_JSON. '
        'Use exactly these fields: schema, parent_unit_ids, changed_units, claim_type, mechanism, '
        'speed_used_for_search, search_effort_link, limitations. '
        f'schema must be "{SCHEMA}". '
        'parent_unit_ids is the unique list of exact catalog IDs you plan to modify or remove, at most sixteen. '
        'changed_units has one to sixteen objects, each exactly {parent_unit_id, proposed_name, change}. '
        'This bounded declaration MUST NOT constrain how many source units you may edit. '
        'Additional or unlisted source edits are audited explicitly; an invalid declaration never discards otherwise valid code. '
        'For modify, both IDs are the same existing catalog ID; for remove, proposed_name is null; '
        'for add, parent_unit_id is null and proposed_name is the proposed new qualified ID. '
        'Declare a rename as remove plus add. Parent IDs must exactly match the modified/removed entries. '
        'claim_type is quality, speed or search_effort. mechanism is your proposed effect: aim for 350 characters or fewer, hard limit 600. '
        'speed_used_for_search is a boolean: true only when a speed claim asserts that saved resources enable more search. '
        'search_effort_link is required for that case and for a search_effort claim: '
        'exactly {resource_saved, additional_work, budget_condition}, each at most 240 characters. '
        'State which loop, restart count or step count can actually increase, and under what binding budget; '
        'if no resource is saved, say so explicitly. Otherwise search_effort_link must be null. '
        'limitations contains one to three strings, each at most 240 characters, identifying what remains untested. '
        'This declaration is committed before evaluation; it is not proof of semantic alignment, complete edit coverage, '
        'speedup, extra search or causality. Literal unit IDs are checked; prose is not certified by keyword matching. '
        'The host audits the actual source after any repair and entry-point grafting. '
        'An invalid declaration is recorded separately and does not by itself exclude valid code from this diagnostic.\n'
        'PARENT_UNIT_CATALOG_JSON:\n' + encoded + '\n')
