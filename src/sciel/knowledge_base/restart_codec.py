"""Pack and restore complete restart observations without changing their values."""
from __future__ import annotations

import copy
from ..hypothesis import restart_observations as observations
from .restart_encoding import (
    EXPANDED_SCHEMA,
    LEGACY_SCHEMA,
    PREVIOUS_FACTORED_SCHEMA,
    PREVIOUS_SCHEMA,
    SCHEMA,
    _COLUMNS,
    _COLUMNS4,
    _COMMON_FIELDS,
    _ROLES,
    _check_wire,
    _common_fields,
    _decode_field,
    _encode_field,
    _require,
    _seal,
    _text,
    _wire,
)


def _pack_legacy(value):
    """Factor only these declared schema slots, preserving whole observations."""
    rows = {(r['group_id'], r['role'], r['seed']): r for r in value['cards'] + value['omissions']}
    refs, indexes = [], {}
    for gi, group in enumerate(value['groups']):
        for pi, seed in enumerate(group['seeds']):
            for ri, role in enumerate(_ROLES):
                key = (group['id'], role, seed)
                row = rows[key]
                indexes[key] = len(refs)
                refs.append([gi, pi, ri, row['receipt_sha256'], row['receipt_status']])
    result = {k: copy.deepcopy(v) for k, v in value.items()
              if k not in ('schema', 'cards', 'omissions', 'rendered_chars', 'context_sha256')}
    result.update(schema=LEGACY_SCHEMA, columns=copy.deepcopy(_COLUMNS), receipt_refs=refs, cards=[], omissions=[])
    for card in value['cards']:
        view = card['observation_view']
        result['cards'].append(dict(receipt_ref=indexes[(card['group_id'], card['role'], card['seed'])],
            observation=copy.deepcopy(view['observation']) if view is not None else None,
            view_sha256=view['view_sha256'] if view is not None else None,
            unavailable_reason=card['unavailable_reason']))
    for row in value['omissions']:
        result['omissions'].append([indexes[(row['group_id'], row['role'], row['seed'])], row['reason']])
    return _seal(result)

def _pack_common(value):
    """Losslessly factor only the declared header of whole admitted observations."""
    result = _pack_legacy(value)
    result.update(schema=PREVIOUS_SCHEMA, observation_common=[])
    encoded = []
    for card in result['cards']:
        observation = card['observation']
        card['observation_common_ref'] = None
        if observation is None:
            continue
        fields = _common_fields(observation['schema'])
        common = {key: observation.pop(key) for key in sorted(fields)}
        key = _wire(common)
        if key not in encoded:
            encoded.append(key)
            result['observation_common'].append(common)
        card['observation_common_ref'] = encoded.index(key)
    return _seal(result)

def _pack(value):
    """Share equal named fields without trimming or ranking any case facts."""
    result = _pack_legacy(value)
    result.update(schema=SCHEMA, columns=copy.deepcopy(_COLUMNS4), observation_catalog=[])
    indexes = {}
    for card in result['cards']:
        observation = card.pop('observation')
        card['observation_refs'] = None
        if observation is None:
            continue
        refs = []
        for field, data in sorted(observation.items()):
            entry = [field, _encode_field(field, data)]
            encoded = _wire(entry)
            if encoded not in indexes:
                indexes[encoded] = len(result['observation_catalog'])
                result['observation_catalog'].append(entry)
            refs.append(indexes[encoded])
        card['observation_refs'] = refs
    return _seal(result)

def _unpack(value):
    if type(value) is dict and value.get('schema') == LEGACY_SCHEMA:
        _check_wire(value, LEGACY_SCHEMA)
        return _expand_wire(value)
    if type(value) is dict and value.get('schema') in (SCHEMA, PREVIOUS_FACTORED_SCHEMA):
        return _unpack_fields(value)
    _check_wire(value, PREVIOUS_SCHEMA, ('observation_common',))
    common = value['observation_common']
    _require(type(common) is list and len(common) <= 384, 'common observation catalog')
    encoded = []
    for entry in common:
        _require(type(entry) is dict and set(entry) == _common_fields(entry.get('schema')),
            'common observation fields')
        key = _wire(entry)
        _require(key not in encoded, 'duplicate common observation')
        encoded.append(key)
    restored = copy.deepcopy(value)
    restored.pop('observation_common')
    seen = set()
    for card in restored['cards']:
        _require(type(card) is dict and set(card) == {'receipt_ref', 'observation', 'view_sha256',
            'unavailable_reason', 'observation_common_ref'}, 'wire card fields')
        ref = card.pop('observation_common_ref')
        remainder = card['observation']
        if remainder is None:
            _require(ref is None, 'unavailable observation common reference')
            continue
        _require(type(remainder) is dict and not (_COMMON_FIELDS & set(remainder)),
            'observation remainder repeats common field')
        _require(type(ref) is int and 0 <= ref < len(common), 'observation common reference')
        if ref not in seen:
            _require(ref == len(seen), 'common observation first-use order')
            seen.add(ref)
        card['observation'] = {**copy.deepcopy(common[ref]), **remainder}
    _require(len(seen) == len(common), 'unused common observation')
    # The actual /3 wire already passed its own size/hash checks. This is only
    # an internal full-object decode; no virtual /2 size is claimed or budgeted.
    return _expand_wire(restored)

def _unpack_fields(value):
    _require(value['schema'] in (SCHEMA, PREVIOUS_FACTORED_SCHEMA), 'factored wire schema')
    _check_wire(value, value['schema'], ('observation_catalog',))
    catalog = value['observation_catalog']
    _require(type(catalog) is list and len(catalog) <= 24576, 'observation field catalog')
    encoded = set()
    for entry in catalog:
        _require(type(entry) is list and len(entry) == 2 and _text(entry[0]),
            'observation field entry')
        key = _wire(entry)
        _require(key not in encoded, 'duplicate observation field')
        encoded.add(key)
    restored = copy.deepcopy(value)
    restored.pop('observation_catalog')
    seen = set()
    for card in restored['cards']:
        _require(type(card) is dict and set(card) == {'receipt_ref', 'observation_refs',
            'view_sha256', 'unavailable_reason'}, 'field-catalog card fields')
        refs = card.pop('observation_refs')
        if refs is None:
            card['observation'] = None
            continue
        _require(type(refs) is list and 0 < len(refs) <= 64, 'observation field references')
        observation, fields = {}, []
        for ref in refs:
            _require(type(ref) is int and 0 <= ref < len(catalog), 'observation field reference')
            if ref not in seen:
                _require(ref == len(seen), 'observation field first-use order')
                seen.add(ref)
            field, data = catalog[ref]
            _require(field not in observation, 'duplicate observation field name')
            fields.append(field)
            observation[field] = _decode_field(field, data)
        _require(fields == sorted(fields), 'observation field order')
        card['observation'] = observation
    _require(len(seen) == len(catalog), 'unused observation field')
    return _expand_wire(restored)

def _expand_wire(value):
    """Shared label/view inverse after the actual wire's schema/hash/budget check."""
    refs = []
    expected = [(gi, pi, ri) for gi, g in enumerate(value['groups']) for pi, _ in enumerate(g['seeds']) for ri in range(2)]
    _require(len(value['receipt_refs']) == len(expected), 'receipt reference coverage')
    for row, index in zip(value['receipt_refs'], expected):
        _require(type(row) is list and len(row) == 5 and all(type(i) is int for i in row[:3])
            and tuple(row[:3]) == index, 'receipt reference order')
        gi, pi, ri, receipt_sha, status = row
        group, role = value['groups'][gi], _ROLES[ri]
        refs.append(dict(group_id=group['id'], role=role, seed=group['seeds'][pi],
            source_sha256=group[role+'_source_sha256'], receipt_sha256=receipt_sha, receipt_status=status))
    def reference(index):
        _require(type(index) is int and 0 <= index < len(refs), 'receipt reference')
        return copy.deepcopy(refs[index])
    result = {k: copy.deepcopy(v) for k, v in value.items()
              if k not in ('schema', 'columns', 'receipt_refs', 'cards', 'omissions', 'rendered_chars', 'context_sha256')}
    result.update(schema=EXPANDED_SCHEMA, cards=[], omissions=[])
    for card in value['cards']:
        _require(type(card) is dict and set(card) == {'receipt_ref', 'observation', 'view_sha256', 'unavailable_reason'}, 'wire card fields')
        observation = card['observation']
        if observation is None:
            _require(card['view_sha256'] is None, 'unavailable view digest')
            view = None
        else:
            _require(type(observation) is dict, 'whole observation')
            view = dict(schema=observations.VIEW_SCHEMA, observation_sha256=observation['observation_sha256'],
                bindings=copy.deepcopy(observation['bindings']), observation_status=observation['status'],
                included=True, omission_reason=None, observation=copy.deepcopy(observation), view_sha256=card['view_sha256'])
        result['cards'].append(dict(reference(card['receipt_ref']), observation_view=view, unavailable_reason=card['unavailable_reason']))
    for row in value['omissions']:
        _require(type(row) is list and len(row) == 2, 'wire omission row')
        result['omissions'].append(dict(reference(row[0]), reason=row[1]))
    return _seal(result)
