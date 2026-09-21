"""Admit whole restart-diagnostic cards and validate their context budget."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from ..hypothesis import restart_observations as observations
from ..hypothesis import restart_motion_observations as motion_observations
from .paired_comparisons import SCHEMA as GROUP_SCHEMA, UNKNOWN_RECEIPT
from .restart_codec import (
    _expand_wire,
    _pack,
    _pack_common,
    _pack_legacy,
    _unpack,
    _unpack_fields,
)
from .restart_encoding import (
    EXPANDED_SCHEMA,
    LEGACY_SCHEMA,
    MAX_CARD_CHARS,
    PREFIX,
    PREVIOUS_FACTORED_SCHEMA,
    PREVIOUS_SCHEMA,
    RULES,
    SCHEMA,
    _ADMISSION_POLICY,
    _COLUMNS,
    _COLUMNS4,
    _COMMON_FIELDS,
    _FIELD_COLUMNS,
    _MAX_VIEW_CHARS,
    _ORIGINS,
    _ORIGIN_COLUMNS,
    _QUALITY_COLUMNS,
    _REF_FIELDS,
    _ROLES,
    _WIRE_FIELDS,
    _check_wire,
    _common_fields,
    _decode_field,
    _digest,
    _encode_field,
    _record_from_values,
    _record_values,
    _require,
    _seal,
    _sha,
    _text,
    _wire,
)


def admission_identity():
    """Bind the admission semantics separately from the lossless observation."""
    source = Path(__file__)
    implementation = {
        name: hashlib.sha256(source.with_name(name).read_bytes()).hexdigest()
        for name in ('restart_encoding.py', 'restart_codec.py')
    }
    return dict(policy=copy.deepcopy(_ADMISSION_POLICY),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                implementation=implementation)

def _factored_card_chars(value, index):
    card = value['cards'][index]
    refs = card['observation_refs']
    entries = [] if refs is None else [value['observation_catalog'][ref] for ref in refs]
    return len(_wire(dict(card=card, observation_catalog=entries)))

def build_context(groups, *, max_chars=12000, max_items=6):
    """Return a detached whole-card context, or None when metadata cannot fit.

    Input order is preserved, followed by each planned seed and comparator then
    challenger. Selection never ranks gains or favorable outcomes. The character
    limit covers the complete JSON; callers additionally budget RULES and PREFIX.
    Historical groups with no restart_trace key anywhere remain exactly absent.
    """
    _require(type(groups) is list and len(groups) <= 64, 'invalid groups')
    _require(type(max_chars) is int and 0 <= max_chars <= 131072, 'invalid character limit')
    _require(type(max_items) is int and 0 <= max_items <= 384, 'invalid item limit')
    _require(all(type(g) is dict and g.get('schema') == GROUP_SCHEMA for g in groups), 'group schema')
    if not max_chars or not any(type(p[role]) is dict and 'restart_trace' in p[role]
                               for g in groups for p in g['pairs'] for role in _ROLES):
        return None
    catalog, selected = [], []
    for group in groups:
        adapter = group['adapter_identity']
        catalog.append({key: copy.deepcopy(group[key]) for key in
                        ('id', 'branch', 'origin_experiment_id', 'comparator_source_sha256', 'challenger_source_sha256')})
        catalog[-1].update(group_sha256=_digest(group), seeds=[p['seed'] for p in group['pairs']],
                           adapter_identity_sha256=adapter.get('identity_sha256', _digest(adapter)))
        for pair in group['pairs']:
            for role in _ROLES:
                receipt = pair[role]
                reference = dict(group_id=group['id'], role=role, seed=pair['seed'],
                    source_sha256=group[role+'_source_sha256'],
                    receipt_sha256=_digest(receipt) if receipt is not None else None,
                    receipt_status=receipt['status'] if receipt is not None else 'not_dispatched')
                selected.append((reference, receipt, adapter))
    _require(len({g['id'] for g in catalog}) == len(catalog), 'duplicate group ID')
    cards = []
    omissions = [dict(ref, reason='item_limit') for ref, _, _ in selected]

    def context():
        return _pack(dict(schema=EXPANDED_SCHEMA, groups=catalog, cards=cards, omissions=omissions,
            selected_count=len(selected), presented_count=len(cards), omitted_count=len(omissions),
            max_items=max_items, max_chars=max_chars))

    if context()['rendered_chars'] > max_chars:
        return None
    for reference, receipt, adapter in selected:
        if len(cards) >= max_items:
            continue
        position = next(i for i, row in enumerate(omissions)
                        if all(row[key] == reference[key] for key in _REF_FIELDS))
        card = dict(reference, observation_view=None, unavailable_reason=None)
        if receipt is None or receipt == UNKNOWN_RECEIPT:
            card['unavailable_reason'] = ('receipt_not_dispatched' if receipt is None else 'dispatch_unknown')
        else:
            try:
                observation = motion_observations.build_observation(receipt, source_sha256=reference['source_sha256'],
                    seed=reference['seed'], adapter_identity=adapter)
            except ValueError:
                # Numeric groups were validated by the caller. A broken optional
                # diagnostic cannot invalidate their independently measured quality.
                card['unavailable_reason'] = 'invalid_diagnostic'
            else:
                try:
                    view = observations.observation_view(observation, max_chars=_MAX_VIEW_CHARS)
                except ValueError:
                    omissions[position]['reason'] = 'card_char_limit'
                    continue
                if not view['included']:
                    omissions[position]['reason'] = 'card_char_limit'
                    continue
                card['observation_view'] = view
        omitted = omissions.pop(position)
        cards.append(card)
        packed = context()
        # Admission measures the representation actually sent. The old view
        # envelope duplicated bindings/hashes before field factoring and could
        # reject a whole card that fits both unchanged output limits.
        card_chars = (_factored_card_chars(packed, len(cards) - 1) if packed['schema'] == SCHEMA
                      else len(_wire(card['observation_view'])))
        if card_chars > MAX_CARD_CHARS:
            cards.pop()
            omissions.insert(position, dict(omitted, reason='card_char_limit'))
        elif packed['rendered_chars'] > max_chars:
            cards.pop()
            omissions.insert(position, dict(omitted, reason='memory_char_limit'))
    result = context()
    # Longer omission reasons can increase metadata after attempted admission.
    if result['rendered_chars'] > max_chars:
        return None
    return validate_context(result)

def validate_context(value):
    """Return validated detached wire, never an expanded over-budget surrogate."""
    try:
        value = json.loads(_wire(value))
        _validate_context(_unpack(value), factored_admission=value.get('schema') == SCHEMA)
        if value.get('schema') == SCHEMA:
            _require(all(_factored_card_chars(value, i) <= MAX_CARD_CHARS
                         for i in range(len(value['cards']))), 'factored card character limit')
        return value
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError('restart context: malformed value') from exc

def expand_context(value):
    """Expand exact labels/views; only the original wire obeys its byte budget.

    The expanded form has its own schema, length and digest. It is a convenient
    decoded representation, not a substitute for actual prompt bytes.
    """
    return _unpack(validate_context(value))

def _validate_context(value, *, factored_admission=False):
    value = json.loads(_wire(value))
    _require(type(value) is dict and set(value) == {'schema', 'groups', 'cards', 'omissions', 'selected_count',
        'presented_count', 'omitted_count', 'max_items', 'max_chars', 'rendered_chars', 'context_sha256'}
        and value['schema'] == EXPANDED_SCHEMA, 'schema fields')
    for key in ('selected_count', 'presented_count', 'omitted_count', 'max_items', 'max_chars', 'rendered_chars'):
        _require(type(value[key]) is int and value[key] >= 0, 'invalid count or limit')
    _require(value['max_items'] <= 384 and value['max_chars'] <= 131072
        and value['rendered_chars'] == len(_wire(value)), 'expanded rendered length')
    _require(value['context_sha256'] == _digest({k: v for k, v in value.items() if k != 'context_sha256'}), 'digest')
    _require(type(value['groups']) is list and 0 < len(value['groups']) <= 64, 'group catalog')
    groups, expected = {}, set()
    for group in value['groups']:
        _require(type(group) is dict and set(group) == {'id', 'branch', 'origin_experiment_id',
            'comparator_source_sha256', 'challenger_source_sha256', 'group_sha256', 'seeds',
            'adapter_identity_sha256'}, 'group fields')
        _require(_text(group['id']) and group['id'] not in groups and _text(group['branch'])
            and (group['origin_experiment_id'] is None or _text(group['origin_experiment_id'])), 'group identity')
        _require(all(_sha(group[k]) for k in ('group_sha256', 'comparator_source_sha256',
            'challenger_source_sha256', 'adapter_identity_sha256')), 'group hashes')
        _require(type(group['seeds']) is list and len(group['seeds']) == len(set(group['seeds'])) == 3
            and all(type(seed) is int and 0 <= seed < 2**32 for seed in group['seeds']), 'group seeds')
        groups[group['id']] = group
        expected.update((group['id'], role, seed) for role in _ROLES for seed in group['seeds'])
    _require(type(value['cards']) is list and type(value['omissions']) is list, 'card/omission lists')
    seen = set()
    for key in ('cards', 'omissions'):
        for row in value[key]:
            extra = {'observation_view', 'unavailable_reason'} if key == 'cards' else {'reason'}
            _require(type(row) is dict and set(row) == _REF_FIELDS | extra, 'entry fields')
            ident = (row['group_id'], row['role'], row['seed'])
            _require(type(row['seed']) is int and ident in expected and ident not in seen, 'entry membership')
            seen.add(ident)
            group = groups[row['group_id']]
            _require(row['source_sha256'] == group[row['role']+'_source_sha256'], 'role/source binding')
            _require(row['receipt_status'] in ('valid', 'candidate_failure', 'infrastructure', 'unknown', 'not_dispatched'), 'receipt status')
            _require((row['receipt_sha256'] is None and row['receipt_status'] == 'not_dispatched')
                or (_sha(row['receipt_sha256']) and row['receipt_status'] != 'not_dispatched'), 'receipt hash')
            if key == 'omissions':
                _require(row['reason'] in ('item_limit', 'card_char_limit', 'memory_char_limit'), 'omission reason')
            elif row['observation_view'] is None:
                _require((row['receipt_status'], row['unavailable_reason']) in
                    (('not_dispatched', 'receipt_not_dispatched'), ('unknown', 'dispatch_unknown'),
                     ('valid', 'invalid_diagnostic'), ('candidate_failure', 'invalid_diagnostic'),
                     ('infrastructure', 'invalid_diagnostic')), 'unavailable reason')
            else:
                _require(row['unavailable_reason'] is None, 'unexpected unavailable reason')
                view = observations.validate_observation_view(row['observation_view'])
                _require(view['included'] and len(_wire(view)) <= (
                    _MAX_VIEW_CHARS if factored_admission else MAX_CARD_CHARS), 'whole view required')
                binding, observation = view['bindings'], view['observation']
                _require(binding['source_sha256'] == row['source_sha256'] and binding['seed'] == row['seed']
                    and binding['adapter_identity_sha256'] == group['adapter_identity_sha256'], 'view binding')
                _require(observation['receipt_sha256'] == row['receipt_sha256']
                    and observation['receipt_status'] == row['receipt_status'], 'view receipt binding')
    _require(seen == expected and value['selected_count'] == len(expected)
        and value['presented_count'] == len(value['cards']) <= value['max_items']
        and value['omitted_count'] == len(value['omissions'])
        and value['selected_count'] == value['presented_count'] + value['omitted_count'], 'coverage/counts')
    return value
