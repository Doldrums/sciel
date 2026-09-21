"""Whole, host-gated review text for one immutable generated paired study.

The separate generated-paired block owns complete measured facts. This view
binds that study and the stored review/catalog without repeating the facts or
turning interpretations into observations, lessons or execution authority.
"""
from __future__ import annotations

import copy
import re

from .catalogs import canonical_json, canonical_sha


SCHEMA = 'sciel-generated-paired-review-context/1'
SCOPED_SCHEMA = 'sciel-generated-paired-review-context/2'
PREFIX = 'GENERATED_PAIRED_REVIEWS_JSON:\n'
MAX_CONTEXT_CHARS = 8000
RULES = (
    'GENERATED PAIRED REVIEWS: stored_card_sha256 binds the full authored card; '
    'catalog_sha256 binds its checked host facts in the separately supplied generated '
    'paired study. Original comparison branches are preserved; this block is routed '
    'through the receiving root. unverified_interpretation includes selected claims, '
    'conjecture, limitations, prospective target and reason. All of it is model text; '
    'an accepted host delivery decision does not prove prose, mechanisms or future '
    'quality, and gives no lesson credit or permission to run. Pending, rejected, stale '
    'and text-OFF interpretations are wholly hidden. Whole-card selection reserves '
    'the full ON text regardless of eligibility or text mode. Missing text is not '
    'negative evidence or a failed experiment.\n'
)
_OMISSION = 'comparison_review_text_ineligible'
SCOPED_RULES = RULES + (
    'Scoped evidence_bindings resolve model-selected references to exact facts in the '
    'sealed study, including their original comparison and source-role bindings. '
    'Resolution does not prove the prose, a mechanism, or future quality. References '
    'and their resolved values are wholly hidden with ineligible or text-OFF '
    'interpretations. The reviewed comparison is distinct from a prospective '
    'component target; the future base and comparator may be different study sources.\n'
)


def _require(condition, message):
    if not condition:
        raise ValueError('generated paired review context: ' + message)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _render(value):
    rules = SCOPED_RULES if value.get('schema') == SCOPED_SCHEMA else RULES
    return rules + PREFIX + canonical_json(value) + '\n'


def _sized(value, *, reserve=None):
    """Count the complete block, including fixed-point size and digest fields."""
    value['rendered_chars'] = 0
    value['reserved_chars'] = 0 if reserve is None else reserve
    value['context_sha256'] = '0' * 64
    # Reserve maximum-size eligibility/omission metadata before any masking.
    # This also makes pending/accepted/rejected/stale selection independent.
    allowance = 1  # false has one more character than true.
    for card in value['cards']:
        decision = card['text_eligibility']
        allowance += 8 - len(decision['status'])
        allowance += len(canonical_json('0' * 64)) - len(canonical_json(decision['decision_sha256']))
        allowance += len(canonical_json(_OMISSION)) - len('null')
    while True:
        size = len(_render(value))
        reserved = size + allowance if reserve is None else reserve
        if value['rendered_chars'] == size and value['reserved_chars'] == reserved:
            break
        value['rendered_chars'], value['reserved_chars'] = size, reserved
    value['context_sha256'] = canonical_sha({k: v for k, v in value.items() if k != 'context_sha256'})
    return value


def build_context(component, cards, *, branch, max_chars=8000, text_policy,
                  policy_sha256, text_enabled=True, enabled=True):
    """Validate all cards, then select against their full-text reservations."""
    _require(type(branch) is str and bool(branch.strip()) and len(branch) <= 128,
             'invalid receiving branch')
    _require(type(max_chars) is int and 0 <= max_chars <= MAX_CONTEXT_CHARS,
             'invalid character limit')
    _require(type(enabled) is bool and type(text_enabled) is bool, 'invalid delivery flags')
    _require(_sha(policy_sha256), 'invalid current policy digest')
    from ..hypothesis.review_eligibility import validate_policy, decision_for
    from ..hypothesis.generated_paired_review import validate_cards
    checked = validate_cards(cards, component)
    _require(branch == component['root_parent']['id'], 'receiving branch differs from study root')
    policy = validate_policy(text_policy)
    decisions = {ident: decision_for(card, policy, policy_sha256=policy_sha256)
                 for ident, card in checked.items()}
    candidates = []
    for group in reversed(component['comparisons']):
        ident = group['id']
        if ident not in checked:
            continue
        card = checked[ident]
        candidates.append(dict(comparison_id=ident,
            comparison_sha256=card['comparison_sha256'],
            branch=card['branch'],
            comparator_source_sha256=card['comparator_source_sha256'],
            challenger_source_sha256=card['challenger_source_sha256'],
            stored_card_sha256=canonical_sha(card), catalog_sha256=canonical_sha(card['host_facts']),
            writer=copy.deepcopy(card['writer']), text_eligibility=decisions[ident],
            unverified_interpretation=copy.deepcopy(card['unverified_interpretation']),
            interpretation_omission=None))
        if card['schema'] == 'sciel-generated-paired-review-card/2':
            candidates[-1]['evidence_bindings'] = copy.deepcopy(card['evidence_bindings'])
    count = len(candidates)
    view_schema = SCOPED_SCHEMA if any('evidence_bindings' in card for card in candidates) else SCHEMA
    view = _sized(dict(schema=view_schema, source_component_sha256=canonical_sha(component),
        branch=branch, enabled=enabled, text_enabled=True,
        policy_sha256=policy_sha256, decisions_sha256=canonical_sha(policy),
        max_chars=max_chars, cards=[], omitted=[], retained_count=count, presented_count=0,
        omitted_count=count, omission_details_omitted_count=count))
    if view['reserved_chars'] > max_chars:
        return None
    omissions = []
    for candidate in candidates:
        reason = None if enabled else 'disabled'
        if reason is None:
            trial = copy.deepcopy(view)
            trial['cards'].append(candidate)
            trial['presented_count'] += 1
            trial['omitted_count'] -= 1
            trial['omission_details_omitted_count'] -= 1
            if _sized(trial)['reserved_chars'] <= max_chars:
                view = trial
                continue
            reason = 'memory_char_limit'
        omissions.append(dict(comparison_id=candidate['comparison_id'], reason=reason,
            stored_card_sha256=candidate['stored_card_sha256']))
    for omission in omissions:
        trial = copy.deepcopy(view)
        trial['omitted'].append(omission)
        trial['omission_details_omitted_count'] -= 1
        if _sized(trial)['reserved_chars'] <= max_chars:
            view = trial
    reservation = view['reserved_chars']
    view['text_enabled'] = text_enabled
    for card in view['cards']:
        eligible = card['text_eligibility']['status'] == 'accepted'
        if not eligible or not text_enabled:
            card['unverified_interpretation'] = None
            if 'evidence_bindings' in card:
                card['evidence_bindings'] = None
            card['interpretation_omission'] = (_OMISSION if not eligible
                                              else 'comparison_review_text_disabled')
    view = _sized(view, reserve=reservation)
    _require(view['rendered_chars'] <= view['reserved_chars'] <= max_chars,
             'masked view exceeds full-text reservation')
    return copy.deepcopy(view)


def validate_context(value, component, cards, *, branch, max_chars=8000, text_policy,
                     policy_sha256, text_enabled=True, enabled=True):
    """Rebuild from authoritative cards and caller-owned allocation, even if omitted."""
    expected = build_context(component, cards, branch=branch, max_chars=max_chars,
        text_policy=text_policy, policy_sha256=policy_sha256,
        text_enabled=text_enabled, enabled=enabled)
    _require(canonical_json(value) == canonical_json(expected),
             'view differs from authoritative cards or allocation')
    return expected


def block(value):
    """Render a sealed view; semantic authority still requires validate_context."""
    if value is None:
        return ''
    _require(type(value) is dict and value.get('schema') in (SCHEMA, SCOPED_SCHEMA), 'invalid view schema')
    _require(_sha(value.get('context_sha256'))
        and value['context_sha256'] == canonical_sha({k: v for k, v in value.items() if k != 'context_sha256'}),
        'view digest differs')
    rendered = _render(value)
    _require(type(value.get('rendered_chars')) is int
        and type(value.get('reserved_chars')) is int and type(value.get('max_chars')) is int
        and len(rendered) == value['rendered_chars'] <= value['reserved_chars'] <= value['max_chars']
        and 0 <= value['max_chars'] <= MAX_CONTEXT_CHARS, 'view size differs')
    return rendered
