"""Whole host-gated interpretation of separately delivered measured study facts."""
from __future__ import annotations
import copy
import re
from .catalogs import canonical_json, canonical_sha

SCHEMA = 'sciel-evaluated-study-review-context/1'
PREFIX = 'EVALUATED_STUDY_REVIEW_JSON:\n'
MAX_CONTEXT_CHARS = 8000
RULES = (
    'EVALUATED STUDY REVIEW: stored_card_sha256 binds the complete model card and '
    'catalog_sha256 its checked host facts in the separate evaluated study block. '
    'Interpretation, selected claims/references, resolved evidence and prospective '
    'target are unverified model choices. Host acceptance permits text delivery, '
    'not semantic truth, causal proof, lesson credit, native score, G, evaluation '
    'or promotion. Pending, rejected, stale and text-OFF content is wholly hidden. '
    'Selection reserves the complete ON card in every mode; unused hidden capacity '
    'is not reassigned. If the complete measured study facts are omitted, the review '
    'is also omitted. Missing text is not a negative measurement.\n'
)
OMISSION = 'comparison_review_text_ineligible'


def require(ok, message):
    if not ok:
        raise ValueError('evaluated review context: ' + message)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _render(value):
    return RULES + PREFIX + canonical_json(value) + '\n'


def _sized(value, reserve=None):
    value.update(rendered_chars=0, reserved_chars=0 if reserve is None else reserve, context_sha256='0'*64)
    allowance = 1  # The text-OFF boolean has one extra character.
    for card in value['cards']:
        decision = card['text_eligibility']
        allowance += 8 - len(decision['status'])
        allowance += len(canonical_json('0'*64)) - len(canonical_json(decision['decision_sha256']))
        allowance += len(canonical_json(OMISSION)) - len('null')
    while True:
        actual = len(_render(value)); reserved = actual + allowance if reserve is None else reserve
        if (actual, reserved) == (value['rendered_chars'], value['reserved_chars']):
            break
        value.update(rendered_chars=actual, reserved_chars=reserved)
    value['context_sha256'] = canonical_sha({k:v for k,v in value.items() if k != 'context_sha256'})
    return value


def build_context(bundle, card, *, branch, text_policy, policy_sha256,
                  max_chars=8000, text_enabled=True, enabled=True, source_facts_presented=True):
    require(type(branch) is str and branch == bundle['capsule']['receiving_root']['id'], 'root branch differs')
    require(type(max_chars) is int and 0 <= max_chars <= MAX_CONTEXT_CHARS, 'invalid character budget')
    require(all(type(v) is bool for v in (text_enabled, enabled, source_facts_presented)), 'invalid delivery flags')
    require(_sha(policy_sha256), 'invalid current policy digest')
    from ..hypothesis.evaluated_study_review import validate_card
    from ..hypothesis.review_eligibility import validate_policy, decision_for
    checked = validate_card(card, bundle)
    policy = validate_policy(text_policy)
    decision = decision_for(checked, policy, policy_sha256=policy_sha256)
    candidate = dict(comparison_id=checked['comparison_id'], comparison_sha256=checked['comparison_sha256'],
        branch=checked['branch'], comparator_source_sha256=checked['comparator_source_sha256'],
        challenger_source_sha256=checked['challenger_source_sha256'],
        stored_card_sha256=canonical_sha(checked), catalog_sha256=canonical_sha(checked['host_facts']),
        writer=copy.deepcopy(checked['writer']), text_eligibility=decision,
        unverified_interpretation=copy.deepcopy(checked['unverified_interpretation']),
        evidence_bindings=copy.deepcopy(checked['evidence_bindings']),
        resolved_target=copy.deepcopy(checked['resolved_target']), interpretation_omission=None)
    view = _sized(dict(schema=SCHEMA, source_study_sha256=canonical_sha(bundle['capsule']),
        author_authority_sha256=canonical_sha(bundle['author_authority']), branch=branch,
        policy_sha256=policy_sha256, decisions_sha256=canonical_sha(policy),
        max_chars=max_chars, enabled=enabled, text_enabled=True, source_facts_presented=source_facts_presented,
        retained_count=1, presented_count=0, omitted_count=1, cards=[], omissions=[],
        omission_details_omitted_count=1))
    if view['reserved_chars'] > max_chars:
        return None
    reason = 'disabled' if not enabled else ('source_study_facts_omitted' if not source_facts_presented else None)
    if reason is None:
        trial = copy.deepcopy(view)
        trial.update(cards=[candidate], presented_count=1, omitted_count=0, omission_details_omitted_count=0)
        if _sized(trial)['reserved_chars'] <= max_chars:
            view = trial
        else:
            reason = 'memory_char_limit'
    if reason is not None:
        trial = copy.deepcopy(view)
        trial.update(omissions=[dict(stored_card_sha256=canonical_sha(checked), reason=reason)],
                     omission_details_omitted_count=0)
        if _sized(trial)['reserved_chars'] <= max_chars:
            view = trial
    reserve = view['reserved_chars']
    view['text_enabled'] = text_enabled
    for shown in view['cards']:
        accepted = shown['text_eligibility']['status'] == 'accepted'
        if not accepted or not text_enabled:
            for key in ('unverified_interpretation', 'evidence_bindings', 'resolved_target'):
                shown[key] = None
            shown['interpretation_omission'] = OMISSION if not accepted else 'comparison_review_text_disabled'
    view = _sized(view, reserve)
    require(view['rendered_chars'] <= view['reserved_chars'] <= max_chars, 'masked view exceeds full reservation')
    return copy.deepcopy(view)


def validate_context(value, bundle, card, **options):
    expected = build_context(bundle, card, **options)
    require(canonical_json(value) == canonical_json(expected), 'context differs from authoritative card or allocation')
    return expected


def block(value):
    if value is None:
        return ''
    require(type(value) is dict and value.get('schema') == SCHEMA, 'invalid context schema')
    require(_sha(value.get('context_sha256')) and value['context_sha256'] ==
        canonical_sha({k:v for k,v in value.items() if k != 'context_sha256'}), 'context digest differs')
    rendered = _render(value)
    require(all(type(value.get(k)) is int for k in ('rendered_chars','reserved_chars','max_chars'))
        and len(rendered) == value['rendered_chars'] <= value['reserved_chars'] <= value['max_chars']
        and 0 <= value['max_chars'] <= MAX_CONTEXT_CHARS, 'invalid context size')
    return rendered
