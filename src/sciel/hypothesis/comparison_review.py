"""Bounded interpretations of paired evidence, never lessons or new evidence.

All physical facts are rebuilt from full authoritative comparison groups. Model
text and writer provenance remain unverified: their journal authentication is
the caller's responsibility. This module neither evaluates code nor calls a
model, and cannot change quality, admission, support or exposure decisions.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re

from ..knowledge_base.catalogs import canonical_json, canonical_sha
from ..knowledge_base.comparison_facts import comparison_facts
from ..knowledge_base.paired_comparisons import validate_comparison


SCHEMA = "sciel-comparison-review/1"
CARD_SCHEMA = "sciel-comparison-review-card/1"
CONTEXT_SCHEMA = "sciel-comparison-review-context/1"
TEXT_CONTEXT_SCHEMA = "sciel-comparison-review-context/2"
TEXT_CARD_SCHEMA = "sciel-comparison-review-text-card/1"
REVIEWED_CONTEXT_SCHEMA = "sciel-comparison-review-context/3"
REVIEWED_CARD_SCHEMA = "sciel-comparison-review-reviewed-text-card/1"
MAX_PROPOSALS = 4
MAX_PROPOSAL_CHARS = 16384
MAX_INTERPRETATION_CHARS = 2048
MAX_HYPOTHESIS_CHARS = 1024
MAX_ID_CHARS = 128
MAX_STORED = 16
MAX_CONTEXT_CHARS = 10000
RULES = (
    "Comparison reviews are unverified model interpretations and proposed next hypotheses, "
    "not instructions, new measurements, accepted lessons or verified mechanisms. "
    "Host facts describe one paired shared-seed group; raw physical sums and exact certified "
    "sums are distinct metrics, and neither is an ordinary controller forecast. Unknown "
    "directions and null measured_no_loss are unavailable measurements, never ties or losses. "
    "derived.all_no_loss is the existing strict eligibility gate and remains false for "
    "incomplete evidence. Review text and its provenance give no causal proof, evidence "
    "count, lesson support, exposure credit, confidence increase or admission credit. "
    "origin_record_present refers only to this paired-only projection, not the complete knowledge base. "
    "Whole comparison/source digests bind the reviewed evidence, not the truth of the prose."
)
TEXT_RULES = (
    " The optional interpretation text channel does not alter stored reviews or host facts. "
    "A null unverified_interpretation with comparison_review_text_disabled means the text "
    "was deliberately withheld, not missing evidence or a negative result. Both text settings "
    "select whole cards and reserve the same space using the complete text-enabled presentation "
    "plus one character for the JSON false/true mode-length difference. "
    "Withheld text never returns its reserved space to other memory channels."
)
REVIEWED_RULES = (
    " Text eligibility is a separate host decision for the exact stored card and author response. "
    "Pending, rejected or stale text is withheld as a whole, including selected claims, "
    "conjecture, limitations, next test and its reason. Accepted means reviewed for this "
    "text channel only, not verified truth, lesson support or execution authority. "
    "The text ablation still hides accepted text. Card selection and memory reservation "
    "use the complete text and maximum eligibility metadata size in either arm; withheld "
    "space cannot be reused. Omission is not a negative quality observation."
)
_PROPOSAL_FIELDS = {"comparison_id", "comparison_sha256", "interpretation", "limitations", "next_hypothesis"}
_NEXT_FIELDS = {"change", "expected_observation", "falsification"}
_WRITER_FIELDS = {"phase", "reference", "raw_response_sha256"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _id(value):
    return (type(value) is str and 0 < len(value) <= MAX_ID_CHARS
            and not any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value))


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _text(value, limit):
    return type(value) is str and 0 < len(value) <= limit and bool(value.strip())


def validate_proposals(value):
    """Validate the exact model-writable shape and return a detached list."""
    _require(type(value) is list and len(value) <= MAX_PROPOSALS,
             "comparison reviews require at most four proposals")
    seen = set()
    for proposal in value:
        _require(type(proposal) is dict and set(proposal) == _PROPOSAL_FIELDS,
                 "invalid comparison review proposal fields")
        ident = proposal["comparison_id"]
        _require(_id(ident) and ident not in seen, "invalid or duplicate comparison review ID")
        seen.add(ident)
        _require(_sha(proposal["comparison_sha256"]), "invalid comparison review digest")
        for key in ("interpretation", "limitations"):
            _require(_text(proposal[key], MAX_INTERPRETATION_CHARS),
                     "invalid comparison review " + key)
        hypothesis = proposal["next_hypothesis"]
        _require(hypothesis is None or (type(hypothesis) is dict and set(hypothesis) == _NEXT_FIELDS
                 and all(_text(hypothesis[k], MAX_HYPOTHESIS_CHARS) for k in _NEXT_FIELDS)),
                 "invalid comparison review next hypothesis")
    _require(len(canonical_json(value)) <= MAX_PROPOSAL_CHARS,
             "comparison review proposals exceed character limit")
    return copy.deepcopy(value)


def _writer(value):
    _require(type(value) is dict and set(value) == _WRITER_FIELDS
             and type(value["phase"]) is str and value["phase"] == "reflection" and type(value["reference"]) is int
             and value["reference"] > 0 and _sha(value["raw_response_sha256"]),
             "invalid comparison review writer")
    return copy.deepcopy(value)


def _groups(groups):
    _require(type(groups) is list, "comparison reviews require full comparison groups")
    result = {}
    for supplied in groups:
        group = validate_comparison(supplied)
        ident = group["id"]
        _require(_id(ident) and ident not in result, "invalid or duplicate authoritative comparison ID")
        _require(_id(group["branch"]), "invalid comparison review branch")
        result[ident] = group
    return result


def _card(proposal, groups, writer):
    ident = proposal["comparison_id"]
    _require(ident in groups, "comparison review references unknown group")
    group = groups[ident]
    _require(proposal["comparison_sha256"] == canonical_sha(group),
             "comparison review group digest mismatch")
    facts = comparison_facts([group])["paired"][ident]
    return {
        "schema": CARD_SCHEMA, "comparison_id": ident,
        "comparison_sha256": proposal["comparison_sha256"], "branch": group["branch"],
        "origin_experiment_id": group["origin_experiment_id"],
        "comparator_source_sha256": group["comparator_source_sha256"],
        "challenger_source_sha256": group["challenger_source_sha256"],
        "seeds": [row["seed"] for row in group["pairs"]],
        "host_facts": {
            "derived": copy.deepcopy(group["derived"]), "signs": facts,
            "measured_no_loss": [{"seed": row["seed"], "no_loss": row["no_loss"]
                if row["comparator_status"] == row["challenger_status"] == "valid" else None}
                for row in group["derived"]["pairs"]],
        },
        "unverified_interpretation": {key: copy.deepcopy(proposal[key]) for key in
                                      ("interpretation", "limitations", "next_hypothesis")},
        "writer": copy.deepcopy(writer),
    }


def build_cards(proposals, groups, writer, *, version=1, records=None, seen=None):
    """Bind at most four proposals to independently validated full groups."""
    _require(type(version) is int and version in (1, 2, 3), 'unsupported comparison review version')
    if version in (2, 3):
        if version == 2:
            from . import grounded_review as contract
        else:
            from . import reasoned_review as contract
        proposals = contract.validate_proposals(proposals)
        groups = _groups(groups)
        _require(all(p['comparison_id'] in groups for p in proposals), 'comparison review references unknown group')
        return {p['comparison_id']: contract.build_card(p, groups[p['comparison_id']], writer,
                    records=records, seen=seen) for p in proposals}
    proposals = validate_proposals(proposals)
    writer = _writer(writer)
    groups = _groups(groups)
    return {proposal["comparison_id"]: _card(proposal, groups, writer) for proposal in proposals}


def validate_cards(cards, groups, *, seen=None):
    """Rebuild every retained card; prose authenticity needs the writer journal."""
    _require(type(cards) is dict and len(cards) <= MAX_STORED,
             "comparison reviews exceed retained card limit")
    canonical_json(cards)
    groups = _groups(groups)
    rebuilt = {}
    for ident, card in cards.items():
        _require(_id(ident) and type(card) is dict and card.get("comparison_id") == ident,
                 "invalid retained comparison review ID")
        if card.get('schema') in ('sciel-comparison-review-card/2', 'sciel-comparison-review-card/3'):
            if card['schema'] == 'sciel-comparison-review-card/2':
                from .grounded_review import validate_card
            else:
                from .reasoned_review import validate_card
            _require(ident in groups, 'comparison review references unknown group')
            rebuilt[ident] = validate_card(card, groups[ident], seen=seen)
            continue
        text = card.get("unverified_interpretation")
        _require(type(text) is dict and set(text) == {"interpretation", "limitations", "next_hypothesis"},
                 "invalid retained comparison interpretation")
        proposal = {"comparison_id": ident, "comparison_sha256": card.get("comparison_sha256"), **text}
        validate_proposals([proposal])
        rebuilt[ident] = _card(proposal, groups, _writer(card.get("writer")))
        _require(canonical_json(card) == canonical_json(rebuilt[ident]),
                 "comparison review card differs from authoritative group")
    return rebuilt


def _sized(value):
    value["rendered_chars"] = 0
    while value["rendered_chars"] != len(canonical_json(value)):
        value["rendered_chars"] = len(canonical_json(value))
    return value


def _text_card(card, *, text_enabled):
    """A presentation-only projection; the durable card schema is unchanged."""
    result = copy.deepcopy(card)
    result["schema"] = TEXT_CARD_SCHEMA
    result["interpretation_omission"] = None if text_enabled else "comparison_review_text_disabled"
    if not text_enabled:
        result["unverified_interpretation"] = None
    return result


def _sized_text(value, *, reserved_chars=None):
    """Count the actual wire and the full ON wire plus its shared mode allowance.

    JSON false is one character longer than true. The same one-character
    allowance in both arms keeps empty or fully omitted views within reserve.
    """
    value["rendered_chars"] = 0
    value["reserved_chars"] = 0 if reserved_chars is None else reserved_chars
    while True:
        size = len(canonical_json(value))
        if value["rendered_chars"] == size and (reserved_chars is not None or value["reserved_chars"] == size + 1):
            return value
        value["rendered_chars"] = size
        if reserved_chars is None:
            value["reserved_chars"] = size + 1


def _text_context(candidates, branch, max_chars, max_items, enabled, text_enabled):
    # Build the complete ON view first, including omission metadata. Both arms
    # use exactly this admission decision and reserve, even when OFF is shorter.
    candidates = [_text_card(card, text_enabled=True) for card in candidates]
    count = len(candidates)
    block = _sized_text({"schema": TEXT_CONTEXT_SCHEMA, "rules": RULES + TEXT_RULES,
        "branch": branch, "enabled": enabled, "text_enabled": True,
        "cards": [], "omitted": [], "retained_count": count,
        "presented_count": 0, "omitted_count": count, "omission_details_omitted_count": count,
        "max_items": max_items, "max_chars": max_chars})
    if block["reserved_chars"] > max_chars:
        return None
    omissions = []
    for card in candidates:
        reason = ("disabled" if not enabled else "item_limit" if len(block["cards"]) >= max_items else None)
        if reason is None:
            trial = copy.deepcopy(block)
            trial["cards"].append(card)
            trial["presented_count"] += 1
            trial["omitted_count"] -= 1
            trial["omission_details_omitted_count"] -= 1
            if _sized_text(trial)["reserved_chars"] <= max_chars:
                block = trial
                continue
            reason = "memory_char_limit"
        omissions.append({"comparison_id": card["comparison_id"], "reason": reason,
            "card_chars": len(canonical_json(card)), "card_sha256": canonical_sha(card)})
    for omission in omissions:
        trial = copy.deepcopy(block)
        trial["omitted"].append(omission)
        trial["omission_details_omitted_count"] -= 1
        if _sized_text(trial)["reserved_chars"] <= max_chars:
            block = trial
    if text_enabled:
        return block
    reserved = block["reserved_chars"]
    block["text_enabled"] = False
    block["cards"] = [_text_card(card, text_enabled=False) for card in block["cards"]]
    block = _sized_text(block, reserved_chars=reserved)
    _require(block["rendered_chars"] <= reserved,
             "disabled review text exceeds its full-text reservation")
    return block


def _sized_reviewed(value, *, reserved_chars=None):
    """Reserve full text plus maximum-size gate/omission metadata, not its view.

    The accepted/rejected labels have eight characters and decision hashes have
    64. Reserving those maxima prevents pending/stale status from admitting an
    extra card. The masks are applied only after this selection is complete.
    """
    value['rendered_chars'] = 0
    value['reserved_chars'] = 0 if reserved_chars is None else reserved_chars
    allowance = 1  # false/true
    for card in value['cards']:
        status = card['text_eligibility']
        allowance += 8 - len(status['status'])
        allowance += (len(canonical_json('0' * 64))
                      - len(canonical_json(status['decision_sha256'])))
        allowance += len(canonical_json('comparison_review_text_ineligible')) - len('null')
    while True:
        size = len(canonical_json(value))
        reserve = size + allowance if reserved_chars is None else reserved_chars
        if value['rendered_chars'] == size and value['reserved_chars'] == reserve:
            return value
        value['rendered_chars'], value['reserved_chars'] = size, reserve


def _reviewed_context(candidates, decisions, stored_hashes, branch, max_chars,
                      max_items, enabled, text_enabled, policy_sha256, policy):
    candidates = [dict(copy.deepcopy(card), schema=REVIEWED_CARD_SCHEMA,
        stored_card_sha256=stored_hashes[card['comparison_id']],
        text_eligibility=decisions[card['comparison_id']], interpretation_omission=None)
        for card in candidates]
    count = len(candidates)
    block = _sized_reviewed(dict(schema=REVIEWED_CONTEXT_SCHEMA,
        rules=RULES + REVIEWED_RULES, branch=branch, enabled=enabled, text_enabled=True,
        text_policy_sha256=policy_sha256, decisions_sha256=canonical_sha(policy),
        cards=[], omitted=[], retained_count=count, presented_count=0,
        omitted_count=count, omission_details_omitted_count=count,
        max_items=max_items, max_chars=max_chars))
    if block['reserved_chars'] > max_chars:
        return None
    omissions = []
    for card in candidates:
        reason = ('disabled' if not enabled else
                  'item_limit' if len(block['cards']) >= max_items else None)
        if reason is None:
            trial = copy.deepcopy(block)
            trial['cards'].append(card)
            trial['presented_count'] += 1
            trial['omitted_count'] -= 1
            trial['omission_details_omitted_count'] -= 1
            if _sized_reviewed(trial)['reserved_chars'] <= max_chars:
                block = trial
                continue
            reason = 'memory_char_limit'
        omissions.append(dict(comparison_id=card['comparison_id'], reason=reason,
            card_chars=len(canonical_json(card)), card_sha256=card['stored_card_sha256']))
    for omission in omissions:
        trial = copy.deepcopy(block)
        trial['omitted'].append(omission)
        trial['omission_details_omitted_count'] -= 1
        if _sized_reviewed(trial)['reserved_chars'] <= max_chars:
            block = trial
    reserve = block['reserved_chars']
    block['text_enabled'] = text_enabled
    for card in block['cards']:
        eligible = card['text_eligibility']['status'] == 'accepted'
        if not eligible or not text_enabled:
            card['unverified_interpretation'] = None
            card['interpretation_omission'] = ('comparison_review_text_ineligible' if not eligible
                                              else 'comparison_review_text_disabled')
    block = _sized_reviewed(block, reserved_chars=reserve)
    _require(block['rendered_chars'] <= reserve, 'gated review text exceeds its full-text reservation')
    return block


def context(cards, groups, branch, max_chars, max_items=MAX_PROPOSALS, enabled=True,
            text_enabled=None, *, seen=None, text_policy=None, policy_sha256=None):
    """Present newest whole cards in authoritative group order with omissions.

    Even an empty or disabled channel returns explicit metadata when that fits.
    Counts stay exact when individual omission details cannot fit. Budgets count
    the entire canonical JSON, including escaped text and rendered_chars itself.
    """
    _require(_id(branch), "invalid comparison review context branch")
    _require(type(max_chars) is int and 0 <= max_chars <= MAX_CONTEXT_CHARS,
             "invalid comparison review context character limit")
    _require(type(max_items) is int and 0 <= max_items <= MAX_PROPOSALS,
             "invalid comparison review context item limit")
    _require(type(enabled) is bool, "invalid comparison review enabled flag")
    _require(text_enabled is None or type(text_enabled) is bool,
             "invalid comparison review text enabled flag")
    cards = validate_cards(cards, groups, seen=seen)
    decisions, stored_hashes = None, None
    if text_policy is not None:
        from .review_eligibility import validate_policy, decision_for
        _require(type(text_enabled) is bool and _sha(policy_sha256),
                 'gated comparison reviews require explicit text mode and policy SHA')
        text_policy = validate_policy(text_policy)
        decisions = {ident: decision_for(card, text_policy, policy_sha256=policy_sha256)
                     for ident, card in cards.items()}
        stored_hashes = {ident: canonical_sha(card) for ident, card in cards.items()}
    else:
        _require(policy_sha256 is None, 'comparison review policy SHA supplied without text policy')
    from .grounded_review import presentation_card as grounded_presentation
    from .reasoned_review import presentation_card as reasoned_presentation
    cards = {ident: grounded_presentation(card) if card['schema'] == 'sciel-comparison-review-card/2'
             else reasoned_presentation(card) if card['schema'] == 'sciel-comparison-review-card/3'
             else card for ident, card in cards.items()}
    candidates = [cards[group["id"]] for group in reversed(groups)
                  if group["id"] in cards and group["branch"] == branch]
    if text_policy is not None:
        return _reviewed_context(candidates, decisions, stored_hashes, branch, max_chars,
                                 max_items, enabled, text_enabled, policy_sha256, text_policy)
    if text_enabled is not None:
        return _text_context(candidates, branch, max_chars, max_items, enabled, text_enabled)
    count = len(candidates)
    block = _sized({"schema": CONTEXT_SCHEMA, "rules": RULES, "branch": branch,
        "enabled": enabled, "cards": [], "omitted": [], "retained_count": count,
        "presented_count": 0, "omitted_count": count, "omission_details_omitted_count": count,
        "max_items": max_items, "max_chars": max_chars})
    if block["rendered_chars"] > max_chars:
        return None
    omissions = []
    for card in candidates:
        reason = ("disabled" if not enabled else "item_limit" if len(block["cards"]) >= max_items else None)
        if reason is None:
            trial = copy.deepcopy(block)
            trial["cards"].append(card)
            trial["presented_count"] += 1
            trial["omitted_count"] -= 1
            trial["omission_details_omitted_count"] -= 1
            if _sized(trial)["rendered_chars"] <= max_chars:
                block = trial
                continue
            reason = "memory_char_limit"
        omissions.append({"comparison_id": card["comparison_id"], "reason": reason,
            "card_chars": len(canonical_json(card)), "card_sha256": canonical_sha(card)})
    for omission in omissions:
        trial = copy.deepcopy(block)
        trial["omitted"].append(omission)
        trial["omission_details_omitted_count"] -= 1
        if _sized(trial)["rendered_chars"] <= max_chars:
            block = trial
    return block


def validate_context(value, cards, groups, branch, enabled=True, text_enabled=None, *, seen=None,
                     text_policy=None, policy_sha256=None):
    """Rebuild a prompt snapshot from authoritative cards, groups and budgets."""
    _require(type(value) is dict, "invalid comparison review context")
    expected = context(cards, groups, branch, value.get("max_chars"),
                       value.get("max_items"), enabled, text_enabled, seen=seen,
                       text_policy=text_policy, policy_sha256=policy_sha256)
    _require(expected is not None and canonical_json(value) == canonical_json(expected),
             "comparison review context differs from authoritative cards")
    return expected


def identity():
    """Bind this complete storage/presentation contract to its source bytes."""
    return {"schema": SCHEMA, "card_schema": CARD_SCHEMA, "context_schema": CONTEXT_SCHEMA,
        "text_context_schema": TEXT_CONTEXT_SCHEMA, "text_card_schema": TEXT_CARD_SCHEMA,
        "reviewed_context_schema": REVIEWED_CONTEXT_SCHEMA, "reviewed_card_schema": REVIEWED_CARD_SCHEMA,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "max_proposals": MAX_PROPOSALS, "max_proposal_chars": MAX_PROPOSAL_CHARS,
        "max_stored": MAX_STORED, "max_context_chars": MAX_CONTEXT_CHARS}
