"""Opt-in paired reviews whose prospective contrast can use the full study.

Evidence references bind model-selected prose to named saved facts, without
proving the prose. A future source pair is independent of the reviewed group's
historical pair. Legacy generated-paired cards keep their group-local contract.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from . import generated_paired_review as legacy
from . import reasoned_review
from ..knowledge_base.catalogs import canonical_json, canonical_sha

CARD_SCHEMA = 'sciel-generated-paired-review-card/2'
PRESENTATION_SCHEMA = 'sciel-generated-paired-review-presentation/2'
SCOPE_SCHEMA = 'sciel-generated-paired-review-scope/1'
PREFIX = 'GENERATED_PAIRED_REVIEW_SCOPE_JSON:\n'
PROSE_FIELDS = ('conjecture', 'limitations', 'next_hypothesis_reason')
MODEL_FIELDS = reasoned_review.MODEL_FIELDS + ('evidence_refs',)
MAX_PROPOSALS = reasoned_review.MAX_PROPOSALS
MAX_PROPOSAL_CHARS = reasoned_review.MAX_PROPOSAL_CHARS
MAX_STORED = legacy.MAX_STORED
_FIELDS = {'comparison_id', 'comparison_sha256', *MODEL_FIELDS}

RULES = legacy.RULES + (
    'For scoped review version 2, comparison_id identifies the historical group being '
    'reviewed, not the future target comparator. Each conjecture, limitations and '
    'next_hypothesis_reason field supplies 1 to 4 distinct named evidence references, '
    'including at least one fact from that reviewed group. Other references may use '
    'any group in this same sealed component. Host evidence_bindings resolve those '
    'references to exact saved values; they do not verify that the prose follows from '
    'them, establish causality, or make a missing observation false. Model-selected '
    'references remain wholly gated with the interpretation. A non-null next target '
    'binds the exact component, receiving root and policy; its base and comparator '
    'may use the full authenticated source registry of this study. Its motivating '
    'evidence is separate from that proposed source pair. This permits a new '
    'prospective contrast, never a reinterpretation of an old measured comparison. '
    'No HOST-only source or outcome outside this component is available. The strict '
    'paired raw and exact zero-loss gate remains unchanged, including known-loss '
    'refutation with incomplete data. No target, reference or review authorizes '
    'execution, gives native-outcome or lesson credit, or guarantees improvement.\n'
)
INSTRUCTION = (
    'Review the sealed generated paired study using scoped review version 2. '
    'Keep every named claim exact and attach bounded evidence_refs separately to '
    'conjecture, limitations and next_hypothesis_reason. Clearly distinguish saved '
    'observations, unknown observations and prospective assumptions. Propose one '
    'justified bounded implementable future test, or null with a concrete reason. '
    'A target may choose its base and comparator from the full authenticated study; '
    'its motivating references must include the reviewed group and are not the '
    'planned comparator. Do not fabricate a target, a result, source code, ordinary '
    'outcome or lesson. All prose remains unverified even when references resolve.\n'
) + RULES


def _require(condition, message):
    if not condition:
        raise ValueError('scoped paired review: ' + message)


def validate_proposals(value):
    """Check shape only; factual and component bindings belong to build_cards."""
    from . import component_targets
    _require(type(value) is list and len(value) <= MAX_PROPOSALS, 'invalid proposal list')
    base = []
    for proposal in value:
        _require(type(proposal) is dict and set(proposal) == _FIELDS, 'invalid proposal fields')
        refs = proposal['evidence_refs']
        _require(type(refs) is dict and set(refs) == set(PROSE_FIELDS), 'invalid evidence fields')
        for field in PROSE_FIELDS:
            component_targets.validate_references_shape(refs[field])
        if proposal['next_hypothesis'] is not None:
            component_targets.validate_target_shape(proposal['next_hypothesis'])
        old = {key: copy.deepcopy(proposal[key]) for key in proposal if key != 'evidence_refs'}
        # Reuse unchanged legacy claim/prose/list validators, while the new target
        # has its own schema and never enters the old group-local decoder.
        old['next_hypothesis'] = None
        base.append(old)
    reasoned_review.validate_proposals(base)
    _require(len(canonical_json(value)) <= MAX_PROPOSAL_CHARS, 'proposal character limit exceeded')
    return copy.deepcopy(value)


def _card(proposal, component, catalog, catalogs, writer):
    from . import component_targets
    _require(proposal['comparison_sha256'] == catalog['comparison_sha256'], 'group binding differs')
    for claim in proposal['claims']:
        _require(catalog['facts'].get(claim['fact_id']) == claim['value'],
                 'claim differs from host fact')
    ident = proposal['comparison_id']
    evidence = {field: component_targets._resolve_references(
        proposal['evidence_refs'][field], catalogs, comparison_id=ident) for field in PROSE_FIELDS}
    evidence['target_motivation'] = None
    if proposal['next_hypothesis'] is not None:
        component_targets._validate_target(proposal['next_hypothesis'], component, catalogs, ident)
        evidence['target_motivation'] = component_targets._resolve_references(
            proposal['next_hypothesis']['motivating_evidence'], catalogs, comparison_id=ident)
    bindings = ('comparison_id', 'comparison_sha256', 'branch', 'origin_experiment_id',
        'comparator_source_sha256', 'challenger_source_sha256', 'seeds',
        'source_component_sha256', 'receiving_root')
    return dict(schema=CARD_SCHEMA, **{key: copy.deepcopy(catalog[key]) for key in bindings},
        host_facts=copy.deepcopy(catalog), evidence_bindings=evidence,
        unverified_interpretation={key: copy.deepcopy(proposal[key]) for key in MODEL_FIELDS},
        writer=copy.deepcopy(writer))


def build_cards(proposals, component, writer):
    """Bind all claims/references/targets to one authenticated study, without execution."""
    from .comparison_review import _writer
    proposals = validate_proposals(proposals)
    writer = _writer(writer)
    catalogs = legacy._catalogs(component)
    _require(all(p['comparison_id'] in catalogs for p in proposals), 'unknown comparison ID')
    return {p['comparison_id']: _card(p, component, catalogs[p['comparison_id']], catalogs, writer)
            for p in proposals}


def validate_cards(cards, component):
    """Rebuild version-2 cards exactly; the sidecar separately authenticates raw R."""
    from .comparison_review import _writer
    _require(type(cards) is dict and len(cards) <= MAX_STORED, 'invalid retained cards')
    catalogs = legacy._catalogs(component)
    rebuilt = {}
    for ident, card in cards.items():
        _require(type(card) is dict and card.get('schema') == CARD_SCHEMA
            and ident == card.get('comparison_id') and ident in catalogs, 'invalid retained card identity')
        text = card.get('unverified_interpretation')
        _require(type(text) is dict and set(text) == set(MODEL_FIELDS), 'invalid retained model fields')
        proposal = dict(comparison_id=ident, comparison_sha256=card.get('comparison_sha256'), **text)
        validate_proposals([proposal])
        rebuilt[ident] = _card(proposal, component, catalogs[ident], catalogs, _writer(card.get('writer')))
    _require(canonical_json(rebuilt) == canonical_json(cards), 'retained cards differ from bound evidence')
    return rebuilt


def presentation_card(card):
    result = copy.deepcopy(card)
    result['schema'] = PRESENTATION_SCHEMA
    result['rules'] = RULES
    return result


def source_block(component):
    """Expose fixed scope bindings; the caller also supplies full source/fact blocks."""
    from .component_targets import _source_hashes
    catalogs = legacy._catalogs(component)
    sources = sorted(_source_hashes(component))
    value = dict(schema=SCOPE_SCHEMA, component_sha256=canonical_sha(component),
        receiving_root=copy.deepcopy(component['root_parent']), policy_sha256=canonical_sha(component['policy']),
        available_source_sha256=sources,
        comparisons=[dict(comparison_id=key, comparison_sha256=catalog['comparison_sha256'])
                     for key, catalog in catalogs.items()])
    return RULES + PREFIX + canonical_json(value) + '\n'


def json_schema(component=None):
    """A distinct authoring wire; enums are bounded unions, admission checks scope."""
    from . import component_targets
    result = reasoned_review.json_schema()
    proposal = result['items']
    props = proposal['properties']
    target = component_targets.json_schema(component)
    target['type'] = ['object', 'null']
    props['next_hypothesis'] = target
    ref = dict(type='object', properties={
        'comparison_id': dict(type='string'), 'fact_id': dict(type='string')},
        required=['comparison_id', 'fact_id'], additionalProperties=False)
    if component is not None:
        vocabulary = legacy.fact_vocabulary(component)
        props['comparison_id']['enum'] = vocabulary['comparison_ids']
        props['comparison_sha256']['enum'] = vocabulary['comparison_sha256']
        props['claims']['items']['properties']['fact_id']['enum'] = vocabulary['fact_ids']
        ref['properties']['comparison_id']['enum'] = vocabulary['comparison_ids']
        ref['properties']['fact_id']['enum'] = vocabulary['fact_ids']
    props['evidence_refs'] = dict(type='object', properties={field: dict(type='array',
        items=copy.deepcopy(ref), description=(
            '1 to 4 distinct exact named references, including at least one from the reviewed '
            'comparison_id; other references may use this same component. Enforced locally. '
            'References do not establish that the associated prose is true.')) for field in PROSE_FIELDS},
        required=list(PROSE_FIELDS), additionalProperties=False)
    proposal['required'].append('evidence_refs')
    result['description'] = ('At most 4 distinct reviews and 16384 canonical JSON characters '
                             'including targets and references, enforced locally. ' + RULES)
    return result


def identity():
    from . import component_targets
    return dict(card_schema=CARD_SCHEMA, presentation_schema=PRESENTATION_SCHEMA,
        scope_schema=SCOPE_SCHEMA, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        legacy_policy=legacy.identity(), target_policy=component_targets.identity(),
        max_stored=MAX_STORED, max_proposal_chars=MAX_PROPOSAL_CHARS)
