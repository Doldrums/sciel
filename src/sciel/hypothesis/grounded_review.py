"""Checked factual claims and prospective tests for comparison reviews v2.

Only exact catalog claims are checked. Conjectures, limitations and proposed
source changes remain unverified, including factual assertions hidden in prose.
An origin snapshot is retained solely to reproduce its named forecast feedback;
it never becomes knowledge-base evidence, support or a new observation.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from ..knowledge_base.catalogs import canonical_json, canonical_sha
from ..knowledge_base.comparison_facts import comparison_facts
from ..knowledge_base.paired_comparisons import validate_comparison
from ..knowledge_base.restart_prediction_context import _card as prediction_card, NAMES

CARD_SCHEMA = 'sciel-comparison-review-card/2'
CATALOG_SCHEMA = 'sciel-comparison-review-facts/2'
MAX_CLAIMS = 24
MAX_ORIGIN_CHARS = 32768
MAX_PROPOSALS = 4
MAX_PROPOSAL_CHARS = 16384
_FIELDS = {'comparison_id', 'comparison_sha256', 'claims', 'conjecture', 'limitations', 'next_hypothesis'}
MODEL_FIELDS = ('claims', 'conjecture', 'limitations', 'next_hypothesis')
RULES = (
    'Only claims with exact fact_id/value matches have been checked against bound host facts. '
    'Conjecture, limitations and source-change text are unverified even when they mention facts. '
    'A checked claim is not a verified interpretation or causal mechanism. The two metric objects '
    'are candidate raw radii and a radius-decreased certified packing, not two representations '
    'of an identical packing. Forecast verdicts concern named observables on retained challenger '
    'calls only; INCONCLUSIVE is unknown. Next-test targets are prospective and carry no execution, '
    'preregistration, quality, lesson or promotion credit. The immutable quality gate requires '
    'all paired raw and exact deltas nonnegative and a strict gain in each metric, without epsilon.'
)


def _require(ok, message):
    if not ok:
        raise ValueError('grounded comparison review: ' + message)


def validate_proposals(value):
    from .comparison_review import _id, _sha, _text
    from .comparison_targets import validate_target
    _require(type(value) is list and len(value) <= MAX_PROPOSALS, 'invalid proposal list')
    seen = set()
    for row in value:
        _require(type(row) is dict and set(row) == _FIELDS, 'invalid proposal fields')
        _require(_id(row['comparison_id']) and row['comparison_id'] not in seen, 'invalid or duplicate group ID')
        seen.add(row['comparison_id'])
        _require(_sha(row['comparison_sha256']), 'invalid group digest')
        _require(_text(row['conjecture'], 1024) and _text(row['limitations'], 2048), 'invalid unverified prose')
        claims = row['claims']
        _require(type(claims) is list and 1 <= len(claims) <= MAX_CLAIMS, 'one to 24 claims required')
        refs = set()
        for claim in claims:
            _require(type(claim) is dict and set(claim) == {'fact_id', 'value'}
                     and _id(claim['fact_id']) and _text(claim['value'], 128), 'invalid factual claim')
            _require(claim['fact_id'] not in refs, 'duplicate factual claim')
            refs.add(claim['fact_id'])
        if row['next_hypothesis'] is not None:
            validate_target(row['next_hypothesis'])
    _require(len(canonical_json(value)) <= MAX_PROPOSAL_CHARS, 'proposal character limit exceeded')
    return copy.deepcopy(value)


def _origin(group, records, seen, supplied):
    ident = group['origin_experiment_id']
    snapshot = supplied if supplied is not None else (records or {}).get(ident)
    if snapshot is None:
        return None
    _require(type(snapshot) is dict and snapshot.get('id') == ident
             and 'code_fragment' not in snapshot and 'code_fragment_sha256' in snapshot,
             'origin must be a normalized observation snapshot')
    _require(len(canonical_json(snapshot)) <= MAX_ORIGIN_CHARS, 'origin snapshot exceeds character limit')
    _require(type(seen) is dict and seen.get(ident) == canonical_sha(snapshot),
             'origin snapshot differs from lifetime observation digest')
    return copy.deepcopy(snapshot)


def source_catalog(group, *, records=None, seen=None, origin_snapshot=None):
    """Recompute named facts from one full group and an authenticated origin.

    ``records`` must be normalized KB evidence, not recent full code records.
    A missing origin produces explicit INCONCLUSIVE forecast facts. A supplied
    but unauthenticated origin is rejected rather than silently ignored.
    """
    group = validate_comparison(group)
    origin = _origin(group, records, seen, origin_snapshot)
    signs = comparison_facts([group])['paired'][group['id']]
    facts = {'quality:outcome': group['derived']['outcome'],
             'quality:all_valid': str(group['derived']['all_valid']).lower(),
             'quality:all_no_loss': str(group['derived']['all_no_loss']).lower(),
             'metric:raw_object': 'candidate_raw_radii',
             'metric:certified_object': 'radius_decreased_certified_packing'}
    for row, measured in zip(signs['pairs'], group['derived']['pairs']):
        seed = row['seed']
        facts[f'pair:{seed}:raw_direction'] = row['raw_physical_sum_direction']
        facts[f'pair:{seed}:certified_direction'] = row['exact_certified_sum_direction']
        facts[f'pair:{seed}:measured_no_loss'] = (str(measured['no_loss']).lower()
            if measured['comparator_status'] == measured['challenger_status'] == 'valid' else 'unknown')
    feedback = prediction_card(group, {origin['id']: origin} if origin is not None else {})
    rows = {r[0]: r for r in feedback.get('rows', [])}
    for pair in group['pairs']:
        seed = pair['seed']
        verdicts = rows[seed][4] if seed in rows else ['INCONCLUSIVE'] * len(NAMES)
        for name, verdict in zip(NAMES, verdicts):
            facts[f'restart:{seed}:{name}:verdict'] = verdict
    return dict(schema=CATALOG_SCHEMA, comparison_id=group['id'], comparison_sha256=canonical_sha(group),
        comparator_source_sha256=group['comparator_source_sha256'],
        challenger_source_sha256=group['challenger_source_sha256'],
        origin_sha256=canonical_sha(origin) if origin is not None else None,
        restart_feedback=feedback, facts=facts)


def build_card(proposal, group, writer, *, records=None, seen=None):
    from .comparison_review import _writer
    from .comparison_targets import validate_target
    proposal = validate_proposals([proposal])[0]
    group = validate_comparison(group)
    _require(proposal['comparison_id'] == group['id']
             and proposal['comparison_sha256'] == canonical_sha(group), 'group binding mismatch')
    origin = _origin(group, records, seen, None)
    catalog = source_catalog(group, origin_snapshot=origin, seen=seen)
    for claim in proposal['claims']:
        _require(catalog['facts'].get(claim['fact_id']) == claim['value'],
                 'claim differs from host fact: ' + claim['fact_id'])
    if proposal['next_hypothesis'] is not None:
        validate_target(proposal['next_hypothesis'], sources={group['comparator_source_sha256'],
            group['challenger_source_sha256']})
    return dict(schema=CARD_SCHEMA, comparison_id=group['id'], comparison_sha256=canonical_sha(group),
        branch=group['branch'], origin_experiment_id=group['origin_experiment_id'],
        comparator_source_sha256=group['comparator_source_sha256'],
        challenger_source_sha256=group['challenger_source_sha256'], seeds=[p['seed'] for p in group['pairs']],
        origin_snapshot=origin, host_facts=catalog,
        unverified_interpretation={k: copy.deepcopy(proposal[k]) for k in MODEL_FIELDS},
        writer=_writer(writer))


def validate_card(card, group, *, seen=None):
    _require(type(card) is dict and card.get('schema') == CARD_SCHEMA, 'unsupported card schema')
    text = card.get('unverified_interpretation')
    _require(type(text) is dict and set(text) == set(MODEL_FIELDS), 'invalid retained proposal')
    origin = _origin(group, None, seen, card.get('origin_snapshot'))
    proposal = dict(comparison_id=card.get('comparison_id'), comparison_sha256=card.get('comparison_sha256'), **text)
    rebuilt = build_card(proposal, group, card.get('writer'),
        records={origin['id']: origin} if origin is not None else {}, seen=seen)
    _require(canonical_json(card) == canonical_json(rebuilt), 'retained card differs from bound sources')
    return rebuilt


def presentation_card(card):
    """Exclude the private origin snapshot, keeping its digest and named facts."""
    result = copy.deepcopy(card)
    result.pop('origin_snapshot')
    result['schema'] = 'sciel-comparison-review-grounded-presentation/1'
    result['rules'] = RULES
    return result


def source_block(groups, *, records=None, seen=None):
    _require(type(groups) is list and len(groups) <= MAX_PROPOSALS, 'invalid source group list')
    _require(len({g['id'] for g in groups}) == len(groups), 'duplicate source groups')
    catalogs = [source_catalog(g, records=records, seen=seen) for g in groups]
    return RULES + '\nCOMPARISON_REVIEW_SOURCES_JSON:\n' + canonical_json(catalogs) + '\n'


def json_schema():
    from .comparison_targets import json_schema as target_schema
    def obj(properties):
        return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)
    def text(limit):
        return dict(type='string', description=f'Nonblank, at most {limit} characters; enforced locally.')
    target = target_schema()
    target['type'] = ['object', 'null']
    proposal = obj(dict(comparison_id=text(128), comparison_sha256=text(64),
        claims=dict(type='array', minItems=1, items=obj(dict(fact_id=text(128), value=text(128))),
                    description='One to 24 distinct named catalog claims; exact fact IDs and values checked locally.'),
        conjecture=text(1024), limitations=text(2048), next_hypothesis=target))
    return dict(type='array', items=proposal, description='At most 4 distinct comparison reviews and '
        '16384 canonical JSON characters, enforced locally. ' + RULES)


def identity():
    from . import comparison_targets
    return dict(card_schema=CARD_SCHEMA, catalog_schema=CATALOG_SCHEMA,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        target_source_sha256=hashlib.sha256(Path(comparison_targets.__file__).read_bytes()).hexdigest())
