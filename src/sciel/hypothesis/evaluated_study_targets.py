"""Prospective targets compiled from one authenticated detached study.

Model source choices and motivation are not authorization. The host supplies all
study, authority, comparison and quality-policy bindings; no legacy component
or native observation is invented.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from . import comparison_targets as legacy
from ..knowledge_base.catalogs import canonical_json, canonical_sha

SCHEMA = 'sciel-evaluated-study-target/1'
MAX_REFERENCES = 4
MAX_FACT_ID_CHARS = 256
PROPOSAL_FIELDS = {'base_source_sha256', 'comparator_source_sha256', 'change',
                   'diagnostics', 'motivating_evidence'}
SCOPE = ('A prospective source edit within the exact authenticated study registry. '
    'Motivating evidence is separate from the proposed source pair. Resolved facts '
    'do not prove prose, source-change fidelity, causality, execution, native '
    'measurement, lesson support or improvement. A later candidate must prove its '
    'edit from the declared base and use a separately frozen paired evaluation.')


def _require(ok, message):
    if not ok:
        raise ValueError('evaluated study target: ' + message)


def validate_references(refs):
    _require(type(refs) is list and 1 <= len(refs) <= MAX_REFERENCES, 'invalid reference count')
    _require(all(type(ref) is str and 0 < len(ref) <= MAX_FACT_ID_CHARS
        and not any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in ref) for ref in refs),
        'invalid fact reference')
    _require(len(set(refs)) == len(refs), 'duplicate fact reference')
    return copy.deepcopy(refs)


def validate_proposal(value):
    _require(type(value) is dict and set(value) == PROPOSAL_FIELDS, 'invalid proposal fields')
    validate_references(value['motivating_evidence'])
    legacy.validate_target(dict(schema=legacy.SCHEMA, quality_gate=legacy.QUALITY_GATE,
        **{key: value[key] for key in PROPOSAL_FIELDS - {'motivating_evidence'}}))
    return copy.deepcopy(value)


def _resolve(refs, catalog):
    """Private: catalog must be freshly authenticated within this call chain."""
    refs = validate_references(refs)
    _require(all(ref in catalog['facts'] for ref in refs), 'unknown motivating fact')
    return [dict(fact_id=ref, **copy.deepcopy(catalog['facts'][ref])) for ref in refs]


def _compile(value, bundle, catalog):
    """Private: bundle must be the input from which catalog was authenticated."""
    value = validate_proposal(value)
    capsule = bundle['capsule']
    allowed = set(capsule['source_registry'])
    _require(value['base_source_sha256'] in allowed and value['comparator_source_sha256'] in allowed,
             'target source outside exact study registry')
    return dict(schema=SCHEMA, study_sha256=canonical_sha(capsule),
        author_authority_sha256=canonical_sha(bundle['author_authority']),
        policy_sha256=canonical_sha(capsule['validation_policy']),
        receiving_root=copy.deepcopy(capsule['receiving_root']),
        comparison_id=catalog['comparison_id'], comparison_sha256=catalog['comparison_sha256'],
        **{key: copy.deepcopy(value[key]) for key in PROPOSAL_FIELDS - {'motivating_evidence'}},
        quality_gate=legacy.QUALITY_GATE,
        motivating_evidence=_resolve(value['motivating_evidence'], catalog))


def compile_target(proposal, *, bundle):
    from .evaluated_study_review import source_catalog
    return _compile(proposal, bundle, source_catalog(bundle))


def validate_target(target, *, bundle):
    from .evaluated_study_review import source_catalog
    _require(type(target) is dict and target.get('schema') == SCHEMA, 'unsupported target schema')
    try:
        proposal = {key: copy.deepcopy(target[key]) for key in PROPOSAL_FIELDS - {'motivating_evidence'}}
        proposal['motivating_evidence'] = [row['fact_id'] for row in target['motivating_evidence']]
        rebuilt = _compile(proposal, bundle, source_catalog(bundle))
        _require(canonical_json(target) == canonical_json(rebuilt), 'target differs from exact study bindings')
        return rebuilt
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError('evaluated study target: malformed target') from exc


def json_schema():
    old = legacy.json_schema()['properties']
    properties = {key: copy.deepcopy(old[key]) for key in PROPOSAL_FIELDS - {'motivating_evidence'}}
    properties['motivating_evidence'] = dict(type='array', items={'type': 'string'},
        description='1 to 4 unique exact named fact IDs from this study; local validation checks membership.')
    return dict(type='object', additionalProperties=False, required=sorted(PROPOSAL_FIELDS),
                properties=properties, description=SCOPE)


def instruction():
    tail = ('The caller must verify' + legacy.instruction().split('The caller must verify', 1)[1])
    tail = tail.replace('quality_gate must be ', 'The host-compiled quality gate is ', 1)
    return (SCOPE + ' Supply only base_source_sha256, comparator_source_sha256, change, '
        'diagnostics and motivating_evidence. The host compiles all study/root/authority '
        'bindings and the immutable quality gate. Base and comparator may coincide. ' +
        tail)


def identity():
    return dict(schema=SCHEMA, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        legacy_source_sha256=hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest(),
        schema_sha256=canonical_sha(json_schema()), max_references=MAX_REFERENCES, scope=SCOPE)
