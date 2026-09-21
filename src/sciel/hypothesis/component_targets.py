"""Study-scoped prospective targets, never execution or semantic authority.

Only the generated-paired-history/1 family is supported. Public validators
rebuild its source proofs and factual catalogs. The external configured digest
and raw author authentication remain the caller's responsibility. No target
creates a candidate, native outcome, measurement, budget or dispatch permission.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from . import comparison_targets as legacy
from ..knowledge_base.catalogs import canonical_json, canonical_sha

SCHEMA = 'sciel-component-comparison-target/1'
MAX_REFERENCES = 4
MAX_TARGET_CHARS = 4096
MAX_FACT_ID_CHARS = 256
_FIELDS = {'schema', 'component_sha256', 'receiving_root', 'policy_sha256',
           'motivating_evidence', 'base_source_sha256', 'comparator_source_sha256',
           'change', 'quality_gate', 'diagnostics'}
SCOPE = ('Prospective source-bound intent within one authenticated generated-paired study. '
         'Named references establish linkage and exact saved values, not truth of prose, '
         'causality, implementation of the proposed change, new observations, execution '
         'authority or preregistration. A later candidate requires an edit proof from the '
         'declared base and a separately frozen fresh paired plan. Quality stays unmeasured.')


def _require(ok, reason):
    if not ok:
        raise ValueError('component target: ' + reason)


def _text(value, limit):
    return type(value) is str and 0 < len(value) <= limit and bool(value.strip())


def validate_references_shape(refs):
    """Bound only literal reference syntax; unknown references require a catalog."""
    _require(type(refs) is list and 1 <= len(refs) <= MAX_REFERENCES, 'invalid reference count')
    seen = set()
    for ref in refs:
        _require(type(ref) is dict and set(ref) == {'comparison_id', 'fact_id'}, 'invalid reference fields')
        _require(_text(ref['comparison_id'], 128) and _text(ref['fact_id'], MAX_FACT_ID_CHARS),
                 'invalid reference ID')
        key = (ref['comparison_id'], ref['fact_id'])
        _require(key not in seen, 'duplicate reference')
        seen.add(key)
    return copy.deepcopy(refs)


def validate_target_shape(value):
    """Shape only, suitable for decoding; does not authenticate any source."""
    _require(type(value) is dict and set(value) == _FIELDS, 'invalid target fields')
    _require(type(value['schema']) is str and value['schema'] == SCHEMA, 'unsupported target schema')
    for key in ('component_sha256', 'policy_sha256'):
        _require(legacy._sha(value[key]), 'invalid ' + key)
    root = value['receiving_root']
    _require(type(root) is dict and set(root) == {'id', 'source_sha256'}
             and _text(root['id'], 128) and legacy._sha(root['source_sha256']), 'invalid receiving root')
    validate_references_shape(value['motivating_evidence'])
    # Preserve the existing strict dual-quality and bounded diagnostic semantics.
    legacy.validate_target(dict(schema=legacy.SCHEMA, **{key: value[key] for key in
        ('base_source_sha256', 'comparator_source_sha256', 'change', 'quality_gate', 'diagnostics')}))
    _require(len(canonical_json(value)) <= MAX_TARGET_CHARS, 'target character limit exceeded')
    return copy.deepcopy(value)


def _catalogs(component):
    # Lazy import avoids a review -> target -> review import cycle. This performs
    # full source/proof/receipt validation, not merely a self-hash check.
    from .generated_paired_review import _catalogs as authenticated_catalogs
    try:
        return authenticated_catalogs(component)
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError('component target: malformed or unsupported component') from exc


def _resolve_references(refs, catalogs, comparison_id=None):
    """Private: catalogs MUST come from _catalogs(component) in this call chain."""
    refs = validate_references_shape(refs)
    if comparison_id is not None:
        _require(_text(comparison_id, 128) and comparison_id in catalogs, 'unknown reviewed comparison')
        _require(any(ref['comparison_id'] == comparison_id for ref in refs),
                 'references do not anchor the reviewed comparison')
    result = []
    for ref in refs:
        _require(ref['comparison_id'] in catalogs, 'unknown referenced comparison')
        catalog = catalogs[ref['comparison_id']]
        _require(ref['fact_id'] in catalog['facts'], 'unknown fact in referenced comparison')
        result.append(dict(**ref, comparison_sha256=catalog['comparison_sha256'],
                           value=catalog['facts'][ref['fact_id']]))
    return result


def resolve_references(refs, *, component, comparison_id=None):
    """Return host-resolved exact facts; quoted prose remains unverified."""
    return copy.deepcopy(_resolve_references(refs, _catalogs(component), comparison_id))


def validate_references(refs, *, component, comparison_id=None):
    """Authenticate references, preserving their exact authored representation."""
    resolve_references(refs, component=component, comparison_id=comparison_id)
    return copy.deepcopy(refs)


def _source_hashes(component):
    # Private: whole-component authentication precedes this projection. Complete
    # edit parents remain known even when no measured pair uses that parent.
    return {hashlib.sha256(proposal[key].encode('utf-8')).hexdigest()
            for proposal in component['proposals'] for key in ('source', 'parent_source')}


def _validate_target(value, component, catalogs, comparison_id):
    """Private: the full component must already have produced these catalogs."""
    value = validate_target_shape(value)
    _require(_text(comparison_id, 128) and comparison_id in catalogs, 'unknown reviewed comparison')
    _require(value['component_sha256'] == canonical_sha(component), 'component digest differs')
    _require(canonical_json(value['receiving_root']) == canonical_json(component['root_parent']),
             'receiving root differs')
    _require(value['policy_sha256'] == canonical_sha(component['policy']), 'component policy differs')
    _resolve_references(value['motivating_evidence'], catalogs, comparison_id)
    # Every catalog is built only after complete proposal parent/child sources,
    # typed source reconstruction and every group source_pair were authenticated.
    sources = _source_hashes(component)
    _require(value['base_source_sha256'] in sources and value['comparator_source_sha256'] in sources,
             'target source is outside the authenticated study')
    return value


def validate_target(value, *, component, comparison_id):
    """Validate bound intent, with no claim of trusted external admission."""
    validate_target_shape(value)  # Reject malformed model input before evidence work.
    return _validate_target(value, component, _catalogs(component), comparison_id)


def references_json_schema(component=None):
    ref = dict(type='object', additionalProperties=False,
        required=['comparison_id', 'fact_id'], properties={
            'comparison_id': dict(type='string', description='Exact comparison ID within this component.'),
            'fact_id': dict(type='string', description='Exact fact key in that referenced comparison catalog; metadata is not a fact.')})
    if component is not None:
        catalogs = _catalogs(component)
        ref['properties']['comparison_id']['enum'] = sorted(catalogs)
        ref['properties']['fact_id']['enum'] = sorted({key for c in catalogs.values() for key in c['facts']})
    return dict(type='array', items=ref, description='One to four unique references, checked locally against their own comparison. '
        'At least one reference must anchor the reviewed comparison. Union enums do not establish cross-group membership or prose truth.')


def json_schema(component=None):
    """Generic syntax or exact study-bound wire; local validation remains required."""
    old = legacy.json_schema()['properties']
    properties = {key: copy.deepcopy(old[key]) for key in
                  ('base_source_sha256', 'comparator_source_sha256', 'change', 'quality_gate', 'diagnostics')}
    properties.update(schema={'type': 'string', 'enum': [SCHEMA]},
        component_sha256=dict(type='string', description='Canonical SHA256 of the complete supplied component.'),
        receiving_root=dict(type='object', additionalProperties=False, required=['id', 'source_sha256'],
            properties={'id': {'type': 'string'}, 'source_sha256': {'type': 'string'}}),
        policy_sha256=dict(type='string', description='Canonical SHA256 of the supplied component policy, not the host text-eligibility policy.'),
        motivating_evidence=references_json_schema())
    if component is not None:
        catalogs = _catalogs(component)
        sources = sorted(_source_hashes(component))
        for key in ('base_source_sha256', 'comparator_source_sha256'):
            properties[key]['enum'] = sources
            properties[key]['description'] = ('Exact authenticated full-source SHA from this component, independently of the reviewed pair. '
                'Base and comparator may coincide. Later candidate derivation and fresh paired execution require separate checks.')
        properties['component_sha256']['enum'] = [canonical_sha(component)]
        properties['policy_sha256']['enum'] = [canonical_sha(component['policy'])]
        for key in ('id', 'source_sha256'):
            properties['receiving_root']['properties'][key]['enum'] = [component['root_parent'][key]]
        refs = properties['motivating_evidence']['items']['properties']
        refs['comparison_id']['enum'] = sorted(catalogs)
        refs['fact_id']['enum'] = sorted({key for c in catalogs.values() for key in c['facts']})
    return dict(type='object', additionalProperties=False, required=sorted(_FIELDS), properties=properties,
                description='At most4096 canonical JSON characters. ' + SCOPE)


def identity():
    return dict(schema=SCHEMA, max_references=MAX_REFERENCES, max_target_chars=MAX_TARGET_CHARS,
        max_fact_id_chars=MAX_FACT_ID_CHARS, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        legacy_target_source_sha256=hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest(),
        schema_sha256=canonical_sha(json_schema()), scope=SCOPE)
