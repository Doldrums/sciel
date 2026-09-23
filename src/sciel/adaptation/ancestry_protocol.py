"""Resolve delivered campaign records from authenticated native commitments.

Program identities are explicit admissions, never matches by source or score.
Unsupported projections remain unknown by raising ResolutionUnknown; they must
not be replaced by empty visibility. This module never accesses audit banks.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from .types import InferenceSpec, content_hash, require_hash, require_identifier, sha256_bytes
from .visibility import validate_capsule

SCHEMA = 'sciel-adaptation-ancestry/1'


class ResolutionUnknown(ValueError):
    pass


def _need(condition, message):
    if not condition:
        raise ResolutionUnknown(message)


def _sha(text):
    return sha256_bytes(text.encode('utf-8'))


def episode_id(event):
    require_identifier(event.get('segment_id'), 'episode segment')
    _need(type(event.get('event_seq')) is int and event['event_seq'] > 0, 'missing episode sequence')
    return f"{event['segment_id']}-{event['event_seq']}"


def _binding(row):
    provenance = row['provenance']
    value = dict(episode_id=row['episode_id'], owner=row['owner'], run_id=provenance['run_id'],
        segment_id=provenance['segment_id'], event_seq=provenance['event_seq'],
        source_sha256=row['source_sha256'])
    _validate_binding(value)
    return value


def _validate_binding(value):
    _need(type(value) is dict and set(value) == {'episode_id', 'owner', 'run_id', 'segment_id',
          'event_seq', 'source_sha256'}, 'invalid ancestor episode binding')
    for field in ('episode_id', 'owner', 'run_id', 'segment_id'):
        require_identifier(value[field], field)
    require_hash(value['source_sha256'], 'ancestor source')
    _need(value['episode_id'] == episode_id(value), 'ancestor episode identity differs')


def validate_resolution(value, *, capsule_sha256, owner):
    fields = {'schema', 'capsule_sha256', 'prompt_episode_ids', 'parameter_episode_ids',
        'ancestor_episode_ids', 'donor_episode_ids', 'episode_bindings', 'adapter_bindings', 'sha256'}
    from .auxiliary_ancestry import SCHEMA as AUX_SCHEMA, validate_graph
    auxiliary = type(value) is dict and value.get('schema') == AUX_SCHEMA
    if auxiliary:
        fields |= {'auxiliary_bindings', 'prompt_auxiliary_ids', 'parameter_auxiliary_ids'}
    _need(type(value) is dict and set(value) == fields and value.get('schema') in (SCHEMA, AUX_SCHEMA),
          'missing authenticated ancestry resolution')
    _need(value['sha256'] == content_hash({k: v for k, v in value.items() if k != 'sha256'}),
          'ancestry resolution hash differs')
    _need(value['capsule_sha256'] == capsule_sha256, 'ancestry belongs to another visibility capsule')
    bindings = value['episode_bindings']
    _need(type(bindings) is list, 'invalid ancestor collection')
    for binding in bindings:
        _validate_binding(binding)
    ids = [b['episode_id'] for b in bindings]
    _need(ids == sorted(set(ids)), 'duplicate or unordered ancestor identities')
    for key in ('prompt_episode_ids', 'parameter_episode_ids', 'ancestor_episode_ids', 'donor_episode_ids'):
        values = value[key]
        _need(type(values) is list and all(type(v) is str for v in values)
              and values == sorted(set(values)) and set(values) <= set(ids), 'invalid ' + key)
    _need(value['ancestor_episode_ids'] == ids, 'ancestor bindings are incomplete')
    _need(value['donor_episode_ids'] == [b['episode_id'] for b in bindings if b['owner'] != owner],
          'donor ancestry differs from producer ownership')
    if auxiliary:
        validate_graph(value['auxiliary_bindings'], value['prompt_auxiliary_ids'],
                       value['parameter_auxiliary_ids'], ids)
    revisions = value['adapter_bindings']
    _need(type(revisions) is list, 'invalid adapter ancestry collection')
    seen = set()
    for revision in revisions:
        _need(type(revision) is dict and set(revision) == {'owner', 'adapter_id', 'adapter_sha256',
            'dataset_sha256', 'training_receipt_sha256', 'dataset_manifest_sha256'},
            'invalid adapter ancestry binding')
        require_identifier(revision['owner'], 'adapter owner')
        require_identifier(revision['adapter_id'], 'adapter id')
        for key in ('adapter_sha256', 'dataset_sha256', 'training_receipt_sha256', 'dataset_manifest_sha256'):
            require_hash(revision[key], key)
        key = revision['owner'], revision['adapter_id']
        _need(key not in seen and revision['owner'] == owner, 'duplicate/foreign adapter lineage')
        seen.add(key)
    return deepcopy(value)


def validate_episode_resolution(row, *, ancestry_sources=None):
    """Dataset verification must retain the exporter proof, including donors.

    Legacy host-authored episodes without capsules retain their old contract;
    a capsule-bearing native episode may not drop or contradict its resolution.
    This validates commitments, not an arbitrary writer's factual authority.
    """
    provenance = row['provenance']
    capsule = provenance.get('adaptation_visibility')
    resolution = provenance.get('ancestry_resolution')
    if capsule is None and resolution is None:
        _need(provenance.get('visibility_policy') != 'authenticated-native-ancestry',
              'authenticated native episode lost its visibility/ancestry commitments')
        return
    validate_capsule(capsule, prompt=row['prompt'], inference_spec=provenance['inference_spec'])
    value = validate_resolution(resolution, capsule_sha256=capsule['sha256'], owner=row['owner'])
    binding = _binding(row)
    _need(binding['episode_id'] not in value['ancestor_episode_ids'], 'episode appears in its own ancestry')
    _need(row['visible_episode_ids'] == value['prompt_episode_ids'], 'visible episodes differ from authenticated delivery')
    spec = InferenceSpec.from_dict(provenance['inference_spec'])
    if spec.adapter_id is None:
        _need(not value['parameter_episode_ids'] and not value['adapter_bindings']
              and not value.get('parameter_auxiliary_ids'),
              'base request unexpectedly declares parameter ancestry')
    else:
        matches = [r for r in value['adapter_bindings'] if r['owner'] == spec.owner and r['adapter_id'] == spec.adapter_id]
        _need(len(matches) == 1 and matches[0]['adapter_sha256'] == spec.adapter_sha256
              and matches[0]['dataset_sha256'] == spec.dataset_sha256 and value['parameter_episode_ids'],
              'parameter ancestry differs from exact inference revision')
    if value.get('auxiliary_bindings'):
        producer = validate_producer_binding(row, value)
        from .adapter_ancestry import context
        context(ancestry_sources).verify(value, producer=producer)


def validate_producer_binding(row, resolution):
    from .auxiliary_ancestry import validate_node, validate_call_revision
    node = validate_node(row['provenance'].get('producer_call_binding'))
    provenance = row['provenance']
    spec = InferenceSpec.from_dict(provenance['inference_spec'])
    if spec.adapter_id is not None:
        _need(node['proof'].get('inference_spec') == spec.to_dict(),
              'adapted auxiliary producer requires a retained typed adapter revision proof')
        validate_call_revision(node, {n['node_id']: n for n in resolution['auxiliary_bindings']})
        _need(node['proof']['adapter_revision_id'] in resolution['parameter_auxiliary_ids'],
              'adapted producer parameter revision is not a direct root')
    else:
        _need('inference_spec' not in node['proof'], 'base producer has an adapted proof')
    _need(node['kind'] == 'llm_call' and node['owner_scope'] == 'learner'
          and node['owner'] == row['owner'] and node['run_id'] == provenance['run_id']
          and node['identity'] == dict(call_id=provenance['call_id'], role=spec.role,
                                      segment_id=provenance['segment_id'])
          and node['proof']['prompt_sha256'] == _sha(row['prompt'])
          and node['proof']['inference_spec_sha256'] == content_hash(spec.to_dict())
          and node['proof']['commit_event']['segment_id'] == provenance['segment_id']
          and node['proof']['commit_event']['event_seq'] == provenance['event_seq'],
          'retained producer call differs from exported episode')
    _need(node['episode_inputs'] == sorted(set(resolution['prompt_episode_ids'])
          | set(resolution['parameter_episode_ids']))
          and node['auxiliary_inputs'] == sorted(set(resolution.get('prompt_auxiliary_ids', []))
          | set(resolution.get('parameter_auxiliary_ids', []))), 'producer call inputs differ from resolution')
    return node
