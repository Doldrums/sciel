"""Read-only native tool/cell ancestry, from actual committed input receipts.

No tool Python is imported or executed here. Publication is not evidence of
usefulness; unsuccessful and replayed observations keep their actual ancestry.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

from .auxiliary_ancestry import GraphFragment, call_records, need, _commit
from .types import content_hash, require_hash, sha256_bytes
# Tool producers escape Unicode; adaptation graph/event hashes retain their own format.
from ..methods.models import content_hash as native_hash


def _sha(text):
    return sha256_bytes(text.encode('utf-8'))


def _event(row):
    return dict(segment_id=row['segment_id'], event_seq=row['event_seq'], event_sha256=content_hash(row))


def _order_like(value, template):
    if isinstance(value, dict) and isinstance(template, dict):
        keys = [key for key in template if key in value] + [key for key in value if key not in template]
        return {key: _order_like(value[key], template.get(key)) for key in keys}
    if isinstance(value, list) and isinstance(template, list) and len(value) == len(template):
        return [_order_like(a, b) for a, b in zip(value, template)]
    return deepcopy(value)


def validate_node(node):
    kind, identity, proof = node['kind'], node['identity'], node['proof']
    need(node['owner_scope'] == 'learner', 'tool producer is not a learner')
    if kind == 'tool_version':
        need(set(identity) == {'tool_id'} and set(proof) == {
            'source_sha256', 'manifest_sha256', 'proposal_artifact_id', 'registration'}, 'invalid tool version proof')
        require_hash(identity['tool_id'], 'tool version')
        for key in ('source_sha256', 'manifest_sha256', 'proposal_artifact_id'):
            require_hash(proof[key], key)
        _commit(proof['registration'])
    elif kind == 'tool_invocation':
        need(set(identity) == {'receipt_id', 'invocation_id'} and set(proof) == {
            'receipt_sha256', 'input_sha256', 'output_sha256', 'runtime_sha256', 'source_sha256',
            'dispatch_status', 'host_status', 'result_status', 'commit_event', 'finalization_event'},
            'invalid tool invocation proof')
        for key in identity:
            require_hash(identity[key], key)
        for key in ('receipt_sha256', 'input_sha256', 'runtime_sha256', 'source_sha256'):
            require_hash(proof[key], key)
        if proof['output_sha256'] is not None:
            require_hash(proof['output_sha256'], 'tool output')
        need(proof['dispatch_status'] in ('invoked', 'not_dispatched')
             and proof['host_status'] in ('ok', 'failed')
             and proof['result_status'] in ('success', 'tool_error', 'host_failed', 'invalid_output'), 'invalid tool outcome status')
        _commit(proof['commit_event'])
        _commit(proof['finalization_event'])
    elif kind == 'research_card':
        need(set(identity) == {'cell_id', 'revision_id', 'card_sha256'} and set(proof) == {
            'card', 'renderer_sha256', 'availability_event', 'finalization_event'}, 'invalid research card proof')
        for key in identity:
            require_hash(identity[key], key)
        from ..engine.research_cells_context import _card, _json
        card = _card(proof['card'])
        need(card is not None and identity == dict(cell_id=card['cell_id'], revision_id=card['revision_id'],
             card_sha256=_sha(_json(card))), 'research card proof differs from projection')
        require_hash(proof['renderer_sha256'], 'card renderer')
        _commit(proof['availability_event'])
        _commit(proof['finalization_event'])
    else:
        need(False, 'unsupported tool auxiliary kind')
