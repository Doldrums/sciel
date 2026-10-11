"""Shared research contracts and pure helpers."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from .prompt.sources import (SCHEMA as SOURCE_SCHEMA, fragment_reference_views,
                             source_digest)
from ..knowledge_base.catalogs import canonical_json
from .journal_anchor import assert_no_uncheckpointed_research


_OUTCOME_CAPTURES = ContextVar('sciel_research_outcome_captures', default=())

def digest(code):
    return hashlib.sha256(code.encode('utf-8', errors='surrogatepass')).hexdigest()

def _fragment_views(snapshot, prompt):
    """Bind every displayed fragment to the actual full-source catalog line."""
    fragments = snapshot.research_context['fragments']
    if not fragments:
        return []
    views = fragment_reference_views(snapshot)
    try:
        # source_catalog_block emits one JSON line. A retry may append factual
        # validation feedback after it, so the catalog need not end the prompt.
        # Require the standalone marker: newlines inside source/response JSON
        # strings are escaped and cannot masquerade as framing boundaries.
        raw = prompt.rsplit('\nSOURCE_CATALOG_JSON:\n', 1)[1].split('\n', 1)[0]
        catalog = json.loads(raw)
    except (IndexError, ValueError):
        raise ValueError('research fragments require a complete source catalog') from None
    if (type(catalog) is not dict or set(catalog) != {'schema', 'catalog'} or
            catalog['schema'] != SOURCE_SCHEMA or type(catalog['catalog']) is not dict):
        raise ValueError('invalid research source catalog')
    sources = catalog['catalog']
    if any(type(code) is not str or source_digest(code) != key
           for key, code in sources.items()):
        raise ValueError('research source catalog hash mismatch')
    for fragment, view in zip(fragments, views):
        if sources.get(view['source_ref']) != fragment['code']:
            raise ValueError('research fragment is missing its exact source')
    return views

@dataclass(frozen=True)
class _FrozenPayload(Mapping):
    """JSON-backed immutable transport/job data; nested reads are detached copies."""
    payload: str

    @classmethod
    def from_dict(cls, value):
        return cls(canonical_json(value))

    def __getitem__(self, key):
        return json.loads(self.payload)[key]

    def __iter__(self):
        return iter(json.loads(self.payload))

    def __len__(self):
        return len(json.loads(self.payload))

