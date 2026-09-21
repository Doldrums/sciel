"""Shared circle generation contracts and pure helpers."""
from __future__ import annotations

from .llm.client import response_accounting
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


CONTRACT = 'sciel-circle-generation-worker/1'

CONTEXT_SCOPE = 'circle_restart_body/1'

_COUNTERS = ('attempts', 'finished', 'calls_started', 'calls_returned',
             'invalid_responses', 'repeated_proposals', 'evaluations')

_PAIR_COUNTERS = ('returned_groups', 'entries_present', 'forecast_valid_entries',
                  'body_valid_entries', 'eligible_entries', 'selected_entries',
                  'no_eligible_requests', 'evaluated_entries')

def _json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)

def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False

def _journal_known_cost(event):
    response = SimpleNamespace(cost=event.get('cost'),
        cost_provenance=event.get('cost_provenance'),
        finish_reason=event.get('finish_reason'))
    expected = response_accounting(response)
    recorded = event.get('cost_accounting')
    if recorded is not None or expected['unpriced_calls']:
        if (type(recorded) is not dict or set(recorded) != set(expected)
                or not _finite(recorded.get('known_usd'))
                or type(recorded.get('unpriced_calls')) is not int
                or recorded != expected):
            raise ValueError('Circle generation journal accounting differs from response')
    return expected['known_usd']

def _integration_identity():
    # The opt-in transport, parsing, source assembly and cost/restore policies
    # jointly define a resumable run. Historical/default Engine identity stays
    # unchanged, while a modified policy cannot force-reuse old paid decisions.
    from .source_identity import package_source_hashes
    return package_source_hashes(files=(
        'eval/native_geometry.py', 'eval/packing_geometry.py',
        'campaign/budget.py', 'config.py', 'config_validation.py',
    ))
