"""Provider-recovery receipts kept in checkpoints and the transient-failure classification used by parallel recovery."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

SCHEMA = 'sciel-provider-recovery/1'
_AUX_STOP = 'research auxiliary request stopped: LLMError'
_GEN_STOP = 'generation outcome unknown'
_IDEA_STOP = 'two-stage idea stopped: LLMError'


class ProviderRecoveryError(ValueError):
    """A stop cannot be abandoned under this deliberately narrow protocol."""


def _require(condition, message):
    if not condition:
        raise ProviderRecoveryError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def _hex(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_recoveries(value):
    """Validate optional checkpoint metadata without loading execution modules."""
    _require(type(value) is list, 'invalid provider recovery history')
    seen = set()
    for row in value:
        _require(type(row) is dict and row.get('schema') == SCHEMA,
                 'invalid provider recovery receipt schema')
        ident = row.get('recovery_id')
        _require(_hex(ident) and ident not in seen, 'invalid or duplicate provider recovery id')
        seen.add(ident)
        _require(_hex(row.get('source_checkpoint_sha256'))
                 and _hex(row.get('terminal_event_sha256')),
                 'invalid provider recovery evidence hashes')
        _require(row.get('unknown_cost') is True and row.get('retry') is False
                 and row.get('stop_reason') in (_AUX_STOP, _GEN_STOP, _IDEA_STOP)
                 and type(row.get('auto_recoverable')) is bool,
                 'invalid provider recovery accounting/stop metadata')
        cost = row.get('known_cost_total')
        _require(type(cost) in (int, float) and math.isfinite(cost) and cost >= 0,
                 'invalid provider recovery known cost')
        _require(type(row.get('request')) is dict and type(row.get('source_anchor')) is dict
                 and type(row.get('provider_failure')) is dict,
                 'invalid provider recovery provenance')
        if row['stop_reason'] == _IDEA_STOP:
            request = row['request']
            _require(request.get('kind') == 'idea' and request.get('stage') in ('card', 'code')
                     and type(request.get('reference')) is str and request['reference'].startswith('idea')
                     and type(request.get('call_index')) is int and request['call_index'] > 0
                     and _hex(row.get('idea_stop_event_sha256'))
                     and type(row.get('known_attempt_cost')) in (int, float)
                     and math.isfinite(row['known_attempt_cost']) and row['known_attempt_cost'] >= 0
                     and row.get('auto_recoverable') is True,
                     'invalid stage-bound idea recovery receipt')
    return deepcopy(value)


def _provider_failure(metadata, *, legacy, allow_legacy):
    if metadata is None:
        _require(legacy and allow_legacy, 'provider diagnostics are absent; automatic recovery is forbidden')
        return {'category': 'legacy_generation_unknown'}, False
    _require(type(metadata) is dict and metadata.get('stage') == 'request',
             'only request-stage provider failures can be abandoned')
    reason, status = metadata.get('reason'), metadata.get('http_status')
    transient = (reason in ('timeout', 'connection_error') and status is None) or (
        reason == 'http_error' and type(status) is int and (status == 429 or 500 <= status <= 599))
    _require(transient, 'provider failure is not a classified transient transport failure')
    info = {'category': reason}
    if status is not None:
        info['http_status'] = status
    return info, True
