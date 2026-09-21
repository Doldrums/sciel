"""Bounded structured error identifiers/fingerprints; no message or param text."""
from __future__ import annotations

import hashlib
import json
import re

SCHEMA = 'sciel-codex-protocol-diagnostic/4'
IDENTIFIER_SCHEMA = 'sciel-codex-protocol-diagnostic/3'
PREVIOUS_SCHEMA = 'sciel-codex-protocol-diagnostic/2'
LEGACY_SCHEMA = 'sciel-codex-protocol-diagnostic/1'
ENDPOINT = 'https://chatgpt.com/backend-api/codex/responses'
ENDPOINT_SHA256 = hashlib.sha256(ENDPOINT.encode('ascii')).hexdigest()
MAX_BYTES = 8 * 1024 * 1024
READ_SIZE = 65536

RECOVERABLE_CODES = frozenset({
    'stream_eof', 'stream_limit', 'done_without_terminal', 'invalid_utf8',
    'invalid_json', 'invalid_event', 'invalid_index', 'invalid_item',
    'invalid_content', 'invalid_text', 'invalid_delta', 'invalid_refusal',
    'invalid_terminal',
    'transport_incomplete_read', 'transport_tls_eof',
    'provider_server_error', 'provider_overloaded',
})
CODES = RECOVERABLE_CODES | frozenset({
    'response_identity', 'model_identity', 'item_identity', 'text_conflict',
    'delta_after_done', 'provisional_text_unconfirmed',
    # A generic SSE error can describe authentication or configuration failure;
    # without an independently validated category it is never auto-abandonable.
    'stream_error',
})
EVENTS = frozenset({
    'none', 'other', 'response.created', 'response.in_progress',
    'response.output_item.added', 'response.output_item.done',
    'response.content_part.added', 'response.content_part.done',
    'response.output_text.delta', 'response.output_text.done',
    'response.refusal.delta', 'response.refusal.done',
    'response.completed', 'response.incomplete', 'response.failed', 'error',
})
TERMINALS = frozenset({'none', 'completed', 'incomplete', 'failed'})
_FIELDS = frozenset({'schema', 'code', 'endpoint_sha256', 'body_sha256', 'model',
                     'bytes_read', 'events_seen', 'last_event', 'terminal_status'})
_ERROR_FIELDS = frozenset({'layout', 'code', 'code_sha256', 'event_sha256',
                         'event_bytes', 'message_sha256', 'message_chars', 'param_kind'})
_PROVIDER_CODES = frozenset({'server_error', 'rate_limit_exceeded', 'insufficient_quota',
    'invalid_api_key', 'authentication_error', 'permission_denied', 'invalid_request_error',
    'model_not_found', 'context_length_exceeded', 'usage_limit_reached', 'unauthorized',
    'access_denied'})


def _hex(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _canonical(value):
    # ASCII escaping also handles unpaired surrogate escapes in a provider event.
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':')).encode('ascii')


def _code_identifier(value):
    """Trust only a short structured category, never arbitrary provider prose.

    This is an observability field, not a recovery category. Only the explicit
    parser category and envelope checks authorize recovery. A lexical filter cannot distinguish every
    possible short secret from an identifier; conspicuous credential forms and
    opaque IDs are excluded, and the original fingerprint is always retained.
    """
    if (type(value) is not str or not 1 <= len(value) <= 64
            or re.fullmatch(r'[A-Za-z0-9]+(?:[_.-][A-Za-z0-9]+)*', value) is None):
        return None
    lower = value.lower()
    if lower.startswith(('sk-', 'sk_', 'sess-', 'sess_', 'bearer', 'eyj',
                         'ghp_', 'github_pat_', 'xoxb-', 'xoxp-')):
        return None
    if (any(len(part) > 24 for part in re.split(r'[_.-]', value))
            or (value.isdigit() and len(value) > 6)
            or re.fullmatch(r'[0-9a-fA-F]{16,}', value)
            or re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value)):
        return None
    return value


def _provider_error(event):
    """Retain bounded public categories/fingerprints, never provider text or params.

    Only exact semantic error cores classify server_is_overloaded. Unrelated
    extension metadata contributes to the event hash, never to exposed text.
    Historical permissive nested layouts remain distinct from the new overload
    envelope; an identifier alone never supplies its missing shape evidence.
    """
    allowed = {'type', 'code', 'message', 'param', 'sequence_number'}
    sequence_ok = ('sequence_number' not in event or
        (type(event['sequence_number']) is int and event['sequence_number'] >= 0))
    flat = (set(event) <= allowed and {'type', 'code', 'message', 'param'} <= set(event)
        and type(event['message']) is str and sequence_ok)
    source = event
    layout = 'flat' if flat else 'other'
    if type(event.get('error')) is dict and not ({'code', 'message', 'param', 'response'} & set(event)):
        source, layout = event['error'], 'nested'
        if (event.get('type') == 'error'
                and sequence_ok and {'code', 'message', 'param'} <= set(source)
                and not ({'error', 'response'} & set(source))
                and ('type' not in source or source['type'] == 'server_error')
                and all('status' not in carrier or carrier['status'] in ('error', 'failed')
                        for carrier in (event, source))
                and ('sequence_number' not in source or
                     (type(source['sequence_number']) is int and source['sequence_number'] >= 0))
                and source['code'] == 'server_is_overloaded'
                and type(source['message']) is str and source['param'] is None):
            layout = 'nested_overload'
    code, message = source.get('code'), source.get('message')
    encoded = _canonical(event)
    return {'layout': layout,
        'code': code if type(code) is str and code in _PROVIDER_CODES else None,
        'code_identifier': _code_identifier(code),
        'code_sha256': hashlib.sha256(_canonical(code)).hexdigest() if type(code) is str else None,
        'event_sha256': hashlib.sha256(encoded).hexdigest(), 'event_bytes': len(encoded),
        'message_sha256': hashlib.sha256(_canonical(message)).hexdigest() if type(message) is str else None,
        'message_chars': len(message) if type(message) is str else None,
        'param_kind': 'missing' if 'param' not in source else 'null' if source['param'] is None else 'present'}


def _validate_provider_error(value, *, previous=False, overload=False):
    fields = _ERROR_FIELDS if previous else _ERROR_FIELDS | {'code_identifier'}
    if type(value) is not dict or set(value) != fields:
        return None
    layouts = ('flat', 'nested', 'other', 'nested_overload') if overload else ('flat', 'nested', 'other')
    if (value['layout'] not in layouts
            or value['param_kind'] not in ('missing', 'null', 'present')
            or (value['code'] is not None and
                (type(value['code']) is not str or value['code'] not in _PROVIDER_CODES))
            or not _hex(value['event_sha256'])):
        return None
    for key in ('code_sha256', 'message_sha256'):
        if value[key] is not None and not _hex(value[key]):
            return None
    if (value['code'] is not None and
            value['code_sha256'] != hashlib.sha256(_canonical(value['code'])).hexdigest()):
        return None
    if not previous:
        identifier = value['code_identifier']
        if identifier is not None:
            if (_code_identifier(identifier) != identifier
                    or value['code_sha256'] != hashlib.sha256(_canonical(identifier)).hexdigest()
                    or value['code'] != (identifier if identifier in _PROVIDER_CODES else None)):
                return None
        elif value['code'] is not None:
            return None  # Every finite allowlisted category has a safe identifier.
    if type(value['event_bytes']) is not int or not 0 < value['event_bytes'] <= 6 * MAX_BYTES:
        return None
    length = value['message_chars']
    if ((length is None) != (value['message_sha256'] is None)
            or (length is not None and (type(length) is not int or not 0 <= length <= MAX_BYTES))
            or (value['layout'] == 'flat' and (length is None or value['param_kind'] == 'missing'))):
        return None
    if value['layout'] == 'nested_overload' and not (
            value['code_identifier'] == 'server_is_overloaded' and value['code'] is None
            and value['param_kind'] == 'null' and length is not None):
        return None
    return dict(value)


def validate_codex_protocol(value):
    """Return a detached, exact safe envelope, or None for any invalid field.

    This authenticates shape only. Recovery must additionally bind the body,
    model and endpoint to the immutable physical request and check its code.
    """
    if type(value) is not dict:
        return None
    legacy = value.get('schema') == LEGACY_SCHEMA
    fields = _FIELDS if legacy else _FIELDS | {'provider_error'}
    if (set(value) != fields or value.get('schema') not in (
            SCHEMA, IDENTIFIER_SCHEMA, PREVIOUS_SCHEMA, LEGACY_SCHEMA)
            or value.get('endpoint_sha256') != ENDPOINT_SHA256):
        return None
    if value.get('code') == 'provider_overloaded' and value['schema'] != SCHEMA:
        return None  # No historical parser attested the strict overload shape.
    for name, allowed in (('code', CODES), ('last_event', EVENTS), ('terminal_status', TERMINALS)):
        if type(value[name]) is not str or value[name] not in allowed:
            return None
    if (type(value['body_sha256']) is not str
            or re.fullmatch(r'[0-9a-f]{64}', value['body_sha256']) is None
            or type(value['model']) is not str
            or re.fullmatch(r'[A-Za-z0-9_.:/-]{1,256}', value['model']) is None):
        return None
    for name, limit in (('bytes_read', MAX_BYTES + READ_SIZE), ('events_seen', MAX_BYTES)):
        if type(value[name]) is not int or not 0 <= value[name] <= limit:
            return None
    result = dict(value)
    if legacy:
        return None if value['code'] == 'provider_server_error' else result
    detail = value['provider_error']
    if value['code'] in ('stream_error', 'provider_server_error', 'provider_overloaded'):
        detail = _validate_provider_error(detail, previous=value['schema'] == PREVIOUS_SCHEMA,
                                         overload=value['schema'] == SCHEMA)
        if detail is None or value['last_event'] != 'error' or value['terminal_status'] != 'none':
            return None
        if value['code'] == 'provider_overloaded' and detail['layout'] != 'nested_overload':
            return None
        if value['code'] == 'provider_server_error' and not (
                detail['layout'] == 'flat' and detail['code'] == 'server_error'
                and detail['param_kind'] == 'null'):
            return None
    elif detail is not None:
        return None
    result['provider_error'] = detail
    return result


class _CodexProtocolError(ValueError):
    """Explicit parser/read-boundary checks only; text is a fixed safe code."""
    def __init__(self, code):
        if code not in CODES:
            raise ValueError('unknown Codex parser diagnostic code')
        self.code = code
        super().__init__(code)


class _ProtocolState:
    def __init__(self):
        self.bytes_read = 0
        self.events_seen = 0
        self.last_event = 'none'
        self.terminal_status = 'none'
        self.provider_error = None

    def server_error(self, event):
        self.provider_error = _provider_error(event)
        detail = self.provider_error
        if (detail['layout'] == 'flat' and detail['code'] == 'server_error'
                and detail['param_kind'] == 'null' and self.terminal_status == 'none'):
            return 'provider_server_error'
        if detail['layout'] == 'nested_overload' and self.terminal_status == 'none':
            return 'provider_overloaded'
        return 'stream_error'

    def event(self, kind):
        self.events_seen += 1
        self.last_event = kind if kind in EVENTS else 'other'
        if kind in ('response.completed', 'response.incomplete', 'response.failed'):
            # An observed terminal kind is not proof that its payload is valid.
            self.terminal_status = kind.split('.')[-1]

    def diagnostic(self, code, *, body_sha256, model):
        return validate_codex_protocol({
            'schema': SCHEMA, 'code': code, 'endpoint_sha256': ENDPOINT_SHA256,
            'body_sha256': body_sha256, 'model': model,
            'bytes_read': self.bytes_read, 'events_seen': self.events_seen,
            'last_event': self.last_event, 'terminal_status': self.terminal_status,
            'provider_error': self.provider_error,
        })
