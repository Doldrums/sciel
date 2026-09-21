"""Sanitized provider failures and bounded retry scheduling hints."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import re

from .codex_protocol import validate_codex_protocol


_ERROR_CAUSES = frozenset({
    'TimeoutError', 'ConnectionError', 'OSError', 'RuntimeError', 'ValueError',
    'TypeError', 'AttributeError', 'IndexError', 'KeyError', 'APIError',
    'APIConnectionError', 'APITimeoutError', 'APIStatusError', 'BadRequestError',
    'AuthenticationError', 'PermissionDeniedError', 'NotFoundError',
    'ConflictError', 'UnprocessableEntityError', 'RateLimitError',
    'InternalServerError', 'APIResponseValidationError',
    '_CodexProtocolError',
})
_ERROR_REASONS = frozenset({
    'timeout', 'connection_error', 'http_error', 'request_failed',
    'empty_choices', 'malformed_response',
})
MAX_RETRY_AFTER_SECONDS = 3600.0


def _retry_after(cause):
    """Read one public throttling header, never arbitrary headers or bodies.

    The bounded value is a scheduling hint, not evidence of zero billing or
    permission to retry an uncertain request. Existing attempt limits remain.
    """
    try:
        response = getattr(cause, 'response', None)
        headers = getattr(response, 'headers', None)
        value = headers.get('retry-after') if headers is not None else None
        if value is None and headers is not None:
            value = headers.get('Retry-After')
        if type(value) is not str or len(value) > 256:
            return {}
        value = value.strip()
        if re.fullmatch(r'\d+(?:\.\d+)?', value):
            seconds = float(value)
        else:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
        if not math.isfinite(seconds) or seconds < 0:
            return {}
    except Exception:
        return {}
    result = {'retry_after_seconds': min(seconds, MAX_RETRY_AFTER_SECONDS)}
    if seconds > MAX_RETRY_AFTER_SECONDS:
        result['retry_after_clamped'] = True
    return result


def _error_details(cause, *, stage, reason):
    """Fixed categories plus the single whitelisted HTTP429 Retry-After hint."""
    name = type(cause).__name__
    details = {'stage': stage, 'reason': reason,
               'cause_type': name if name in _ERROR_CAUSES else 'OtherError'}
    try:
        status = getattr(cause, 'status_code', None)
    except Exception:
        status = None
    if type(status) is int and 100 <= status <= 599:
        details['http_status'] = status
        if status == 429 and stage == 'request':
            details.update(_retry_after(cause))
    if reason == 'request_failed':
        if isinstance(cause, TimeoutError) or name == 'APITimeoutError':
            details['reason'] = 'timeout'
        elif isinstance(cause, ConnectionError) or name == 'APIConnectionError':
            details['reason'] = 'connection_error'
        elif 'http_status' in details:
            details['reason'] = 'http_error'
    return details


class LLMError(Exception):
    def __init__(self, message='LLM request failed', *, stage='request',
                 reason='request_failed', cause=None, codex_protocol=None):
        super().__init__(message)
        self._metadata = _error_details(cause,
            stage=stage if type(stage) is str and stage in ('request', 'response') else 'request',
            reason=reason if type(reason) is str and reason in _ERROR_REASONS else 'request_failed')
        diagnostic = validate_codex_protocol(codex_protocol)
        if (diagnostic is not None and self._metadata['stage'] == 'response'
                and self._metadata['reason'] == 'malformed_response'
                and self._metadata['cause_type'] == '_CodexProtocolError'):
            self._metadata['codex_protocol'] = diagnostic

    @property
    def metadata(self):
        value = dict(self._metadata)
        if 'codex_protocol' in value:
            value['codex_protocol'] = validate_codex_protocol(value['codex_protocol'])
        return value


def llm_error_metadata(error):
    """Project safe diagnostics even for a provider implementation outside this client."""
    if not isinstance(error, LLMError):
        return _error_details(error, stage='request', reason='request_failed')
    try:
        value = error.metadata
    except Exception:
        value = None
    if type(value) is not dict:
        return _error_details(error, stage='request', reason='request_failed')
    # Re-project rather than passing a mutable or subclass-supplied mapping through.
    stage, reason, cause = (value.get(key) for key in ('stage', 'reason', 'cause_type'))
    result = {'stage': stage if type(stage) is str and stage in ('request', 'response') else 'request',
              'reason': reason if type(reason) is str and reason in _ERROR_REASONS else 'request_failed',
              'cause_type': cause if type(cause) is str and cause in _ERROR_CAUSES else 'OtherError'}
    diagnostic = validate_codex_protocol(value.get('codex_protocol'))
    if (diagnostic is not None and result == {'stage': 'response',
            'reason': 'malformed_response', 'cause_type': '_CodexProtocolError'}):
        result['codex_protocol'] = diagnostic
    status = value.get('http_status')
    if type(status) is int and 100 <= status <= 599:
        result['http_status'] = status
        seconds = value.get('retry_after_seconds')
        if (status == 429 and result['stage'] == 'request'
                and type(seconds) in (int, float) and math.isfinite(seconds)
                and 0 <= seconds <= MAX_RETRY_AFTER_SECONDS):
            result['retry_after_seconds'] = float(seconds)
            if value.get('retry_after_clamped') is True:
                result['retry_after_clamped'] = True
    return result
