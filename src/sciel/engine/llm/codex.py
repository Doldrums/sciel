"""One Codex Responses SSE dispatch, with explicit unsupported wire options.

Credentials never enter request metadata. A terminal response is required even
when the preceding deltas happen to contain complete-looking JSON.
"""
from __future__ import annotations

from copy import deepcopy
from http.client import IncompleteRead
import hashlib
import json
import math
from pathlib import Path
from ssl import SSLEOFError
from time import monotonic
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, HTTPRedirectHandler, build_opener

from ...campaign.budget import RequestNotDispatched
from .client import (LLMError, LLMResponse, COST_PROVENANCE_SCHEMA, _retry_after,
                     _tokens, validate_response_format)
from .codex_protocol import (ENDPOINT, MAX_BYTES, READ_SIZE, _CodexProtocolError,
                             _ProtocolState)

CONTRACT = 'sciel-codex-responses/6'
INSTRUCTIONS = 'Follow the user task and requested output format exactly.'


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def validate_config(value):
    """No I/O, credentials or provider probes during configuration validation."""
    cfg = deepcopy(value)
    if cfg.get('provider') != 'codex_responses':
        raise ValueError('Codex requires provider=codex_responses')
    if cfg.get('api_base', ENDPOINT) != ENDPOINT:
        raise ValueError('Codex requires the fixed chatgpt.com Responses endpoint')
    models = cfg.get('models') or [cfg.get('primary_model')]
    if (type(models) is not list or not models or any(type(m) is not str or not m.strip() for m in models)
            or len(models) != len(set(models))):
        raise ValueError('Codex requires explicit distinct model IDs')
    weights = cfg.get('weights') or [1.0] * len(models)
    if (type(weights) is not list or len(weights) != len(models)
            or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in weights)
            or not any(weights)):
        raise ValueError('Codex requires finite model weights')
    if type(cfg.get('max_retries', 1)) is not int or cfg.get('max_retries', 1) != 1:
        raise ValueError('Codex requires max_retries=1; no hidden physical retry')
    for name, default in (('timeout', 120.), ('temperature', .7)):
        number = cfg.get(name, default)
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0 or (name == 'timeout' and number == 0):
            raise ValueError('invalid Codex ' + name)
    if type(cfg.get('max_tokens', 32000)) is not int or cfg.get('max_tokens', 32000) <= 0:
        raise ValueError('Codex max_tokens must be a positive requested limit')
    effort = cfg.get('reasoning_effort', 'low')
    if effort not in ('low', 'medium', 'high', 'xhigh'):
        raise ValueError('invalid Codex reasoning_effort')
    instructions = cfg.get('codex_instructions', INSTRUCTIONS)
    if type(instructions) is not str or not instructions.strip() or len(instructions) > 65536:
        raise ValueError('invalid Codex instructions')
    auth_file = cfg.get('codex_auth_file')
    if auth_file is not None and (type(auth_file) is not str or not auth_file.strip() or '\x00' in auth_file):
        raise ValueError('codex_auth_file must be an explicit path or null')
    key = cfg.get('api_key_env', 'CODEX_TOKEN')
    if type(key) is not str or not key.strip():
        raise ValueError('invalid Codex api_key_env')
    caps = cfg.get('model_capabilities', {})
    if (type(caps) is not dict or any(m not in models or entry != {'temperature':False} for m, entry in caps.items())):
        raise ValueError('Codex does not support temperature')
    cfg.update(api_base=ENDPOINT, models=models, weights=weights, api_key_env=key,
        codex_auth_file=auth_file, reasoning_effort=effort, codex_instructions=instructions,
        model_capabilities={m:{'temperature':False} for m in models})
    return cfg


def transport_identity(config):
    cfg = validate_config(config)
    return {'schema':CONTRACT, 'provider':'codex_responses', 'endpoint':ENDPOINT,
        'models':list(cfg['models']), 'weights':list(cfg['weights']),
        'reasoning_effort':cfg['reasoning_effort'],
        'instructions_sha256':_sha(cfg['codex_instructions'].encode('utf-8')),
        'structured_output':'native_text_format', 'usd_accounting':'subscription_unpriced',
        'temperature_sent':False, 'output_token_cap_sent':False,
        'requested_temperature':cfg.get('temperature', .7), 'requested_max_tokens':cfg.get('max_tokens', 32000),
        'timeout_seconds':cfg.get('timeout', 120.), 'max_response_bytes':MAX_BYTES,
        'implementation_sha256':_sha(Path(__file__).read_bytes()),
        'protocol_implementation_sha256':_sha(Path(__file__).with_name('codex_protocol.py').read_bytes()),
        'auth':{'strategy':'file_per_request' if cfg['codex_auth_file'] else 'environment',
                'path':cfg['codex_auth_file'], 'api_key_env':cfg['api_key_env']}}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


class CodexResponsesLLM:
    def __init__(self, config, api_key=None, *, opener=None):
        self.config = validate_config(config)
        self.models, self.weights = self.config['models'], self.config['weights']
        self.model_capabilities = self.config['model_capabilities']
        self.api_base = ENDPOINT
        self.temperature = self.config.get('temperature', .7)
        self.max_tokens = self.config.get('max_tokens', 32000)
        self.timeout = self.config.get('timeout', 120.)
        self.max_retries = 1
        self._api_key = api_key
        self._opener = opener or build_opener(_NoRedirect())

    def choose_model(self, rng):
        return rng.choices(self.models, weights=self.weights, k=1)[0]

    @property
    def transport_identity(self):
        return transport_identity({**self.config, 'models':self.models, 'weights':self.weights,
            'temperature':self.temperature, 'max_tokens':self.max_tokens, 'timeout':self.timeout})

    def _body(self, prompt, model, temperature=None, *, response_format=None, max_tokens=None):
        if type(prompt) is not str or model not in self.models:
            raise ValueError('Codex requires a text prompt and a configured model')
        requested = self.max_tokens if max_tokens is None else max_tokens
        if type(requested) is not int or requested <= 0:
            raise ValueError('max_tokens must be a positive integer or null')
        sampling = self.temperature if temperature is None else temperature
        if type(sampling) not in (int, float) or not math.isfinite(sampling):
            raise ValueError('invalid requested temperature')
        spec = validate_response_format(response_format)
        body = {'model':model, 'instructions':self.config['codex_instructions'],
            'input':[{'role':'user','content':[{'type':'input_text','text':prompt}]}],
            'store':False, 'stream':True, 'reasoning':{'effort':self.config['reasoning_effort']}}
        if spec is not None:
            body['text'] = {'format':{'type':'json_schema', **spec['json_schema']}}
        _bytes(body)  # Fail before dispatch on invalid Unicode/JSON.
        return body, requested, sampling

    def request_metadata(self, prompt, model, temperature=None, **options):
        body, requested, sampling = self._body(prompt, model, temperature, **options)
        return {'schema':CONTRACT, 'provider':'codex_responses', 'endpoint':ENDPOINT,
            'body_sha256':_sha(_bytes(body)), 'transport_identity':self.transport_identity,
            'model':model, 'reasoning_effort':body['reasoning']['effort'],
            'response_format':deepcopy(body.get('text', {}).get('format')),
            'temperature':{'requested':sampling,'sent':False},
            'output_token_cap':{'requested':requested,'sent':False,'enforced':False}}

    def _credential(self):
        try:
            if self.config['codex_auth_file'] is not None:
                path = Path(self.config['codex_auth_file']).expanduser()
                with path.open('rb') as source:
                    raw = source.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ValueError('oversized auth file')
                token = json.loads(raw)['tokens']['access_token']
            else:
                token = self._api_key
            if type(token) is not str or not token.strip() or '\r' in token or '\n' in token:
                raise ValueError('unavailable token')
            return token
        except (OSError, ValueError, KeyError, TypeError):
            raise RequestNotDispatched('Codex authentication unavailable before HTTP dispatch') from None

    def generate(self, prompt, model, temperature=None, *, response_format=None, max_tokens=None):
        body, _, _ = self._body(prompt, model, temperature, response_format=response_format, max_tokens=max_tokens)
        token = self._credential()  # Fresh file read each HTTP request; never refresh/retry over the network.
        request = Request(ENDPOINT, data=_bytes(body), method='POST', headers={
            'Authorization':'Bearer ' + token, 'Content-Type':'application/json', 'Accept':'text/event-stream'})
        started = monotonic()
        state = _ProtocolState()
        try:
            with self._opener.open(request, timeout=self.timeout) as stream:
                # The live endpoint can omit Content-Type. Require the bounded
                # SSE protocol and a matching terminal response, not a MIME
                # header; generic HTML/JSON still cannot pass the parser.
                parsed = _consume(stream, timeout=self.timeout, started=started, state=state)
            if parsed[0].get('model') != model:
                raise _CodexProtocolError('model_identity')
        except RequestNotDispatched:
            raise
        except HTTPError as exc:
            error = LLMError()
            error._metadata.update(reason='http_error', cause_type='APIStatusError', http_status=exc.code)
            if exc.code == 429:
                error._metadata.update(_retry_after(SimpleNamespace(response=SimpleNamespace(headers=exc.headers))))
            raise error from None
        except _CodexProtocolError as exc:
            raise LLMError(stage='response', reason='malformed_response', cause=exc,
                codex_protocol=state.diagnostic(exc.code,
                    body_sha256=_sha(request.data), model=model)) from None
        except Exception as exc:
            cause = exc.reason if isinstance(exc, URLError) and isinstance(exc.reason, Exception) else exc
            # Unexpected local exceptions are not evidence of a protocol error
            # and never acquire the response-abandonment diagnostic.
            raise LLMError(stage='request', reason='request_failed', cause=cause) from None
        terminal, text, refusal = parsed
        status = terminal['status']
        usage = terminal.get('usage') if type(terminal.get('usage')) is dict else {}
        finish = {'completed':'stop', 'incomplete':'length', 'failed':'error'}[status]
        provenance = {'schema':COST_PROVENANCE_SCHEMA, 'source':'unavailable'}
        if status == 'completed':
            provenance = {'schema':COST_PROVENANCE_SCHEMA, 'source':'subscription_unpriced',
                'response_status':'completed', 'provider':'codex_responses',
                'endpoint_sha256':_sha(ENDPOINT.encode('utf-8'))}
        return LLMResponse(text=text, model=model, cost=None,
            prompt_tokens=_tokens(usage.get('input_tokens')), completion_tokens=_tokens(usage.get('output_tokens')),
            finish_reason=finish, refusal=refusal, cost_provenance=provenance)


def _events(stream, *, timeout, started, state=None):
    state = _ProtocolState() if state is None else state
    buffer = b''; data = []
    while True:
        if monotonic() - started > timeout:
            raise TimeoutError('Codex stream deadline exceeded')
        try:
            chunk = stream.read1(READ_SIZE)
        except (IncompleteRead, SSLEOFError, URLError) as exc:
            # Only classify known interruptions while reading an opened response.
            # Handshake/open/close failures and generic SSL errors stay distinct.
            # IncompleteRead.partial may contain plausible JSON; it is untrusted
            # and must not become a returned response or enter diagnostics.
            cause = exc.reason if isinstance(exc, URLError) else exc
            if isinstance(cause, IncompleteRead):
                raise _CodexProtocolError('transport_incomplete_read') from None
            if isinstance(cause, SSLEOFError):
                raise _CodexProtocolError('transport_tls_eof') from None
            raise
        if monotonic() - started > timeout:
            raise TimeoutError('Codex stream deadline exceeded')
        if not chunk:
            raise _CodexProtocolError('stream_eof')
        state.bytes_read += len(chunk)
        if state.bytes_read > MAX_BYTES:
            raise _CodexProtocolError('stream_limit')
        buffer += chunk
        while b'\n' in buffer:
            line, buffer = buffer.split(b'\n', 1)
            line = line.rstrip(b'\r')
            if not line:
                if data:
                    raw = b'\n'.join(data); data = []
                    if raw == b'[DONE]':
                        raise _CodexProtocolError('done_without_terminal')
                    try:
                        decoded = raw.decode('utf-8')
                    except UnicodeDecodeError:
                        raise _CodexProtocolError('invalid_utf8') from None
                    try:
                        value = json.loads(decoded)
                    except (json.JSONDecodeError, RecursionError):
                        raise _CodexProtocolError('invalid_json') from None
                    if type(value) is not dict or type(value.get('type')) is not str:
                        raise _CodexProtocolError('invalid_event')
                    state.event(value['type'])
                    yield value
            elif line.startswith(b'data:'):
                data.append(line[5:].lstrip(b' '))


def _consume(stream, *, timeout, started, state=None):
    state = _ProtocolState() if state is None else state
    fragments = {}; complete = {}; items = {}; response_id = None; refusals = {}
    provisional = set()

    def index(value):
        if type(value) is not int or not 0 <= value < 10000:
            raise _CodexProtocolError('invalid_index')
        return value

    def finish_text(key, text):
        if type(text) is not str:
            raise _CodexProtocolError('invalid_text')
        previous = complete.get(key)
        if (previous is not None and previous != text) or (key in fragments and ''.join(fragments[key]) != text):
            raise _CodexProtocolError('text_conflict')
        complete[key] = text
        provisional.discard(key)

    def item(number, value, *, finished):
        if type(value) is not dict:
            raise _CodexProtocolError('invalid_item')
        prior = items.get(number)
        if prior is not None and prior.get('id') != value.get('id'):
            raise _CodexProtocolError('item_identity')
        items[number] = value
        if value.get('type') == 'message' and value.get('role') == 'assistant':
            content = value.get('content', [])
            if type(content) is not list:
                raise _CodexProtocolError('invalid_content')
            for content_index, part in enumerate(content):
                if type(part) is not dict:
                    raise _CodexProtocolError('invalid_content')
                if part.get('type') == 'output_text':
                    text = part.get('text')
                    if type(text) is not str:
                        raise _CodexProtocolError('invalid_text')
                    key = number, content_index
                    if finished:
                        finish_text(key, text)
                    elif text:
                        # An added snapshot is not a delta or a completed text.
                        # Require an authoritative final value before using a
                        # nonempty provisional snapshot; never concatenate it.
                        provisional.add(key)
                elif part.get('type') == 'refusal':
                    if type(part.get('refusal')) is not str:
                        raise _CodexProtocolError('invalid_refusal')
                    if finished:
                        refusals[(number, content_index)] = part['refusal']

    for event in _events(stream, timeout=timeout, started=started, state=state):
        kind = event['type']
        response = event.get('response')
        if type(response) is dict:
            ident = response.get('id')
            if type(ident) is not str or not ident or (response_id is not None and response_id != ident):
                raise _CodexProtocolError('response_identity')
            response_id = ident
        if kind in ('response.output_item.added', 'response.output_item.done'):
            item(index(event.get('output_index')), event.get('item'),
                 finished=kind == 'response.output_item.done')
        elif kind in ('response.output_text.delta', 'response.output_text.done'):
            key = index(event.get('output_index')), index(event.get('content_index'))
            if kind.endswith('.delta'):
                delta = event.get('delta')
                if type(delta) is not str:
                    raise _CodexProtocolError('invalid_delta')
                if key in complete:
                    raise _CodexProtocolError('delta_after_done')
                fragments.setdefault(key, []).append(delta)
            else:
                finish_text(key, event.get('text'))
        elif kind == 'response.refusal.done':
            key = index(event.get('output_index')), index(event.get('content_index'))
            if type(event.get('refusal')) is not str:
                raise _CodexProtocolError('invalid_refusal')
            refusals[key] = event['refusal']
        if kind in ('response.completed', 'response.incomplete', 'response.failed'):
            if type(response) is not dict or response.get('status') != kind.split('.')[-1]:
                raise _CodexProtocolError('invalid_terminal')
            output = response.get('output', [])
            if type(output) is not list:
                raise _CodexProtocolError('invalid_terminal')
            for number, value in enumerate(output):
                item(number, value, finished=True)
            if provisional:
                raise _CodexProtocolError('provisional_text_unconfirmed')
            for key, parts in fragments.items():
                if key not in complete:
                    complete[key] = ''.join(parts)
            def final(key):
                meta = items.get(key[0], {})
                return (meta.get('type', 'message') == 'message' and meta.get('role', 'assistant') == 'assistant'
                        and meta.get('phase') in (None, 'final_answer'))
            text = ''.join(value for key, value in sorted(complete.items()) if final(key))
            refusal = '\n'.join(value for key, value in sorted(refusals.items()) if final(key)) or None
            if response['status'] == 'completed' and response.get('error') is not None:
                raise _CodexProtocolError('invalid_terminal')
            return response, text, refusal
        if kind == 'error':
            raise _CodexProtocolError(state.server_error(event))
