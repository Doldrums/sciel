"""Local, exact request/response transcripts; never record transport credentials."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from time import monotonic
import uuid

from ...campaign.budget import RequestNotDispatched
from .client import (LLMResponse, known_cost, llm_error_metadata, response_cost_provenance, transport_parameters,
                     response_accounting, response_resolved_model)

_log = logging.getLogger(__name__)


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class TracedLLM:
    """Transparent provider adapter. Recording never consumes search RNG state."""

    def __init__(self, provider, run_dir, context, on_response=None):
        object.__setattr__(self, '_provider', provider)
        object.__setattr__(self, '_directory', Path(run_dir) / 'artifacts')
        object.__setattr__(self, '_context', context)
        object.__setattr__(self, '_on_response', on_response)

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def __setattr__(self, name, value):
        setattr(self._provider, name, value)

    def _text(self, call_id, kind, text):
        directory = self._directory / 'llm_calls'
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f'{call_id}.{kind}.txt').open('x', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())

    def _emit(self, record):
        self._directory.mkdir(parents=True, exist_ok=True)
        record = dict(record, timestamp=datetime.now(timezone.utc).isoformat(),
                      schema='sciel-llm-call/1')
        with (self._directory / 'llm_calls.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def generate(self, prompt, model, temperature=None, **options):
        max_tokens = options.get('max_tokens')
        if max_tokens is None:
            max_tokens = getattr(self._provider, 'max_tokens', None)
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
            raise RequestNotDispatched('max_tokens must be a positive integer or null')
        call_id = uuid.uuid4().hex
        context = self._context(prompt)
        from .local import bind_inference_options
        options = bind_inference_options(self._provider, model, options, context)
        visibility = options.pop('adaptation_visibility', None)
        spec = options.get('inference_spec')
        if spec is not None and spec['role'] in ('G', 'repair'):
            from ...adaptation.visibility import unknown_capsule, validate_capsule
            try:
                visibility = validate_capsule(visibility if visibility is not None
                    else unknown_capsule(prompt, spec), prompt=prompt, inference_spec=spec)
            except (ValueError, TypeError) as exc:
                raise RequestNotDispatched('Cannot bind adaptation visibility') from exc
        elif visibility is not None:
            raise RequestNotDispatched('Adaptation visibility requires a generation role')
        common = dict(context, call_id=call_id, model=model)
        if visibility is not None:
            common['adaptation_visibility'] = visibility
        if options.get('inference_route') is not None:
            common['inference_route'] = options['inference_route']
        if options.get('inference_deployment') is not None:
            common['inference_deployment'] = options['inference_deployment']
        if options.get('inference_spec') is not None:
            common['inference_spec'] = options['inference_spec']
            common['role'] = options['inference_spec']['role']
        request = dict(common, event='request', prompt_sha256=_sha(prompt),
                       prompt_chars=len(prompt), temperature=temperature)
        parameters = transport_parameters(self._provider, model, temperature,
            inference_deployment=options.get('inference_deployment'))
        if parameters is not None:
            request['transport_parameters'] = parameters
        prepare_metadata = getattr(self._provider, 'request_metadata', None)
        if callable(prepare_metadata):
            request['provider_request'] = prepare_metadata(prompt, model, temperature, **options)
        if type(max_tokens) is int:
            request['max_tokens'] = max_tokens
        # These are body options, never client headers, API keys or SDK objects.
        if options.get('response_format') is not None:
            request['response_format'] = options['response_format']
        try:
            self._text(call_id, 'prompt', prompt)
            self._emit(request)
        except (OSError, ValueError, TypeError):
            raise RequestNotDispatched('Cannot persist LLM request transcript') from None
        started = monotonic()
        try:
            response = self._provider.generate(prompt, model, temperature, **options)
        except BaseException as exc:
            dispatched = not isinstance(exc, RequestNotDispatched)
            try:
                self._emit(dict(common, event='error', elapsed_seconds=monotonic() - started,
                    generation_returned=False, generation_dispatch='unknown' if dispatched else 'not_dispatched',
                    cost=None if dispatched else 0.0, current_call_cost_unknown=dispatched,
                    llm_error=llm_error_metadata(exc)))
            except (OSError, ValueError, TypeError):
                _log.warning('Could not persist LLM error transcript')
            raise
        if isinstance(response, LLMResponse) and getattr(self._provider, 'local_inference', False) is True:
            response.call_id = call_id
            response.adaptation_visibility = visibility
            if options.get('inference_route') is not None:
                response.inference_route = options['inference_route']
            if options.get('inference_deployment') is not None:
                response.inference_deployment = options['inference_deployment']
        try:
            cost = known_cost(response)
        except ValueError:
            cost = None
        try:
            accounting = response_accounting(response)
        except ValueError:
            accounting = None
        try:
            text = response.text
            provenance = response_cost_provenance(response)
            resolved = response_resolved_model(response)
            self._text(call_id, 'response', text)
            self._emit(dict(common, event='response', elapsed_seconds=monotonic() - started,
                generation_returned=True, generation_dispatch='returned',
                response_sha256=_sha(text), response_chars=len(text),
                cost=cost, current_call_cost_unknown=accounting is None,
                **({'unpriced_calls':accounting['unpriced_calls'], 'known_usd':accounting['known_usd']}
                   if accounting is not None and accounting['unpriced_calls'] else {}),
                prompt_tokens=getattr(response, 'prompt_tokens', None),
                completion_tokens=getattr(response, 'completion_tokens', None),
                finish_reason=getattr(response, 'finish_reason', None),
                **({'served_model': response.served_model}
                   if getattr(response, 'served_model', None) is not None else {}),
                **({'deployment_response': response.deployment_response}
                   if getattr(response, 'deployment_response', None) is not None else {}),
                **({'cost_provenance': provenance} if provenance is not None else {}),
                **({'resolved_model': resolved} if resolved is not None else {}),
                **({'response_error': response.response_error}
                   if getattr(response, 'response_error', None) is not None else {})))
        except (OSError, ValueError, TypeError, AttributeError):
            # The paid response must still reach its ordinary accounting owner.
            _log.warning('Could not persist LLM response transcript')
        if self._on_response is not None:
            self._on_response(response)
        return response
