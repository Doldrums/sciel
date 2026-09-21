"""Durable, immutable I/O receipts for cooperative parallel search.

Only leaf transport/evaluation functions execute in pool threads. No Engine,
research controller, database, RNG or journal is handed to a worker thread.
"""
from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    Future,
    TimeoutError as FutureTimeout,
    wait,
    FIRST_COMPLETED,
)
from contextlib import ExitStack
from dataclasses import asdict
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
import stat
import threading
from pathlib import Path
from time import monotonic

from ..campaign.budget import CampaignStopped, RequestNotDispatched
from .llm.client import (LLMError, LLMResponse, llm_error_metadata, response_cost_provenance,
                         response_resolved_model)
from ..eval.cascade import CascadeOutcome
from ..eval.facade import setup_failure
from ..eval.facade import (
    EvalResult,
    EvaluationInfrastructureError,
    EvaluationSetupError,
    validate_setup_failure,
)


_EVALUATION_OPERATION=ContextVar('sciel_evaluation_operation',default=None)


def current_evaluation_operation():
    """Host-owned leaf context; no Engine/state enters the worker thread."""
    return _EVALUATION_OPERATION.get()


JSON_CONTROL_KINDS = frozenset({'tool_job_submit', 'tool_job_poll', 'tool_job_cancel'})


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def durable_json(path, value, *, ensure_ascii=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    data = (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                       allow_nan=False).encode('ascii') if ensure_ascii else canonical(value))
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _encode(kind, value):
    if kind == 'tool_execution' or kind in JSON_CONTROL_KINDS:
        if type(value) is not dict:
            raise TypeError(kind + ' requires a JSON object receipt')
        return json.loads(canonical(value))
    if kind == 'evaluation':
        encoded=asdict(value)
        if getattr(value,'execution',None) is not None:
            encoded['execution']=value.execution
        return encoded
    fields = ('text', 'model', 'cost', 'prompt_tokens', 'completion_tokens',
              'finish_reason', 'refusal')
    # Malformed metadata remains unusable on replay; never stringify SDK objects.
    result = {key: getattr(value, key, None) for key in fields}
    provenance = response_cost_provenance(value)
    if provenance is not None:
        result['cost_provenance'] = provenance
    resolved = response_resolved_model(value)
    if resolved is not None:
        result['resolved_model'] = resolved
    if getattr(value, 'response_error', None) is not None:
        result['response_error'] = value.response_error
    for key, item in result.items():
        try:
            canonical(item)
        except (TypeError, ValueError):
            result[key] = None
    return result


def _decode(kind, value):
    if kind == 'tool_execution' or kind in JSON_CONTROL_KINDS:
        if type(value) is not dict:
            raise ValueError('invalid ' + kind + ' receipt')
        return json.loads(canonical(value))
    if kind == 'evaluation':
        try:
            outcome = CascadeOutcome(result=EvalResult(**value['result']),
                raw_first_score=value['raw_first_score'], reached=value['reached'],
                confirmed=value['confirmed'])
            if 'execution' in value:
                from ..eval.candidate_backend import validate_execution
                outcome.execution = validate_execution(value['execution'])
            return outcome
        except Exception as error:
            if type(value) is dict and 'execution' in value:
                raise EvaluationInfrastructureError('malformed candidate evaluation result') from error
            raise
    response = LLMResponse(**value)
    response.cost_provenance = response_cost_provenance(response)
    response.resolved_model = response_resolved_model(response)
    return response


from .parallel_control import (
    ParallelAborted,
    ParallelPause,
    UnknownParallelCall,
    _paused_error,
    dispatch_evidence,
)


class CallLedger:
    def __init__(self, directory, capacity, provider):
        self.directory = Path(directory) / 'calls'
        self.capacity = capacity
        self.provider = provider

    def prepare(self, key, kind, request):
        try:
            return self._prepare(key,kind,request)
        except BaseException as error:
            if kind=='evaluation' and request.get('evaluation_backend') is not None:
                from ..eval.facade import EvaluationInfrastructureError
                if isinstance(error,EvaluationInfrastructureError): raise
                raise EvaluationInfrastructureError('candidate evaluation durable evidence is unresolved') from error
            raise

    def _prepare(self, key, kind, request):
        ident = hashlib.sha256(key.encode()).hexdigest()
        directory = self.directory / ident
        request_hash = digest({'kind': kind, 'request': request})
        intent_path = directory / 'intent.json'
        if intent_path.exists():
            intent = json.loads(intent_path.read_text())
            if (intent['schema'] != 'sciel-parallel-call/1' or intent['key'] != key
                    or intent['kind'] != kind or intent['request_sha256'] != request_hash
                    or digest({'kind': intent['kind'], 'request': intent['request']}) != request_hash):
                raise ValueError('Parallel replay request differs from its durable intent: ' + key)
        else:
            durable_json(intent_path, {'schema': 'sciel-parallel-call/1', 'key': key,
                'kind': kind, 'request': request, 'request_sha256': request_hash})
        receipt_path = directory / 'receipt.json'
        if receipt_path.exists():
            dispatch_evidence(directory, request_hash)
            receipt = json.loads(receipt_path.read_text())
            payload = receipt['payload']
            if receipt['sha256'] != digest(payload) or payload['request_sha256'] != request_hash:
                raise ValueError('Parallel call receipt checksum mismatch')
            if kind == 'evaluation' and request.get('evaluation_backend') is not None:
                from ..eval.candidate_backend import verify_execution
                execution = (payload.get('value', {}).get('execution') if payload.get('status') == 'returned'
                    else payload.get('error', {}).get('candidate_evaluation', {}).get('execution'))
                if execution is None:
                    raise ValueError('candidate evaluation WAL lacks bound execution evidence')
                verify_execution(execution, directory, request, request_hash,
                    outcome=payload.get('value') if payload.get('status')=='returned' else None)
            if 'abandonment' in payload:
                raise ValueError('Physical receipt cannot contain a recovery authorization')
            if payload.get('status') == 'error':
                from .parallel_recovery import load_abandonment
                abandonment = load_abandonment(directory, request_hash, receipt['sha256'])
                if abandonment is not None:
                    if kind != 'llm':
                        raise ValueError('Only a provider request can be abandoned')
                    payload = {**payload, 'abandonment': abandonment}
            return directory, request_hash, payload
        if (directory / 'dispatched.json').exists():
            raise UnknownParallelCall('Submitted parallel call has no terminal receipt: ' + key)
        return directory, request_hash, None

    def perform(self, directory, request_hash, kind, function, cancel=None, abort=None):
        def cancelled(message):
            if abort is not None and abort.is_set():
                return ParallelAborted(message)
            return ParallelPause(message)
        if cancel is not None and cancel.is_set():
            raise cancelled('Parallel call cancelled before capacity acquisition')
        queued = monotonic()
        resource = ('research_evaluator' if kind == 'tool_execution' else
                    'evaluator' if kind == 'evaluation' else 'llm')
        options = {'provider': self.provider} if resource == 'llm' else {}
        lease = ExitStack()
        try:
            # These are bounded local metadata operations. The detached worker
            # owns the evaluator lease for its actual execution lifetime.
            if kind not in JSON_CONTROL_KINDS:
                lease.enter_context(self.capacity.acquire(resource, **options,
                    **({'cancel': cancel} if cancel is not None else {})))
        except BaseException as error:
            # No dispatch marker or provider/evaluator invocation exists yet.
            # Do not turn host-capacity failures into an unknown paid request.
            lease.close()
            from .capacity import CapacityCancelled
            if isinstance(error, CapacityCancelled):
                raise cancelled('Parallel call cancelled while waiting for capacity') from None
            raise RequestNotDispatched('Cannot acquire parallel call capacity') from None
        with lease:
            if cancel is not None and cancel.is_set():
                raise cancelled('Parallel call cancelled before dispatch')
            started = monotonic()
            started_at = datetime.now(timezone.utc).isoformat()
            durable_json(directory / 'dispatched.json', {'request_sha256': request_hash,
                'timestamp': started_at, 'pid': os.getpid()})
            try:
                token = _EVALUATION_OPERATION.set((Path(directory),request_hash)) if kind == 'evaluation' else None
                try:
                    value = function()
                finally:
                    if token is not None:
                        _EVALUATION_OPERATION.reset(token)
                if kind in JSON_CONTROL_KINDS:
                    # A malformed acknowledgement is an unresolved operation,
                    # never an empty LLMResponse or an implicitly successful job.
                    value = _encode(kind, value)
            except BaseException as error:
                metadata = llm_error_metadata(error)
                if kind == 'evaluation' and isinstance(error, EvaluationSetupError):
                    metadata['cause_type'] = 'EvaluationSetupError'
                    metadata['evaluation_setup'] = validate_setup_failure(error.diagnostic)
                if kind == 'evaluation':
                    from ..eval.facade import EvaluationInfrastructureError
                    if isinstance(error, EvaluationInfrastructureError):
                        metadata['cause_type'] = type(error).__name__
                    from ..eval.candidate_backend import EvaluationBackendError
                    if isinstance(error, EvaluationBackendError):
                        metadata['cause_type'] = 'EvaluationBackendError'
                        metadata['candidate_evaluation'] = error.diagnostic
                payload = {'status': 'error', 'request_sha256': request_hash,
                    'error': metadata,
                    'not_dispatched': isinstance(error, RequestNotDispatched),
                    'queue_seconds': started - queued,
                    'elapsed_seconds': monotonic() - started, 'started_at': started_at,
                    'completed_at': datetime.now(timezone.utc).isoformat()}
            else:
                payload = {'status': 'returned', 'request_sha256': request_hash,
                    'value': _encode(kind, value), 'queue_seconds': started - queued,
                    'elapsed_seconds': monotonic() - started, 'started_at': started_at,
                    'completed_at': datetime.now(timezone.utc).isoformat()}
            durable_json(directory / 'receipt.json', {'sha256': digest(payload), 'payload': payload})
            if payload['status'] == 'error' and abort is not None:
                abort.set()
                if cancel is not None:
                    cancel.set()
            error = payload.get('error', {})
            delay = error.get('retry_after_seconds')
            if kind == 'llm' and error.get('http_status') == 429 and type(delay) in (int, float):
                try:
                    self.capacity.defer_provider(self.provider, delay)
                except (OSError, ValueError):
                    # Capacity telemetry cannot replace an already durable
                    # provider receipt or change its unknown-charge status.
                    pass
            return payload

    @staticmethod
    def result(kind, payload):
        if payload['status'] == 'returned':
            return _decode(kind, payload['value'])
        if payload.get('abandonment') is not None:
            from .provider_abandonment import AbandonedProviderRequest
            if kind != 'llm' or payload.get('not_dispatched') is not False:
                raise ValueError('Invalid abandoned operation')
            raise AbandonedProviderRequest(payload['abandonment'], payload['error'])
        if payload.get('not_dispatched'):
            raise RequestNotDispatched('Parallel transport was not dispatched')
        if kind == 'evaluation':
            if payload['error'].get('cause_type') == 'EvaluationInfrastructureError':
                from ..eval.facade import EvaluationInfrastructureError
                raise EvaluationInfrastructureError('evaluation infrastructure invariant failed')
            if ('candidate_evaluation' in payload['error']
                    or payload['error'].get('cause_type') == 'EvaluationBackendError'):
                from ..eval.candidate_backend import EvaluationBackendError
                from ..eval.facade import EvaluationInfrastructureError
                try:
                    failure = EvaluationBackendError(payload['error']['candidate_evaluation'])
                except (ValueError,TypeError,KeyError):
                    raise EvaluationInfrastructureError('malformed candidate evaluation failure evidence') from None
                raise failure
            metadata = payload['error']
            if type(metadata) is dict and ('evaluation_setup' in metadata
                    or metadata.get('cause_type') == 'EvaluationSetupError'):
                try:
                    failure = EvaluationSetupError(metadata.get('evaluation_setup'))
                except (TypeError, ValueError, OverflowError):
                    # A malformed known-family diagnostic is still a stop,
                    # never an ordinary failed candidate measurement. Limits
                    # remain unknown; the saved request and elapsed are intact.
                    failure = EvaluationSetupError(setup_failure('malformed_report', 'unknown'))
                raise failure
            raise RuntimeError('Isolated parallel evaluator failed: ' + payload['error']['cause_type'])
        if kind == 'tool_execution':
            raise UnknownParallelCall('Research tool execution has no usable outcome: '
                                      + payload['error']['cause_type'])
        if kind in JSON_CONTROL_KINDS:
            raise UnknownParallelCall('Research tool job control has no usable outcome: '
                                      + payload['error']['cause_type'])
        error = LLMError()
        error._metadata = payload['error']
        raise error

from .cooperative_io import (CooperativeIO)
