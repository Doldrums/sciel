"""A separate trusted scorer container; no candidate source or task files.

The caller durably owns intent, receipt persistence, deadlines and replay. This
module performs one dispatch only; it does not import arbitrary saved receipts.
"""
import base64
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import re

from ..methods.models import content_hash, canonical_bytes
from ..methods.executor import ExecutionRefused
from .facade import EvalResult
from .trusted_scoring.adapter import HostBinding, _binding
from .trusted_scoring.transport import MAX_BYTES

ROOT = Path(__file__).resolve().parent
BOUND = ('method_id', 'invocation_id', 'plan_id', 'reservation_id', 'source_sha256',
         'runtime_sha256', 'input_sha256', 'request_sha256')
SCORER_FILES = ('__init__.py', 'adapter.py', 'circle.py', 'heil.py', 'beat.py', 'transport.py')
OUTPUT_SCHEMA = 'sciel-trusted-scorer-output/1'


class ScorerInfrastructureError(RuntimeError):
    def __init__(self, reason, receipt=None, *, not_dispatched=False):
        # Only an authenticated owned executor receipt is retained here. Missing
        # or unbound transport is unknown, not a fabricated zero-CPU receipt.
        self.receipt = receipt
        self.reason = reason
        self.not_dispatched = not_dispatched
        super().__init__('Trusted scorer infrastructure failed: ' + reason)


def prepare_scorer(executor, kind, payload, binding, *, limits, invocation_id,
                   plan_id, reservation_id):
    if type(kind) is not str or kind not in ('circle', 'heil', 'beat'):
        raise ValueError('unsupported trusted scorer')
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_BYTES:
        raise ValueError('scorer requires bounded raw bytes')
    if type(binding) is not HostBinding:
        raise ValueError('exact primitive HostBinding required')
    _binding(binding)
    if (type(invocation_id) is not str or re.fullmatch('[a-f0-9]{32}', invocation_id) is None
            or any(type(value) is not str or not 0 < len(value) <= 256
                   or any(ord(char) < 32 for char in value) for value in (plan_id, reservation_id))):
        raise ValueError('invalid scorer invocation ownership')
    runtime = executor.identity
    image = runtime.get('image_id')
    if type(image) is not str or re.fullmatch('sha256:[a-f0-9]{64}', image) is None:
        raise ValueError('immutable scorer image required')
    limits = deepcopy(limits)
    inputs = dict(kind=kind, payload_base64=base64.b64encode(payload).decode('ascii'),
                  binding=asdict(binding))
    files = {'worker.py': (ROOT / '_scorer_worker.py').read_text()}
    files.update({'trusted_scoring/' + name: (ROOT / 'trusted_scoring' / name).read_text()
                  for name in SCORER_FILES})
    binding_properties = {key: {'type': 'string', 'minLength': size, 'maxLength': size}
                          for key, size in (('source_sha256', 64), ('evaluator_sha256', 64), ('invocation_id', 32))}
    binding_properties.update(stage={'type': 'string', 'enum': ['smoke', 'full']},
                              elapsed_seconds={'type': 'number', 'minimum': 0})
    result_schema = {'type': 'object', 'properties': {
            'ok': {'type': 'boolean'}, 'score': {'type': 'number'},
            'cases': {'type': 'object', 'additionalProperties': {'type': 'number'}},
            'artifacts': {'type': 'object', 'additionalProperties': {'type': 'string'}}},
            'required': ['ok', 'score', 'cases', 'artifacts'], 'additionalProperties': False}
    manifest = dict(name='trusted_candidate_scorer_' + kind, kind='adapter', author_lab='trusted-host',
        entrypoint='worker.py:run',
        input_schema={'type': 'object', 'properties': {
            'kind': {'type': 'string', 'enum': [kind]},
            'payload_base64': {'type': 'string', 'minLength': 1, 'maxLength': 4 * ((MAX_BYTES + 2) // 3)},
            'binding': {'type': 'object', 'properties': binding_properties,
                        'required': list(binding_properties), 'additionalProperties': False}},
            'required': ['kind', 'payload_base64', 'binding'], 'additionalProperties': False},
        output_schema={'type': 'object', 'properties': {
            'schema': {'type': 'string', 'enum': [OUTPUT_SCHEMA]}, 'result': result_schema,
            'runtime': {'type': 'object', 'properties': {key: {'type': 'string', 'minLength': 1,
                'maxLength': 128} for key in ('python', 'numpy')},
                'required': ['python', 'numpy'], 'additionalProperties': False}},
            'required': ['schema', 'result', 'runtime'], 'additionalProperties': False},
        protocol='Trusted scoring of bounded numeric bytes; no candidate execution or source import.',
        runtime_lock={'image_id': image}, dependencies=[], parents=[], resources=limits,
        scope='Separate trusted scorer container; candidate and task files are not in this bundle.', examples=[])
    bundle = dict(id=content_hash(dict(schema='sciel-method/1', manifest=manifest, files=files)),
                  manifest=manifest, files=files)
    request = dict(schema='sciel-method-request/1', method_id=bundle['id'], invocation_id=invocation_id,
        plan_id=plan_id, reservation_id=reservation_id, input=inputs, input_sha256=content_hash(inputs),
        source_sha256=content_hash(files), runtime_sha256=content_hash(runtime), limits=limits,
        methods={bundle['id']: dict(root='/bundle', entrypoint='worker.py:run')}, dependency_ids=[])
    request['request_sha256'] = content_hash(request)
    executor.preflight(bundle, inputs, limits)
    return {'bundle': bundle, 'input': inputs, 'request': request}


def _bound_receipt(executor, request, receipt, container_name):
    if (type(receipt) is not dict or receipt.get('schema') != 'sciel-method-execution/1'
            or receipt.get('receipt_sha256') != content_hash({k:v for k,v in receipt.items() if k != 'receipt_sha256'})
            or any(receipt.get(key) != request[key] for key in BOUND)
            or canonical_bytes(receipt.get('runtime')) != canonical_bytes(executor.identity)
            or canonical_bytes(receipt.get('limits')) != canonical_bytes(request['limits'])
            or receipt.get('dependency_ids') != []
            or type(receipt.get('container')) is not dict
            or receipt['container'].get('name') != container_name
            or receipt['container'].get('image_id') != executor.identity['image_id']):
        raise ValueError('unbound scorer receipt')
    if receipt.get('status') not in ('ok', 'failed', 'unknown'):
        raise ValueError('invalid scorer status')
    resources, measurement = receipt.get('resources'), receipt.get('resource_measurement')
    if (type(resources) is not dict or type(measurement) is not dict
            or type(receipt['container'].get('removed')) is not bool
            or receipt.get('dispatch_status') not in ('invoked', 'unknown', 'not_dispatched')
            or (receipt.get('reason') is not None and type(receipt['reason']) is not str)):
        raise ValueError('malformed scorer accounting')
    def nonnegative(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    if not nonnegative(resources.get('wall_seconds')):
        raise ValueError('missing observed scorer wall time')
    cpu_kind, cpu = measurement.get('cpu'), resources.get('cpu_seconds')
    if cpu_kind == 'trusted_cgroup_observation':
        if (receipt['dispatch_status'] != 'invoked' or not nonnegative(cpu)
                or not nonnegative(resources.get('method_wall_seconds'))):
            raise ValueError('invalid observed scorer CPU')
    elif cpu_kind == 'unavailable':
        if cpu is not None or receipt['dispatch_status'] != 'unknown':
            raise ValueError('unavailable scorer CPU has no numeric value')
    elif cpu_kind == 'not_dispatched':
        if type(cpu) not in (int, float) or cpu != 0 or receipt['dispatch_status'] != 'not_dispatched':
            raise ValueError('invalid scorer predispatch accounting')
    else:
        raise ValueError('unknown scorer accounting authority')
    return receipt


def score_in_container(executor, prepared, *, container_name):
    """Dispatch once. The caller must have persisted this exact scorer intent."""
    try:
        if type(prepared) is not dict or set(prepared) != {'bundle', 'input', 'request'}:
            raise ValueError('invalid scorer preparation')
        request, inputs = prepared['request'], prepared['input']
        current = prepare_scorer(executor, inputs['kind'],
            base64.b64decode(inputs['payload_base64'], validate=True), HostBinding(**inputs['binding']),
            limits=request['limits'], invocation_id=request['invocation_id'],
            plan_id=request['plan_id'], reservation_id=request['reservation_id'])
        if canonical_bytes(current) != canonical_bytes(prepared):
            raise ValueError('scorer preparation differs from trusted source')
        if type(container_name) is not str or re.fullmatch('sciel-eval-[a-f0-9]{64}', container_name) is None:
            raise ValueError('invalid scorer container ownership')
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise ScorerInfrastructureError('invalid_preparation', not_dispatched=True) from None
    try:
        receipt = executor.execute(prepared['bundle'], inputs, invocation_id=request['invocation_id'],
            plan_id=request['plan_id'], reservation_id=request['reservation_id'], limits=request['limits'],
            container_name=container_name)
    except ExecutionRefused:
        raise ScorerInfrastructureError('executor_refused', not_dispatched=True) from None
    except Exception as exc:
        raise ScorerInfrastructureError('executor_' + type(exc).__name__) from None
    try:
        _bound_receipt(executor, request, receipt, container_name)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise ScorerInfrastructureError('unbound_executor_receipt') from None
    if receipt['status'] != 'ok' or receipt['container'].get('removed') is not True:
        raise ScorerInfrastructureError('scorer_not_completed', receipt)
    try:
        resources = receipt['resources']
        if (receipt.get('dispatch_status') != 'invoked' or receipt.get('reason') is not None
                or receipt['resource_measurement']['cpu'] != 'trusted_cgroup_observation'
                or any(type(resources[key]) not in (float, int) or not math.isfinite(resources[key])
                       or resources[key] < 0 for key in ('cpu_seconds', 'wall_seconds', 'method_wall_seconds'))
                or receipt.get('output_sha256') != content_hash(receipt.get('output'))):
            raise ValueError('missing trusted scorer accounting/output')
        output = receipt['output']
        if (type(output) is not dict or set(output) != {'schema', 'result', 'runtime'}
                or output['schema'] != OUTPUT_SCHEMA or type(output['runtime']) is not dict
                or set(output['runtime']) != {'python', 'numpy'}
                or any(type(value) is not str or re.fullmatch(r'[A-Za-z0-9.+_-]{1,128}', value) is None
                       for value in output['runtime'].values())):
            raise ValueError('invalid bound scorer runtime envelope')
        result = output['result']
        if type(result) is not dict or set(result) != {'ok', 'score', 'cases', 'artifacts'}:
            raise ValueError('invalid primitive scorer output')
        EvalResult.from_json(result)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        raise ScorerInfrastructureError('invalid_completed_scorer', receipt) from None
    return {'result': deepcopy(result), 'receipt': receipt}
