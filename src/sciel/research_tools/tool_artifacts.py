"""Persist authenticated tool invocations and account for provider responses."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from time import monotonic
from ..campaign.budget import CampaignStopped
from ..engine.parallel_calls import durable_json
from ..engine.research_targets import input_binding
from ..methods.models import content_hash, require_hash
from ..methods.executor import ExecutionRefused
from .config import effective_limits
from .result_contract import RESULT_SCHEMA, assess_receipt
from .observation_cards import _clip, _clip_json
from .artifact_reading import read_json
from .tool_protocol import ARTIFACT_SCHEMA

def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _artifact(tools, kind, payload):
    artifact = {'schema': ARTIFACT_SCHEMA, 'kind': kind, 'payload': payload}
    ident = content_hash(artifact)
    path = tools.path / (ident + '.json')
    if path.is_symlink():
        raise ValueError('research tool artifact path is a symlink')
    if path.exists():
        if content_hash(read_json(path, 32 * 1024 * 1024)) != ident:
            raise ValueError('research tool artifact was modified')
    else:
        durable_json(path, artifact)
    return ident


def _read_artifact(tools, ident, kind):
    require_hash(ident, 'research artifact')
    path = tools.path / (ident + '.json')
    value = read_json(path, 32 * 1024 * 1024)
    if (type(value) is not dict or value.get('schema') != ARTIFACT_SCHEMA or value.get('kind') != kind
            or content_hash(value) != ident):
        raise ValueError('research artifact identity differs')
    return value['payload']


def _charge_once(tools, charge_id, response):
    from ..engine.llm.client import response_accounting, response_cost_provenance
    try:
        accounting = response_accounting(response)
    except ValueError:
        raise CampaignStopped('research tool response has unknown cost; reconcile its durable call receipt') from None
    cost = accounting['known_usd']
    charge = ({'known_usd': cost, 'unpriced_calls': 1,
               'cost_provenance': response_cost_provenance(response)}
              if accounting['unpriced_calls'] else cost)
    if charge_id in tools._state['charges']:
        if tools._state['charges'][charge_id] != charge:
            raise ValueError('research tool charge changed on replay')
        return cost
    tools.charge(cost)
    tools._state['charges'][charge_id] = charge
    return cost


def _invoke(tools, bundle, inputs, invocation_id, action_id, limits=None):
    begun = monotonic()
    limits = effective_limits(tools.config, limits)
    try:
        return tools.library.execute(bundle, inputs, invocation_id=invocation_id,
            plan_id=action_id, reservation_id=invocation_id, limits=limits)
    except ExecutionRefused as error:
        # This exception is exclusively a proven pre-dispatch refusal. A
        # container/transport exception or an unknown receipt still escapes
        # to the durable I/O layer and stops the wave without a hidden retry.
        receipt = {'schema': 'sciel-research-tool-refusal/1',
            'method_id': bundle['id'], 'invocation_id': invocation_id,
            'plan_id': action_id, 'reservation_id': invocation_id,
            'input_sha256': content_hash(inputs), 'source_sha256': content_hash(bundle['files']),
            'runtime_sha256': content_hash(tools.library.identity['executor']),
            'limits': deepcopy(limits),
            'dependency_ids': sorted(row['id'] for row in tools.library.dependency_closure(bundle['id'])),
            'status': 'failed', 'dispatch_status': 'not_dispatched',
            'reason': 'execution_refused_before_dispatch', 'diagnostic': _clip(str(error), 1000),
            'output': None, 'output_sha256': None, 'stderr': '',
            'resources': {'cpu_seconds': 0.0, 'wall_seconds': monotonic() - begun},
            'resource_measurement': {'cpu': 'candidate_not_dispatched',
                'wall': 'host_monotonic_total', 'scope': 'No candidate CPU; host setup CPU is unmetered.'}}
        receipt['receipt_sha256'] = content_hash(receipt)
        return receipt


def _run_tool(tools, plan, snapshot, bundle, inputs, attempt, io, *, question=None, limits=None, proposal_id=None):
    input_binding(inputs)
    limits = effective_limits(tools.config, limits)
    ident = bundle['id']
    # Pin the compact question once in the immutable receipt. Presentation
    # must never silently change its identity during compaction or resume.
    question = _clip_json(_clip(question or plan['question'], 600),
                          min(620, tools.config['max_context_chars'] // 12))
    invocation_id = content_hash({'action_id': plan['action_id'], 'attempt': attempt, 'tool_id': ident})
    request = {'tool_id': ident, 'inputs': inputs, 'invocation_id': invocation_id,
        'plan_id': plan['action_id'], 'reservation_id': invocation_id,
        'limits': limits, 'library_identity': tools.library.identity}
    author = bundle['manifest']['protocol']['author_island']
    tools._event(plan, 'invocation_started', invocation_id=invocation_id, tool_id=ident,
                author_island_id=author, input_sha256=content_hash(inputs))
    receipt = io.call('tool_execution', request, lambda bundle=bundle, inputs=inputs,
        invocation_id=invocation_id: tools._invoke(bundle, inputs, invocation_id, plan['action_id'], limits))
    return tools._record_invocation(plan, bundle, inputs, invocation_id, receipt, limits, question, proposal_id=proposal_id)


def _record_invocation(tools, plan, bundle, inputs, invocation_id, receipt, limits, question, *, proposal_id=None):
    ident = bundle['id']
    author = bundle['manifest']['protocol']['author_island']
    if (type(receipt) is not dict or receipt.get('method_id') != ident
            or receipt.get('invocation_id') != invocation_id
            or receipt.get('plan_id') != plan['action_id']
            or receipt.get('reservation_id') != invocation_id
            or receipt.get('source_sha256') != content_hash(bundle['files'])
            or receipt.get('runtime_sha256') != content_hash(tools.library.identity['executor'])
            or receipt.get('input_sha256') != content_hash(inputs)
            or receipt.get('limits') != limits
            or receipt.get('dependency_ids') != sorted(row['id'] for row in tools.library.dependency_closure(ident))
            or receipt.get('receipt_sha256') != content_hash(
                {key: value for key, value in receipt.items() if key != 'receipt_sha256'})):
        raise CampaignStopped('unbound research tool execution receipt')
    binding = input_binding(inputs)
    result = assess_receipt(receipt)
    policy = bundle['manifest']['protocol']['replay_policy']
    record = {'action_id': plan['action_id'], 'island_id': plan['island_id'],
        'parent_program_id': plan['parent_program_id'], 'iteration': plan['iteration'],
        'tool_id': ident, 'inputs': inputs, 'receipt': receipt,
        'cell_id':plan['cell_id'], 'cell_revision_id':plan['cell_revision_id'],
        'stage':plan['stage'], 'question':question,
        'input_program_id':binding['candidate_id'], 'target_binding':binding,
        'result': result, 'replay_policy': policy, 'experiment_proposal_id':proposal_id}
    from .comparison_assessment import from_record
    # Validate the saved declaration before binding it into the artifact.
    from_record(record, '0' * 64, lambda ident: tools._read_artifact(ident, 'proposal'))
    artifact = tools._artifact('invocation', record)
    tools._event(plan, 'invocation_result', invocation_id=invocation_id, tool_id=ident,
        author_island_id=author, receipt_id=artifact, input_sha256=receipt['input_sha256'],
        output_sha256=receipt.get('output_sha256'), dispatch_status=receipt.get('dispatch_status'),
        status=receipt.get('status'), resources=receipt.get('resources'), error=receipt.get('reason'),
        result_schema=RESULT_SCHEMA, tool_result_status=result['status'],
        tool_result_error=result['error'], replay_policy=policy, target_binding=binding)
    if receipt.get('status') == 'unknown' or receipt.get('dispatch_status') == 'unknown':
        raise CampaignStopped('research tool execution outcome unknown; reconcile its durable receipt')
    if receipt.get('status') not in ('ok', 'failed'):
        raise CampaignStopped('research tool receipt has unsupported status')
    if receipt.get('dispatch_status') not in ('invoked', 'not_dispatched'):
        raise CampaignStopped('research tool receipt has unsupported dispatch status')
    resources = receipt.get('resources')
    if type(resources) is not dict or any(type(resources.get(key)) not in (int, float)
            or not math.isfinite(resources[key]) or resources[key] < 0
            for key in ('cpu_seconds', 'wall_seconds')):
        raise CampaignStopped('research tool receipt lacks measured resource accounting')
    invocation = {'tool_id': ident, 'invocation_id': invocation_id, 'receipt_id': artifact,
        'status': receipt['status'], 'dispatch_status': receipt['dispatch_status'],
        'author_island_id': author, 'resources': receipt.get('resources', {}),
        'output_sha256': receipt.get('output_sha256'), 'input_sha256': receipt['input_sha256'],
        'candidate_id': binding['candidate_id'], 'target_binding':binding, 'result':result, 'replay_policy':policy}
    observation = {'tool_id': ident, 'invocation_id': invocation_id, 'receipt_id': artifact,
        'status': receipt['status'], 'reason': receipt.get('reason'),
        'diagnostic': receipt.get('diagnostic'),
        'output': receipt.get('output'), 'stderr': receipt.get('stderr'),
        'resources': receipt.get('resources'), 'candidate_id': binding['candidate_id'], 'target_binding':binding,
        'question':question, 'result':result, 'replay_policy':policy,
        'comparison_assessment':from_record(record, artifact, lambda ident: tools._read_artifact(ident, 'proposal'))}
    if receipt['status'] == 'ok':
        if receipt.get('dispatch_status') != 'invoked' or receipt.get('output_sha256') != content_hash(receipt.get('output')):
            raise CampaignStopped('research tool success lacks a bound actual invocation')
    return invocation, observation
