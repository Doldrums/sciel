"""Exact JSON handoff between saved research invocations, without LLM copying."""
from copy import deepcopy

from ..methods.models import canonical_bytes, content_hash, require_hash
from ..engine.research_targets import input_binding
from .result_contract import assess_receipt, parse_result

SCHEMA = 'sciel-research-stage-input/1'
MAX_INPUT_BYTES = 2 * 1024 * 1024


def attach_stage_input(inputs, invocation, *, receipt_id):
    """Copy a previously authorized successful receipt into the next tool input.

    The controller authorizes receipt visibility before calling this helper.
    The old input remains in its immutable artifact; embedding it recursively
    would duplicate the entire stage history at every invocation.
    """
    require_hash(receipt_id, 'stage receipt')
    input_binding(inputs)
    if type(invocation) is not dict or type(invocation.get('receipt')) is not dict:
        raise ValueError('stage handoff requires a saved invocation')
    if receipt_id != content_hash(dict(schema='sciel-research-tool-artifact/1',
                                       kind='invocation', payload=invocation)):
        raise ValueError('stage invocation differs from its immutable artifact identity')
    previous = invocation['inputs']
    binding = input_binding(previous)
    receipt = invocation['receipt']
    if (assess_receipt(receipt)['status'] != 'success'
            or receipt.get('input_sha256') != content_hash(previous)
            or receipt.get('output_sha256') != content_hash(receipt.get('output'))
            or receipt.get('method_id') != invocation.get('tool_id')
            or receipt.get('receipt_sha256') != content_hash(
                {key: value for key, value in receipt.items() if key != 'receipt_sha256'})):
        raise ValueError('stage handoff requires an authenticated successful output')
    if (previous['task'] != inputs['task']
            or previous['research_targets']['run_id'] != inputs['research_targets']['run_id']):
        raise ValueError('stage data belongs to another run or task')
    output = parse_result(receipt['output'])
    evaluation_conditions = {}
    for role in ('selected', 'population_best'):
        target = previous['research_targets'][role]
        if target is None:
            evaluation_conditions[role] = None
            continue
        saved = target['saved_evaluation']
        evaluation_conditions[role] = dict(program_id=target['program_id'],
            source_sha256=target['source_sha256'], availability=saved['status'],
            reference=deepcopy(saved['reference']),
            conditions=deepcopy(saved['record']['conditions']) if saved['record'] is not None else None)
    copied = deepcopy(inputs)
    copied['research_stage'] = dict(schema=SCHEMA, prior_receipt_id=receipt_id,
        prior_invocation_id=receipt['invocation_id'], prior_tool_id=invocation['tool_id'],
        prior_input_sha256=receipt['input_sha256'], prior_output_sha256=receipt['output_sha256'],
        prior_source_sha256=receipt['source_sha256'], prior_runtime_sha256=receipt['runtime_sha256'],
        observations=deepcopy(output['observations']),
        observations_sha256=content_hash(output['observations']),
        conditions=deepcopy(receipt['limits']), resources=deepcopy(receipt.get('resources')),
        arguments=deepcopy(previous.get('arguments', {})), target_binding=binding,
        target_evaluation_conditions=evaluation_conditions,
        origin_task=deepcopy(previous['task']))
    if 'continuation_state' in output['observations']:
        copied['continuation_state'] = deepcopy(output['observations']['continuation_state'])
    else:
        copied.pop('continuation_state', None)
    if len(canonical_bytes(copied)) > MAX_INPUT_BYTES:
        raise ValueError('exact stage data exceeds the 2 MiB tool input bound; no data was truncated')
    return copied
