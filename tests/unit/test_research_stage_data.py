"""Trusted handoff helpers only; the tool source is never imported or executed."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from sciel.engine.research_targets import freeze_targets, tool_inputs
from sciel.engine.types import Snapshot
from sciel.methods.models import canonical_bytes, content_hash
from sciel.research_tools.stage_data import MAX_INPUT_BYTES, attach_stage_input


def inputs():
    snapshot = Snapshot(0, 'diff', 'fixture', 4, 'p1', 'def solve(): return 1\n', .5)
    program = SimpleNamespace(id='p1', code=snapshot.parent_code, score=.5)
    engine = SimpleNamespace(log=SimpleNamespace(run_id='run'), task=SimpleNamespace(task_hash='task'),
        observed_changes=None, iteration_no=4, B_evals=3,
        db=SimpleNamespace(pop_best_id='p1', get=lambda _:program))
    snapshot.research_targets = freeze_targets(engine, program, program.code)
    return tool_inputs(snapshot, {'task_hash': 'task'}, {'seed': 41})


def invocation(previous, observations):
    output = dict(schema='sciel-tool-result/1', status='success', observations=observations,
                  diagnostics={}, error=None)
    receipt = dict(method_id='a' * 64, invocation_id='b' * 64,
        status='ok', dispatch_status='invoked', input_sha256=content_hash(previous),
        output=output, output_sha256=content_hash(output), source_sha256='c' * 64,
        runtime_sha256='d' * 64, limits={'cpu_seconds': 20, 'wall_seconds': 30},
        resources={'cpu_seconds': 1.5, 'wall_seconds': 2.5})
    receipt['receipt_sha256'] = content_hash(receipt)
    body = dict(inputs=deepcopy(previous), receipt=receipt, tool_id=receipt['method_id'])
    return body, content_hash(dict(schema='sciel-research-tool-artifact/1', kind='invocation', payload=body))


def test_multistage_values_seed_and_state_are_exact_without_recursive_history():
    original = inputs()
    data = {'best': [1., -0., .12345678901234567],
            'continuation_state': {'rng': [42, 0, 19], 'counter': 100, 'note': '雪🙂'}}
    first, first_id = invocation(original, data)
    next_inputs = attach_stage_input(inputs(), first, receipt_id=first_id)
    assert canonical_bytes(next_inputs['research_stage']['observations']) == canonical_bytes(data)
    assert next_inputs['continuation_state'] == data['continuation_state']
    assert next_inputs['research_stage']['arguments'] == {'seed': 41}
    assert next_inputs['research_stage']['conditions'] == {'cpu_seconds': 20, 'wall_seconds': 30}
    assert next_inputs['research_stage']['prior_receipt_id'] == first_id
    second, second_id = invocation(next_inputs, {'next': 'stage output'})
    third_inputs = attach_stage_input(next_inputs, second, receipt_id=second_id)
    assert 'continuation_state' not in third_inputs
    assert third_inputs['research_stage']['observations'] == {'next': 'stage output'}
    assert 'research_stage' not in third_inputs['research_stage']
    assert 'inputs' not in third_inputs['research_stage']
    assert third_inputs['research_stage']['prior_input_sha256'] == content_hash(next_inputs)
    next_inputs['research_stage']['observations']['best'][0] = 99
    assert first['receipt']['output']['observations'] == data


@pytest.mark.parametrize('changed', ['output', 'inputs', 'receipt_id', 'failure', 'other_run'])
def test_handoff_refuses_unbound_or_failed_or_cross_run_data(changed):
    current = inputs()
    old, ident = invocation(current, {'measurement': .25})
    if changed == 'output': old['receipt']['output']['observations']['measurement'] = .5
    if changed == 'inputs': old['inputs']['arguments']['seed'] = 12
    if changed == 'receipt_id': ident = 'f' * 64
    if changed == 'failure': old['receipt']['status'] = 'failed'
    if changed == 'other_run':
        current['research_targets']['run_id'] = 'other-run'
        current['research_targets']['sha256'] = content_hash(
            {k: v for k, v in current['research_targets'].items() if k != 'sha256'})
    with pytest.raises(ValueError):
        attach_stage_input(current, old, receipt_id=ident)


def test_large_state_refused_without_prefix_clipping():
    current = inputs()
    old, ident = invocation(current, {'continuation_state': {'value': 'x' * MAX_INPUT_BYTES}})
    with pytest.raises(ValueError, match='2 MiB.*no data was truncated'):
        attach_stage_input(current, old, receipt_id=ident)
