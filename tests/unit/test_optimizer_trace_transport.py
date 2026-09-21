"""Inert sidecar transport and task preparation, without candidate execution."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from sciel.eval import repeat_draw as adapter
from sciel.hypothesis import restart_optimizer_trace as codec
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.engine import repeat_support
from sciel.eval.registry import load_task
from test_repeat_trace_transport import fake_draw
import test_repeat_trace_transport as transport
from test_traced_circle_support import preparation, CANONICAL, hashes


def status_trace(binding):
    identity = {key: binding[key] for key in ('source_sha256', 'seed', 'runtime_sha256', 'invocation_id')}
    identity['scaffold_sha256'] = transport.SCAFFOLD
    trace = codec.RestartTrace(identity)
    values = np.zeros(78)
    values[:52] = .5
    args = {name: values.copy() for name in ('best', 'initial', 'noise', 'ordinary')}
    trace.begin_start(0, incumbent_sum=0., improvements=0)
    trace.begin_helper(args)
    trace.helper_end(raw_kind='none', post_inputs=args, effective=values,
        origin='saved_ordinary_fallback', incumbent_sum=0., improvements=0)
    trace.optimizer_enter(incumbent_sum=0., improvements=0)
    trace.optimizer_exit(incumbent_sum=0., improvements=0)
    trace.optimizer_outcome(False, SimpleNamespace(status=9))
    trace.optimizer_result(values)
    trace.end_start('completed', incumbent_sum=0., improvements=0)
    return trace.finish('restart_limit', incumbent_sum=0., improvements=0)


@pytest.mark.parametrize('tamper', [False, True])
def test_real_adapter_decoder_accepts_v3_and_rejects_rehashed_outcome_without_losing_quality(fake_draw, monkeypatch, tamper):
    monkeypatch.setattr(transport, 'empty_trace', status_trace)
    draw, behavior, calls = fake_draw
    if tamper:
        def alter(envelope, _directory, _setup):
            trace = envelope['trace']
            trace['calls'][0]['optimizer_outcome']['success']['value'] = 0
            trace['trace_sha256'] = adapter._sha(adapter._encoded({k:v for k,v in trace.items() if k != 'trace_sha256'}))
        behavior['mutate'] = alter
    receipt = draw(restart_trace=True)
    assert receipt['status'] == 'valid' and receipt['raw_sum'] == 0.
    assert [mode for mode, _ in calls] == ['candidate', 'certificate']
    assert receipt['restart_trace']['status'] == ('invalid' if tamper else 'captured')
    if not tamper:
        trace = receipt['restart_trace']['trace']
        assert trace['schema'] == codec.TRACE_SCHEMA
        assert trace['calls'][0]['optimizer_outcome']['status'] == dict(value=9, reason=None)
    else:
        assert receipt['restart_trace']['trace'] is None


def test_explicit_profile_two_cli_prepares_distinct_bound_task_without_execution(tmp_path, capsys):
    before = hashes(CANONICAL)
    target = tmp_path/'optimizer-task'
    preparation.main(['--output', str(target), '--execution-profile', builder.EXECUTION_PROFILE])
    receipt = json.loads(capsys.readouterr().out)
    task = load_task(target)
    source, _ = repeat_support.validate_circle_task(task, initial_source_policy=builder.OPERATION)
    assert source == builder.source_for().encode()
    assert receipt['task_hash'] == task.task_hash == repeat_support.OPTIMIZER_TRACED_TASK_HASH
    assert receipt['execution_profile'] == builder.EXECUTION_PROFILE
    assert receipt['initial_source_policy'] == builder.OPERATION and receipt['executed'] is False
    assert receipt['files'] == hashes(target) and hashes(CANONICAL) == before
    with pytest.raises(ValueError):
        repeat_support.validate_circle_task(task, initial_source_policy='circle_restart_body_traced/1')


@pytest.mark.parametrize('profile', [None, False, 'unknown'])
def test_preparation_rejects_invalid_profile_before_publication(tmp_path, profile):
    target = tmp_path/'never-created'
    with pytest.raises(ValueError):
        preparation.prepare(CANONICAL, target, execution_profile=profile)
    assert not target.exists()
