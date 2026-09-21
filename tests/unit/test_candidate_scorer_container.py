"""Trusted scorer boundary; normal cases mock transport, never import child code.

The opt-in Docker cases use only authored primitive values, no candidate module.
They are separate from campaign evaluation and require an explicit local image.
"""
import base64
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import uuid

import pytest

from sciel.eval.scorer_container import (BOUND, ROOT, SCORER_FILES, OUTPUT_SCHEMA,
    ScorerInfrastructureError, prepare_scorer, score_in_container)
from sciel.eval.trusted_scoring.adapter import HostBinding
from sciel.eval.trusted_scoring.transport import MAX_BYTES, SCHEMA
from sciel.methods.executor import DockerExecutor, ExecutionRefused
from sciel.methods.models import content_hash

IMAGE = 'sha256:' + 'a' * 64
NAME = 'sciel-eval-' + 'b' * 64
LIMITS = dict(wall_seconds=15.0, cpu_seconds=10.0, memory_bytes=1024**3,
              pids=32, max_output_bytes=6 * MAX_BYTES + 2, max_stderr_bytes=4096)
BINDING = HostBinding('1' * 64, '2' * 64, '3' * 32, 'full', .125)
RESULT = dict(ok=True, score=.25, cases={'success_prob': .25}, artifacts={})
SCORER_RUNTIME = dict(python='3.11.9', numpy='2.2.6')


def payload(kind='beat'):
    values = ({'support': [0.0, 1.0], 'weights': [1.0, 1.0]} if kind == 'beat'
              else {'points': [[.5, .5]] * 11} if kind == 'heil'
              else {'centers': [[.5, .5]] * 26, 'radii': [0.0] * 26,
                    'centers_dtype': 'float64', 'radii_dtype': 'float64',
                    'reported_sum': 'fixture-diagnostic'})
    return json.dumps(dict(schema=SCHEMA, kind=kind, values=values)).encode()


@pytest.fixture
def executor(monkeypatch):
    responses = iter([subprocess.CompletedProcess([], 0, (IMAGE + ' linux arm64\n').encode(), b''),
                      subprocess.CompletedProcess([], 0, b'fixture-version\n', b'')])
    monkeypatch.setattr('sciel.methods.executor._cli', lambda *_a, **_k: next(responses))
    return DockerExecutor(IMAGE)


def prepare(executor, **kw):
    args = dict(kind='beat', payload=payload(), binding=BINDING, limits=LIMITS,
                invocation_id='4' * 32, plan_id='fixture-plan', reservation_id='fixture-reservation')
    args.update(kw)
    return prepare_scorer(executor, **args)


def terminal(request, output=None, *, removed=True, mutate=None, name=NAME):
    output = dict(schema=OUTPUT_SCHEMA, result=deepcopy(RESULT if output is None else output),
                  runtime=deepcopy(SCORER_RUNTIME))
    raw = dict(schema='sciel-method-container-receipt/1',
        **{key: request[key] for key in BOUND}, status='ok', reason=None,
        invoked=True, exit_code=0, output=output, output_sha256=content_hash(output),
        limits=request['limits'], resources=dict(cpu_seconds=.125, wall_seconds=.25,
            memory_peak_bytes=12345, output_bytes=len(json.dumps(output)), stderr_bytes=0),
        stderr='', stderr_truncated=False)
    if mutate:
        mutate(raw)
    return dict(stdout=json.dumps(raw).encode(), stderr='', exit_code=0, timed_out=False,
                overflow=False, dispatched=True, removed=removed, name=name)


def install_terminal(executor, monkeypatch, **kw):
    calls = []
    def transport(directory, request, *, container_name):
        calls.append(deepcopy(request))
        assert container_name == NAME
        assert json.loads((directory / 'request.json').read_bytes()) == request
        assert not list((directory / 'dependencies').iterdir())
        return terminal(request, name=container_name, **kw)
    monkeypatch.setattr(executor, '_container', transport)
    return calls


def test_exact_trusted_bundle_and_request_no_candidate_or_task_mount(executor, monkeypatch):
    prepared = prepare(executor)
    expected = {'worker.py', *('trusted_scoring/' + name for name in SCORER_FILES)}
    assert set(prepared['bundle']['files']) == expected
    for name in SCORER_FILES:
        assert prepared['bundle']['files']['trusted_scoring/' + name] == (ROOT / 'trusted_scoring' / name).read_text()
    assert prepared['bundle']['files']['worker.py'] == (ROOT / '_scorer_worker.py').read_text()
    assert prepared['request']['input'] == prepared['input']
    assert prepared['request']['input_sha256'] == content_hash(prepared['input'])
    assert prepared['request']['source_sha256'] == content_hash(prepared['bundle']['files'])
    assert prepared['request']['methods'] == {prepared['bundle']['id']: {'root': '/bundle', 'entrypoint': 'worker.py:run'}}
    calls = install_terminal(executor, monkeypatch)
    completed = score_in_container(executor, prepared, container_name=NAME)
    assert calls == [prepared['request']]
    assert completed['result'] == RESULT
    assert completed['receipt']['resources']['cpu_seconds'] == .125
    assert completed['receipt']['resource_measurement']['cpu'] == 'trusted_cgroup_observation'
    completed['result']['cases']['success_prob'] = 99
    assert completed['receipt']['output']['result']['cases']['success_prob'] == .25
    assert completed['receipt']['output']['runtime'] == SCORER_RUNTIME


def test_duplicate_json_keys_are_preserved_not_host_decoded(executor, monkeypatch):
    raw = b'{"kind":"beat","kind":"circle"}'
    prepared = prepare(executor, payload=raw)
    assert base64.b64decode(prepared['input']['payload_base64']) == raw
    invalid_candidate = dict(ok=False, score=0.0, cases={'success_prob': 0.0}, artifacts={'error': 'duplicate key'})
    calls = install_terminal(executor, monkeypatch, output=invalid_candidate)
    result = score_in_container(executor, prepared, container_name=NAME)
    assert result['result'] == invalid_candidate and len(calls) == 1
    assert result['receipt']['status'] == 'ok'  # A completed scorer can reject candidate values.


@pytest.mark.parametrize('kind', ['circle', 'heil', 'beat'])
def test_each_supported_kind_exact_input_and_binding(executor, kind):
    prepared = prepare(executor, kind=kind, payload=payload(kind))
    assert prepared['input']['kind'] == kind
    assert prepared['input']['binding']['elapsed_seconds'] == .125
    assert prepared['bundle']['manifest']['input_schema']['properties']['kind']['enum'] == [kind]


@pytest.mark.parametrize('kw', [dict(kind='oracle'), dict(payload=b''), dict(payload='{}'),
    dict(payload=b'x' * (MAX_BYTES + 1)), dict(binding=replace(BINDING, elapsed_seconds=True)),
    dict(binding=replace(BINDING, elapsed_seconds=float('nan'))),
    dict(binding=replace(BINDING, stage='confirm')), dict(invocation_id='bad'),
    dict(plan_id=''), dict(reservation_id='a\nb'), dict(limits={**LIMITS, 'cpu_seconds': 0})])
def test_invalid_prepare_refuses_before_dispatch(executor, monkeypatch, kw):
    monkeypatch.setattr(executor, 'execute', lambda *_a, **_k: pytest.fail('dispatch forbidden'))
    with pytest.raises((ValueError, TypeError)):
        prepare(executor, **kw)


@pytest.mark.parametrize('change', ['worker', 'scorer', 'extra_file', 'binding', 'input', 'limits', 'request', 'container'])
def test_changed_preparation_cannot_dispatch(executor, monkeypatch, change):
    prepared = prepare(executor)
    if change == 'worker': prepared['bundle']['files']['worker.py'] += '\n# changed'
    elif change == 'scorer': prepared['bundle']['files']['trusted_scoring/beat.py'] += '\n# changed'
    elif change == 'extra_file': prepared['bundle']['files']['candidate.py'] = 'raise AssertionError()'
    elif change == 'binding': prepared['input']['binding']['source_sha256'] = 'f' * 64
    elif change == 'input': prepared['input']['payload_base64'] = base64.b64encode(b'{}').decode()
    elif change == 'limits': prepared['request']['limits']['cpu_seconds'] = 1
    elif change == 'request': prepared['request']['request_sha256'] = 'f' * 64
    monkeypatch.setattr(executor, 'execute', lambda *_a, **_k: pytest.fail('dispatch forbidden'))
    with pytest.raises(ScorerInfrastructureError) as caught:
        score_in_container(executor, prepared, container_name='other' if change == 'container' else NAME)
    assert caught.value.receipt is None
    assert caught.value.not_dispatched is True


@pytest.mark.parametrize('error,known', [(ExecutionRefused('proven predispatch refusal'), True),
    (OSError('transport uncertain'), False), (RuntimeError('transport uncertain'), False)])
def test_only_proven_executor_refusal_claims_no_dispatch(executor, monkeypatch, error, known):
    prepared = prepare(executor)
    calls = []
    def execute(*_a, **_k):
        calls.append(1)
        raise error
    monkeypatch.setattr(executor, 'execute', execute)
    with pytest.raises(ScorerInfrastructureError) as caught:
        score_in_container(executor, prepared, container_name=NAME)
    assert caught.value.not_dispatched is known
    assert caught.value.receipt is None
    assert calls == [1]


@pytest.mark.parametrize('defect', ['lost', 'bad_request', 'cleanup', 'failed', 'raise'])
def test_one_attempt_unknown_and_failure_never_become_score(executor, monkeypatch, defect):
    prepared, calls = prepare(executor), []
    def transport(_directory, request, *, container_name):
        calls.append(1)
        if defect == 'raise': raise OSError('fixture transport disappeared')
        if defect == 'lost': return dict(stdout=b'', stderr='', dispatched=True, removed=True, name=container_name)
        mutation = (lambda row: row.update(request_sha256='f' * 64)) if defect == 'bad_request' else \
                   (lambda row: row.update(status='failed', reason='cpu_limit', exit_code=71)) if defect == 'failed' else None
        return terminal(request, removed=defect != 'cleanup', mutate=mutation, name=container_name)
    monkeypatch.setattr(executor, '_container', transport)
    with pytest.raises(ScorerInfrastructureError) as caught:
        score_in_container(executor, prepared, container_name=NAME)
    assert calls == [1]
    if defect == 'raise':
        assert caught.value.receipt is None
    else:
        receipt = caught.value.receipt
        assert receipt['status'] == ('failed' if defect == 'failed' else 'unknown')
        assert receipt['resources']['cpu_seconds'] == (.125 if defect in ('cleanup', 'failed') else None)
        assert receipt['container']['removed'] is (defect != 'cleanup')


@pytest.mark.parametrize('field', list(BOUND) + ['runtime', 'limits', 'container', 'cpu', 'measurement'])
def test_resealed_unbound_or_malformed_receipt_never_claims_cpu(executor, monkeypatch, field):
    prepared = prepare(executor)
    install_terminal(executor, monkeypatch)
    receipt = score_in_container(executor, prepared, container_name=NAME)['receipt']
    if field in BOUND: receipt[field] = 'f' * 64
    elif field == 'runtime': receipt['runtime']['docker_server'] = 'other'
    elif field == 'limits': receipt['limits']['cpu_seconds'] = 9.0
    elif field == 'container': receipt['container']['name'] = 'sciel-eval-' + 'c' * 64
    elif field == 'cpu': receipt['resources']['cpu_seconds'] = True
    else: receipt['resource_measurement']['cpu'] = 'child_claim'
    receipt['receipt_sha256'] = content_hash({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    calls = []
    def execute(*_a, **_k):
        calls.append(1)
        return receipt
    monkeypatch.setattr(executor, 'execute', execute)
    with pytest.raises(ScorerInfrastructureError) as caught:
        score_in_container(executor, prepared, container_name=NAME)
    assert caught.value.receipt is None and calls == [1]


@pytest.mark.parametrize('output', [dict(ok=1, score=.25, cases={}, artifacts={}),
    dict(ok=True, score=True, cases={}, artifacts={}),
    dict(ok=True, score=.25, cases={'x': True}, artifacts={}),
    dict(ok=True, score=.25, cases={}, artifacts={'x': 1}),
    dict(ok=True, score=.25, cases={}, artifacts={}, extra=0)])
def test_malformed_scorer_output_is_infrastructure_with_known_charge(executor, monkeypatch, output):
    calls = install_terminal(executor, monkeypatch, output=output)
    with pytest.raises(ScorerInfrastructureError) as caught:
        score_in_container(executor, prepare(executor), container_name=NAME)
    assert len(calls) == 1
    assert caught.value.receipt['resources']['cpu_seconds'] == .125


@pytest.mark.parametrize('runtime', [None, {}, {'python': '3.11.9'},
    {'python': True, 'numpy': '2.2.6'}, {'python': '3.11.9', 'numpy': ''},
    {'python': '3.11.9', 'numpy': 'a\nb'}, {'python': '3.11.9', 'numpy': '2.2.6', 'extra': 'x'}])
def test_missing_or_malformed_runtime_cannot_be_scientific_result(executor, monkeypatch, runtime):
    def mutate(raw):
        raw['output']['runtime'] = runtime
        raw['output_sha256'] = content_hash(raw['output'])
    calls = install_terminal(executor, monkeypatch, mutate=mutate)
    with pytest.raises(ScorerInfrastructureError) as caught:
        score_in_container(executor, prepare(executor), container_name=NAME)
    assert len(calls) == 1 and caught.value.receipt['resources']['cpu_seconds'] == .125


@pytest.mark.parametrize('kind', ['circle', 'heil', 'beat', 'duplicate'])
def test_real_primitive_scorer_container_opt_in(kind, tmp_path):
    image = os.environ.get('SCIEL_TEST_SCORER_IMAGE')
    if not image:
        pytest.skip('explicit local immutable image required; no Docker in normal tests')
    executor = DockerExecutor(image)
    actual_kind = 'beat' if kind == 'duplicate' else kind
    raw = b'{"kind":"beat","kind":"circle"}' if kind == 'duplicate' else payload(kind)
    token = uuid.uuid4().hex
    prepared = prepare(executor, kind=actual_kind, payload=raw, invocation_id=token)
    name = 'sciel-eval-' + content_hash(dict(fixture='primitive-scorer', token=token))
    (tmp_path / 'intent.json').write_text(json.dumps(dict(request=prepared['request'], container_name=name)))
    completed = score_in_container(executor, prepared, container_name=name)
    (tmp_path / 'receipt.json').write_text(json.dumps(completed['receipt']))
    result, receipt = completed['result'], completed['receipt']
    assert receipt['container'] == dict(name=name, removed=True, image_id=image)
    assert receipt['resources']['cpu_seconds'] > 0
    assert receipt['resource_measurement']['cpu'] == 'trusted_cgroup_observation'
    assert receipt['output']['schema'] == OUTPUT_SCHEMA
    assert set(receipt['output']['runtime']) == {'python', 'numpy'}
    assert all(type(value) is str and value for value in receipt['output']['runtime'].values())
    assert result['ok'] is (kind != 'duplicate')
    assert result['score'] == (.25 / .4006 if kind == 'beat' else 0.0)
    if kind == 'beat':
        assert result['cases']['success_prob'] == .25
        assert result['cases']['combined_score'] == result['score']
    if kind == 'circle':
        evidence = json.loads(result['artifacts']['native_geometry'])
        assert evidence['source_sha256'] == BINDING.source_sha256
        assert result['artifacts']['reported_sum'] == 'fixture-diagnostic'
    if kind == 'duplicate':
        assert 'duplicate' in result['artifacts']['error']
