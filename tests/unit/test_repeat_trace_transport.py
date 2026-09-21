"""Inert transport tests: fake process/runtime; no helper, solver, Docker or API."""
from copy import deepcopy
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from sciel.eval import _repeat_runner as runner
from sciel.eval import repeat_draw as adapter
from sciel.eval import repeat_docker as docker
from sciel.hypothesis.restart_trace import RestartTrace


SCAFFOLD = 'a' * 64
SOURCE = 'SCIEL_RESTART_TRACE_SCAFFOLD_SHA256 = ' + repr(SCAFFOLD) + '\n# inert fixture; never imported\n'
LIMITS = dict(seed=19, timeout=4., cpu_seconds=3, mem_bytes=2**30, certification_seconds=2.)
PACKING = {'centers': [[.5, .5]] * 26, 'radii': [0.] * 26}


def caps():
    return {'cpu_seconds': [3, 3], 'mem_bytes': [2**30, 2**30], 'file_bytes': [1048576, 1048576]}


def empty_trace(binding):
    identity = {key: binding[key] for key in ('source_sha256', 'seed', 'runtime_sha256', 'invocation_id')}
    identity['scaffold_sha256'] = SCAFFOLD
    return RestartTrace(identity).finish('restart_limit', incumbent_sum=0., improvements=0)


@pytest.fixture
def fake_draw(monkeypatch):
    runtime = {'python_executable': '/fake/python', 'platform': 'inert transport fixture'}
    identity = {'schema': 'sciel-repeat-adapter/1', 'source_sha256': adapter._source_hashes(),
                'runtime_config': {'backend': 'native'}, 'runtime': runtime}
    identity['identity_sha256'] = adapter._sha(adapter._encoded(identity))
    monkeypatch.setattr(adapter, '_python', lambda _python: '/fake/python')
    monkeypatch.setattr(adapter, 'adapter_identity', lambda *a, **k: pytest.fail('runtime probe forbidden'))
    calls = []
    behavior = {'sidecar': 'valid', 'candidate_failure': False, 'mutate': None, 'packing': deepcopy(PACKING)}

    def process(mode, request, directory, **kwargs):
        calls.append((mode, deepcopy(request)))
        if mode == 'certificate':
            return {'exit_code': 0, 'timed_out': False, 'stderr_tail': '',
                    'setup': {'status': 'ready', 'limits_applied': caps()},
                    'output': {'certificate': adapter.geometry.certify_packing(behavior['packing'])}}
        assert mode == 'candidate'
        setup = {'status': 'ready', 'source_sha256': request['source_sha256'],
                 'runtime': request['expected_runtime'], 'limits_applied': caps(),
                 'seed_applied': {'python_random': 19, 'numpy_legacy': 19, 'python_hash_seed': '19',
                                 'before_candidate_import': True, 'determinism_proven': False}}
        if request.get('restart_trace'):
            binding = {'source_sha256': request['source_sha256'], 'seed': request['seed'],
                       'runtime_sha256': adapter._sha(adapter._encoded(runtime)),
                       'adapter_identity_sha256': request['adapter_identity_sha256'],
                       'invocation_id': request['restart_trace_invocation_id']}
            envelope = {'schema': runner.TRACE_SCHEMA, **binding, 'status': 'captured',
                        'reason': None, 'trace': empty_trace(binding)}
            if behavior['mutate']:
                behavior['mutate'](envelope, directory, setup)
            path = directory / 'candidate_trace.json'
            kind = behavior['sidecar']
            if kind == 'valid':
                path.write_bytes(adapter._encoded(envelope))
            elif kind == 'oversize':
                path.write_bytes(b' ' * (96 * 1024 + 1))
            elif kind == 'duplicate':
                path.write_text('{"status":"captured","status":"missing"}')
            elif kind == 'nonfinite':
                path.write_text('{"value":NaN}')
            elif kind == 'symlink':
                other = directory / 'other.json'; other.write_bytes(adapter._encoded(envelope))
                path.symlink_to(other)
            elif kind != 'missing':
                raise AssertionError(kind)
        output = ({'status': 'candidate_failure', 'reason': 'inert fixture failure'}
                  if behavior['candidate_failure'] else {'status': 'ok', 'packing': deepcopy(behavior['packing'])})
        return {'exit_code': 0, 'timed_out': False, 'stderr_tail': 'unchanged marker',
                'setup': setup, 'output': output}

    monkeypatch.setattr(adapter, '_process', process)
    def draw(source=SOURCE, **kwargs):
        return adapter.draw(source, **(LIMITS | {'expected_identity': identity} | kwargs))
    return draw, behavior, calls


def test_default_has_no_trace_request_or_receipt_fields(fake_draw):
    draw, _, calls = fake_draw
    result = draw()
    assert result['status'] == 'valid'
    assert 'restart_trace' not in result
    assert set(calls[0][1]) == {'source_path', 'source_sha256', 'seed', 'cpu_seconds', 'mem_bytes', 'expected_runtime'}
    assert [mode for mode, _ in calls] == ['candidate', 'certificate']


@pytest.mark.parametrize('value', [None, 0, 1, 'true'])
def test_opt_in_is_strict_boolean_before_any_dispatch(monkeypatch, value):
    monkeypatch.setattr(adapter, 'adapter_identity', lambda *a, **k: pytest.fail('preflight forbidden'))
    with pytest.raises(ValueError, match='restart_trace must be a boolean'):
        adapter.draw(SOURCE, **LIMITS, restart_trace=value)


def test_valid_trace_fully_validated_and_not_duplicated_in_process(fake_draw):
    draw, _, calls = fake_draw
    result = draw(restart_trace=True)
    assert result['status'] == 'valid'
    assert result['raw_sum'] == 0 and result['certified_sum'] == {'numerator': 0, 'denominator': 1}
    captured = result['restart_trace']
    assert captured['status'] == 'captured' and captured['reason'] is None
    assert len(captured['sidecar_sha256']) == 64
    assert captured['trace']['identity']['source_sha256'] == result['source_sha256']
    assert captured['trace']['identity']['invocation_id'] == calls[0][1]['restart_trace_invocation_id']
    assert 'trace' not in result['candidate_process'] and 'restart_trace' not in result['candidate_process']
    assert not any('restart_trace' in request for mode, request in calls if mode == 'certificate')


@pytest.mark.parametrize('kind', ['missing', 'oversize', 'duplicate', 'nonfinite', 'symlink'])
def test_bad_sidecar_preserves_certified_quality(fake_draw, kind):
    draw, behavior, _ = fake_draw; behavior['sidecar'] = kind
    result = draw(restart_trace=True)
    assert result['status'] == 'valid' and result['raw_sum'] == 0
    assert result['restart_trace']['status'] == ('missing' if kind == 'missing' else 'invalid')
    assert result['restart_trace']['trace'] is None


@pytest.mark.parametrize('field', ['source_sha256', 'seed', 'runtime_sha256', 'adapter_identity_sha256', 'invocation_id'])
def test_envelope_tamper_is_not_valid_facts(fake_draw, field):
    draw, behavior, _ = fake_draw
    def change(envelope, _directory, _setup):
        envelope[field] = True if field == 'seed' else 'f' * len(envelope[field])
    behavior['mutate'] = change
    result = draw(restart_trace=True)
    assert result['status'] == 'valid' and result['restart_trace']['status'] == 'invalid'
    assert result['restart_trace']['trace'] is None


@pytest.mark.parametrize('mutation', ['source', 'scaffold', 'collector_digest', 'inner_invocation', 'setup'])
def test_inner_trace_source_and_setup_joins_are_required(fake_draw, mutation):
    draw, behavior, _ = fake_draw
    def change(envelope, directory, setup):
        if mutation == 'source':
            (directory / 'candidate.py').write_text('changed')
        elif mutation == 'scaffold':
            envelope['trace']['identity']['scaffold_sha256'] = 'b' * 64
        elif mutation == 'inner_invocation':
            envelope['trace']['identity']['invocation_id'] = 'c' * 32
        elif mutation == 'setup':
            setup['seed_applied']['before_candidate_import'] = False
        else:
            envelope['trace']['unexpected'] = 'tampered'
        trace = envelope['trace']
        trace['trace_sha256'] = adapter._sha(adapter._encoded({k: v for k, v in trace.items() if k != 'trace_sha256'}))
    behavior['mutate'] = change
    result = draw(restart_trace=True)
    assert result['status'] == ('infrastructure' if mutation == 'setup' else 'valid')
    assert result['restart_trace']['status'] == 'invalid' and result['restart_trace']['trace'] is None


def test_source_scaffold_must_be_unique_literal_without_execution(fake_draw):
    draw, _, _ = fake_draw
    result = draw(source="SCIEL_RESTART_TRACE_SCAFFOLD_SHA256 = 'a' * 64\n", restart_trace=True)
    assert result['status'] == 'valid' and result['restart_trace']['status'] == 'invalid'
    assert result['restart_trace']['trace'] is None


def test_candidate_failure_retains_valid_partial_diagnostic_without_certificate(fake_draw):
    draw, behavior, calls = fake_draw; behavior['candidate_failure'] = True
    result = draw(restart_trace=True)
    assert result['status'] == 'candidate_failure' and result['raw_sum'] is None
    assert result['restart_trace']['status'] == 'captured'
    assert [mode for mode, _ in calls] == ['candidate']


@pytest.mark.parametrize('kind', ['wrong_sum', 'error_status', 'small_roundoff', 'zero_not_exact'])
def test_rehashed_trace_must_join_same_valid_packing_without_changing_quality(fake_draw, kind):
    draw, behavior, _ = fake_draw
    if kind != 'zero_not_exact':
        behavior['packing'] = {'centers': [[(i % 5 + .5) / 5, (i // 5 + .5) / 5] for i in range(25)] + [[0., 0.]],
                               'radii': [.01] * 25 + [0.]}
    raw = math.fsum(behavior['packing']['radii'])
    total = (raw + .01 if kind == 'wrong_sum' else
             raw + 4 * math.ulp(raw) if kind == 'small_roundoff' else
             math.ulp(0.) if kind == 'zero_not_exact' else raw)
    def change(envelope, _directory, _setup):
        trace = envelope['trace']
        trace['initial_quality']['sum'] = total
        trace['final_quality']['sum'] = total
        if kind == 'error_status': trace['status'] = 'error'
        trace['trace_sha256'] = adapter._sha(adapter._encoded({k: v for k, v in trace.items() if k != 'trace_sha256'}))
    behavior['mutate'] = change
    result = draw(restart_trace=True)
    assert result['status'] == 'valid' and result['raw_sum'] == raw
    diagnostic = result['restart_trace']
    assert diagnostic['status'] == ('captured' if kind == 'small_roundoff' else 'invalid')
    assert diagnostic['quality_join']['maximum_roundoff'] == (0. if raw == 0 else 32 * math.ulp(raw))
    assert diagnostic['quality_join']['trace_sum'] == total
    if kind != 'small_roundoff':
        assert diagnostic['trace'] is None and diagnostic['reason'] == 'trace_result_mismatch'


@pytest.fixture
def runner_call(tmp_path, monkeypatch):
    python_random_state, numpy_random_state = random.getstate(), np.random.get_state()
    runtime = {'inert_runtime': 'same'}
    source = tmp_path / 'candidate.py'; source.write_text(SOURCE)
    request = {'source_path': str(source), 'source_sha256': adapter._sha(source.read_bytes()),
               'seed': 19, 'cpu_seconds': 3, 'mem_bytes': 2**30, 'expected_runtime': runtime,
               'restart_trace': True, 'adapter_identity_sha256': 'b' * 64,
               'restart_trace_invocation_id': 'c' * 32}
    monkeypatch.setenv('PYTHONHASHSEED', '19')
    monkeypatch.setattr(runner, '_limits', lambda *a: caps())
    monkeypatch.setattr(runner, '_runtime', lambda: runtime)
    module = SimpleNamespace(); called = []
    def candidate():
        called.append(1)
        module.SCIEL_RESTART_TRACE = empty_trace(module.SCIEL_RESTART_TRACE_BINDING)
        return PACKING['centers'], PACKING['radii'], 999
    module.run_packing = candidate
    monkeypatch.setattr(runner, '_load', lambda *a: module)
    def run():
        path = tmp_path / 'request.json'; path.write_text(json.dumps(request))
        result = runner.main(['candidate', str(path), str(tmp_path / 'setup.json'), str(tmp_path / 'candidate_output.json')])
        return result, json.loads((tmp_path / 'candidate_output.json').read_text())
    yield run, request, module, called, tmp_path
    random.setstate(python_random_state)
    np.random.set_state(numpy_random_state)


def test_runner_one_invocation_same_binding_separate_output(runner_call):
    run, request, module, called, path = runner_call
    result, output = run()
    assert result == 0 and called == [1]
    assert output == {'status': 'ok', 'packing': PACKING}
    sidecar = json.loads((path / 'candidate_trace.json').read_text())
    assert sidecar['status'] == 'captured' and sidecar['invocation_id'] == request['restart_trace_invocation_id']
    assert sidecar['trace']['identity']['invocation_id'] == sidecar['invocation_id']
    assert module.SCIEL_RESTART_TRACE_BINDING['runtime_sha256'] == adapter._sha(adapter._encoded(request['expected_runtime']))


def test_runner_collects_global_after_exception_without_retry(runner_call):
    run, _, module, called, path = runner_call
    original = module.run_packing
    def fail():
        original()
        raise ValueError('inert candidate failure')
    module.run_packing = fail
    result, output = run()
    assert result == 0 and called == [1] and output['status'] == 'candidate_failure'
    assert json.loads((path / 'candidate_trace.json').read_text())['status'] == 'captured'


@pytest.mark.parametrize('kind', ['missing', 'nonfinite', 'oversize'])
def test_runner_trace_capture_failure_does_not_change_numeric_output(runner_call, kind):
    run, _, module, called, path = runner_call
    def candidate():
        called.append(1)
        if kind == 'nonfinite': module.SCIEL_RESTART_TRACE = {'value': float('nan')}
        if kind == 'oversize': module.SCIEL_RESTART_TRACE = {'value': 'x' * (96 * 1024)}
        return PACKING['centers'], PACKING['radii'], 0
    module.run_packing = candidate
    result, output = run()
    sidecar = json.loads((path / 'candidate_trace.json').read_text())
    assert result == 0 and called == [1] and output == {'status': 'ok', 'packing': PACKING}
    assert sidecar['status'] == ('missing' if kind == 'missing' else 'invalid')
    assert sidecar['trace'] is None and (path / 'candidate_trace.json').stat().st_size <= 96 * 1024


def test_runner_sidecar_io_failure_does_not_replace_packing(runner_call):
    run, _, _, called, path = runner_call
    (path / 'candidate_trace.json.tmp').write_text('existing exclusive path')
    result, output = run()
    assert result == 0 and called == [1] and output['status'] == 'ok'
    assert not (path / 'candidate_trace.json').exists()


def test_runner_source_mutation_invalidates_trace_only(runner_call):
    run, _, module, called, path = runner_call; original = module.run_packing
    def mutate():
        result = original(); (path / 'candidate.py').write_text('changed'); return result
    module.run_packing = mutate
    result, output = run()
    assert result == 0 and called == [1] and output['status'] == 'ok'
    sidecar = json.loads((path / 'candidate_trace.json').read_text())
    assert sidecar['status'] == 'invalid' and sidecar['reason'] == 'source_changed' and sidecar['trace'] is None


def test_docker_uses_same_work_sidecar_without_extra_numeric_fields(tmp_path, monkeypatch):
    requests = []
    monkeypatch.setattr(docker, '_cli', lambda *a, **k: SimpleNamespace(returncode=0, stderr=b''))
    monkeypatch.setattr(docker, '_attach', lambda *a, **k: (0, False, ''))
    monkeypatch.setattr(docker, '_remove', lambda *a: None)
    request = {'source_path': str(tmp_path / 'candidate.py'), 'source_sha256': 'a' * 64,
               'seed': 19, 'cpu_seconds': 3, 'mem_bytes': 2**30, 'restart_trace': True,
               'adapter_identity_sha256': 'b' * 64, 'restart_trace_invocation_id': 'c' * 32}
    result = docker.process('candidate', request, tmp_path, timeout=4., seed=19,
        image='sha256:' + 'd' * 64, files=adapter._FILES,
        write_json=lambda path, value: requests.append((path, value)),
        read_json=lambda path: pytest.fail('no output fixture exists'))
    translated = requests[0][1]
    assert translated['source_path'] == '/work/candidate.py'
    assert translated['restart_trace_invocation_id'] == request['restart_trace_invocation_id']
    assert 'trace' not in result and result['output'] is None
    # Runner derives candidate_trace.json from this same output parent, so no
    # caller-controlled host path or new Docker mount is needed.
    assert Path('/work/candidate_output.json').with_name('candidate_trace.json') == Path('/work/candidate_trace.json')
