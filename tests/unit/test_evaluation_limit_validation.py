"""Pure/mocked malformed-limit guards; no process or submitted code is run."""
from types import SimpleNamespace
import resource
import sys
from unittest.mock import Mock

import pytest
import yaml

from sciel.eval import registry, sandbox
from sciel.eval.facade import EvalResult


INVALID_TIMEOUTS = [float('nan'), float('inf'), float('-inf'), 0, -1, True, False,
                    '1', None, [], {}, 10**500]
INVALID_INTEGERS = [0, -1, True, False, 1.5, '3', float('nan'), float('inf')]


@pytest.mark.parametrize('timeout', INVALID_TIMEOUTS)
def test_direct_stage_rejects_invalid_deadline_before_file_or_process(monkeypatch, timeout):
    allocate = Mock(side_effect=AssertionError('must reject before allocating transport'))
    spawn = Mock(side_effect=AssertionError('must not start any process'))
    monkeypatch.setattr(sandbox.tempfile, 'mkstemp', allocate)
    monkeypatch.setattr(sandbox.subprocess, 'Popen', spawn)
    with pytest.raises(ValueError, match='timeout'):
        sandbox.run_stage('unused', 'smoke', 'unused', timeout=timeout)
    allocate.assert_not_called()
    spawn.assert_not_called()


@pytest.mark.parametrize('field', ['n', 'cpu_seconds', 'mem_bytes'])
@pytest.mark.parametrize('value', INVALID_INTEGERS)
def test_direct_stage_rejects_malformed_caps_before_dispatch(monkeypatch, field, value):
    allocate = Mock(side_effect=AssertionError('must reject before allocating transport'))
    monkeypatch.setattr(sandbox.tempfile, 'mkstemp', allocate)
    with pytest.raises(ValueError, match=field):
        sandbox.run_stage('unused', 'smoke', 'unused', timeout=1., **{field:value})
    allocate.assert_not_called()


@pytest.mark.parametrize('limits', [{}, {'n':None, 'cpu_seconds':None, 'mem_bytes':None},
    {'n':1, 'cpu_seconds':2, 'mem_bytes':1048576}])
def test_valid_deadline_and_optional_caps_keep_exact_transport_arguments(monkeypatch, limits):
    calls = []
    class Process:
        returncode = 0
        stderr = object()
        def __init__(self, argv, **kwargs):
            calls.append((argv, kwargs))
        def wait(self, timeout):
            calls.append(timeout)
            return 0
    monkeypatch.setattr(sandbox.subprocess, 'Popen', Process)
    monkeypatch.setattr(sandbox, '_kill_group', lambda _: None)
    monkeypatch.setattr(sandbox, '_StderrTail', lambda _: SimpleNamespace(finish=lambda: ''))
    monkeypatch.setattr(sandbox, '_read_result', lambda _: EvalResult(ok=True, score=.25))
    result = sandbox.run_stage('evaluator', 'confirm', 'candidate', timeout=.25,
                               python='trusted-python', **limits)
    assert result.ok and result.score == .25 and calls[1] == .25
    argv = calls[0][0]
    assert argv[:5] == ['trusted-python', sandbox._RUNNER, 'evaluator', 'confirm', 'candidate']
    if limits.get("n") is not None:
        assert argv[6:] == ['1', '--cpu-seconds', '2', '--mem-bytes', '1048576']
    else:
        assert len(argv) == 6


def write_task(path, **overrides):
    cfg = dict(suite='test', name='inert', origin='math', primary_metric='score')
    cfg.update(overrides)
    (path/'task.yaml').write_text(yaml.safe_dump(cfg))
    (path/'evaluator.py').write_text('# inert, never imported\n')
    (path/'initial_program.py').write_text('# inert, never imported\n')
    return cfg


@pytest.mark.parametrize('field', ['smoke_timeout','full_timeout','confirm_timeout','per_run_timeout'])
@pytest.mark.parametrize('value', [float('nan'),float('inf'),float('-inf'),0,-1,True,'2'])
def test_loader_rejects_nonfinite_or_coerced_deadlines(tmp_path, field, value):
    write_task(tmp_path, **{field:value})
    with pytest.raises(ValueError, match=field):
        registry.load_task(str(tmp_path))


@pytest.mark.parametrize('field', ['cpu_seconds','mem_bytes','confirm_n'])
@pytest.mark.parametrize('value', INVALID_INTEGERS)
def test_loader_requires_positive_integer_counts_and_caps(tmp_path, field, value):
    write_task(tmp_path, **{field:value})
    with pytest.raises(ValueError, match=field):
        registry.load_task(str(tmp_path))


@pytest.mark.parametrize('value', ['false','true',0,1,None,[],{}])
def test_loader_does_not_infer_stochastic_from_truthiness(tmp_path, value):
    write_task(tmp_path, stochastic=value)
    with pytest.raises(ValueError, match='stochastic'):
        registry.load_task(str(tmp_path))


@pytest.mark.parametrize('value', [[], ['suite'], 'task', True, 42])
def test_loader_requires_a_yaml_mapping(tmp_path, value):
    (tmp_path/'task.yaml').write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match='mapping'):
        registry.load_task(str(tmp_path))


def test_valid_task_limits_and_content_identity_are_preserved(tmp_path):
    cfg = write_task(tmp_path, stochastic=False, confirm_n=1, smoke_timeout=.25,
                     full_timeout=2, confirm_timeout=3, cpu_seconds=4,
                     mem_bytes=1048576, per_run_timeout=.125)
    spec = registry.load_task(str(tmp_path))
    assert spec.stochastic is False and spec.confirm_n == 1
    assert (spec.smoke_timeout,spec.full_timeout,spec.confirm_timeout) == (.25,2.,3.)
    assert spec.cpu_seconds == 4 and spec.mem_bytes == 1048576
    assert spec.per_run_timeout == .125
    assert spec.task_hash == registry.compute_task_hash(str(tmp_path), cfg)


def test_omitted_caps_and_optional_per_run_deadline_remain_none(tmp_path):
    write_task(tmp_path)
    spec = registry.load_task(str(tmp_path))
    assert spec.cpu_seconds is spec.mem_bytes is spec.per_run_timeout is None
    assert spec.stochastic is True and spec.confirm_n == 10


def test_determinism_probe_forwards_both_caps_to_each_invocation(monkeypatch):
    spec = registry.TaskSpec(root='inert',suite='test',name='inert',origin='math',
        primary_metric='score',smoke_timeout=.5,cpu_seconds=3,mem_bytes=1048576)
    call = Mock(return_value=EvalResult(ok=True,score=.5))
    monkeypatch.setattr(sandbox,'run_stage',call)
    result = registry.determinism_check(spec, python='trusted-python', program_path='chosen')
    assert result['valid'] and result['runs']
    assert call.call_count == 2
    for saved in call.call_args_list:
        assert saved.args == (spec.evaluator_path,'smoke','chosen')
        assert saved.kwargs == dict(timeout=.5,cpu_seconds=3,mem_bytes=1048576,python='trusted-python')


@pytest.mark.parametrize('field,value', [
    ('mem_bytes', sys.maxsize), ('mem_bytes', sys.maxsize + 1),
    ('mem_bytes', 10**500), ('cpu_seconds', sys.maxsize - 1),
    ('cpu_seconds', sys.maxsize), ('cpu_seconds', sys.maxsize + 1),
    ('cpu_seconds', 10**500),
])
def test_nonfinite_resource_representations_rejected_before_transport(monkeypatch, field, value):
    allocate = Mock(side_effect=AssertionError('must reject before allocating transport'))
    spawn = Mock(side_effect=AssertionError('must not start any process'))
    monkeypatch.setattr(sandbox.tempfile, 'mkstemp', allocate)
    monkeypatch.setattr(sandbox.subprocess, 'Popen', spawn)
    with pytest.raises(ValueError, match=field + '.*finite resource range'):
        sandbox.run_stage('unused', 'smoke', 'unused', timeout=1., **{field:value})
    allocate.assert_not_called()
    spawn.assert_not_called()


@pytest.mark.parametrize('field,value', [
    ('mem_bytes', sys.maxsize), ('mem_bytes', 10**500),
    ('cpu_seconds', sys.maxsize - 1), ('cpu_seconds', 10**500),
])
def test_loader_rejects_unbounded_or_unrepresentable_resource_caps(tmp_path, field, value):
    write_task(tmp_path, **{field:value})
    with pytest.raises(ValueError, match=field + '.*finite resource range'):
        registry.load_task(str(tmp_path))


@pytest.mark.parametrize('field', ['cpu_seconds', 'mem_bytes'])
def test_local_rlim_infinity_cannot_be_a_declared_finite_cap(monkeypatch, field):
    # On macOS this is positive sys.maxsize; on Linux it is typically -1.
    allocate = Mock(side_effect=AssertionError('must reject before allocating transport'))
    monkeypatch.setattr(sandbox.tempfile, 'mkstemp', allocate)
    with pytest.raises(ValueError, match=field):
        sandbox.run_stage('unused', 'smoke', 'unused', timeout=1.,
                          **{field:resource.RLIM_INFINITY})
    allocate.assert_not_called()


def test_largest_portable_finite_caps_and_optional_none_are_accepted():
    sandbox.validate_stage_limits(1., n=1, cpu_seconds=sys.maxsize - 2,
                                  mem_bytes=sys.maxsize - 1)
    sandbox.validate_stage_limits(1., n=None, cpu_seconds=None, mem_bytes=None)
    # This establishes input representability, not that every OS will enforce
    # every accepted cap; _runner's existing warning policy is unchanged.
