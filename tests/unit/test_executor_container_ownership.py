"""A reused container name is not authority to start or delete that container."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import uuid
from unittest.mock import patch

import pytest

from sciel.methods.executor import DockerExecutor, _cli


NAME = 'sciel-eval-' + 'a' * 64
CID = 'b' * 64
LIMITS = dict(wall_seconds=5, cpu_seconds=2, memory_bytes=128 * 1024 * 1024,
              pids=32, max_output_bytes=65536, max_stderr_bytes=2048)


def executor():
    result = object.__new__(DockerExecutor)
    result._identity = {'image_id': 'sha256:' + 'c' * 64}
    return result


def completed(code=0, stdout=b'', stderr=b''):
    return subprocess.CompletedProcess([], code, stdout, stderr)


def invoke(ex=None, **kwargs):
    return (ex or executor())._container(Path('/tmp/unused-fixture'), {'limits': LIMITS},
                                       container_name=NAME, **kwargs)


def test_cancel_before_create_does_not_touch_preexisting_name():
    with patch('sciel.methods.executor._cli') as cli:
        result = invoke(cancel_requested=lambda: True)
    cli.assert_not_called()
    assert result['removed'] and not result['dispatched']


@pytest.mark.parametrize('failure', ['conflict', 'timeout', 'malformed_id'])
def test_ambiguous_or_failed_create_does_not_remove_foreign_container(failure):
    calls = []
    def cli(args, **kwargs):
        calls.append(args)
        if args[0] == 'create':
            if failure == 'timeout':
                raise subprocess.TimeoutExpired('docker create', 1)
            return completed(1, stderr=b'Conflict') if failure == 'conflict' else completed(stdout=b'bad-id')
        assert args[0] == 'inspect'
        return completed(stdout=json.dumps({'id': CID, 'owner': 'another-invocation'}).encode())
    with patch('sciel.methods.executor._cli', side_effect=cli), patch('sciel.methods.executor._attach') as attach:
        result = invoke()
    attach.assert_not_called()
    assert [args[0] for args in calls] == ['create', 'inspect']
    assert result['removed'] and not result['dispatched']


def test_create_timeout_recovers_only_own_nonce_and_removes_by_id():
    calls, owner = [], None
    def cli(args, **kwargs):
        nonlocal owner
        calls.append(args)
        if args[0] == 'create':
            owner = next(a.split('=', 2)[2] for a in args if a.startswith('--label=sciel.invocation-owner='))
            raise subprocess.TimeoutExpired('docker create', 1)
        if args[0] == 'inspect':
            return completed(stdout=json.dumps({'id': CID, 'owner': owner}).encode())
        assert args == ['rm', '--force', CID]
        return completed()
    with patch('sciel.methods.executor._cli', side_effect=cli), patch('sciel.methods.executor._attach') as attach:
        result = invoke()
    attach.assert_not_called()
    assert [args[0] for args in calls] == ['create', 'inspect', 'rm']
    assert result['removed'] and not result['dispatched']


@pytest.mark.parametrize('cancel_after_create', [False, True])
def test_successful_create_attaches_and_cleans_up_by_returned_id(cancel_after_create):
    calls = []
    def cli(args, **kwargs):
        calls.append(args)
        return completed(stdout=CID.encode())
    with patch('sciel.methods.executor._cli', side_effect=cli), patch('sciel.methods.executor._attach', return_value={}) as attach:
        result = invoke(cancel_requested=lambda: cancel_after_create and bool(calls))
    assert calls[-1] == ['rm', '--force', CID]
    if cancel_after_create:
        attach.assert_not_called()
    else:
        assert attach.call_args.args[0] == CID
    assert result['removed']
    assert result['dispatched'] is not cancel_after_create


@pytest.mark.parametrize('lookup', [completed(1, stderr=b'daemon unavailable'), completed(stdout=b'{}'), completed(stdout=b'garbage')])
def test_unresolved_create_cleanup_stays_unknown(lookup):
    with patch('sciel.methods.executor._cli', side_effect=[completed(1), lookup]) as cli:
        result = invoke()
    assert not result['removed'] and not result['dispatched']
    assert cli.call_count == 2


def test_failed_create_with_no_existing_container_needs_no_removal():
    with patch('sciel.methods.executor._cli', side_effect=[completed(1), completed(1, stderr=b'No such object')]) as cli:
        result = invoke()
    assert result['removed'] and not result['dispatched']
    assert cli.call_count == 2


@pytest.mark.skipif(not os.environ.get('SCIEL_TEST_METHOD_IMAGE'), reason='explicit immutable local Docker image required')
def test_real_existing_container_survives_name_collision():
    image = os.environ['SCIEL_TEST_METHOD_IMAGE']
    name = 'sciel-eval-' + uuid.uuid4().hex + uuid.uuid4().hex
    created = _cli(['create', '--pull=never', '--name', name, '--network=none', image, '/bin/true'])
    assert created.returncode == 0, created.stderr
    container_id = created.stdout.decode().strip()
    try:
        ex = DockerExecutor(image)
        with tempfile.TemporaryDirectory(prefix='sciel-collision-test-') as temp:
            root = Path(temp)
            for path in ['bundle', 'dependencies', 'runtime']:
                (root / path).mkdir()
            (root / 'request.json').write_text('{}')
            result = ex._container(root, {'limits': LIMITS}, container_name=name)
        assert not result['dispatched'] and result['removed']
        found = _cli(['inspect', '--format', '{{.Id}} {{.State.Status}}', container_id])
        assert found.returncode == 0
        assert found.stdout.decode().strip() == container_id + ' created'
    finally:
        removed = _cli(['rm', '--force', container_id])
        assert removed.returncode == 0
