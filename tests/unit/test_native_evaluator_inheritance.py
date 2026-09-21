"""Host launch-channel checks with synthetic env and inert Popen receipts only.

These do not prove hostile-code isolation: native candidates retain the host UID
and execute in their evaluator's interpreter. No candidate or child is executed.
"""
import json
from pathlib import Path

import pytest

from sciel.eval import sandbox
from sciel.eval.facade import EvalResult


@pytest.mark.parametrize('key', [
    'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'AWS_SECRET_ACCESS_KEY',
    'GOOGLE_APPLICATION_CREDENTIALS', 'SSH_AUTH_SOCK',
    'CODEX_AUTH_FILE', 'SCIEL_AUTH_FILE', 'HTTP_AUTHORIZATION',
    'SERVICE_AUTH', 'SERVICE_AUTHFILE', 'CI_JOB_JWT', 'CI_JOB_JWT_V2',
    'SERVICE_PASSWD', 'SERVICE_PRIVATE_KEY', 'SERVICE_PRIVATEKEY',
    'AWS_ACCESS_KEY_ID', 'KUBECONFIG', 'KRB5CCNAME', 'NETRC',
    'GIT_ASKPASS', 'SSH_ASKPASS',
    'PYTHONSTARTUP', 'PYTHONINSPECT', 'PYTHONBREAKPOINT',
    'LD_PRELOAD', 'LD_AUDIT', 'DYLD_INSERT_LIBRARIES',
    'codex_auth_file', 'mixed_Service_Authorization',
])
def test_known_credentials_and_automatic_hooks_are_not_inherited(monkeypatch, key):
    # Never inspect the process's real environment or any credential file.
    monkeypatch.setattr(sandbox.os, 'environ', {key: 'inert-fixture-value', 'LANG': 'C.UTF-8'})
    assert sandbox._evaluation_env() == {'LANG': 'C.UTF-8'}


def test_runtime_and_task_configuration_remain_compatible(monkeypatch):
    environment = {
        'PATH': '/fixture/bin', 'HOME': '/fixture/home', 'TMPDIR': '/fixture/tmp',
        'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'PYTHONHASHSEED': '91',
        'PYTHONPATH': '/fixture/declared-local-dependency',
        'OPENBLAS_NUM_THREADS': '3', 'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '5', 'VECLIB_MAXIMUM_THREADS': '6',
        'NUMEXPR_NUM_THREADS': '7', 'BLIS_NUM_THREADS': '8',
        # The trusted Oracle evaluator invokes the host Docker CLI. None of
        # these values is passed into its candidate container by that API.
        'DOCKER_HOST': 'unix:///fixture/docker.sock', 'DOCKER_CONTEXT': 'fixture',
        'DOCKER_CONFIG': '/fixture/docker-config',
        'SCIEL_ORACLE_IMAGE': 'sha256:' + 'a' * 64,
        'SCIEL_ORACLE_PRIVATE_DIR': '/fixture/private',
        'SCIEL_ORACLE_COMMITMENT_DIR': '/fixture/public',
        'UNCLASSIFIED_TASK_SETTING': 'fixture-value',
    }
    monkeypatch.setattr(sandbox.os, 'environ', environment)
    filtered = sandbox._evaluation_env()
    assert filtered == environment and filtered is not environment


@pytest.mark.parametrize('returncode', [0, 1])
def test_evaluator_detaches_host_input_and_unrelated_descriptors(monkeypatch, returncode):
    launches, cleaned = [], []

    class Process:
        pid = 999999999
        stderr = None

        def __init__(self):
            self.returncode = returncode

        def wait(self, timeout=None):
            return self.returncode

    class Tail:
        def __init__(self, stream):
            assert stream is None

        def finish(self):
            return ''

    def launch(argv, **options):
        launches.append((argv, options))
        Path(argv[5]).write_text(json.dumps(EvalResult(True, .25).to_json()))
        return Process()

    monkeypatch.setattr(sandbox.subprocess, 'Popen', launch)
    monkeypatch.setattr(sandbox, '_StderrTail', Tail)
    monkeypatch.setattr(sandbox, '_kill_group', lambda proc: cleaned.append(proc.pid))
    monkeypatch.setattr(sandbox.os, 'environ', {'LANG': 'C.UTF-8', 'CODEX_AUTH_FILE': '/inert/not-opened'})
    result = sandbox.run_stage('unread-evaluator.py', 'confirm', 'unread-candidate.py',
                              2., n=3, cpu_seconds=4, mem_bytes=1024, python='/fixture/python')
    assert len(launches) == 1 and cleaned
    argv, options = launches[0]
    assert argv[:5] == ['/fixture/python', sandbox._RUNNER, 'unread-evaluator.py',
                        'confirm', 'unread-candidate.py']
    assert argv[6:] == ['3', '--cpu-seconds', '4', '--mem-bytes', '1024']
    assert options['stdin'] == sandbox.subprocess.DEVNULL
    assert options['close_fds'] is True
    assert options['stdout'] == sandbox.subprocess.DEVNULL
    assert options['stderr'] == sandbox.subprocess.PIPE
    assert options['env'] == {'LANG': 'C.UTF-8'}
    assert options['start_new_session'] is (sandbox.os.name == 'posix')
    assert result.ok is (returncode == 0)
    assert not Path(argv[5]).exists()
