"""External JSON method execution in fresh, immutable-image Docker containers.

No registered code is imported or executed on the host. There is deliberately no
subprocess-only fallback. Dependency code shares the top-level invocation's
container and budget; its inclusion is not a separately attested invocation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import signal
import subprocess
import tempfile
import time
import uuid

from .models import canonical_bytes, content_hash, validate_manifest, validate_value

SCHEMA = 'sciel-method-execution/1'
_IMAGE = re.compile(r'sha256:[0-9a-f]{64}\Z')
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_LIMITS = ('wall_seconds', 'cpu_seconds', 'memory_bytes', 'pids', 'max_output_bytes', 'max_stderr_bytes')
_BOUND_KEYS = ('method_id', 'invocation_id', 'plan_id', 'reservation_id', 'source_sha256',
               'runtime_sha256', 'input_sha256', 'request_sha256')
POLICY = dict(network='none', rootfs_read_only=True, bundle_mounts_read_only=True,
    candidate_uid=10001, candidate_gid=10001, cpus=1, credentials='no host environment passed',
    watchdog='root PID1 exits private PID namespace', swap='disabled',
    cpu_accounting='cgroup-v2 CPU observed before namespace exit; excludes Docker daemon',
    cpu_limit_scope='child phase cgroup delta; startup CPU is additionally measured',
    wall_accounting='host invocation total including staging/create/cleanup',
    dependency_invocations='not independently attested; top-level resource accounting only',
    tmpfs_bytes=33554432)


class ExecutionRefused(ValueError):
    """Proven refusal before entering Docker transport; no method was dispatched."""
    dispatch_status = 'not_dispatched'


def _environment():
    # Docker CLI needs only its local endpoint/config, never provider keys.
    return {k: v for k, v in os.environ.items() if k in
        ('PATH', 'HOME', 'DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_CONFIG', 'LANG', 'LC_ALL')}


def _cli(args, timeout=10):
    return subprocess.run(['docker', *args], stdin=subprocess.DEVNULL,
        capture_output=True, timeout=timeout, env=_environment())


def _remove_owned_container(name, container_id, owner):
    """Never remove a colliding name; resolve ambiguous creates by a fresh nonce."""
    deadline = time.monotonic() + 10
    try:
        if container_id is None:
            template = '{"id":{{json .Id}},"owner":{{json (index .Config.Labels "sciel.invocation-owner")}}}'
            found = _cli(['inspect', '--type=container', '--format', template, name], timeout=10)
            if found.returncode:
                return b'No such object' in found.stderr or b'No such container' in found.stderr
            identity = json.loads(found.stdout)
            if not isinstance(identity, dict) or not _hash(identity.get('id')):
                return False
            if identity.get('owner') != owner:
                # This invocation did not create the object using this name.
                return True
            container_id = identity['id']
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        removed = _cli(['rm', '--force', container_id], timeout=remaining)
        return removed.returncode == 0 or b'No such container' in removed.stderr
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return False


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _identifier(value):
    return isinstance(value, str) and 0 < len(value) <= 256 and all(ord(c) >= 32 for c in value)


def _hash(value):
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _validate_bundle(bundle):
    if not isinstance(bundle, dict) or set(bundle) != {'id', 'manifest', 'files'} or not _hash(bundle['id']):
        raise ValueError('invalid immutable method bundle')
    manifest = validate_manifest(bundle['manifest'], bundle['files'])
    files = deepcopy(bundle['files'])
    if content_hash(dict(schema='sciel-method/1', manifest=manifest, files=files)) != bundle['id']:
        raise ValueError('method bundle hash differs')
    if len(files) > 128 or sum(len(value.encode('utf-8')) for value in files.values()) > 16 * 1024 * 1024:
        raise ValueError('method bundle exceeds executor materialization limit')
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or ',' in name:
            raise ValueError('unsafe bundle path')
    return dict(id=bundle['id'], manifest=manifest, files=files)


def _limits(value, declared):
    if not isinstance(value, dict) or set(value) != set(_LIMITS):
        raise ValueError('executor requires all six resource limits')
    for key in _LIMITS:
        v = value[key]
        if not _finite(v) or v <= 0 or (key not in ('wall_seconds', 'cpu_seconds') and type(v) is not int):
            raise ValueError('invalid execution limit: ' + key)
        if v > declared[key]:
            raise ValueError('execution limit exceeds manifest resource contract: ' + key)
    if value['pids'] < 2 or value['pids'] > 256 or value['memory_bytes'] < 16 * 1024 * 1024:
        raise ValueError('unsupported process/memory allocation')
    if value['max_output_bytes'] > 16 * 1024 * 1024 or value['max_stderr_bytes'] > 256 * 1024:
        raise ValueError('executor JSON/diagnostic hard limit exceeded')
    if value['wall_seconds'] > 7200 or value['cpu_seconds'] > 3600:
        raise ValueError('executor invocation hard limit exceeded (one CPU hour/two wall hours)')
    return deepcopy(value)


def _attach(name, timeout, stdout_limit, stderr_limit, cancel_requested=None):
    process = subprocess.Popen(['docker', 'start', '--attach', name], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_environment(), start_new_session=True)
    selector = selectors.DefaultSelector()
    for pipe, kind in ((process.stdout, 'stdout'), (process.stderr, 'stderr')):
        os.set_blocking(pipe.fileno(), False)
        selector.register(pipe, selectors.EVENT_READ, kind)
    output, stderr = bytearray(), bytearray()
    begun, timed_out, overflow, cancelled = time.monotonic(), False, False, False
    try:
        while selector.get_map():
            if cancel_requested is not None and cancel_requested():
                cancelled = True
                break
            remaining = timeout - (time.monotonic() - begun)
            if remaining <= 0:
                timed_out = True
                break
            for key, _ in selector.select(min(.05, remaining)):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                elif key.data == 'stdout':
                    output.extend(chunk[:stdout_limit + 1 - len(output)])
                    if len(output) > stdout_limit:
                        overflow = True
                        break
                else:
                    stderr.extend(chunk)
                    del stderr[:-stderr_limit]
            if overflow:
                break
        if not timed_out and not overflow and not cancelled:
            try:
                process.wait(timeout=max(.001, timeout - (time.monotonic() - begun)))
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=2)
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return dict(stdout=bytes(output), stderr=stderr.decode('utf-8', 'replace'),
                exit_code=process.returncode, timed_out=timed_out, overflow=overflow,
                cancelled=cancelled)


class DockerExecutor:
    # Create/attach are inside the requested wall allowance. CLI teardown has
    # these additional bounds; materialization is size-bounded and charged too.
    overhead_wall_bound = 12

    def __init__(self, image_id):
        if not isinstance(image_id, str) or _IMAGE.fullmatch(image_id) is None:
            raise ValueError('an immutable sha256 Docker image ID is required')
        image = _cli(['image', 'inspect', '--format', '{{.Id}} {{.Os}} {{.Architecture}}', image_id])
        parts = image.stdout.decode('utf-8', 'replace').strip().split()
        if image.returncode or len(parts) != 3 or parts[0] != image_id or parts[1] != 'linux':
            raise RuntimeError('required immutable Linux Docker image is unavailable')
        version = _cli(['version', '--format', '{{.Server.Version}}'])
        if version.returncode or not version.stdout.strip():
            raise RuntimeError('Docker server is unavailable')
        self._runner = Path(__file__).with_name('_runner.py').read_bytes()
        self._loader = Path(__file__).with_name('_loader.py').read_bytes()
        self._identity = dict(schema='sciel-method-executor/1', image_id=image_id,
            architecture=parts[2], docker_server=version.stdout.decode().strip(),
            runner_sha256=hashlib.sha256(self._runner).hexdigest(),
            loader_sha256=hashlib.sha256(self._loader).hexdigest(), policy=deepcopy(POLICY))

    @property
    def identity(self):
        return deepcopy(self._identity)

    def _container(self, directory, request, cancel_requested=None, container_name=None):
        name = container_name or ('sciel-method-' + uuid.uuid4().hex)
        owner = uuid.uuid4().hex
        container_id, create_attempted = None, False
        limits = request['limits']
        args = ['create', '--pull=never', '--name', name, '--network=none', '--read-only', '--cpus=1',
            '--label=sciel.invocation-owner=' + owner,
            '--user=0:0', '--cgroupns=private',
            '--pids-limit=' + str(limits['pids']), '--memory=' + str(limits['memory_bytes']),
            '--memory-swap=' + str(limits['memory_bytes']), '--cap-drop=ALL', '--cap-add=SETUID', '--cap-add=SETGID',
            '--security-opt=no-new-privileges', '--log-driver=none', '--workdir=/tmp',
            '--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=33554432,mode=1777',
            '--entrypoint=/usr/local/bin/python']
        for local, target in [('bundle', '/bundle'), ('dependencies', '/dependencies'),
                              ('runtime', '/runtime'), ('request.json', '/request.json')]:
            args += ['--mount', f'type=bind,src={directory / local},dst={target},readonly']
        args += [self._identity['image_id'], '-I', '-B', '/runtime/_runner.py', '--supervise', '/request.json']
        result = dict(stdout=b'', stderr='', exit_code=None, timed_out=False, overflow=False,
                      dispatched=False, removed=False, name=name)
        begun = time.monotonic()
        try:
            if cancel_requested is not None and cancel_requested():
                result['reason'] = 'cancelled_before_dispatch'
                return result
            create_attempted = True
            created = _cli(args, timeout=min(10, limits['wall_seconds']))
            if created.returncode:
                result['stderr'] = created.stderr[-limits['max_stderr_bytes']:].decode('utf-8', 'replace')
                result['reason'] = 'container_create_failed'
            else:
                container_id = created.stdout.decode('ascii').strip()
                if not _hash(container_id):
                    container_id = None
                    result['reason'] = 'container_create_identity_invalid'
                    return result
                remaining = limits['wall_seconds'] - (time.monotonic() - begun)
                if cancel_requested is not None and cancel_requested():
                    result['reason'] = 'cancelled_before_dispatch'
                elif remaining <= 0:
                    result['reason'] = 'container_start_budget_exhausted'
                else:
                    # From here a lost attach can hide real execution; never call
                    # this a known failure or automatically replay the request.
                    result['dispatched'] = True
                    options = {'cancel_requested': cancel_requested} if cancel_requested is not None else {}
                    result.update(_attach(container_id, remaining,
                        limits['max_output_bytes'] + 6 * limits['max_stderr_bytes'] + 65536,
                        limits['max_stderr_bytes'], **options))
                    if result.get('cancelled'):
                        result['reason'] = 'cancel_requested_without_terminal_receipt'
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            result['reason'] = 'container_transport_' + type(exc).__name__
        finally:
            result['removed'] = (not create_attempted or
                                  _remove_owned_container(name, container_id, owner))
        return result

    def _prepare(self, bundle, inputs, limits, dependency_bundles):
        bundle = _validate_bundle(bundle)
        dependencies = [_validate_bundle(value) for value in dependency_bundles]
        if len(dependencies) > 64 or sum(len(text.encode('utf-8')) for value in [bundle, *dependencies]
                for text in value['files'].values()) > 32 * 1024 * 1024:
            raise ValueError('dependency materialization bound exceeded')
        by_id = {value['id']: value for value in dependencies}
        if len(by_id) != len(dependencies) or bundle['id'] in by_id:
            raise ValueError('duplicate dependency bundle identity')
        visiting, closure = set(), set()
        def visit(value):
            if value['id'] in visiting:
                raise ValueError('cyclic dependency closure')
            visiting.add(value['id'])
            for ident in value['manifest']['dependencies']:
                if ident not in by_id:
                    raise ValueError('missing pinned dependency bundle')
                if ident not in closure:
                    visit(by_id[ident])
                    closure.add(ident)
            visiting.remove(value['id'])
        visit(bundle)
        if closure != set(by_id):
            raise ValueError('unused dependency bundle outside exact closure')
        for value in [bundle, *dependencies]:
            if set(value['manifest']['runtime_lock']) != {'image_id'}:
                raise ValueError('unsupported runtime lock fields; executor supports exact image_id only')
            if value['manifest']['runtime_lock']['image_id'] != self._identity['image_id']:
                raise ValueError('method runtime lock differs from executor image')
        limits = _limits(limits, bundle['manifest']['resources'])
        validate_value(inputs, bundle['manifest']['input_schema'])
        if len(canonical_bytes(inputs)) > 16 * 1024 * 1024:
            raise ValueError('executor input hard limit exceeded')
        return bundle, dependencies, closure, limits

    def preflight(self, bundle, inputs, limits, dependency_bundles=()):
        """Validate and detach inputs without staging files or dispatching code."""
        try:
            return self._prepare(bundle, inputs, limits, dependency_bundles)
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            raise ExecutionRefused(str(exc)[:1000]) from exc

    def execute(self, bundle, inputs, *, invocation_id, plan_id, reservation_id, limits,
                dependency_bundles=(), cancel_requested=None, container_name=None):
        if container_name is not None and (type(container_name) is not str or
                re.fullmatch(r"sciel-eval-[0-9a-f]{64}", container_name) is None):
            raise ExecutionRefused("invalid owned evaluation container identity")
        begun = time.monotonic()
        for value in (invocation_id, plan_id, reservation_id):
            if not _identifier(value):
                raise ExecutionRefused('invocation, plan and reservation IDs are required')
        bundle, dependencies, closure, limits = self.preflight(bundle, inputs, limits, dependency_bundles)
        methods = {bundle['id']: dict(root='/bundle', entrypoint=bundle['manifest']['entrypoint'])}
        for value in dependencies:
            methods[value['id']] = dict(root='/dependencies/m_' + value['id'] + '/files',
                                       entrypoint=value['manifest']['entrypoint'])
        request = dict(schema='sciel-method-request/1', method_id=bundle['id'], invocation_id=invocation_id,
            plan_id=plan_id, reservation_id=reservation_id, input=deepcopy(inputs), input_sha256=content_hash(inputs),
            source_sha256=content_hash(bundle['files']), runtime_sha256=content_hash(self._identity),
            limits=limits, methods=methods, dependency_ids=sorted(closure))
        request['request_sha256'] = content_hash(request)
        entered_transport = False
        try:
            with tempfile.TemporaryDirectory(prefix='sciel-method-') as temp:
                directory = Path(temp)
                def write_files(root, files):
                    root.mkdir(parents=True, exist_ok=True)
                    for name, text in files.items():
                        target = root / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(text, encoding='utf-8')
                write_files(directory / 'bundle', bundle['files'])
                (directory / 'dependencies').mkdir()
                for value in dependencies:
                    root = directory / 'dependencies' / ('m_' + value['id'])
                    write_files(root / 'files', value['files'])
                    (root / '__init__.py').write_text('from sciel_method_runtime import invoke_dependency as _invoke\n'
                        'METHOD_ID = ' + repr(value['id']) + '\ndef invoke(value):\n    return _invoke(METHOD_ID, value)\n')
                (directory / 'runtime').mkdir()
                (directory / 'runtime' / '_runner.py').write_bytes(self._runner)
                (directory / 'runtime' / '_loader.py').write_bytes(self._loader)
                (directory / 'request.json').write_bytes(canonical_bytes(request))
                entered_transport = True
                options = {'cancel_requested': cancel_requested} if cancel_requested is not None else {}
                if container_name is not None:
                    options['container_name'] = container_name
                transport = self._container(directory, request, **options)
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            if not entered_transport:
                raise ExecutionRefused('local method materialization failed: ' + type(exc).__name__) from exc
            raise
        receipt = dict(schema=SCHEMA, **{key: request[key] for key in _BOUND_KEYS},
            status='unknown' if transport['dispatched'] else 'failed',
            reason=transport.get('reason') or 'missing_trusted_container_receipt',
            dispatch_status='unknown' if transport['dispatched'] else 'not_dispatched',
            output=None, output_sha256=None, stderr=transport.get('stderr', '')[:limits['max_stderr_bytes']],
            limits=limits, runtime=self.identity, dependency_ids=sorted(closure),
            dependency_use_attested=False, resources=dict(cpu_seconds=None if transport['dispatched'] else 0.0,
                wall_seconds=time.monotonic() - begun,
                memory_peak_bytes=None, output_bytes=None, stderr_bytes=None),
            container=dict(name=transport.get('name'), removed=transport['removed'], image_id=self._identity['image_id']))
        receipt['resource_measurement'] = dict(cpu='unavailable' if transport['dispatched'] else 'not_dispatched',
            wall='host_monotonic_total', scope=POLICY['cpu_accounting'])
        if transport['dispatched']:
            try:
                raw = json.loads(transport['stdout'], parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
                if (not isinstance(raw, dict) or raw.get('schema') != 'sciel-method-container-receipt/1'
                        or any(raw.get(key) != request[key] for key in _BOUND_KEYS)
                        or raw.get('limits') != limits or raw.get('invoked') is not True
                        or raw.get('status') not in ('ok', 'failed') or not isinstance(raw.get('stderr'), str)):
                    raise ValueError('unbound container receipt')
                resources = raw.get('resources', {})
                if not isinstance(resources, dict) or any(not _finite(resources.get(k)) or resources[k] < 0
                    for k in ('cpu_seconds', 'wall_seconds', 'output_bytes', 'stderr_bytes')):
                    raise ValueError('missing actual resource accounting')
                if raw['status'] == 'ok':
                    if raw.get('exit_code') != 0 or raw.get('reason') is not None or raw.get('output_sha256') != content_hash(raw.get('output')):
                        raise ValueError('invalid successful output binding')
                    if len(canonical_bytes(raw['output'])) > limits['max_output_bytes']:
                        raise ValueError('output limit exceeded')
                receipt.update(status=raw['status'], reason=raw.get('reason'), dispatch_status='invoked',
                    output=raw.get('output'), output_sha256=raw.get('output_sha256'),
                    stderr=raw['stderr'][-limits['max_stderr_bytes']:], stderr_truncated=raw.get('stderr_truncated') is True,
                    resources={**resources, 'wall_seconds': time.monotonic() - begun,
                               'method_wall_seconds': resources['wall_seconds']})
                receipt['resource_measurement']['cpu'] = 'trusted_cgroup_observation'
                if receipt['status'] == 'ok':
                    try:
                        validate_value(receipt['output'], bundle['manifest']['output_schema'])
                    except ValueError as exc:
                        receipt.update(status='failed', reason='output_schema_invalid',
                                       diagnostic=str(exc)[:1000])
            except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
                receipt['reason'] = ('cancel_requested_without_terminal_receipt' if transport.get('cancelled')
                                     else 'invalid_or_missing_trusted_container_receipt')
        if not transport['removed']:
            receipt.update(status='unknown', reason='container_cleanup_unconfirmed')
        receipt['receipt_sha256'] = content_hash(receipt)
        return receipt
