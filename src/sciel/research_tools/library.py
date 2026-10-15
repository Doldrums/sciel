"""Immutable, run-wide tool versions for the native island engine.

The controller owns visibility, trial/publication decisions, durable external
call caching and accounting. An object on disk is not automatically available
to an island. This module never imports submitted Python on the host and never
retries an execution. No laboratory or research-practice coordinator is created.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import stat
import tempfile
import threading

from ..methods.executor import DockerExecutor, ExecutionRefused, _limits
from ..methods.models import (canonical_bytes, content_hash, detached, hash_list,
    require_hash, validate_manifest, validate_runtime_lock, validate_value)
from .result_contract import RESULT_SCHEMA, validate_policy
from ..engine.research_targets import SCHEMA as TARGET_SCHEMA

SCHEMA = 'sciel-research-tool-library/3'
TOOL_SCHEMA = 'sciel-research-tool/3'
MAX_CODE_BYTES = 256 * 1024
MAX_OBJECT_BYTES = MAX_CODE_BYTES * 6 + 65536
MAX_DEPENDENCIES = 64
MAX_PARENTS = 16
MAX_CLOSURE_BYTES = 32 * 1024 * 1024
MAX_INPUT_BYTES = 16 * 1024 * 1024
_OBJECT = {'type': 'object', 'additionalProperties': True}


def _text(value, label, limit):
    if not isinstance(value, str) or not value.strip() or len(value.encode('utf-8')) > limit:
        raise ValueError(f'{label} must be nonempty UTF-8 text within {limit} bytes')
    return value


def _island(value):
    if type(value) is int and 0 <= value <= 2**63 - 1:
        return value
    if (isinstance(value, str) and value.strip() and len(value.encode('utf-8')) <= 128
            and all(ord(character) >= 32 for character in value)):
        return value
    raise ValueError('author_island must be a nonnegative integer or bounded nonempty identifier')


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate tool artifact JSON key')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite tool artifact JSON')))
    canonical_bytes(value)
    return value


def _entrypoint_error(reason, functions=()):
    names = [('async ' if isinstance(node, ast.AsyncFunctionDef) else '')
             + node.name[:64] for node in functions[:8]]
    found = ', '.join(names) or '(none)'
    if len(functions) > 8:
        found += f' (+{len(functions) - 8} omitted)'
    return ValueError('create.code: expected module-level synchronous def run(payload) '
        'accepting one positional JSON input; ' + reason + '. Found top-level functions: '
        + found + '. Candidate helpers do not replace the tool entrypoint. '
        'Define run(payload) to perform the research check and return a sciel-tool-result/1 '
        'object with exactly schema, status, observations, diagnostics, error. '
        'Use status=tool_error with a nonempty error if the check cannot be performed.')


def validate_tool_entrypoint(code):
    """Check the tool-specific calling convention without importing source.

    This is not proof that decorators, function bodies or module execution will
    work, nor that the returned value satisfies the runtime result contract.
    The generic manifest validator and isolated execution guards still apply.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, UnicodeError, RecursionError) as error:
        location = f' at line {error.lineno}' if isinstance(error, SyntaxError) else ''
        detail = str(getattr(error, 'msg', type(error).__name__))[:160]
        raise _entrypoint_error('invalid Python' + location + ': ' + detail) from error
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    entrypoints = [node for node in functions if node.name == 'run']
    if not entrypoints:
        raise _entrypoint_error('missing run', functions)
    entrypoint = entrypoints[-1]
    if isinstance(entrypoint, ast.AsyncFunctionDef):
        raise _entrypoint_error('run is async, but this runtime calls it synchronously', functions)
    args = entrypoint.args
    positional = len(args.posonlyargs) + len(args.args)
    if ((positional == 0 and args.vararg is None)
            or positional - len(args.defaults) > 1
            or any(default is None for default in args.kw_defaults)):
        raise _entrypoint_error('run cannot be called with one positional input', functions)


def data_executor_factory(data_env):
    """Executor factory mounting the directory named by data_env; None keeps the default."""
    if data_env is None:
        return None
    data_dir = os.environ.get(data_env)
    if not data_dir:
        raise ValueError(f'research_tools.data_env names an unset variable: {data_env}')
    return lambda image_id: DockerExecutor(image_id, data_dir=data_dir)


class ToolLibrary:
    """Source storage and one-shot isolated execution, without mutable selection.

    ``executor`` is an optional injected backend for tests. Otherwise the lazy
    factory receives the exact image ID; the default is ``DockerExecutor``.
    ``identity`` initializes that backend so a controller can freeze its identity
    before committing an external-call intent. Stored bundle existence alone
    grants no permission: the caller must enforce its checkpointed visible IDs.
    """
    def __init__(self, run_dir, runtime_lock, limits, *, executor=None, executor_factory=None):
        if executor is not None and executor_factory is not None:
            raise ValueError('provide executor or executor_factory, not both')
        validate_runtime_lock(runtime_lock)
        if set(runtime_lock) != {'image_id'}:
            raise ValueError('tool runtime_lock supports only an immutable image_id')
        self._runtime_lock = detached(runtime_lock)
        self._limits = _limits(limits, limits)
        self._configuration = dict(schema=SCHEMA, runtime_lock=self.runtime_lock, limits=self.limits)
        supplied = Path(run_dir).expanduser().absolute()
        if supplied.is_symlink():
            raise ValueError('tool run directory cannot be a symbolic link')
        # Resolve OS aliases such as macOS /var -> /private/var before defining
        # the trusted run root; no links are followed inside that root.
        supplied = supplied.parent.resolve() / supplied.name
        self._refuse_links(supplied)
        supplied.mkdir(parents=True, exist_ok=True)
        self.run_dir = supplied.resolve()
        self.path = self.run_dir / 'research_tools'
        self.objects = self.path / 'objects'
        for directory in (self.path, self.objects):
            self._refuse_links(directory)
            directory.mkdir(exist_ok=True)
            if not directory.is_dir():
                raise ValueError('tool storage must be a directory')
        self._write_immutable(self.path / 'identity.json', canonical_bytes(self._configuration))
        self._executor = executor
        self._factory = executor_factory or DockerExecutor
        self._executor_lock = threading.Lock()

    @property
    def runtime_lock(self):
        return detached(self._runtime_lock)

    @property
    def limits(self):
        return detached(self._limits)

    @staticmethod
    def _refuse_links(path):
        if any(parent.is_symlink() for parent in (path, *path.parents)):
            raise ValueError('tool storage refuses symbolic links')

    def _read(self, path):
        self._refuse_links(path)
        if not path.is_relative_to(self.path):
            raise ValueError('tool artifact is outside run storage')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_OBJECT_BYTES:
                raise ValueError('tool artifact is oversized or not a regular file')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                raw = stream.read(MAX_OBJECT_BYTES + 1)
            after = os.fstat(fd)
            if (len(raw) > MAX_OBJECT_BYTES or before.st_size != len(raw)
                    or (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                raise ValueError('tool artifact changed during read')
            return raw
        finally:
            os.close(fd)

    def _write_immutable(self, path, raw):
        self._refuse_links(path)
        if len(raw) > MAX_OBJECT_BYTES:
            raise ValueError('tool artifact exceeds byte limit')
        fd, temporary = tempfile.mkstemp(prefix='.tool-write-', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if self._read(path) != raw:
                    raise ValueError('immutable tool artifact already exists with different bytes')
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def _validate(self, body):
        if not isinstance(body, dict) or set(body) != {'schema', 'manifest', 'files'} or body['schema'] != 'sciel-method/1':
            raise ValueError('unsupported tool bundle envelope')
        files = body['files']
        if not isinstance(files, dict) or set(files) != {'tool.py'}:
            raise ValueError('a research tool must contain exactly tool.py')
        code = _text(files['tool.py'], 'tool source', MAX_CODE_BYTES)
        # Explain the actual tool API before a generic manifest error can hide
        # the expected function name behind "entrypoint function is not defined".
        validate_tool_entrypoint(code)
        manifest = validate_manifest(body['manifest'], files)
        protocol = manifest['protocol']
        if (manifest['kind'] != 'tool' or manifest['entrypoint'] != 'tool.py:run'
                or manifest['input_schema'] != _OBJECT or manifest['output_schema'] != _OBJECT
                or manifest['runtime_lock'] != self.runtime_lock or manifest['resources'] != self.limits
                or not isinstance(protocol, dict) or protocol.get('schema') != TOOL_SCHEMA):
            raise ValueError('tool manifest differs from the frozen library contract')
        _text(manifest['name'], 'tool name', 256)
        _text(protocol.get('description'), 'tool description', 4096)
        if (protocol.get('result_schema') != RESULT_SCHEMA
                or protocol.get('research_targets_schema') != TARGET_SCHEMA):
            raise ValueError('tool result schema differs from the frozen contract')
        validate_policy(protocol.get('replay_policy'))
        author = _island(protocol.get('author_island'))
        if manifest['author_lab'] != 'island:' + str(author):
            raise ValueError('tool author metadata is inconsistent')
        if len(manifest['parents']) > MAX_PARENTS or len(manifest['dependencies']) > MAX_DEPENDENCIES:
            raise ValueError('tool ancestry/dependency bound exceeded')
        return dict(manifest=manifest, files=detached(files))

    def register(self, name, description, code, author_island, parents=(), *, dependencies=(),
                 replay_policy='exact_observations'):
        author = _island(author_island)
        parent_ids = hash_list(list(parents), 'parents')
        dependency_ids = hash_list(list(dependencies), 'dependencies')
        manifest = dict(name=name, kind='tool', author_lab='island:' + str(author),
            entrypoint='tool.py:run', input_schema=detached(_OBJECT), output_schema=detached(_OBJECT),
            protocol=dict(schema=TOOL_SCHEMA, description=description, author_island=author,
                result_schema=RESULT_SCHEMA, research_targets_schema=TARGET_SCHEMA, replay_policy=validate_policy(replay_policy),
                source_inclusion='Pinned dependency source is not proof of invocation.'),
            runtime_lock=detached(self.runtime_lock), dependencies=dependency_ids, parents=parent_ids,
            resources=detached(self.limits), scope=description, examples=[])
        body = dict(schema='sciel-method/1', manifest=manifest, files={'tool.py':code})
        self._validate(body)
        for ident in parent_ids + dependency_ids:
            self.get(ident)
        self._closure(dependency_ids)
        ident = content_hash(body)
        self._write_immutable(self.objects / (ident + '.json'), canonical_bytes(body))
        return self.get(ident)

    def get(self, ident):
        require_hash(ident, 'tool ID')
        body = _decode(self._read(self.objects / (ident + '.json')))
        if content_hash(body) != ident:
            raise ValueError('tool source/manifest hash mismatch')
        return dict(id=ident, **self._validate(body))

    def _closure(self, identifiers):
        visited, visiting, result = set(), set(), []
        total_bytes = 0
        def visit(ident):
            nonlocal total_bytes
            if ident in visiting:
                raise ValueError('cyclic tool dependency closure')
            if ident in visited:
                return
            if len(visited) + len(visiting) >= MAX_DEPENDENCIES:
                raise ValueError('tool dependency closure exceeds version limit')
            visiting.add(ident)
            bundle = self.get(ident)
            total_bytes += sum(len(source.encode('utf-8')) for source in bundle['files'].values())
            if total_bytes > MAX_CLOSURE_BYTES:
                raise ValueError('tool dependency closure exceeds source byte limit')
            for dependency in bundle['manifest']['dependencies']:
                visit(dependency)
            visiting.remove(ident)
            visited.add(ident)
            result.append(bundle)
        for ident in identifiers:
            visit(ident)
        return result

    def dependency_closure(self, ident):
        """Return exact dependencies, dependency-first, excluding the root tool."""
        return self._closure(self.get(ident)['manifest']['dependencies'])

    def _backend(self):
        with self._executor_lock:
            if self._executor is None:
                self._executor = self._factory(self.runtime_lock['image_id'])
            identity = self._executor.identity
            if not isinstance(identity, dict) or identity.get('image_id') != self.runtime_lock['image_id']:
                raise ExecutionRefused('executor identity differs from frozen tool image')
            return self._executor

    @property
    def identity(self):
        return dict(detached(self._configuration), executor=detached(self._backend().identity))

    def execute(self, bundle, inputs, *, invocation_id, plan_id, reservation_id,
                limits=None, cancel_requested=None):
        """Execute once; the caller must durably cache this operation first.

        Known preflight refusals raise ``ExecutionRefused``. Failures after the
        backend is entered are never relabelled safe to retry. Actual/unknown
        resources and dependency-use attribution remain exactly as receipted.
        """
        try:
            if not isinstance(bundle, dict) or set(bundle) != {'id', 'manifest', 'files'}:
                raise ValueError('invalid tool bundle argument')
            saved = self.get(bundle['id'])
            if canonical_bytes(bundle) != canonical_bytes(saved):
                raise ValueError('supplied tool differs from immutable registered source')
            for value in (invocation_id, plan_id, reservation_id):
                if not isinstance(value, str) or not 0 < len(value) <= 256 or any(ord(character) < 32 for character in value):
                    raise ValueError('bounded invocation, plan and reservation IDs are required')
            validate_value(inputs, _OBJECT)
            if len(canonical_bytes(inputs)) > MAX_INPUT_BYTES:
                raise ValueError('tool input exceeds executor byte bound')
            inputs = detached(inputs)
            dependencies = self.dependency_closure(saved['id'])
            limits = _limits(self.limits if limits is None else limits, saved['manifest']['resources'])
        except (OSError, ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
            raise ExecutionRefused(str(exc)[:1000]) from exc
        backend = self._backend()
        # Docker's additional sandbox validation runs before transport. A test
        # backend may expose only execute; all public library contracts above
        # still apply and no candidate code executes during their validation.
        if callable(getattr(backend, 'preflight', None)):
            backend.preflight(saved, inputs, limits, dependencies)
        options = {'cancel_requested': cancel_requested} if cancel_requested is not None else {}
        receipt = backend.execute(saved, inputs, invocation_id=invocation_id, plan_id=plan_id,
            reservation_id=reservation_id, limits=detached(limits), dependency_bundles=dependencies, **options)
        if not isinstance(receipt, dict):
            raise ValueError('executor returned no receipt object')
        return detached(receipt)
