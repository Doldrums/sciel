"""Durable background tool intents. Disk existence does not grant island visibility.

Submit/cancel return immutable acknowledgements; only poll observes mutable state.
The detached service never repeats a dispatch marker without a terminal receipt.
Submitted Python is executed only by the existing isolated ToolLibrary backend.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time

from ..engine.runtime import process_matches
from ..methods.executor import ExecutionRefused, _limits
from ..methods.models import canonical_bytes, content_hash, detached, require_hash
from .library import _decode, MAX_INPUT_BYTES
from .result_contract import assess_receipt, parse_result

SCHEMA = 'sciel-research-tool-job/1'
TERMINAL = frozenset({'completed', 'failed', 'cancelled', 'unknown'})
MAX_BYTES = 20 * 1024 * 1024


def now():
    return datetime.now(timezone.utc).isoformat()


def _root(run_dir):
    supplied = Path(run_dir).expanduser().absolute()
    if supplied.is_symlink():
        raise ValueError('job run directory cannot be a symbolic link')
    supplied = supplied.parent.resolve() / supplied.name
    _no_links(supplied)
    return supplied / 'research_tools' / 'jobs'


def _no_links(path):
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('job storage refuses symbolic links')


def read_json(path):
    _no_links(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_BYTES:
            raise ValueError('job object must be a bounded regular file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(fd)
        if (len(raw) > MAX_BYTES or len(raw) != before.st_size or
                (before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError('job object changed during read')
        return _decode(raw)
    finally:
        os.close(fd)


def write_json(path, value, *, immutable=False):
    _no_links(path)
    raw = canonical_bytes(value)
    if len(raw) > MAX_BYTES:
        raise ValueError('job object exceeds byte bound')
    fd, name = tempfile.mkstemp(prefix='.job-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(name, path)
            except FileExistsError:
                if canonical_bytes(read_json(path)) != raw:
                    raise ValueError('immutable job object differs')
        else:
            os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


@contextmanager
def lock(path, *, blocking=True):
    _no_links(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('job lock is not a regular file')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _spec(root, job_id):
    require_hash(job_id, 'job ID')
    value = read_json(root / job_id / 'spec.json')
    if (value.get('schema') != SCHEMA or value.get('job_id') != job_id or
            value.get('request_hash') != content_hash(value.get('request')) or
            job_id != content_hash({'schema': SCHEMA, 'job_key': value.get('job_key')})):
        raise ValueError('job spec identity differs')
    return value


def _receipt(spec, root, value=None):
    value = read_json(root / spec['job_id'] / 'receipt.json') if value is None else value
    request = spec['request']
    if (value.get('method_id') != request['tool_id'] or
            value.get('invocation_id') != spec['job_id'] or
            value.get('plan_id') != request['owner'].get('action_id', spec['request_hash']) or
            value.get('reservation_id') != spec['job_id'] or
            value.get('input_sha256') != content_hash(request['inputs']) or
            value.get('source_sha256') != request['source_sha256'] or
            value.get('runtime_sha256') != request['runtime_sha256'] or
            value.get('dependency_ids') != request['dependency_ids'] or
            value.get('limits') != request['limits'] or
            value.get('status') not in ('ok', 'failed', 'unknown') or
            value.get('dispatch_status') not in ('invoked', 'not_dispatched', 'unknown')):
        raise ValueError('job receipt differs from its immutable request')
    if value.get('receipt_sha256') != content_hash({k:v for k,v in value.items() if k != 'receipt_sha256'}):
        raise ValueError('job receipt checksum differs')
    if value['status'] == 'ok' and (value.get('dispatch_status') != 'invoked' or
            value.get('output_sha256') != content_hash(value.get('output'))):
        raise ValueError('job successful output is not bound')
    resources = value.get('resources')
    if type(resources) is not dict:
        raise ValueError('job receipt lacks resource accounting')
    if value['status'] != 'unknown' and any(type(resources.get(key)) not in (int,float)
            or not math.isfinite(resources[key]) or resources[key] < 0 for key in ('cpu_seconds','wall_seconds')):
        raise ValueError('job terminal receipt lacks measured resources')
    return value


def read_job(run_dir, job_id, *, include_receipt=False):
    """Pure bounded dashboard read. Never starts a service, executes or reconciles."""
    root = _root(run_dir)
    spec = _spec(root, job_id)
    state = read_json(root / job_id / 'status.json')
    if (state.get('schema') != SCHEMA or state.get('job_id') != job_id or
            state.get('request_hash') != spec['request_hash'] or
            state.get('status') not in TERMINAL | {'queued', 'running'}):
        raise ValueError('job status differs from its immutable spec')
    request = spec['request']
    result = dict(state, tool_id=request['tool_id'], owner=detached(request['owner']),
        parent_job_id=request['parent_job_id'], requested_limits=detached(request['limits']))
    cancel = root / job_id / 'cancel.request'
    if cancel.exists() or cancel.is_symlink():
        ack = read_json(cancel)
        if ack != {'job_id':job_id, 'request_hash':spec['request_hash'], 'cancel_requested':True}:
            raise ValueError('job cancellation request differs')
        result['cancel_requested'] = True
    else:
        result['cancel_requested'] = False
    if state.get('receipt_id') is not None:
        receipt = _receipt(spec, root)
        if content_hash(receipt) != state['receipt_id']:
            raise ValueError('job status terminal receipt differs')
        if (state.get('resources') != receipt.get('resources') or
                state.get('dispatch_status') != receipt['dispatch_status'] or
                (state['status']=='completed') != (receipt['status']=='ok') or
                (state['status']=='unknown') != (receipt['status']=='unknown')):
            raise ValueError('job terminal status/accounting differs from receipt')
        if include_receipt:
            result['receipt'] = receipt
    if include_receipt:
        result['request'] = detached(spec['request'])
    return result


def _config(config):
    allowed = {'enabled','max_active','max_pending','max_cpu_seconds','max_wall_seconds', 'capacity'}
    if type(config) is not dict or set(config) - allowed:
        raise ValueError('unsupported background job configuration')
    result = dict(enabled=False, max_active=1, max_pending=16,
        max_cpu_seconds=3600., max_wall_seconds=7200.)
    result.update(config)
    if type(result['enabled']) is not bool:
        raise ValueError('jobs.enabled must be boolean')
    for key, high in [('max_active',64),('max_pending',10000)]:
        if type(result[key]) is not int or not 1 <= result[key] <= high:
            raise ValueError('invalid job '+key)
    for key, high in [('max_cpu_seconds',3600),('max_wall_seconds',7200)]:
        if type(result[key]) not in (int,float) or not math.isfinite(result[key]) or not 0 < result[key] <= high:
            raise ValueError('invalid job '+key)
    capacity=result.get('capacity')
    if capacity is not None:
        if (type(capacity) is not dict or set(capacity) - {'host_research_slots'} !=
                {'capacity_dir','host_llm_slots','host_evaluator_slots'} or
                capacity['capacity_dir'] is not None and type(capacity['capacity_dir']) is not str or
                any(type(capacity[key]) is not int or capacity[key]<1
                    for key in ('host_llm_slots','host_evaluator_slots'))):
            raise ValueError('invalid shared job capacity configuration')
        research_slots = capacity.get('host_research_slots')
        if research_slots is not None and (type(research_slots) is not int or
                not 1 <= research_slots < capacity['host_evaluator_slots']):
            raise ValueError('shared job research capacity must leave ordinary evaluator slots')
    return detached(result)


class BackgroundJobs:
    def __init__(self, run_dir, library, config, *, launch=None):
        self.run_dir = Path(run_dir).expanduser().absolute()
        self.library, self.config = library, _config(config)
        self.root = _root(run_dir)
        _no_links(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'receipts').mkdir(exist_ok=True)
        _no_links(self.root / 'receipts')
        self.identity = {'schema':SCHEMA, 'config':self.config, 'library':library.identity}
        write_json(self.root / 'identity.json', self.identity, immutable=True)
        with lock(self.root/'state.lock'):
            for name in ('queue.json','blocked.json'):
                if not (self.root/name).exists():
                    write_json(self.root/name,[])
        self._launch = launch or self._spawn

    def _spawn(self):
        # Service receives no model credentials and imports only trusted source.
        env = {k:v for k,v in os.environ.items() if k in
            ('PATH','HOME','DOCKER_HOST','DOCKER_CONTEXT','DOCKER_CONFIG','LANG','LC_ALL')}
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        return subprocess.Popen([sys.executable, '-m', 'sciel.research_tools.job_worker',
            '--run-dir', str(self.run_dir)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True, env=env)

    def ensure_service(self):
        with lock(self.root / 'launch.lock'):
            path = self.root / 'service.json'
            if path.exists() and process_matches(read_json(path)) is True:
                return
            launching = self.root / 'launch.json'
            if launching.exists():
                pending = read_json(launching)
                if process_matches(pending) is True:
                    return
            # A just-spawned service can be acquiring its lifetime lock. Waiting
            # is harmless; duplicate starters are excluded by service.lock.
            try:
                process = self._launch()
            except OSError as error:
                # Acceptance is durable even when the dispatcher cannot start.
                # A later poll can retry service startup, never the job invocation.
                write_json(path, {'schema':SCHEMA, 'status':'launch_failed',
                    'heartbeat_at':now(), 'identity_sha256':content_hash(self.identity),
                    'error':type(error).__name__})
                return
            if process is not None and type(getattr(process, 'pid', None)) is int:
                from ..engine.runtime import process_identity
                write_json(launching, {'pid':process.pid, 'hostname':socket.gethostname(),
                    'process_identity':process_identity(process.pid), 'at':now()})

    def submit(self, job_key, bundle, inputs, limits, owner, parent_job_id=None):
        if not self.config['enabled']:
            raise ExecutionRefused('background jobs are disabled')
        if type(job_key) is not str or not 0 < len(job_key) <= 256 or any(ord(c)<32 for c in job_key):
            raise ValueError('bounded job dispatch key required')
        saved = self.library.get(bundle['id'])
        if canonical_bytes(saved) != canonical_bytes(bundle):
            raise ValueError('submitted tool differs from immutable source')
        limits = _limits(limits, saved['manifest']['resources'])
        if limits['cpu_seconds'] > self.config['max_cpu_seconds'] or limits['wall_seconds'] > self.config['max_wall_seconds']:
            raise ExecutionRefused('requested job budget exceeds configured maximum')
        if type(inputs) is not dict or type(owner) is not dict or len(canonical_bytes(owner)) > 16384:
            raise ValueError('job inputs/owner must be bounded JSON objects')
        if len(canonical_bytes(inputs)) > MAX_INPUT_BYTES:
            raise ExecutionRefused('job input exceeds executor byte bound')
        if parent_job_id is not None:
            require_hash(parent_job_id, 'parent job ID')
            if assess_receipt(self.result(parent_job_id))['status'] != 'success':
                raise ValueError('continuation requires a successful prior result')
        job_id = content_hash({'schema':SCHEMA, 'job_key':job_key})
        request = {'tool_id':saved['id'], 'inputs':detached(inputs), 'limits':limits,
            'owner':detached(owner), 'parent_job_id':parent_job_id, 'library_identity':content_hash(self.library.identity),
            'source_sha256':content_hash(saved['files']),
            'runtime_sha256':content_hash(self.library.identity['executor']),
            'dependency_ids':sorted(value['id'] for value in self.library.dependency_closure(saved['id']))}
        spec = {'schema':SCHEMA, 'job_id':job_id, 'job_key':job_key,
            'request_hash':content_hash(request), 'request':request}
        ack = {'job_id':job_id, 'request_hash':spec['request_hash'], 'accepted':True}
        with lock(self.root / 'state.lock'):
            queue=read_json(self.root/'queue.json')
            directory = self.root / job_id
            _no_links(directory)
            if (directory / 'spec.json').exists():
                if _spec(self.root, job_id) != spec:
                    raise ValueError('job dispatch key reused for a different request')
                if (job_id not in queue and len(queue)>=self.config['max_pending'] and
                        (not (directory/'status.json').exists() or
                         read_json(directory/'status.json')['status'] not in TERMINAL)):
                    raise ExecutionRefused('background job pending capacity exhausted')
            else:
                if len(queue) >= self.config['max_pending']:
                    raise ExecutionRefused('background job pending capacity exhausted')
                directory.mkdir(exist_ok=True)
                write_json(directory / 'spec.json', spec, immutable=True)
            if not (directory / 'status.json').exists():
                stamp = now()
                write_json(directory / 'status.json', {'schema':SCHEMA, 'job_id':job_id,
                    'request_hash':spec['request_hash'], 'status':'queued', 'created_at':stamp,
                    'updated_at':stamp, 'started_at':None, 'finished_at':None, 'receipt_id':None,
                    'resources':None, 'dispatch_status':'not_dispatched', 'error':None})
            if (read_json(directory/'status.json')['status'] not in TERMINAL and job_id not in queue):
                queue.append(job_id)
                write_json(self.root/'queue.json',queue)
        self.ensure_service()
        return ack

    def poll(self, job_id):
        # Restart only the service. Its recovery never redispatches marked jobs.
        self.ensure_service()
        return read_job(self.run_dir, job_id)

    def cancel(self, job_id):
        spec = _spec(self.root, job_id)
        ack = {'job_id':job_id, 'request_hash':spec['request_hash'], 'cancel_requested':True}
        write_json(self.root / job_id / 'cancel.request', ack, immutable=True)
        self.ensure_service()
        return ack

    def result(self, job_id):
        spec = _spec(self.root, job_id)
        receipt = _receipt(spec, self.root)
        return detached(receipt)

    def spec(self, job_id):
        return detached(_spec(self.root, job_id))

    def continuation_inputs(self, job_id):
        spec, receipt = _spec(self.root, job_id), self.result(job_id)
        if assess_receipt(receipt)['status'] != 'success':
            raise ValueError('job has no successful continuation result')
        output = parse_result(receipt['output'])['observations']
        if 'continuation_state' not in output:
            raise ValueError('tool did not return explicit continuation_state')
        return {'bundle':self.library.get(spec['request']['tool_id']),
            'inputs':detached(spec['request']['inputs']),
            'continuation_state':detached(output['continuation_state']),
            'parent_job_id':job_id, 'receipt_id':content_hash(receipt)}
