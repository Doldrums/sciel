"""Detached, bounded per-run job service. Never restarts a marked invocation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
from pathlib import Path
import socket
import time

from ..engine.capacity import HostCapacity
from ..engine.runtime import process_identity
from ..methods.executor import ExecutionRefused
from ..methods.models import canonical_bytes, content_hash
from .library import ToolLibrary, data_executor_factory
from .jobs import (SCHEMA, TERMINAL, _root, _spec, _receipt, _no_links,
                   read_json, write_json, lock, now)


def _status(root, ident, **fields):
    path = root / ident / 'status.json'
    with lock(root / 'state.lock'):
        state = read_json(path)
        state.update(fields, updated_at=now())
        write_json(path, state)
    return state


def _host_receipt(spec, library, *, reason, dispatched, elapsed=0.):
    request, ident = spec['request'], spec['job_id']
    bundle = library.get(request['tool_id'])
    receipt = dict(schema='sciel-method-execution/1', method_id=bundle['id'],
        invocation_id=ident, plan_id=request['owner'].get('action_id', spec['request_hash']),
        reservation_id=ident, source_sha256=content_hash(bundle['files']),
        runtime_sha256=content_hash(library.identity['executor']), input_sha256=content_hash(request['inputs']),
        request_sha256=None, status='unknown' if dispatched else 'failed',
        reason=reason, dispatch_status='unknown' if dispatched else 'not_dispatched',
        output=None, output_sha256=None, stderr='', limits=request['limits'],
        dependency_ids=request['dependency_ids'],dependency_use_attested=False,
        runtime=library.identity['executor'], resources=dict(cpu_seconds=None if dispatched else 0.,
        wall_seconds=elapsed, memory_peak_bytes=None, output_bytes=None, stderr_bytes=None),
        resource_measurement=dict(cpu='unavailable' if dispatched else 'not_dispatched',
            wall='host_worker_observed_lower_bound', scope='tool execution only; service overhead excluded'))
    receipt['receipt_sha256'] = content_hash(receipt)
    return receipt


def _finish(root, spec, receipt):
    ident = spec['job_id']
    # The per-job receipt is written first so a crash before status publication
    # can be recovered without executing or scanning arbitrary other artifacts.
    write_json(root / ident / 'receipt.json', receipt, immutable=True)
    receipt = _receipt(spec, root)
    receipt_id = content_hash(receipt)
    write_json(root / 'receipts' / (receipt_id+'.json'), receipt, immutable=True)
    cancelled = (root / ident / 'cancel.request').exists()
    if receipt['status'] == 'ok':
        status = 'completed'
    elif receipt['status'] == 'unknown':
        status = 'unknown'
    elif cancelled and (receipt['dispatch_status'] == 'not_dispatched' or
                        str(receipt.get('reason','')).startswith('cancel')):
        status = 'cancelled'
    else:
        status = 'failed'
    state=read_json(root/ident/'status.json')
    if not (state.get('status')==status and state.get('receipt_id')==receipt_id
            and state.get('finished_at') is not None and state.get('resources')==receipt.get('resources')
            and state.get('dispatch_status')==receipt['dispatch_status']):
        state = _status(root, ident, status=status, phase='terminal', finished_at=now(),
            receipt_id=receipt_id, resources=receipt.get('resources'),
            dispatch_status=receipt['dispatch_status'], error=receipt.get('reason'))
    with lock(root/'state.lock'):
        blocked=read_json(root/'blocked.json')
        if (status=='unknown' and receipt['dispatch_status']!='not_dispatched'
                and receipt.get('container',{}).get('removed') is not True and ident not in blocked):
            blocked.append(ident)
            write_json(root/'blocked.json',blocked)
        # Publish the blocking marker BEFORE removing work from the durable queue.
        # A crash in between leaves a harmless queue entry, recovered next start.
        queue=read_json(root/'queue.json')
        if ident in queue:
            write_json(root/'queue.json',[value for value in queue if value!=ident])
    return state


def _cancelled(root, ident):
    path = root / ident / 'cancel.request'
    if path.exists() or path.is_symlink():
        spec = _spec(root, ident)
        if read_json(path) != {'job_id':ident, 'request_hash':spec['request_hash'], 'cancel_requested':True}:
            raise ValueError('job cancellation identity differs')
        return True
    return False


def run_job(run_dir, ident, library, capacity=None):
    """One immutable intent; callable with a fake library in isolated tests."""
    root = _root(run_dir)
    with lock(root / ident / 'worker.lock', blocking=False) as acquired:
        if not acquired:
            return
        spec = _spec(root, ident)
        request = spec['request']
        if request['library_identity'] != content_hash(library.identity):
            raise ValueError('job library runtime identity changed')
        if (root / ident / 'receipt.json').exists():
            _finish(root, spec, _receipt(spec, root))
            return
        marker = root / ident / 'dispatch.json'
        if marker.exists():
            _finish(root, spec, _host_receipt(spec, library,
                reason='worker_lost_after_dispatch_intent', dispatched=True))
            return
        _status(root, ident, status='running', started_at=now(), phase='waiting_for_capacity',
                dispatch_status='not_dispatched')
        cancel = lambda: _cancelled(root, ident)
        begun = time.monotonic()
        entered = False
        try:
            lease = (capacity.acquire('research_evaluator', cancel=cancel, timeout=None)
                     if capacity is not None else nullcontext())
            with lease:
                if cancel():
                    receipt = _host_receipt(spec, library, reason='cancelled_before_dispatch',
                        dispatched=False, elapsed=0.)
                else:
                    write_json(marker, {'schema':SCHEMA, 'job_id':ident,
                        'request_hash':spec['request_hash'], 'at':now()}, immutable=True)
                    entered = True
                    _status(root, ident, phase='executing', dispatch_status='unknown',
                        queue_wall_seconds=time.monotonic()-begun)
                    execution_begun = time.monotonic()
                    receipt = library.execute(library.get(request['tool_id']), request['inputs'],
                        limits=request['limits'], invocation_id=ident,
                        plan_id=request['owner'].get('action_id',spec['request_hash']),
                        reservation_id=ident, cancel_requested=cancel)
        except ExecutionRefused as error:
            receipt = _host_receipt(spec, library, reason='not_dispatched:'+str(error)[:1000],
                dispatched=False, elapsed=0.)
        except Exception as error:
            # Queue cancellation is known not to have dispatched. Any exception
            # after intent is conservative, even if it might be local transport.
            receipt = _host_receipt(spec, library,
                reason=('cancelled_before_dispatch' if not entered and cancel() else
                        'worker_error:'+type(error).__name__), dispatched=entered,
                elapsed=max(0.,time.monotonic()-execution_begun) if entered and 'execution_begun' in locals() else 0.)
        try:
            _receipt(spec, root, receipt)
        except (ValueError, TypeError, KeyError):
            receipt = _host_receipt(spec, library, reason='invalid_executor_receipt',
                dispatched=entered, elapsed=max(0.,time.monotonic()-begun))
        _finish(root, spec, receipt)


def run_service(run_dir, *, library_factory=None, stop_when_idle=False, poll_seconds=.2):
    """One service process, at most max_active executor threads/containers.

    The optional factory is a Python-only test hook, never loaded from a job or
    source artifact. No submitted code or config can select a host executor.
    """
    root = _root(run_dir)
    with lock(root / 'service.lock', blocking=False) as acquired:
        if not acquired:
            return
        identity = read_json(root / 'identity.json')
        lib = identity['library']
        if library_factory is None:
            library = ToolLibrary(run_dir, lib['runtime_lock'], lib['limits'],
                                  executor_factory=data_executor_factory(identity['config'].get('data_env'),
                                                                         identity['config'].get('task_root')))
        else:
            library = library_factory(run_dir, lib['runtime_lock'], lib['limits'])
        if library.identity != lib:
            raise ValueError('background service library identity changed')
        cfg = identity['config']
        capacity_cfg = cfg.get('capacity')
        capacity = None
        if capacity_cfg is not None:
            capacity = HostCapacity(capacity_cfg.get('capacity_dir'),
                llm_slots=capacity_cfg['host_llm_slots'],
                evaluator_slots=capacity_cfg['host_evaluator_slots'],
                research_slots=capacity_cfg.get('host_research_slots'), provider_limits={})
        process = {'schema':SCHEMA, 'pid':os.getpid(), 'hostname':socket.gethostname(),
            'process_identity':process_identity(os.getpid()), 'started_at':now(),
            'identity_sha256':content_hash(identity)}
        active = {}
        last_heartbeat, last_blocked = 0., None
        with ThreadPoolExecutor(max_workers=cfg['max_active'], thread_name_prefix='research-job') as pool:
            while True:
                for ident, future in list(active.items()):
                    if future.done():
                        # Failed bookkeeping remains a marked, non-retryable job.
                        try:
                            future.result()
                        except Exception:
                            pass
                        del active[ident]
                pending = []
                # This index is bounded by max_pending, irrespective of history.
                for ident in read_json(root/'queue.json'):
                    path=root/ident/'status.json'
                    spec=_spec(root,ident)
                    state = read_json(path)
                    if state['status'] in TERMINAL and ident not in active:
                        _finish(root,spec,_receipt(spec,root))
                        continue
                    if (state['status'] not in TERMINAL and ident not in active and
                            ((path.parent/'dispatch.json').exists() or (path.parent/'receipt.json').exists())):
                        run_job(run_dir,ident,library,capacity)
                        state=read_json(path)
                    if state['status'] not in TERMINAL and ident not in active:
                        pending.append((state['created_at'],ident))
                # Lost service leases cannot prove that an orphan container has
                # stopped. Hold new work until that unknown execution is reconciled.
                pending.sort()
                blocked=read_json(root/'blocked.json')
                eligible=[ident for _,ident in pending if not blocked or _cancelled(root,ident)]
                ready=eligible[:max(0,cfg['max_active']-len(active))]
                if time.monotonic()-last_heartbeat>=1 or blocked!=last_blocked:
                    write_json(root/'service.json',dict(process,heartbeat_at=now(),status='running',
                        blocked_by_unknown_jobs=blocked))
                    last_heartbeat,last_blocked=time.monotonic(),list(blocked)
                for ident in ready:
                    active[ident] = pool.submit(run_job, run_dir, ident, library, capacity)
                if stop_when_idle and not pending and not active:
                    write_json(root / 'service.json', dict(process, heartbeat_at=now(), status='stopped'))
                    return
                time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run_service(args.run_dir)


if __name__ == '__main__':
    main()
