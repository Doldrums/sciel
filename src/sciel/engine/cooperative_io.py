"""Cooperative wave scheduling and deterministic completion publication."""
from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    Future,
    TimeoutError as FutureTimeout,
    wait,
    FIRST_COMPLETED,
)
import hashlib
import json
import threading
from ..campaign.budget import RequestNotDispatched
from .parallel_control import (
    ParallelAborted,
    ParallelPause,
    UnknownParallelCall,
    _paused_error,
    dispatch_evidence,
)


class CooperativeIO:
    """Stackful suspension keeps all scientific mutations on the caller thread.

    Scoped Engine calls resume ready jobs with a durable, authenticated order.
    Unscoped direct callers retain the legacy logical-round API; their job lists
    historically may change between invocations against the same call ledger.
    """
    def __init__(self, ledger, workers, on_receipt=None, on_state=None, stop_requested=None,
                 schedule_identity=None):
        self.ledger = ledger
        self.schedule_identity = schedule_identity
        self.run_index = 0
        self.workers = workers
        self.on_receipt = on_receipt
        self.on_state = on_state
        if stop_requested is not None and not callable(stop_requested):
            raise TypeError('stop_requested must be callable or None')
        self.stop_requested = stop_requested
        self.stop_event = threading.Event()
        self.abort_event = threading.Event()
        self.cancel_event = threading.Event()
        self.paused = False
        self.current = None
        self.sequences = {}
        self._has_tool_calls = False
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='sciel-io')

    @property
    def aborted(self):
        return self.abort_event.is_set()

    def _abort(self):
        self.abort_event.set()
        self.cancel_event.set()

    def _note_failure(self, receipt):
        if ((isinstance(receipt, BaseException) and not _paused_error(receipt))
                or (isinstance(receipt, dict) and receipt.get('status') == 'error'
                    and receipt.get('abandonment') is None)):
            self._abort()

    def close(self):
        self.executor.shutdown(wait=True)

    def call(self, kind, request, function):
        from greenlet import getcurrent
        if self.current is None:
            raise RuntimeError('Parallel I/O requires a coordinator job')
        if kind == 'tool_execution':
            self._has_tool_calls = True
        key = self.current
        ordinal = self.sequences.get(key, 0)
        self.sequences[key] = ordinal + 1
        operation = {'key': f'{key}:{ordinal}', 'kind': kind, 'request': request, 'function': function}
        payload = getcurrent().parent.switch(operation)
        if isinstance(payload, BaseException):
            raise payload
        return self.ledger.result(kind, payload)

    def run(self, jobs):
        try:
            if self.schedule_identity is not None:
                index, self.run_index = self.run_index, self.run_index + 1
                return self._run_ready(jobs, index)
            return self._run(jobs)
        finally:
            self.current = None
            self._publish_pending({})

    def _publish_pending(self, operations):
        """Optional telemetry counts submitted work, including host-queued I/O."""
        if self.on_state is None:
            return
        try:
            fields = dict(pending_calls=len(operations),
                pending_llm_calls=sum(op['kind'] == 'llm' for op in operations.values()),
                pending_evaluations=sum(op['kind'] == 'evaluation' for op in operations.values()))
            if self._has_tool_calls:
                fields['pending_tool_calls'] = sum(op['kind'] == 'tool_execution' for op in operations.values())
            self.on_state(fields)
        except BaseException:
            # Status is best effort and never replaces a scientific receipt.
            pass

    def _check_stop(self):
        if self.stop_event.is_set() or (self.stop_requested is not None and self.stop_requested()):
            self.stop_event.set()
            self.cancel_event.set()
            self.paused = True
        return self.paused

    def _run(self, jobs):
        from greenlet import greenlet
        self._check_stop()
        fibers = [(key, greenlet(function)) for key, function in jobs]
        results, errors, pending = {}, {}, {}
        for key, fiber in fibers:
            self.current = key
            try:
                value = fiber.switch()
                if fiber.dead:
                    results[key] = value
                else:
                    pending[key] = value
            except BaseException as error:
                errors[key] = error
                self._note_failure(error)
        while pending:
            futures, receipts, prepared = {}, {}, {}
            # Check every immutable receipt before submitting any new work.
            # A cached fatal/unknown call late in dispatch order must prevent an
            # earlier uncached sibling from making another paid request.
            for key, operation in pending.items():
                try:
                    self._check_stop()
                    directory, request_hash, receipt = self.ledger.prepare(
                        operation['key'], operation['kind'], operation['request'])
                    if receipt is not None:
                        receipts[key] = receipt
                    else:
                        prepared[key] = (directory, request_hash, operation)
                except BaseException as error:
                    receipts[key] = error
            for receipt in receipts.values():
                self._note_failure(receipt)
            for key, (directory, request_hash, operation) in prepared.items():
                try:
                    self._check_stop()
                    if self.aborted:
                        receipts[key] = ParallelAborted('Parallel sibling failed before submission')
                    elif self.paused:
                        receipts[key] = ParallelPause('Parallel call paused before submission')
                    else:
                        # submit() can enqueue work and then fail while creating
                        # a thread. A gate proves that no leaf call starts until
                        # submission has returned its future successfully.
                        permission = Future()
                        def submitted(directory=directory, request_hash=request_hash,
                                      operation=operation, permission=permission):
                            if not permission.result():
                                raise RequestNotDispatched('Parallel call submission failed')
                            try:
                                return self.ledger.perform(directory, request_hash,
                                    operation['kind'], operation['function'],
                                    cancel=self.cancel_event, abort=self.abort_event)
                            except BaseException as error:
                                self._note_failure(error)
                                raise
                        try:
                            future = self.executor.submit(submitted)
                        except BaseException:
                            permission.set_result(False)
                            receipts[key] = RequestNotDispatched('Cannot submit parallel call')
                            self._abort()
                        else:
                            futures[key] = future
                            permission.set_result(True)
                except BaseException as error:
                    receipts[key] = error
                    self._note_failure(error)
            # Drain all submitted work, even after a sibling fails. Futures only
            # execute bounded leaf I/O; they never wait for coordinator callbacks.
            self._publish_pending({key: pending[key] for key in futures})
            for key, future in futures.items():
                while True:
                    self._check_stop()
                    try:
                        receipts[key] = future.result(timeout=.1)
                        break
                    except FutureTimeout:
                        if not future.done():
                            continue
                        # A completed future may itself have raised TimeoutError;
                        # it is not a reason to wait forever or lose its receipt.
                        try:
                            receipts[key] = future.result()
                        except BaseException as error:
                            receipts[key] = error
                        break
                    except BaseException as error:
                        receipts[key] = error
                        break
                self._check_stop()
                self._note_failure(receipts[key])
            previous, pending = pending, {}
            for key, fiber in fibers:
                if key not in previous:
                    continue
                self.current = key
                receipt = receipts[key]
                if self.on_receipt is not None and not isinstance(receipt, BaseException):
                    try:
                        self.on_receipt(previous[key], receipt)
                    except BaseException as error:
                        # A transcript/telemetry failure must not replace a paid
                        # receipt, or prevent siblings from booking their cost.
                        errors['coordinator:receipt:' + previous[key]['key']] = error
                        self._note_failure(error)
                try:
                    value = fiber.switch(receipt)
                    if fiber.dead:
                        results[key] = value
                    else:
                        pending[key] = value
                except BaseException as error:
                    errors[key] = error
                    self._note_failure(error)
        return results, errors


    def _cached_frontier_failures(self):
        """Authenticate all existing physical calls before extending paid work.

        Cached failure knowledge must not reorder historical coordinator turns.
        Return it to apply only after the exact saved prefix has been replayed.
        """
        failures = []
        for directory in sorted(self.ledger.directory.glob('*')):
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError('Parallel call history is not a regular directory')
            path = directory / 'intent.json'
            if not path.exists():
                if (directory / 'receipt.json').exists() or (directory / 'dispatched.json').exists():
                    raise ValueError('Parallel call history is missing its intent')
                continue  # Interrupted preparation before dispatch is unpaid.
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError('Parallel call history is a symlink')
            intent = json.loads(path.read_bytes())
            if path.parent.name != hashlib.sha256(intent['key'].encode()).hexdigest():
                raise ValueError('Parallel call history path differs from its key')
            marker = directory / 'dispatched.json'
            if marker.exists() or (directory / 'receipt.json').exists():
                from .parallel_schedule import ResumeOrder
                ResumeOrder.verify_call(self.ledger.directory.parent, directory, intent)
            if marker.exists():
                dispatch_evidence(directory, intent['request_sha256'])
            if (directory / 'receipt.json').is_symlink():
                raise ValueError('Parallel receipt is a symlink')
            try:
                _, _, receipt = self.ledger.prepare(intent['key'], intent['kind'], intent['request'])
            except UnknownParallelCall as error:
                failures.append(error)
            else:
                if receipt is not None and receipt.get('status') == 'error' and not receipt.get('abandonment'):
                    failures.append(receipt)
        return failures

    def validate_resume_complete(self):
        if self.schedule_identity is None:
            return
        root = self.ledger.directory.parent
        expected = {f'phase-{index:06d}.json' for index in range(self.run_index)}
        actual = {p.name for p in (root / 'resume-order-phases').glob('phase-*.json')}
        if actual != expected:
            raise ValueError('Resume-order has unvisited or missing phase history')
        expected_runs = {f'run-{index:06d}' for index in range(self.run_index)}
        if {p.name for p in (root / 'resume-order').glob('run-*')} != expected_runs:
            raise ValueError('Resume-order has unvisited or missing run history')

    def _run_ready(self, jobs, run_index):
        try:
            return self._run_ready_inner(list(jobs), run_index)
        except BaseException:
            # A schedule-authentication/write failure cannot safely resume a
            # fiber. Cancel undispatched work and seal every already paid leaf
            # before returning; the wave rolls back and receipts remain proof.
            self._abort()
            self.executor.shutdown(wait=True)
            raise

    def _run_ready_inner(self, jobs, run_index):
        from greenlet import greenlet
        from .parallel_schedule import ResumeOrder
        cached_failures = self._cached_frontier_failures()
        order = ResumeOrder(self.ledger.directory.parent, self.schedule_identity,
                            run_index, [key for key, _ in jobs])
        self._check_stop()
        fibers = {key: greenlet(function) for key, function in jobs}
        rank = {key: index for index, (key, _) in enumerate(jobs)}
        results, errors, pending, futures = {}, {}, {}, {}

        def advance(key, receipt=None, *, initial=False):
            operation = None if initial else pending.pop(key)
            self.current = key
            if not initial and not isinstance(receipt, BaseException):
                # This durable turn must precede scientific mutations or the
                # next request. Physical receipts themselves stay immutable.
                order.consume(key, operation, receipt)
                if self.on_receipt is not None:
                    try:
                        self.on_receipt(operation, receipt)
                    except BaseException as error:
                        errors['coordinator:receipt:' + operation['key']] = error
                        self._note_failure(error)
            try:
                value = fibers[key].switch() if initial else fibers[key].switch(receipt)
                if fibers[key].dead:
                    results[key] = value
                else:
                    pending[key] = value
            except BaseException as error:
                errors[key] = error
                self._note_failure(error)

        for key in fibers:
            advance(key, initial=True)
        # Reconstruct the exact historical interleaving before looking at newly
        # ready receipts. No new leaf is dispatched while replaying this prefix.
        saved_turns = len(order.turns)
        while order.cursor < saved_turns:
            turn = order.turns[order.cursor]
            key = turn['job_id']
            if key not in pending:
                raise ValueError('Resume-order names a job that is not suspended')
            operation = pending[key]
            _, _, receipt = self.ledger.prepare(operation['key'], operation['kind'], operation['request'])
            if receipt is None:
                raise ValueError('Resume-order physical receipt is missing')
            advance(key, receipt)
        for failure in cached_failures:
            self._note_failure(failure)
        # A cached terminal failure not reached by a caught/revised job still
        # forbids extending the paid frontier until explicit recovery resolves it.
        if cached_failures:
            errors['coordinator:cached_failure'] = ParallelAborted('Cached fatal or unresolved physical call')

        def submit(key, directory, request_hash, operation):
            permission = Future()
            def submitted():
                if not permission.result():
                    raise RequestNotDispatched('Parallel call submission failed')
                try:
                    return self.ledger.perform(directory, request_hash, operation['kind'],
                        operation['function'], cancel=self.cancel_event, abort=self.abort_event)
                except BaseException as error:
                    self._note_failure(error)
                    raise
            try:
                future = self.executor.submit(submitted)
            except BaseException:
                permission.set_result(False)
                self._abort()
                return RequestNotDispatched('Cannot submit parallel call')
            futures[key] = future
            permission.set_result(True)
            return None

        while pending:
            self._check_stop()
            ready, prepared = {}, {}
            # Collect every currently done operation, not the first inserted
            # future. A done future can itself carry TimeoutError; never spin.
            for key, future in list(futures.items()):
                if future.done():
                    try:
                        ready[key] = future.result()
                    except BaseException as error:
                        ready[key] = error
                    del futures[key]
            for key, operation in list(pending.items()):
                if key in futures or key in ready:
                    continue
                try:
                    directory, request_hash, receipt = self.ledger.prepare(
                        operation['key'], operation['kind'], operation['request'])
                    if receipt is not None:
                        ready[key] = receipt
                    else:
                        order.bind_call(directory, operation['key'], request_hash)
                        prepared[key] = (directory, request_hash, operation)
                except BaseException as error:
                    ready[key] = error
            for receipt in ready.values():
                self._note_failure(receipt)
            if ready:
                # Consume the whole known ready batch before paying for any
                # derived request, so a processing/callback failure cancels it.
                for key in sorted(ready, key=rank.get):
                    advance(key, ready[key])
                self._publish_pending({key: pending[key] for key in futures})
                continue
            for key, (directory, request_hash, operation) in prepared.items():
                self._check_stop()
                if self.aborted:
                    ready[key] = ParallelAborted('Parallel sibling failed before submission')
                elif self.paused:
                    ready[key] = ParallelPause('Parallel call paused before submission')
                else:
                    error = submit(key, directory, request_hash, operation)
                    if error is not None:
                        ready[key] = error
            for key in sorted(ready, key=rank.get):
                advance(key, ready[key])
            self._publish_pending({key: pending[key] for key in futures})
            if futures:
                wait(tuple(futures.values()), timeout=.1, return_when=FIRST_COMPLETED)
        return results, errors
