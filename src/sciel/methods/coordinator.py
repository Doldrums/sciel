"""Host-owned research coordinator with explicit durable operation boundaries."""
from __future__ import annotations

import fcntl
import math
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from . import (
    coordinator_context,
    coordinator_episode,
    coordinator_exchange,
    coordinator_execution,
    coordinator_planning,
    coordinator_policy,
)
from .budget import BudgetExceeded, BudgetUnknown
from .coordinator_contracts import Refusal, StopRequested, TrialQuotaExceeded
from .engine_bridge import EngineCascadeBridge
from .models import canonical_bytes, content_hash, detached
from .proposals import PROCEDURE_ID
from .transport import DispatchStore
from ..engine.capacity import CapacityCancelled, CapacityConfigurationError, CapacityTimeout
from ..eval.registry import TaskSpec


class Coordinator:
    def __init__(self, registry, ledger, executor, proposer, task, *, resources,
                 common_methods=(), max_steps=20, max_cases=16, calendar_seconds=3600,
                 storage_bytes=100_000_000, library_policy='verified',
                 transfer_policy='methods', capacity=None, capacity_provider='openrouter.ai', capacity_wait_seconds=60,
                 should_stop=None):
        if library_policy not in ('verified', 'unreplayed') or transfer_policy not in ('methods', 'solution_only'):
            raise ValueError('unsupported library or transfer policy')
        if ((max_steps is not None and (type(max_steps) is not int or max_steps < 1))
                or type(max_cases) is not int or max_cases < 1):
            raise ValueError('positive integer episode/case limits required; episode limit may be null')
        if (calendar_seconds is not None and (type(calendar_seconds) not in (int,float)
                or not math.isfinite(calendar_seconds) or calendar_seconds <= 0)):
            raise ValueError('calendar limit must be null or finite and positive')
        if storage_bytes is not None and (type(storage_bytes) is not int or storage_bytes <= 0):
            raise ValueError('physical storage limit must be null or a positive integer')
        if capacity_wait_seconds is not None and (type(capacity_wait_seconds) not in (int,float)
                or not math.isfinite(capacity_wait_seconds) or not 0<capacity_wait_seconds<=300):
            raise ValueError('capacity wait must be null or finite, positive and at most 300 seconds')
        if should_stop is not None and not callable(should_stop):
            raise ValueError('should_stop must be callable or null')
        self.registry, self.ledger, self.executor = registry, ledger, executor
        self.proposer, self.task, self.resources = proposer, task, detached(resources)
        self.solution_task = (EngineCascadeBridge(task.task)
                              if isinstance(getattr(task, "task", None), TaskSpec) else task)
        self.common_methods = list(common_methods)
        self.max_steps, self.max_cases = max_steps, max_cases
        self.calendar_seconds, self.storage_bytes = calendar_seconds, storage_bytes
        self.library_policy, self.transfer_policy, self.capacity = library_policy, transfer_policy, capacity
        self.capacity_provider=capacity_provider
        self.capacity_wait_seconds=capacity_wait_seconds
        self.should_stop=should_stop
        self.dispatch = DispatchStore(registry.root/'dispatch', registry, ledger)
        self._thread_lock = threading.RLock()
        self._lock_depth = 0
        self.identity = dict(schema='sciel-method-coordinator/1', task=task.identity,
            solution_task=self.solution_task.identity,
            executor=executor.identity, generator=proposer.identity, procedure=PROCEDURE_ID,
            resources=self.resources, common_methods=self.common_methods, max_steps=max_steps,
            max_cases=max_cases, calendar_seconds=calendar_seconds, storage_bytes=storage_bytes,
            library_policy=library_policy, transfer_policy=transfer_policy,
            capacity_provider=capacity_provider,
            capacity_wait_seconds=capacity_wait_seconds,
            implementation=content_hash({p.name:p.read_text() for p in Path(__file__).parent.glob('*.py')}))
        # Initialization participates in the same transaction as later actions.
        # Registry.append alone cannot serialize the preceding absent-read.
        with self._locked():
            events = self._events('coordinator_identity')
            if any(event['identity'] != self.identity for event in events):
                raise ValueError('resume coordinator identity differs from frozen configuration')
            if not events:
                registry.append('coordinator_identity', dict(identity=self.identity, started_at=time.time()))
        for ident in self.common_methods:
            registry.get_method(ident)

    def _events(self, kind):
        return [event['data'] for event in self.registry.events(kind)]

    @contextmanager
    def _locked(self):
        # One coordinator transaction per registry, including across processes.
        # Registry and budget have their own shorter durable-write locks.
        with self._thread_lock:
            if self._lock_depth:
                yield
                return
            with (self.registry.root/'.coordinator.lock').open('a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _size(self):
        return sum(path.stat().st_size for path in self.registry.root.rglob('*') if path.is_file())

    def _headroom(self):
        started = self._events('coordinator_identity')[0]['started_at']
        if self.calendar_seconds is not None and time.time() - started >= self.calendar_seconds:
            raise BudgetExceeded('campaign calendar deadline exhausted')
        if self.storage_bytes is not None and self._size() >= self.storage_bytes:
            raise BudgetExceeded('campaign physical storage limit exhausted')

    def _check_stop(self):
        if self.should_stop is not None and self.should_stop():
            raise StopRequested('operator requested a stop before dispatch')

    def _capacity_lease(self, kind, *, lab_id, request_id, charged):
        """Wait without retrying external work; queue chunks have their own costs.

        An unbounded queue is not an unbounded execution reservation. Each wait
        attempt remains finite and durable; the enclosing call retains its one
        reservation/dispatch identity until the actual provider or executor call.
        """
        self._check_stop()
        if not self.capacity:
            return nullcontext()
        kwargs = dict(provider=self.capacity_provider) if kind == 'llm' else {}
        if self.should_stop is not None:
            kwargs['cancel'] = self.should_stop
        if self.capacity_wait_seconds is not None:
            try:
                return self.capacity.acquire(kind, timeout=self.capacity_wait_seconds, **kwargs)
            except CapacityCancelled:
                self._check_stop()
                raise
        attempt = 0
        account = self._account(lab_id)
        # Carry the previous receipt's final bookkeeping into the next chunk;
        # it must not accumulate against the eventual invocation's wall bound.
        window_wall, window_cpu = time.monotonic(), time.process_time()
        while True:
            self._check_stop()
            self._headroom()
            key = 'capacity-wait:' + content_hash(dict(request_id=request_id, kind=kind, attempt=attempt))
            reservation = self.ledger.reserve(account, key,
                dict(dollars=0,cpu_seconds=10,gpu_seconds=0,wall_seconds=32,storage_bytes=8192,steps=0),
                'capacity_wait')
            if not reservation['dispatch_allowed']:
                raise BudgetUnknown('capacity wait has a durable unresolved/reused reservation')
            wall, cpu = window_wall, window_cpu
            lease, error, status = None, None, 'acquired'
            try:
                lease = self.capacity.acquire(kind, timeout=30, **kwargs)
            except CapacityTimeout as exc:
                status, error = 'waiting', exc
            except (CapacityCancelled, CapacityConfigurationError) as exc:
                status, error = 'cancelled' if isinstance(exc,CapacityCancelled) else 'refused', exc
            except BaseException:
                partial=dict(cpu_seconds=round(time.process_time()-cpu,9),wall_seconds=round(time.monotonic()-wall,9))
                self.ledger.settle(reservation['id'],partial,status='unknown')
                for key in charged:
                    charged[key] += partial[key]
                raise
            receipt = dict(schema='sciel-method-capacity-wait/1',lab_id=lab_id,request_id=request_id,
                kind=kind,attempt=attempt,reservation_id=reservation['id'],status=status,
                dispatch_status='not_dispatched',failure_scope='infrastructure' if error else None,
                reason=type(error).__name__ if error else None)
            window_wall, window_cpu = time.monotonic(), time.process_time()
            actual = dict(dollars=0,gpu_seconds=0,steps=0,storage_bytes=len(canonical_bytes(receipt)),
                cpu_seconds=round(window_cpu-cpu,9),wall_seconds=round(window_wall-wall,9))
            try:
                self.ledger.settle(reservation['id'],actual,status='completed' if status in ('waiting','acquired') else 'failed')
                charged['cpu_seconds'] += actual['cpu_seconds']
                charged['wall_seconds'] += actual['wall_seconds']
                self.registry.put_artifact('capacity_wait_receipt',dict(receipt,actual=actual))
            except BaseException:
                if lease is not None:
                    lease.__exit__(None,None,None)
                raise
            if lease is not None:
                return lease
            if not isinstance(error,CapacityTimeout):
                self._check_stop()
                raise error
            attempt += 1

    def _account(self, lab_id):
        return self.registry.lab(lab_id)['budget_account']

    def _host(self, key, lab_id, category, request, function):
        """Charge preparation, retrieval, assessment and failed validation too."""
        self._headroom()
        bounds = dict(dollars=0, cpu_seconds=10, gpu_seconds=0, wall_seconds=30,
                      storage_bytes=2_000_000, steps=0)
        def work(_):
            wall, cpu, size = time.monotonic(), time.process_time(), self._size()
            try:
                result = dict(status='ok', value=function())
            except (Refusal, ValueError, KeyError) as exc:
                # Validation messages originate in trusted contracts, not code.
                result = dict(status='failed', reason=str(exc)[:1000], error_type=type(exc).__name__)
            actual = dict(dollars=0, gpu_seconds=0, steps=0,
                cpu_seconds=round(time.process_time()-cpu,9), wall_seconds=round(time.monotonic()-wall,9),
                storage_bytes=max(0, self._size()-size)+len(canonical_bytes(result)))
            return result, actual
        value = self.dispatch.run(key, self._account(lab_id), category, request, bounds, work)
        if value['status'] != 'ok':
            raise Refusal(value['reason'])
        return value['value']

    def create_lab(self, lab_id, agenda, account_id, limits, *, mode, parent_revision=None):
        with self._locked():
            self.ledger.create_account(account_id, limits)
            fields = dict(search_procedure=PROCEDURE_ID, generator=self.proposer.identity,
                runtime_lock={'image_id': self.executor.identity['image_id']},
                active_methods=self.common_methods, active_adapters=[], evidence_refs=[],
                capabilities=dict(network=False, executor='docker', resource_contract=self.resources,
                                  library_policy=self.library_policy, transfer_policy=self.transfer_policy))
            if parent_revision:
                parent = self.registry.revision(parent_revision)
                if self.registry.lab(parent['lab_id'])['mode'] != mode:
                    raise Refusal('fork cannot change the experimental mode')
                fields = {key: parent[key] for key in fields}
                fields['parent_revision'] = parent_revision
                # A fork has its own account and cannot spend its parent's
                # direction grant merely by copying revision capabilities.
                fields['capabilities'] = {key:value for key,value in fields['capabilities'].items()
                                          if key != 'active_funding_lineage'}
            return self.registry.create_lab(lab_id, agenda, account_id, mode, fields)

    def _revision(self, lab_id, revision_id=None):
        lab = self.registry.lab(lab_id)
        revision_id = revision_id or lab['active_revision']
        revision = self.registry.revision(revision_id)
        if revision['lab_id'] != lab_id:
            raise Refusal('revision belongs to another lab')
        if (revision['search_procedure']!=PROCEDURE_ID or revision['generator']!=self.proposer.identity
                or revision['runtime_lock']!={'image_id':self.executor.identity['image_id']}):
            raise Refusal('revision procedure/generator/runtime is unsupported by this frozen coordinator')
        return revision_id, revision

    def _bundle_closure(self, roots):
        found = {}
        def visit(ident):
            if ident in found:
                return
            value = self.registry.get_method(ident)
            found[ident] = value
            for child in value['manifest']['dependencies']:
                visit(child)
        for ident in roots:
            visit(ident)
        return found

    def _allowed(self, lab_id, revision, ident, *, purpose, selected):
        method = self.registry.get_method(ident)
        status = self.registry.method_status(ident)['status']
        if status in ('quarantined', 'superseded'):
            raise Refusal('method is quarantined or superseded')
        if ident in revision['active_methods'] + revision['active_adapters']:
            return
        if method['manifest']['author_lab'] == lab_id:
            if self.registry.lab(lab_id)['mode'] == 'A' and ident not in self.common_methods:
                # Ephemeral solution execution bundles are intentionally separate.
                protocol = method['manifest']['protocol']
                if not isinstance(protocol, dict) or protocol.get('permanent_toolbox_method', True):
                    raise Refusal('A cannot extend a permanent method toolbox')
            return
        if purpose == 'replay' and ident in selected:
            return  # explicitly assigned by the trusted replay coordinator
        if (purpose in ('transfer', 'heldout') and ident in selected
                and self.registry.lab(lab_id)['mode'] == 'C' and self.transfer_policy == 'methods'):
            released = status == 'released'
            unreplayed = any(row['method_id'] == ident for row in self._events('unreplayed_publication'))
            if released or (self.library_policy == 'unreplayed' and unreplayed):
                return
        raise Refusal('method is outside this revision and authorized trial')

    def _funding(self):
        return coordinator_policy._funding(self)

    def _save_funding(self, policy):
        return coordinator_policy._save_funding(self, policy)

    def lineage_for_version(self, method_id, *, policy=None):
        return coordinator_policy.lineage_for_version(self, method_id, policy=policy)

    def _active_funding_lineage(self, lab_id, revision=None):
        return coordinator_policy._active_funding_lineage(self, lab_id, revision)

    def _plan_funding_lineage(self, lab_id, revision, treatment):
        return coordinator_policy._plan_funding_lineage(self, lab_id, revision, treatment)

    def grant_direction(self, lab_id, question, evidence_refs):
        return coordinator_policy.grant_direction(self, lab_id, question, evidence_refs)

    def register_draft(self, lab_id, proposal_id, files, manifest):
        return coordinator_planning.register_draft(self, lab_id, proposal_id, files, manifest)

    def plan(self, lab_id, proposal_id, experiment, *, revision_id=None, purpose='local',
             selected=(), source_plan=None, input_origin='author_proposed', history_id=None,
             heldout_protocol_id=None, heldout_assignment_id=None):
        return coordinator_planning.plan(self, lab_id, proposal_id, experiment, revision_id=revision_id, purpose=purpose, selected=selected, source_plan=source_plan, input_origin=input_origin, history_id=history_id, heldout_protocol_id=heldout_protocol_id, heldout_assignment_id=heldout_assignment_id)

    def _invoke(self, plan_id, plan, arm, case, stage, method_id, value):
        return coordinator_execution._invoke(self, plan_id, plan, arm, case, stage, method_id, value)

    def _measurement_adapter(self, plan, arm):
        return coordinator_execution._measurement_adapter(self, plan, arm)

    def execute(self, plan_id):
        return coordinator_execution.execute(self, plan_id)

    def _finish_trial_block(self, lab_id, lineage=None):
        return coordinator_policy._finish_trial_block(self, lab_id, lineage)

    def lab_state(self, lab_id):
        return coordinator_policy.lab_state(self, lab_id)

    def archive_lab(self, lab_id, *, reason):
        return coordinator_policy.archive_lab(self, lab_id, reason=reason)

    def renew_trial(self, lab_id, quota, *, question, evidence_refs, lineage_id=None):
        return coordinator_policy.renew_trial(self, lab_id, quota, question=question, evidence_refs=evidence_refs, lineage_id=lineage_id)

    def reopen_episode(self, lab_id, *, question, evidence_refs):
        return coordinator_policy.reopen_episode(self, lab_id, question=question, evidence_refs=evidence_refs)

    def replay(self, experiment_id, reviewer_lab, *, assignment_id):
        return coordinator_exchange.replay(self, experiment_id, reviewer_lab, assignment_id=assignment_id)

    def freeze_holdouts(self, method_id, assignments, *, treatment,control,scope,criterion):
        return coordinator_exchange.freeze_holdouts(self, method_id, assignments, treatment=treatment, control=control, scope=scope, criterion=criterion)

    def assess_benefit(self, method_id, original_id, heldout_ids, *, protocol_id):
        return coordinator_exchange.assess_benefit(self, method_id, original_id, heldout_ids, protocol_id=protocol_id)

    def release(self, method_id, original_id, replay_id):
        return coordinator_exchange.release(self, method_id, original_id, replay_id)

    def publish_unreplayed(self, method_id, experiment_id):
        return coordinator_exchange.publish_unreplayed(self, method_id, experiment_id)

    def adopt(self, lab_id, method_id, trial_id, *, expected_revision, adapters=()):
        return coordinator_exchange.adopt(self, lab_id, method_id, trial_id, expected_revision=expected_revision, adapters=adapters)

    def _context(self, lab_id, revision_id, revision):
        return coordinator_context._context(self, lab_id, revision_id, revision)

    def transfer_solution(self, recipient, solution_id):
        return coordinator_exchange.transfer_solution(self, recipient, solution_id)

    def step(self, lab_id):
        return coordinator_episode.step(self, lab_id)

    def run(self, lab_id):
        outcomes=[]
        while True:
            result=self.step(lab_id)
            outcomes.append(result)
            if result['status']=='stopped':
                return outcomes

    def checkpoint(self, path):
        with self._locked():
            self.registry.put_artifact('budget_checkpoint',self.ledger.snapshot())
            return self.registry.checkpoint(path)
