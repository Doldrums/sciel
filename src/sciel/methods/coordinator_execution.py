"""Durable external invocation, measurement and paired experiment assessment."""
from __future__ import annotations

import time

from .budget import BudgetExceeded
from .coordinator_contracts import Refusal, StopRequested, TrialQuotaExceeded
from .models import canonical_bytes, content_hash, detached
from ..engine.capacity import CapacityCancelled, CapacityConfigurationError, CapacityTimeout


def _invoke(coordinator, plan_id, plan, arm, case, stage, method_id, value):
    coordinator._headroom()
    bundle = coordinator.registry.get_method(method_id)
    deps = coordinator._bundle_closure(bundle['manifest']['dependencies'])
    limits = bundle['manifest']['resources']
    invocation_id = content_hash(dict(plan=plan_id, arm=arm, case=case, stage=stage))
    bounds = dict(dollars=0, cpu_seconds=limits['cpu_seconds']+10, gpu_seconds=0,
        wall_seconds=limits['wall_seconds']+coordinator.executor.overhead_wall_bound+((coordinator.capacity_wait_seconds or 0) if coordinator.capacity else 0),
        storage_bytes=limits['max_output_bytes']+limits['max_stderr_bytes']+100_000, steps=0)
    def invoke(reservation):
        cpu, wall = time.process_time(), time.monotonic()
        queued = dict(cpu_seconds=0.,wall_seconds=0.)
        dispatched = False
        from .executor import ExecutionRefused
        try:
            if any(coordinator.registry.method_status(i)['status'] in ('quarantined','superseded')
                   for i in [method_id]+list(deps)):
                raise ExecutionRefused('method denied after planning')
            with coordinator._capacity_lease('evaluator',lab_id=plan['lab_id'],request_id='invoke:'+invocation_id,charged=queued):
                coordinator._check_stop()
                dispatched = True
                receipt = coordinator.executor.execute(bundle, value, invocation_id=invocation_id,
                    plan_id=plan_id, reservation_id=reservation, limits=limits,
                    dependency_bundles=list(deps.values()))
        except (ExecutionRefused, CapacityCancelled, CapacityConfigurationError, CapacityTimeout, StopRequested, BudgetExceeded) as exc:
            reason=str(exc)[:1000] if isinstance(exc,ExecutionRefused) else type(exc).__name__
            receipt = dict(status='failed',reason=reason,dispatch_status='not_dispatched',
                failure_scope='operator' if isinstance(exc,StopRequested) else 'budget' if isinstance(exc,BudgetExceeded)
                    else 'contract' if isinstance(exc,ExecutionRefused) else 'infrastructure',
                method_id=method_id,invocation_id=invocation_id,plan_id=plan_id,
                reservation_id=reservation,source_sha256=content_hash(bundle['files']),
                input_sha256=content_hash(value),output=None,output_sha256=None,
                runtime_sha256=content_hash(coordinator.executor.identity),
                resources=dict(cpu_seconds=0,wall_seconds=0),stderr='')
        except BaseException as exc:
            # Preserve the known queue charges if another part of admission
            # or execution fails. DispatchStore's generic fallback otherwise
            # measures the whole callback and would charge queued time twice.
            receipt=dict(status='unknown',reason=type(exc).__name__,
                dispatch_status='unknown' if dispatched else 'not_dispatched',
                failure_scope='execution_outcome_unknown' if dispatched else 'infrastructure',
                method_id=method_id,invocation_id=invocation_id,plan_id=plan_id,
                reservation_id=reservation,source_sha256=content_hash(bundle['files']),
                input_sha256=content_hash(value),output=None,output_sha256=None,
                runtime_sha256=content_hash(coordinator.executor.identity),
                resources=dict(cpu_seconds=None if dispatched else 0,wall_seconds=None),stderr='')
        actual = dict(dollars=0, gpu_seconds=0, steps=0,
            wall_seconds=round(max(0,time.monotonic()-wall-queued['wall_seconds']),9),
            storage_bytes=len(canonical_bytes(receipt)))
        measured = receipt['resources'].get('cpu_seconds')
        if measured is not None:
            actual['cpu_seconds'] = round(measured + max(0,time.process_time()-cpu-queued['cpu_seconds']),9)
        elif receipt['status']=='unknown':
            actual['cpu_seconds'] = round(max(0,time.process_time()-cpu-queued['cpu_seconds']),9)
        return receipt, actual
    request = dict(plan_id=plan_id, invocation_id=invocation_id, method_id=method_id,
                   input=value, limits=limits, runtime=plan['runtime'])
    result = coordinator.dispatch.run('invoke:'+invocation_id, coordinator._account(plan['lab_id']),
                               'replication' if plan['purpose']=='replay' else 'execution', request, bounds, invoke)
    # A cached receipt keeps this identity; it never creates another observation.
    if result.get('reason') == 'StopRequested' and result.get('dispatch_status') == 'not_dispatched':
        coordinator._check_stop()
    return result


def _measurement_adapter(coordinator, plan, arm):
    identity = plan.get('measurement_protocols', {}).get(arm)
    for adapter in (coordinator.task, coordinator.solution_task):
        if identity == adapter.identity:
            return adapter
    raise Refusal('measurement protocol differs from the frozen task/cascade')


def execute(coordinator, plan_id):
    with coordinator._locked():
        plan = coordinator.registry.get_artifact('experiment_plan', plan_id)
        if plan['task'] != coordinator.task.identity or plan['runtime'] != coordinator.executor.identity:
            raise Refusal('executor or evaluator differs from the frozen plan')
        adapters = {arm: coordinator._measurement_adapter(plan, arm) for arm in ('control', 'treatment')}
        existing = [row for row in coordinator._events('experiment_finished') if row['plan_id'] == plan_id]
        if existing:
            coordinator._finish_trial_block(plan['lab_id'],plan.get('funding_lineage','lab:'+plan['lab_id']))
            return existing[-1]['experiment_id']
        if coordinator.lab_state(plan['lab_id'])['status']=='archived':
            raise Refusal('archived laboratory cannot dispatch new work')
        lineage = plan.get('funding_lineage','lab:'+plan['lab_id'])
        local_trials = [row for row in coordinator._events('trial_reserved') if row['lab_id']==plan['lab_id']
                        and row.get('lineage_id','lab:'+row['lab_id'])==lineage]
        if plan['purpose'] not in ('replay', 'heldout'):
            policy = coordinator._funding()
            policy.create_lineage(lineage)
            quota = policy.snapshot(lineage)['next_quota']
            # Quotas are packages of experiments, not reset by code versions.
            reserved = any(row['plan_id']==plan_id for row in local_trials)
            if not reserved:
                since_block = len(local_trials) - sum(row.get('trials', 0) for row in coordinator._events('trial_block')
                    if row['lab_id']==plan['lab_id'] and row.get('lineage_id','lab:'+row['lab_id'])==lineage)
                if quota == 0 or since_block >= quota:
                    raise TrialQuotaExceeded('trial quota exhausted for this method lineage; choose another family or request an explicit evidence-bound renewal',
                        record=dict(lab_id=plan['lab_id'],lineage_id=lineage,plan_id=plan_id))
                coordinator.registry.append('trial_reserved', dict(lab_id=plan['lab_id'],plan_id=plan_id,lineage_id=lineage))
                coordinator._save_funding(policy)
        results, invocations = {}, []
        for arm in ('control', 'treatment'):
            adapter = adapters[arm]
            results[arm] = []
            for case, input_value in enumerate(plan['inputs']):
                output, status = detached(input_value), 'ok'
                for stage, ident in enumerate(plan[arm]):
                    receipt = coordinator._invoke(plan_id, plan, arm, case, stage, ident, output)
                    invocations.append(receipt)
                    if receipt['status'] != 'ok':
                        status = receipt['status']
                        break
                    output = receipt['output']
                measurement = None
                if status == 'ok':
                    measurement = coordinator._host('measure:'+content_hash(dict(plan=plan_id, arm=arm, case=case)),
                        plan['lab_id'], 'measurement', dict(input=input_value, output=output, task=adapter.identity),
                        lambda i=input_value, o=output, a=adapter: a.measure(i, o))
                results[arm].append(dict(input_sha256=content_hash(input_value),
                    output_sha256=content_hash(output) if status=='ok' else None,
                    status=status, measurement=measurement))
        def finish():
            ok = all(item['status']=='ok' for values in results.values() for item in values)
            effects = []
            if ok:
                sign = 1 if coordinator.task.direction=='max' else -1
                effects = [sign*(t['measurement']['metrics'][coordinator.task.metric]-c['measurement']['metrics'][coordinator.task.metric])
                           for t,c in zip(results['treatment'], results['control'])]
            effect = sum(effects)/len(effects) if effects else None
            valid = ok and all(item['measurement']['valid'] for item in results['treatment'])
            body = dict(schema='sciel-method-experiment/1', plan_id=plan_id,
                status='completed' if ok else 'failed', results=results,
                invocation_refs=[r['call_receipt_id'] for r in invocations],
                invoked_methods=sorted({r['method_id'] for r in invocations if r.get('dispatch_status')=='invoked'}),
                diagnostics=[dict(method_id=r['method_id'], status=r['status'], reason=r.get('reason'),
                    dispatch_status=r.get('dispatch_status'), failure_scope=r.get('failure_scope') or
                        ('candidate_execution' if r.get('dispatch_status')=='invoked' else 'infrastructure'),
                    stderr=r.get('stderr','')[-2000:]) for r in invocations if r['status']!='ok'],
                contract_passed=ok, valid_treatment=valid, paired_effects=effects, mean_effect=effect,
                rule_passed=bool(valid and effect is not None and effect>=plan['criterion']['minimum_effect']),
                evidence_level='local_effect_observed' if valid and effect is not None and effect>0 else 'contract_passed' if ok else 'failed',
                uncertainty='Only these assigned inputs; deterministic replay does not establish generalization.')
            ident = coordinator.registry.put_artifact('experiment', body)
            claim_key=content_hash(dict(methods=plan['treatment'],scope=plan['scope'],task=plan['task']))
            previous=[r for r in coordinator._events('claim_revision') if r['key']==claim_key]
            claim=coordinator.registry.put_artifact('knowledge_claim',dict(schema='sciel-method-claim/1',
                previous_revision=previous[-1]['id'] if previous else None,methods=plan['treatment'],
                scope=plan['scope'],task=plan['task'],supporting=[ident] if body['rule_passed'] else [],
                contradicting=[ident] if ok and not body['rule_passed'] else [],
                inconclusive=[ident] if not ok else [],evidence_level=body['evidence_level'],
                uncertainty=body['uncertainty'],open_conditions=['held-out replication','recipient utility']))
            coordinator.registry.append('claim_revision',dict(key=claim_key,id=claim))
            coordinator.registry.append('experiment_finished', dict(plan_id=plan_id, experiment_id=ident))
            return ident
        result=coordinator._host('assess:'+plan_id, plan['lab_id'], 'assessment', dict(plan_id=plan_id), finish)
        coordinator._finish_trial_block(plan['lab_id'],lineage)
        return result
