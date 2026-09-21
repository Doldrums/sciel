"""Independent replay, release, adoption and solution transfer."""
from __future__ import annotations

from .coordinator_contracts import Refusal
from .models import content_hash


def replay(coordinator, experiment_id, reviewer_lab, *, assignment_id):
    original = coordinator.registry.get_artifact('experiment', experiment_id)
    plan = coordinator.registry.get_artifact('experiment_plan', original['plan_id'])
    experiment = {key: plan[key] for key in ('question','treatment','control','inputs','criterion','scope')}
    replay_plan = coordinator.plan(reviewer_lab, assignment_id, experiment, purpose='replay',
        selected=plan['treatment']+plan['control'], source_plan=original['plan_id'],
        input_origin='independent_coordinator_replay_assignment', history_id=reviewer_lab)
    return coordinator.execute(replay_plan)


def freeze_holdouts(coordinator, method_id, assignments, *, treatment,control,scope,criterion):
    from .evidence import EvidenceAssessor
    with coordinator._locked():
        author=coordinator.registry.get_method(method_id)['manifest']['author_lab']
        request=dict(method_id=method_id,assignments=assignments,treatment=treatment,
                     control=control,scope=scope,criterion=criterion)
        return coordinator._host('holdouts:'+content_hash(request),author,'holdout_planning',request,
            lambda:EvidenceAssessor(coordinator.registry).freeze_holdouts(method_id,assignments,
                treatment=treatment,control=control,scope=scope,criterion=criterion,
                task=coordinator.task.identity,runtime=coordinator.executor.identity,resources=coordinator.resources))


def assess_benefit(coordinator, method_id, original_id, heldout_ids, *, protocol_id):
    from .evidence import EvidenceAssessor
    with coordinator._locked():
        author=coordinator.registry.get_method(method_id)['manifest']['author_lab']
        request=dict(method_id=method_id,original_id=original_id,heldout_ids=list(heldout_ids),protocol_id=protocol_id)
        return coordinator._host('benefit:'+content_hash(request),author,'independent_assessment',request,
            lambda:EvidenceAssessor(coordinator.registry).assess_benefit(method_id,original_id,
                list(heldout_ids),protocol_id=protocol_id))


def release(coordinator, method_id, original_id, replay_id):
    original = coordinator.registry.get_artifact('experiment', original_id)
    replay = coordinator.registry.get_artifact('experiment', replay_id)
    source_plan = coordinator.registry.get_artifact('experiment_plan', original['plan_id'])
    check_plan = coordinator.registry.get_artifact('experiment_plan', replay['plan_id'])
    author = coordinator.registry.get_method(method_id)['manifest']['author_lab']
    def publish():
        if (not original['contract_passed'] or not replay['contract_passed']
                or method_id not in original['invoked_methods'] or method_id not in replay['invoked_methods']
                or check_plan['purpose']!='replay' or check_plan['source_plan']!=original['plan_id']
                or source_plan['lab_id']!=author or check_plan['lab_id']==author
                or source_plan['history_id']==check_plan['history_id']):
            raise Refusal('release requires actual author use and an independent assigned replay')
        def observations(value):
            return {arm:[dict(input_sha256=r['input_sha256'],status=r['status'],measurement=r['measurement'])
                         for r in rows] for arm,rows in value['results'].items()}
        if observations(original) != observations(replay):
            raise Refusal('replay did not reproduce the frozen observation')
        evidence = coordinator.registry.put_artifact('method_evidence', dict(method_id=method_id,
            level='replayed', scope=source_plan['scope'], experiments=[original_id,replay_id],
            benefit='not established beyond assigned comparisons'))
        coordinator.registry.status_method(method_id, 'released', [evidence])
        return evidence
    return coordinator._host('release:'+content_hash([method_id,original_id,replay_id]), author,
        'publication', dict(method_id=method_id,original_id=original_id,replay_id=replay_id), publish)


def publish_unreplayed(coordinator, method_id, experiment_id):
    if coordinator.library_policy != 'unreplayed':
        raise Refusal('unreplayed publication is disabled in the verified-transfer arm')
    experiment = coordinator.registry.get_artifact('experiment', experiment_id)
    method = coordinator.registry.get_method(method_id)
    plan = coordinator.registry.get_artifact('experiment_plan', experiment['plan_id'])
    if not experiment['contract_passed'] or method_id not in experiment['invoked_methods'] or plan['lab_id']!=method['manifest']['author_lab']:
        raise Refusal('unreplayed control still requires the author contract trial')
    coordinator.registry.append('unreplayed_publication', dict(method_id=method_id,
        experiment_id=experiment_id, evidence_level='unreplayed'))


def adopt(coordinator, lab_id, method_id, trial_id, *, expected_revision, adapters=()):
    with coordinator._locked():
        trial = coordinator.registry.get_artifact('experiment', trial_id)
        plan = coordinator.registry.get_artifact('experiment_plan', trial['plan_id'])
        old_id, old = coordinator._revision(lab_id, expected_revision)
        request = dict(lab_id=lab_id, method_id=method_id, trial_id=trial_id,
                       expected_revision=old_id, adapters=list(adapters))
        def apply():
            if plan['lab_id']!=lab_id or plan['lab_revision_id']!=old_id or plan['purpose'] not in ('local','transfer','heldout'):
                raise Refusal('adoption requires a recipient trial pinned to this revision')
            components = [method_id]+list(adapters)
            for ident in components:
                if ident not in coordinator.common_methods:
                    if coordinator.registry.lab(lab_id)['mode']=='A':
                        raise Refusal('fixed toolbox cannot adopt persistent created methods')
                    if coordinator.registry.get_method(ident)['manifest']['author_lab']!=lab_id and (
                            coordinator.registry.lab(lab_id)['mode']!='C' or coordinator.transfer_policy!='methods'):
                        raise Refusal('independent verification does not authorize foreign method adoption')
            if not set(components)<=set(trial['invoked_methods']) or not trial['rule_passed']:
                raise Refusal('adoption rule failed or selected component was never invoked')
            if set(components).intersection(coordinator._bundle_closure(plan['control'])):
                raise Refusal('adoption control did not exclude the selected method')
            for ident in components:
                coordinator._allowed(lab_id, old, ident, purpose=plan['purpose'], selected=plan['selected'])
                if coordinator.registry.method_status(ident)['status']!='released' and not (coordinator.library_policy=='unreplayed' and
                        any(r['method_id']==ident for r in coordinator._events('unreplayed_publication'))):
                    raise Refusal('only released or explicitly unreplayed control versions can be adopted')
            for ident in adapters:
                if coordinator.registry.get_method(ident)['manifest']['kind']!='adapter':
                    raise Refusal('selected adaptation is not a versioned adapter')
            basis = coordinator.registry.put_artifact('adoption_basis', dict(**request,
                invocation_refs=trial['invocation_refs'], rule=plan['adoption_rule'],
                compatibility='runtime, resource bounds, sequential JSON schemas and declared units checked before dispatch; semantic preconditions remain contract trials',
                level='transfer_supported' if plan['purpose']=='transfer' and trial['mean_effect']>0 else 'limited_trial_adoption',
                scope=plan['scope']))
            changes = dict(active_methods=list(dict.fromkeys(old['active_methods']+[method_id])),
                active_adapters=list(dict.fromkeys(old['active_adapters']+list(adapters))),
                capabilities={**old['capabilities'],'accepted_pipeline':plan['treatment'],
                              'accepted_scope':plan['scope']},
                evidence_refs=list(dict.fromkeys(old['evidence_refs']+[trial_id,basis])))
            reason = 'adoption:'+basis
            current = coordinator.registry.lab(lab_id)['active_revision']
            if current!=old_id and coordinator.registry.revision(current)['reason']==reason:
                new_id = current  # recover after CAS before durable call receipt
            else:
                new_id = coordinator.registry.revise_lab(lab_id, old_id, changes, reason)
            return coordinator.registry.put_artifact('adoption', dict(**request, basis=basis,
                previous_revision=old_id, new_revision=new_id))
        return coordinator._host('adopt:'+content_hash(request), lab_id, 'adoption', request, apply)


def transfer_solution(coordinator, recipient, solution_id):
    """Explicit solution-only control; no method library or method credit."""
    with coordinator._locked():
        if coordinator.registry.lab(recipient)['mode']!='C' or coordinator.transfer_policy!='solution_only':
            raise Refusal('solution-only transfer control is not enabled')
        solution=coordinator.registry.get_artifact('solution',solution_id)
        if solution['lab_id']==recipient or not solution['evaluation']['ok']:
            raise Refusal('transfer requires a checked solution from another lab')
        request=dict(recipient=recipient,solution_id=solution_id)
        def transfer():
            experiment=coordinator.registry.get_artifact('experiment',solution['experiment_id'])
            plan=coordinator.registry.get_artifact('experiment_plan',experiment['plan_id'])
            call=coordinator.registry.get_artifact('call_receipt',solution['call_receipt_id'])
            execution=call['result']
            if (execution['status']!='ok' or execution['plan_id']!=experiment['plan_id']
                    or solution['call_receipt_id'] not in experiment['invocation_refs']
                    or len(plan['inputs'])!=1 or execution['input_sha256']!=content_hash(plan['inputs'][0])):
                raise Refusal('ready solution export lacks its bound actual output')
            payload=coordinator.solution_task.transfer_payload(plan['inputs'][0],execution['output'])
            receipt=coordinator.registry.put_artifact('solution_transfer',dict(**request,
                author_lab=solution['lab_id'],source_sha256=content_hash(solution['program']['code']),
                payload=payload,payload_sha256=content_hash(payload),
                origin_call_receipt=solution['call_receipt_id'],
                attribution='validated ready output/seed only; no source, method library or utility claim'))
            coordinator.registry.append('solution_transfer',dict(**request,receipt_id=receipt))
            return receipt
        return coordinator._host('solution_transfer:'+content_hash(request),recipient,'solution_transfer',request,transfer)
