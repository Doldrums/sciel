"""Bounded proposal context from methods, experiments and saved solutions."""
from __future__ import annotations

from .engine_bridge import EngineCascadeBridge
from .models import canonical_bytes, content_hash, detached


def _context(coordinator, lab_id, revision_id, revision):
    own = [event['data']['id'] for event in coordinator.registry.events() if event['kind']=='method_registered'
           and coordinator.registry.get_method(event['data']['id'])['manifest']['author_lab']==lab_id]
    methods = list(dict.fromkeys(revision['active_methods']+revision['active_adapters']+own))
    # Compact immutable cards, with explicit source fetches bound in receipt.
    cards, remaining = [], 160_000
    for ident in reversed(methods[-16:]):
        bundle = coordinator.registry.get_method(ident)
        protocol=bundle['manifest']['protocol']
        if isinstance(protocol,dict) and protocol.get('permanent_toolbox_method') is False:
            continue
        card=dict(id=ident,manifest=bundle['manifest'],files=bundle['files'],source_included=True,
                  status=coordinator.registry.method_status(ident))
        if len(canonical_bytes(card))>remaining:
            card=dict(id=ident,name=bundle['manifest']['name'][:256],entrypoint=bundle['manifest']['entrypoint'],
                status=coordinator.registry.method_status(ident),source_included=False,
                omission='immutable full source is retained; this context contains a bounded card')
        remaining-=len(canonical_bytes(card))
        cards.append(card)
    cards.reverse()
    evidence = []
    for ident in revision['evidence_refs'][-8:]:
        for kind in ('experiment','proposal_receipt','adoption_basis','funding_grant'):
            try:
                body=coordinator.registry.get_artifact(kind,ident)
                if kind=='experiment':
                    body={key:body[key] for key in ('plan_id','status','contract_passed','mean_effect',
                          'valid_treatment','rule_passed','evidence_level','uncertainty','diagnostics','invocation_refs')}
                elif kind=='proposal_receipt':
                    body={key:body[key] for key in ('proposal_id','error','diagnostic','novelty','call_receipt_id',
                          'failure_scope','funding_lineage','plan_id') if key in body}
                if len(canonical_bytes(body))>8000:
                    body=dict(summary=canonical_bytes(body).decode()[:7000],truncated=True)
                evidence.append(dict(id=ident,kind=kind,body=body))
                break
            except KeyError:
                continue
    budget=coordinator.ledger.snapshot(coordinator._account(lab_id))
    budget={key:budget[key] for key in ('limits','spent','reserved','remaining','halted','unknown_reservations')}
    library=[]
    if coordinator.registry.lab(lab_id)['mode']=='C' and coordinator.transfer_policy=='methods':
        for event in coordinator.registry.events():
            if event['kind']!='method_registered':
                continue
            ident=event['data']['id']
            bundle=coordinator.registry.get_method(ident)
            status=coordinator.registry.method_status(ident)
            unreplayed=any(r['method_id']==ident for r in coordinator._events('unreplayed_publication'))
            if (bundle['manifest']['author_lab']!=lab_id and ident not in methods
                    and (status['status']=='released' or (coordinator.library_policy=='unreplayed' and unreplayed))):
                manifest=bundle['manifest']
                card=dict(id=ident,name=manifest['name'],scope=manifest['scope'],
                    input_schema=manifest['input_schema'],output_schema=manifest['output_schema'],
                    evidence_refs=status['evidence_refs'],evidence_label='replayed' if status['status']=='released' else 'unreplayed')
                if len(canonical_bytes(card))>8000:
                    card=dict(id=ident,name=manifest['name'][:256],
                        summary=canonical_bytes(card).decode()[:7000],truncated=True)
                library.append(card)
    solutions=[]
    imported={row['solution_id']:row['receipt_id'] for row in coordinator._events('solution_transfer') if row['recipient']==lab_id}
    for event in coordinator.registry.events():
        if event['kind']=='artifact_registered' and event['data']['kind']=='solution':
            ident=event['data']['id']
            value=coordinator.registry.get_artifact('solution',ident)
            if value['lab_id']==lab_id:
                code=value['program']['code']
                solutions.append(dict(id=ident,author_lab=value['lab_id'],score=value['program']['score'],
                    code=code if len(code.encode())<=32_000 else None,
                    source_sha256=content_hash(code),source_omitted=len(code.encode())>32_000,
                    transfer_kind='local_solution'))
            elif ident in imported:
                transferred=coordinator.registry.get_artifact('solution_transfer',imported[ident])
                solutions.append(dict(id=ident,author_lab=value['lab_id'],score=value['program']['score'],
                    payload=transferred['payload'],payload_sha256=transferred['payload_sha256'],
                    source_sha256=transferred['source_sha256'],transfer_kind='solution_only'))
    delivery_revision=detached(revision)
    omitted={}
    for key,limit in (('evidence_refs',8),('active_methods',16),('active_adapters',8)):
        values=delivery_revision[key]
        omitted[key]=max(0,len(values)-limit)
        delivery_revision[key]=values[-limit:]
    return dict(lab_id=lab_id, revision_id=revision_id, revision=delivery_revision,
        revision_delivery=dict(immutable_revision_id=revision_id,omitted_counts=omitted,
            scope='Bounded prompt view only; the immutable lab revision and history remain complete.'),
        agenda=coordinator.registry.lab(lab_id)['agenda'], mode=coordinator.registry.lab(lab_id)['mode'],
        task=coordinator.task.identity, available_methods=cards, evidence=evidence,
        solution_input_schema=coordinator.solution_task.input_schema,
        solution_entrypoint=('run_packing() -> (centers, radii, reported_sum)'
                             if isinstance(coordinator.solution_task,EngineCascadeBridge) else 'solve(values) -> sorted values'),
        canonical_initial_solution=getattr(coordinator.task,'canonical_seed_source',None),
        available_library=library[-8:],previous_solutions=solutions[-4:],
        resources=coordinator.resources, budget=budget,
        protocol_contract=dict(max_cases=coordinator.max_cases, max_pipeline_components=8,
            repeated_pipeline_components=False, adjacent_schemas='exact_equality',
            criterion=dict(metric=coordinator.task.metric, direction=coordinator.task.direction,
                           minimum_effect_lower_bound=0),
            solution=dict(treatment=['$new'], case_count=1,
                          input_schema=coordinator.solution_task.input_schema,
                          empty_control_allowed=True)),
        allowed_actions=['solution','experiment','stop'] if coordinator.registry.lab(lab_id)['mode']=='A'
                        else ['method','solution','experiment','stop'])
