"""One research proposal and its durable experiment/feedback transitions."""
from __future__ import annotations

import time
from dataclasses import asdict

from .budget import BudgetExceeded
from .coordinator_contracts import Refusal, StopRequested, TrialQuotaExceeded, _experiment_shape
from .engine_bridge import EngineCascadeBridge, LabSolutionHistory
from .materialize import materialize
from .models import canonical_bytes, content_hash, detached
from .proposals import (
    PROCEDURE_ID,
    parse_proposal,
    prompt_for,
    proposal_diagnostic,
    response_record,
)
from ..campaign.budget import RequestNotDispatched
from ..engine.capacity import CapacityCancelled, CapacityConfigurationError, CapacityTimeout


def step(coordinator, lab_id):
    with coordinator._locked():
        coordinator._check_stop()
        coordinator._headroom()
        lab_status=coordinator.lab_state(lab_id)['status']
        if lab_status=='dormant':
            raise TrialQuotaExceeded('dormant lab requires explicit evidence-bound trial renewal',
                record=dict(lab_id=lab_id,lineage_id=coordinator._active_funding_lineage(lab_id)))
        if lab_status=='archived':
            raise BudgetExceeded('archived lab requires an explicitly authorized fork')
        done=[row for row in coordinator._events('episode_step') if row['lab_id']==lab_id]
        reopened=bool(done and done[-1]['status']=='stopped' and any(
            row['lab_id']==lab_id and row['stopped_proposal_id']==done[-1]['proposal_id']
            for row in coordinator._events('episode_reopened')))
        if done and done[-1]['status']=='stopped' and not reopened:
            return detached(done[-1])
        index=len(done)
        if coordinator.max_steps is not None and index>=coordinator.max_steps:
            raise BudgetExceeded('episode step cap includes format failures and repairs')
        key=f'proposal:{lab_id}:{index}'
        intents=[row for row in coordinator._events('proposal_intent') if row['key']==key]
        if intents:
            context_id=intents[0]['context_id']
            context=coordinator.registry.get_artifact('proposal_context',context_id)
        else:
            revision_id,revision=coordinator._revision(lab_id)
            context=coordinator._host('context:'+key,lab_id,'retrieval',dict(revision_id=revision_id),
                               lambda:coordinator._context(lab_id,revision_id,revision))
            context['proposal_index']=index
            context_id=coordinator.registry.put_artifact('proposal_context',context)
            coordinator.registry.append('proposal_intent',dict(key=key,context_id=context_id))
        prompt=prompt_for(context)
        bounds=coordinator.proposer.bounds(prompt)
        if coordinator.capacity and coordinator.capacity_wait_seconds is not None:
            bounds['wall_seconds']+=coordinator.capacity_wait_seconds
        def generate(_):
            wall,cpu=time.monotonic(),time.process_time()
            queued=dict(cpu_seconds=0.,wall_seconds=0.)
            try:
                lease=coordinator._capacity_lease('llm',lab_id=lab_id,request_id=key,charged=queued)
            except (CapacityCancelled,CapacityConfigurationError,CapacityTimeout,StopRequested,BudgetExceeded) as exc:
                value=dict(status='failed',reason=type(exc).__name__,text='',cost=0,
                           dispatch_status='not_dispatched',
                           failure_scope='operator' if isinstance(exc,StopRequested) else 'budget' if isinstance(exc,BudgetExceeded) else 'infrastructure',
                           model=coordinator.proposer.identity.get('model','not_dispatched'),
                           prompt_tokens=0,completion_tokens=0)
            except BaseException as exc:
                value=dict(status='unknown',reason=type(exc).__name__,text='',cost=0,
                    dispatch_status='not_dispatched',failure_scope='infrastructure',
                    model=coordinator.proposer.identity.get('model','not_dispatched'),
                    prompt_tokens=0,completion_tokens=0)
            else:
                dispatched=False
                try:
                    with lease:
                        coordinator._check_stop()
                        dispatched=True
                        response=coordinator.proposer.generate(prompt)
                        value=response_record(response)
                except (RequestNotDispatched,StopRequested) as exc:
                    value=dict(status='failed',reason=type(exc).__name__,text='',cost=0,
                        dispatch_status='not_dispatched',failure_scope='operator' if isinstance(exc,StopRequested) else 'provider_admission',
                        model=coordinator.proposer.identity.get('model','not_dispatched'),
                        prompt_tokens=0,completion_tokens=0)
                except BaseException as exc:
                    value=dict(status='unknown',reason=type(exc).__name__,text='',cost=None if dispatched else 0,
                        dispatch_status='unknown' if dispatched else 'not_dispatched',
                        failure_scope='provider_outcome_unknown' if dispatched else 'infrastructure',
                        model=coordinator.proposer.identity.get('model','unknown'),prompt_tokens=0,completion_tokens=0)
            actual=dict(cpu_seconds=round(max(0,time.process_time()-cpu-queued['cpu_seconds']),9),
                gpu_seconds=0,wall_seconds=round(max(0,time.monotonic()-wall-queued['wall_seconds']),9),
                storage_bytes=len(canonical_bytes(value))+len(prompt.encode()),steps=1)
            if value.get('cost') is not None:
                actual['dollars']=value['cost']
            return value,actual
        response=coordinator.dispatch.run(key,coordinator._account(lab_id),'generation',
            dict(prompt=prompt,generator=coordinator.proposer.identity),bounds,generate)
        if response.get('reason')=='StopRequested' and response.get('dispatch_status')=='not_dispatched':
            coordinator._check_stop()
        draft,error,diagnostic=None,None,None
        if response.get('dispatch_status')=='not_dispatched':
            error='proposal_not_dispatched'
            diagnostic=dict(code=error,message='Provider request was not dispatched; this is not a model-format failure.',
                reason=response.get('reason'),failure_scope=response.get('failure_scope'),dispatch_status='not_dispatched')
        else:
            try:
                draft=parse_proposal(response['text'])
            except (ValueError,TypeError,RecursionError) as exc:
                error='invalid_proposal_format'
                diagnostic=proposal_diagnostic(exc)
        proposal_id=coordinator.registry.put_artifact('proposal_receipt',dict(
            schema='sciel-method-proposal/1',lab_id=lab_id,lab_revision_id=context['revision_id'],
            search_procedure=PROCEDURE_ID,generator=coordinator.proposer.identity,
            evidence_refs=[e['id'] for e in context['evidence']],
            method_refs=[m['id'] for m in context['available_methods']],
            request_sha256=content_hash(prompt),response_sha256=content_hash(response['text']),
            call_receipt_id=response['call_receipt_id'],context_id=context_id,
            solution_refs=[v['id'] for v in context.get('previous_solutions',[])],
            library_refs=[v['id'] for v in context.get('available_library',[])],
            source_read=[dict(id=v['id'],source_included=bool(v.get('source_included',False)))
                         for v in context.get('available_methods',[])],
            draft=draft,error=error,diagnostic=diagnostic))
        evidence=[proposal_id]
        result=dict(lab_id=lab_id,index=index,proposal_id=proposal_id,status='continued')
        # One initial duplicate plus at most two subsequent generator calls.
        # Malformed repairs consume the same chain; an explicit different
        # action or a source accepted by the gate ends it.
        retry_count=(done[-1].get('novelty_retry_count',0) if done and not reopened else 0)
        if retry_count and (draft is None or draft['kind']=='solution'):
            result['novelty_retry_count']=retry_count+1
        if draft:
            try:
                if draft['kind']=='stop':
                    result.update(status='stopped',reason=draft['reason'])
                else:
                    experiment=detached(draft.get('experiment'))
                    if experiment is not None:
                        _experiment_shape(experiment)
                    new_id=None
                    if draft['kind']=='method':
                        new_id=coordinator.register_draft(lab_id,proposal_id,draft['files'],draft['manifest'])
                        result['method_id']=new_id
                    if draft['kind']=='solution':
                        if (experiment is None or experiment.get('treatment')!=['$new']
                                or len(experiment.get('inputs',[]))!=1):
                            raise Refusal('solution requires exactly its new execution bundle and one assigned input')
                        _,revision=coordinator._revision(lab_id,context['revision_id'])
                        closure=coordinator._bundle_closure(revision['active_methods']+revision['active_adapters'])
                        assembled=materialize(draft['code'],list(closure.values()))
                        if assembled['status']!='ok':
                            raise Refusal(assembled['reason'])
                        composition=coordinator.registry.put_artifact('composition',assembled['receipt'])
                        bundle=coordinator.solution_task.solution_bundle(assembled['code'],lab_id,revision['runtime_lock'],coordinator.resources)
                        new_id=coordinator.registry.register(bundle['files'],bundle['manifest'])
                        result['composition_id']=composition
                        result['solution_source_id']=coordinator.registry.put_artifact('solution_source',dict(
                            code=assembled['code'],composition_id=composition,method_id=new_id,
                            attribution='composition only; no embedded method invocation credit'))
                    if experiment is not None:
                        experiment['treatment']=[new_id if ident=='$new' else ident for ident in experiment['treatment']]
                        selected=[]
                        if draft['kind']=='experiment':
                            available={v['id'] for v in context.get('available_library',[])}
                            selected=list(dict.fromkeys(i for i in experiment['treatment']+experiment['control']
                                if i in available and coordinator.registry.get_method(i)['manifest']['author_lab']!=lab_id))
                        plan_id=coordinator.plan(lab_id,proposal_id,experiment,revision_id=context['revision_id'],
                            purpose='solution' if draft['kind']=='solution' else 'transfer' if selected else 'local',
                            selected=selected)
                        result['plan_id']=plan_id
                        history=None
                        if draft['kind']=='solution':
                            history=LabSolutionHistory(coordinator.registry,lab_id,coordinator.solution_task)
                            evaluation_id='solution:'+plan_id
                            decision=history.reserve(assembled['code'],evaluation_id,context['revision_id'],plan_id=plan_id)
                            if not decision['accept']:
                                result['novelty_retry_count']=retry_count+1
                                diagnostic=detached(decision)
                                matched=history.state()['terminal'].get(decision.get('matched_evaluation_id'))
                                if matched:
                                    diagnostic['matched_call_receipt_id']=matched['call_receipt_id']
                                    previous=coordinator.registry.get_artifact('call_receipt',matched['call_receipt_id'])['result']
                                    prior_plan=coordinator.registry.get_artifact('experiment_plan',previous['plan_id'])
                                    measured=coordinator.solution_task.measure(prior_plan['inputs'][0],previous['output'])
                                    diagnostic['matched_observation']=dict(call_receipt_id=matched['call_receipt_id'],
                                        status=previous['status'],valid=measured['valid'],
                                        score=measured['metrics'][coordinator.task.metric],cases=measured['metrics'],
                                        observables=measured.get('observables',{}))
                                result['novelty']=diagnostic
                                raise Refusal('solution novelty rejected: '+canonical_bytes(diagnostic).decode())
                            result.pop('novelty_retry_count',None)
                        experiment_id=coordinator.execute(plan_id)
                        evidence.append(experiment_id)
                        result.update(plan_id=plan_id,experiment_id=experiment_id)
                        if draft['kind']=='solution':
                            from ..engine.types import Program
                            from ..eval.facade import EvalResult
                            measured=coordinator.registry.get_artifact('experiment',experiment_id)
                            from ..eval.cascade import CascadeOutcome
                            calls=[(rid,coordinator.registry.get_artifact('call_receipt',rid)['result'])
                                   for rid in measured['invocation_refs']]
                            calls=[(rid,call) for rid,call in calls if call.get('method_id')==new_id]
                            if len(calls)!=1:
                                raise Refusal('solution must have exactly one execution receipt')
                            receipt_id,execution=calls[0]
                            program=None
                            if execution['status']=='ok':
                                if isinstance(coordinator.solution_task,EngineCascadeBridge):
                                    observed=coordinator.solution_task.extract(execution['output'])
                                    if coordinator.solution_task._valid(observed):
                                        program=coordinator.solution_task.program(assembled['code'],execution['output'],
                                            'research-'+proposal_id[:20],gen=index)
                                else:
                                    measurement=coordinator.solution_task.measure(experiment['inputs'][0],execution['output'])
                                    if measurement['valid']:
                                        program=Program(id='research-'+proposal_id[:20],code=assembled['code'],
                                            score=measurement['metrics'][coordinator.task.metric],cases=measurement['metrics'],
                                            island=0,origin='research_solution',gen=index)
                            observed=history.finalize(evaluation_id,receipt_id,
                                                      program_id=program.id if program else None)
                            cascade=observed if isinstance(observed,CascadeOutcome) else None
                            evaluation=cascade.result if cascade else observed
                            body=dict(program=asdict(program) if program else None,
                                evaluation=asdict(evaluation) if evaluation else None,
                                lab_id=lab_id,lab_revision_id=context['revision_id'],
                                source_id=result['solution_source_id'],experiment_id=experiment_id,
                                evaluation_protocol='cascade' if cascade else 'task_adapter',
                                cascade=asdict(cascade) if cascade else None,call_receipt_id=receipt_id)
                            kind='solution' if program else 'solution_evaluation'
                            result[kind+'_id']=coordinator.registry.put_artifact(kind,body)
            except (Refusal,ValueError,KeyError,TrialQuotaExceeded) as exc:
                refusal=coordinator.registry.put_artifact('proposal_receipt',dict(lab_id=lab_id,
                    proposal_id=proposal_id,error='proposal_refused',diagnostic=str(exc)[:1000],
                    novelty=result.get('novelty'),
                    failure_scope='trial_policy' if isinstance(exc,TrialQuotaExceeded) else 'contract',
                    plan_id=result.get('plan_id'),
                    funding_lineage=(exc.record or {}).get('lineage_id') if isinstance(exc,TrialQuotaExceeded) else None))
                evidence.append(refusal)
                result['refusal_id']=refusal
        if result.get('novelty_retry_count',0)>=3:
            result.update(status='stopped',reason='solution_novelty_resamples_exhausted')
        current=coordinator.registry.lab(lab_id)['active_revision']
        reason='proposal_feedback:'+proposal_id
        if coordinator.registry.revision(current)['reason']!=reason:
            old=coordinator.registry.revision(context['revision_id'])
            current=coordinator.registry.revise_lab(lab_id,context['revision_id'],
                dict(evidence_refs=list(dict.fromkeys(old['evidence_refs']+evidence))),reason)
        result['revision_id']=current
        coordinator.registry.append('episode_step',result)
        return result
