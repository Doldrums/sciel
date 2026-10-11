"""Run resumable multi-step research with canonical candidate admission."""
from __future__ import annotations

from copy import deepcopy
import json
from ..campaign.budget import CampaignStopped
from ..engine.provider_abandonment import AbandonedProviderRequest
from ..engine.research_targets import validate_targets, tool_inputs, input_binding
from ..methods.models import canonical_bytes, content_hash
from .config import effective_limits
from .controller_jobs import JobRefused
from .result_contract import RESULT_SCHEMA
from .research_steps import new_state, finish_state, pending_method_trials, complete_method_trial
from .stage_data import attach_stage_input
from . import experiment_protocol
from .planning import _basis
from .tool_actions import _proposal_failure, parse_action, response_format
from .tool_artifacts import _sha
from .episode_evidence import ResearchEpisodeEvidence
from .canonical_evidence import cited_ids

def _execute_sequential(tools, plan, snapshot, llm, *, io, candidate_evaluator):
    if io is None:
        raise ValueError('research tools require durable wave I/O (workers=1 is supported)')
    if (plan.get('identity_sha256') != content_hash(tools._identity)
            or plan.get('snapshot_sha256') != content_hash(_basis(snapshot))
            or plan.get('action_id') != content_hash({k:v for k,v in plan.items() if k != 'action_id'})):
        raise ValueError('research tool episode differs from its frozen plan')
    validate_targets(snapshot.research_targets,snapshot=snapshot,task_hash=tools.task_identity.get('task_hash'))
    if plan['stage'] != 'research':
        return tools._assigned_episode(plan,snapshot,io)
    state = new_state(plan.get('research_state'))
    findings = []
    work_kind = plan.get('work_kind', 'question')
    transfer_only = work_kind == 'method_transfer'
    if work_kind not in ('question','method_transfer'):
        raise ValueError('invalid research work kind')
    transfer_id = plan.get('method_trial_receipt_id')
    pending = next((row for row in pending_method_trials(state)
                    if row['submission_receipt_id'] == transfer_id), None)
    if (transfer_only and (pending is None or pending.get('blocked_reason')
                          or state['transfer_review_pending']
                          or not tools._independent_branch(snapshot,pending))
            or not transfer_only and transfer_id is not None):
        raise ValueError('research transfer differs from its pending intent or source branch')
    state['status'] = 'open'
    available = list(plan['visible_tool_ids'])
    registered, invocations, feedback, submitted_jobs = [], [], [], []
    if state['transfer_review_pending'] and not transfer_only:
        for deferred in pending_method_trials(state):
            if deferred.get('blocked_reason') == 'pipeline_exceeds_scientific_limit':
                feedback.append({'status':'branch_trial_deferred', 'reason':
                    'complete saved method needs '+str(deferred['pipeline_stages'])+
                    ' stages, exceeding this opportunity scientific-step limit; no partial transfer is claimed'})
    known_cost, scientific, repairs, attempt = 0.0, 0, 0, 0
    provider_abandonment = None
    terminal_decision = None
    evidence = ResearchEpisodeEvidence(
        tools, plan, snapshot, state, feedback, candidate_evaluator, pending)

    # A candidate-producing method is tried on a different source, with the
    # original arguments but no inherited computed state. This is execution
    # evidence only; canonical transfer benefit remains a separate question.
    selected = snapshot.research_targets['selected']
    if transfer_only:
        try:
            pipeline = tools._method_pipeline(pending,plan['cell_id'])
            if (len(pipeline) > tools.config['max_scientific_steps']
                    and not (tools.background_enabled and tools.config['independent_research'])):
                pending.update(blocked_reason='pipeline_exceeds_scientific_limit',
                    pipeline_stages=len(pipeline), scientific_step_limit=tools.config['max_scientific_steps'])
                raise ValueError('complete saved method needs '+str(len(pipeline))+
                    ' stages, exceeding this opportunity scientific-step limit; no partial transfer is claimed')
        except ValueError as error:
            pipeline = []
            evidence.append_step({'action':'branch_trial'},'deferred',error=str(error),benefit='not_established')
            feedback.append({'status':'branch_trial_deferred','reason':str(error)})
        transfer_complete = True
        if pipeline and tools.background_enabled and tools.config['independent_research']:
            try:
                job, trial_id, transfer_complete = tools._background_transfer(
                    plan, snapshot, state, pending, pipeline, io)
            except JobRefused as error:
                # Proven refusal before acceptance: retain the exact stage;
                # no paid job or partial execution is silently resubmitted.
                transfer_complete = False
                evidence.append_step({'action':'branch_trial'}, 'deferred', error=str(error), benefit='not_established')
                feedback.append({'status':'branch_trial_deferred','reason':str(error)})
                state['status'] = 'waiting'
            else:
                if job is not None:
                    submitted_jobs.append(job)
                if transfer_complete:
                    complete_method_trial(state, pending['submission_receipt_id'])
                    state.pop('background_transfer', None)
                    if trial_id is not None:
                        trial_record = tools._read_artifact(trial_id, 'invocation')
                        source = trial_record['receipt']['output']['observations'].get('candidate_source')
                        if type(source) is str and source.strip() and candidate_evaluator is not None:
                            evidence.submit_saved({'action':'submit_candidate','candidate_receipt_id':trial_id}, branch_trial=True)
                        else:
                            feedback.append({'status':'branch_candidate_unavailable', 'receipt_id':trial_id,
                                'reason':'Complete background method returned no canonically evaluated candidate; benefit is not established.'})
        else:
            previous_trial = None
            for index, (original_id,original) in enumerate(pipeline):
                bundle = tools.library.get(original['tool_id'])
                inputs = tool_inputs(snapshot,tools.task_identity,original['inputs'].get('arguments',{}))
                if previous_trial is not None:
                    previous_id, previous_record = previous_trial
                    inputs = attach_stage_input(inputs,previous_record,receipt_id=previous_id)
                limits = effective_limits(tools.config,original['receipt']['limits'])
                if tools.background_enabled:
                    inputs['execution_budget'] = deepcopy(limits)
                if len(canonical_bytes(inputs)) > 2*1024*1024:
                    raise ValueError('branch trial tool inputs exceed 2 MiB')
                invocation, observation = tools._run_tool(plan,snapshot,bundle,inputs,-1-index,io,
                    question='Does stage '+str(index+1)+' of the candidate-producing method execute on an independent branch?',limits=limits)
                invocations.append(invocation); feedback.append(observation)
                success = observation['result']['status']=='success'
                evidence.append_step({'action':'branch_trial','limits':limits},observation['result']['status'],
                    observation=observation,scientific=True,benefit='not_established',
                    reference_source_sha256=pending['source_sha256'],reference_receipt_id=original_id,
                    branch_independence='complete_lineages_no_ancestor_relation',
                    pipeline_stage=index+1,pipeline_stages=len(pipeline),
                    whole_method_completed=success and index+1==len(pipeline))
                scientific += 1
                if not success:
                    complete_method_trial(state, pending['submission_receipt_id'])
                    break
                previous_trial = (invocation['receipt_id'],tools._read_artifact(invocation['receipt_id'],'invocation'))
            if pipeline and previous_trial is not None and success and index+1 == len(pipeline):
                complete_method_trial(state, pending['submission_receipt_id'])
                trial_id, trial_record = previous_trial
                source = trial_record['receipt']['output']['observations'].get('candidate_source')
                if type(source) is str and source.strip() and candidate_evaluator is not None:
                    evidence.submit_saved({'action':'submit_candidate','candidate_receipt_id':trial_id},branch_trial=True)
                else:
                    feedback.append({'status':'branch_candidate_unavailable',
                        'reason':('The complete method executed on another branch, but no saved candidate was '
                                  'canonically evaluated. Execution alone does not demonstrate transfer benefit.'),
                        'receipt_id':trial_id})
        # Keep the original scientific question intact. The temporary
        # branch owns only this atomic replay/evaluation; interpretation
        # runs under a separately frozen question plan on its own source.
        if transfer_complete:
            state['transfer_review_pending'] = True
            tools._event(plan, 'transfer_results_ready',
                reference_submission_receipt_id=pending['submission_receipt_id'],
                scope='Transfer execution/evaluation awaits interpretation on the question source.')

    # Each scientific step has its own repair allowance. When configured,
    # a successful foreground trial yields its observations to ordinary
    # search; otherwise the next planner interprets it in this opportunity.
    while not transfer_only and repairs < tools.config['max_requests']:
        prompt = tools._prompt(plan,snapshot,available,feedback,attempt,
                              research_state=state,scientific_step=scientific)
        # A durable returned/abandoned planner attempt consumes the review
        # opportunity. A STOP rolls this transient flag back with the wave.
        state['transfer_review_pending'] = False
        charge_id = content_hash({'action_id':plan['action_id'],'attempt':attempt,'kind':'llm'})
        tools._event(plan,'llm_start',attempt=attempt,charge_id=charge_id,
                    prompt_sha256=_sha(prompt),model=tools.config['model'] or snapshot.model)
        try:
            response = llm.generate(prompt,tools.config['model'] or snapshot.model,tools.config['temperature'],
                response_format=response_format(tools.background_enabled,visible_tool_ids=available,sequential=True,
                    experimental_protocol=tools.config.get('experimental_protocol',False),
                    independent_research=tools.config.get('independent_research',False)))
        except AbandonedProviderRequest as error:
            provider_abandonment = deepcopy(error.receipt)
            artifact = tools._artifact('proposal_abandonment',{'action_id':plan['action_id'],'attempt':attempt,
                'prompt':prompt,'provider_abandonment':provider_abandonment,'llm_error':deepcopy(error._metadata),
                'generation_returned':False,'generation_dispatch':'unknown','cost':None,'current_call_cost_unknown':True})
            state['unknown_llm_calls'] += 1
            state['status'] = 'provider_abandoned'
            evidence.append_step({'action':'provider_abandoned'},'unknown',proposal_id=artifact)
            tools._event(plan,'llm_abandoned',attempt=attempt,charge_id=charge_id,receipt_id=artifact,
                provider_abandonment=provider_abandonment,llm_error=deepcopy(error._metadata),
                generation_returned=False,generation_dispatch='unknown',cost=None,current_call_cost_unknown=True)
            break
        response_data = {key:getattr(response,key,None) for key in (
            'text','model','cost','prompt_tokens','completion_tokens','finish_reason','refusal','cost_provenance')}
        proposal_id = tools._artifact('proposal',{'action_id':plan['action_id'],'attempt':attempt,
            'cell_id':plan['cell_id'],'cell_revision_id':plan['cell_revision_id'],
            'target_binding':input_binding(tool_inputs(snapshot,tools.task_identity,{})),
            'evidence_receipt_ids':list(dict.fromkeys(row['receipt_id'] for row in state['steps']
                                                     if row.get('receipt_id'))),
            'canonical_evaluation_ids':cited_ids(response.text, {row['artifact_id'] for row in
                evidence.dispatch_plan.get('shared_observations',{}).get('canonical_evaluation_index',[])}),
            'scientific_step':scientific,'prompt':prompt,'response':response_data})
        cost = tools._charge_once(charge_id,response)
        known_cost += cost
        tools._event(plan,'llm_result',attempt=attempt,charge_id=charge_id,receipt_id=proposal_id,
                    known_cost=cost,model=response.model)
        this_attempt, attempt = attempt, attempt+1
        action = {'action':'invalid'}
        evidence.candidate_started = False
        try:
            if getattr(response,'refusal',None):
                raise ValueError('model refused this research proposal')
            action = parse_action(response.text,tools.config,visible_tool_ids=available)
            if action['finding']:
                findings.append({'text':action['finding'],'proposal_id':proposal_id,'cell_id':plan['cell_id'],
                    'island_id':plan['island_id'],'receipt_id':next((row['receipt_id'] for row in reversed(state['steps'])
                        if row.get('receipt_id') and row.get('status') == 'success'), None)})
            if tools.config.get('experimental_protocol'):
                from .experiment_protocol import validate_dispatch
                protocol_audit = validate_dispatch(action,evidence.dispatch_plan,selected['source_sha256'],
                    default_limits=tools.config['limits'])
                tools._event(plan,'experiment_protocol',proposal_id=proposal_id,
                    protocol_schema=protocol_audit['schema'],
                    **{key:value for key,value in protocol_audit.items() if key != 'schema'})
            tools._event(plan,'research_decision',proposal_id=proposal_id,attempt=this_attempt,
                action=action['action'],status='planned',
                **{key:action[key] for key in ('question','experiment','decision_if_supported',
                    'decision_if_refuted','next_decision','conclusion')},
                requested_limits=deepcopy(action.get('limits') or tools.config['limits']),
                scope='Model proposal before execution; no result or efficacy is established.')
            state['transfer_review_pending'] = False
            if action['action'] in ('finish','skip'):
                if action['action'] == 'finish' and action['conclusion'] in ('supported','refuted'):
                    if not any(row.get('receipt_id') and row.get('status') == 'success' for row in state['steps']):
                        raise ValueError('a supported/refuted conclusion requires recorded experiment evidence')
                evidence.append_step(action,'closed',proposal_id=proposal_id,cost=cost,
                            interpretation=action.get('reason',''))
                if action['action'] == 'finish':
                    terminal_decision = proposal_id
                state['status'] = 'closed'
                break
            if action['action'] == 'submit_candidate':
                evidence.submit_saved(action,proposal_id=proposal_id,cost=cost)
                repairs = 0
                continue
            if scientific >= tools.config['max_scientific_steps']:
                evidence.append_step(action,'step_limit',proposal_id=proposal_id,cost=cost)
                state['status'] = 'step_limit'
                break
            if action['action'] in ('poll','cancel','continue'):
                outcome = tools._job_action(plan,action,this_attempt,io,proposal_id=proposal_id)
                if action['action'] == 'continue':
                    submitted_jobs.append(outcome)
                    state['status'] = 'waiting'
                else:
                    feedback.append(outcome)
                evidence.append_step(action,'submitted' if action['action']=='continue' else outcome['status'],
                            proposal_id=proposal_id,cost=cost)
                break
            if action['action'] == 'create':
                bundle = tools.library.register(action['name'],action['description'],action['code'],
                    plan['island_id'],parents=action['parents'],dependencies=action['dependencies'],
                    replay_policy=action['replay_policy'])
                ident = bundle['id']
                if ident not in registered:
                    registered.append(ident)
                if ident not in available:
                    available.append(ident)
                tools._event(plan,'registered',tool_id=ident,name=action['name'],description=action['description'],
                    source_sha256=_sha(action['code']),parent_tool_ids=action['parents'],
                    dependency_tool_ids=action['dependencies'],author_island_id=plan['island_id'],
                    result_schema=RESULT_SCHEMA,replay_policy=action['replay_policy'])
            else:
                ident = action['tool_id']
                if ident not in available:
                    raise ValueError('tool_id is not visible in this frozen episode')
                bundle = tools.library.get(ident)
            inputs = tool_inputs(snapshot,tools.task_identity,action['arguments'])
            if action.get('input_receipt_id'):
                inputs = attach_stage_input(inputs,
                    tools._stage_receipt(action['input_receipt_id'],plan,state,feedback),
                    receipt_id=action['input_receipt_id'])
            if tools.background_enabled:
                inputs['execution_budget'] = deepcopy(action['limits'])
            if len(canonical_bytes(inputs)) > 2*1024*1024:
                raise ValueError('tool inputs exceed 2 MiB')
            if tools.background_enabled and action['execution'] == 'background':
                submitted_jobs.append(tools._submit_job(plan,bundle,inputs,action['limits'],this_attempt,io,
                    action['question'],experimental_protocol=action.get('experimental_protocol'),
                    proposal_id=proposal_id))
                evidence.append_step(action,'submitted',proposal_id=proposal_id,cost=cost,tool_id=ident)
                state['status'] = 'waiting'
                break
        except (ValueError,TypeError,KeyError,SyntaxError,RecursionError) as error:
            if evidence.candidate_started:
                raise CampaignStopped('canonical research candidate submission could not be reconciled') from error
            failure = _proposal_failure(error, response, proposal_id)
            feedback.append(failure)
            evidence.append_step({**action,'action':'invalid'},'proposal_invalid',proposal_id=proposal_id,cost=cost,
                        error=failure['reason'])
            tools._event(plan,'rejected',attempt=this_attempt,status='proposal_invalid',error=failure['reason'])
            repairs += 1
            continue
        invocation, observation = tools._run_tool(plan,snapshot,bundle,inputs,this_attempt,io,
            question=action['question'],limits=action.get('limits'),
            proposal_id=proposal_id if tools.config.get('experimental_protocol') else None)
        invocations.append(invocation); feedback.append(observation)
        success = invocation['result']['status'] == 'success'
        comparison_assessment = None
        if tools.config.get('experimental_protocol'):
            comparison_assessment = experiment_protocol.result_assessment_from_output(
                observation.get('output'), action.get('experimental_protocol'))
            if comparison_assessment is not None:
                feedback.append({'status':'comparison_report_assessment','assessment':comparison_assessment,
                    'reason':json.dumps(comparison_assessment,ensure_ascii=True, sort_keys=True),
                    'receipt_id':observation['receipt_id']})
                tools._event(plan,'comparison_report_assessment',receipt_id=observation['receipt_id'],
                    protocol_schema=comparison_assessment['schema'],
                    **{key:value for key,value in comparison_assessment.items() if key != 'schema'})
        evidence.append_step(action,invocation['result']['status'],proposal_id=proposal_id,observation=observation,
                    cost=cost,scientific=success,
                    **({'comparison_report_assessment':comparison_assessment} if comparison_assessment else {}))
        if success:
            scientific += 1
            repairs = 0
            if tools.config['yield_after_success']:
                state['status'] = 'open'
                tools._event(plan,'research_yielded',status='open',reason='successful_foreground_trial',
                    tool_id=ident,receipt_id=invocation['receipt_id'],
                    scientific_steps_this_opportunity=scientific,
                    scope='Observations return to ordinary search; the question is not scientifically closed.')
                break
        else:
            repairs += 1
    if repairs >= tools.config['max_requests']:
        state['status'] = 'repair_exhausted'
        state['transfer_review_pending'] = False
    episode = tools._finish_episode(plan,registered,invocations,feedback,known_cost,attempt,
                                   provider_abandonment=provider_abandonment,
                                   terminal_decision=terminal_decision)
    episode['research_state'] = finish_state(state,state['status'])
    episode['findings'] = findings
    if submitted_jobs:
        episode['submitted_jobs'] = submitted_jobs
    if evidence.candidate_submission is not None:
        episode['candidate_submission'] = evidence.candidate_submission
        episode['candidate_submissions'] = evidence.candidate_submissions
    return episode
