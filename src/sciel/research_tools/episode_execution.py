"""Execute single-step research, replay and recipient method trials."""
from __future__ import annotations

from copy import deepcopy
from ..engine.provider_abandonment import AbandonedProviderRequest
from ..engine.research_targets import validate_targets, tool_inputs
from ..methods.models import canonical_bytes, content_hash
from .config import effective_limits
from .controller_jobs import JobRefused
from .result_contract import RESULT_SCHEMA, assess_receipt, replay_evidence
from .planning import _basis
from .tool_actions import _proposal_failure, parse_action, response_format
from .tool_artifacts import _sha

def _assigned_episode(tools, plan, snapshot, io):
    bundle = tools.library.get(plan['assigned_tool_id'])
    reference = None
    if plan['stage'] == 'independent_replay':
        reference = tools._read_artifact(plan['reference_receipt_id'], 'invocation')
        if (reference['tool_id'] != bundle['id'] or assess_receipt(reference['receipt'])['status'] != 'success'
                or reference['receipt']['dispatch_status'] != 'invoked'):
            raise ValueError('independent replay requires the exact successful local invocation')
        inputs = deepcopy(reference['inputs'])
    else:
        method = tools.lifecycle.method(bundle['id'])
        original = tools._read_artifact(method['local_receipt_id'], 'invocation')
        inputs = tool_inputs(snapshot, tools.task_identity, original['inputs'].get('arguments', {}))
    if len(canonical_bytes(inputs)) > 2 * 1024 * 1024:
        # Admission is bounded for both model-proposed and automatic trials.
        # Keep full frozen sources; never silently cut code to fit a tool input.
        episode = tools._finish_episode(plan, [], [], [], 0.0, 0)
        episode.update(deferred=True, context='Tool inputs exceed the 2 MiB admission limit.')
        if plan.get('sharing_work_id'):
            episode['sharing_deferred_reason'] = 'input_admission_limit'
        tools._event(plan, 'input_deferred', tool_id=bundle['id'], error=episode['context'])
        return episode
    question = {
        'independent_replay':'Does the exact saved tool pass its immutable declared replay policy in an independent execution?',
        'adopt_trial':'Does the released tool execute correctly on this recipient\'s current input?',
        'use_again':'Can the adopted method be invoked again in a later research step?',
    }[plan['stage']]
    limits = effective_limits(tools.config, (reference or original)['receipt']['limits'])
    if tools.background_enabled:
        episode = tools._finish_episode(plan, [], [], [], 0.0, 0)
        try:
            episode['submitted_jobs'] = [tools._submit_job(plan, bundle, inputs, limits, 0, io, question)]
        except JobRefused as error:
            episode['deferred'] = True
            if plan.get('sharing_work_id'):
                episode['sharing_deferred_reason'] = 'submission_deferred'
            episode['context'] = str(error)
            tools._event(plan, 'job_deferred', tool_id=bundle['id'], error=str(error))
        return episode
    invocation, observation = tools._run_tool(plan, snapshot, bundle, inputs, 0, io, question=question, limits=limits)
    evidence = (replay_evidence(reference['receipt'],
        tools._read_artifact(invocation['receipt_id'], 'invocation')['receipt'],
        bundle['manifest']['protocol']['replay_policy']) if reference else None)
    return tools._finish_episode(plan, [], [invocation], [observation], 0.0, 0,
                                replay_result=evidence)


def _execute_single(tools, plan, snapshot, llm, *, io):
    if io is None:
        raise ValueError('research tools require durable wave I/O (workers=1 is supported)')
    if (plan.get('identity_sha256') != content_hash(tools._identity)
            or plan.get('snapshot_sha256') != content_hash(_basis(snapshot))
            or plan.get('action_id') != content_hash({k: v for k, v in plan.items() if k != 'action_id'})):
        raise ValueError('research tool episode differs from its frozen plan')
    validate_targets(snapshot.research_targets, snapshot=snapshot,
                     task_hash=tools.task_identity.get('task_hash'))
    if plan['stage'] != 'research':
        return tools._assigned_episode(plan, snapshot, io)
    available = list(plan['visible_tool_ids'])
    registered, published, invocations, feedback = [], [], [], []
    known_cost = 0.0
    submitted_jobs = []
    provider_abandonment = None
    for attempt in range(tools.config['max_requests']):
        prompt = tools._prompt(plan, snapshot, available, feedback, attempt)
        charge_id = content_hash({'action_id': plan['action_id'], 'attempt': attempt, 'kind': 'llm'})
        tools._event(plan, 'llm_start', attempt=attempt, charge_id=charge_id,
                    prompt_sha256=_sha(prompt), model=tools.config['model'] or snapshot.model)
        # WaveLLM already owns provider capacity, tracing and durable dispatch.
        # Never put it inside a second io.call or provider retry loop.
        try:
            response = llm.generate(prompt, tools.config['model'] or snapshot.model,
                tools.config['temperature'], response_format=response_format(tools.background_enabled,
                    visible_tool_ids=available))
        except AbandonedProviderRequest as error:
            # Only authenticated wave recovery can authorize this typed
            # transition. The paid unknown is neither a response nor an
            # extra proposal retry; earlier execution evidence survives.
            provider_abandonment = deepcopy(error.receipt)
            artifact = tools._artifact('proposal_abandonment', {
                'action_id':plan['action_id'], 'attempt':attempt, 'prompt':prompt,
                'provider_abandonment':provider_abandonment,
                'llm_error':deepcopy(error._metadata), 'generation_returned':False,
                'generation_dispatch':'unknown', 'cost':None, 'current_call_cost_unknown':True})
            feedback.append({'status':'provider_abandoned', 'receipt_id':artifact,
                'reason':'The tool proposal request was explicitly abandoned; its outcome and cost remain unknown. '
                         'No further planner request is made in this episode.',
                'provider_abandonment':provider_abandonment})
            tools._event(plan, 'llm_abandoned', attempt=attempt, charge_id=charge_id,
                receipt_id=artifact, provider_abandonment=provider_abandonment,
                llm_error=deepcopy(error._metadata), generation_returned=False,
                generation_dispatch='unknown', cost=None, current_call_cost_unknown=True)
            break
        response_data = {key: getattr(response, key, None) for key in (
            'text', 'model', 'cost', 'prompt_tokens', 'completion_tokens', 'finish_reason', 'refusal', 'cost_provenance')}
        artifact = tools._artifact('proposal', {'action_id': plan['action_id'],
            'attempt': attempt, 'prompt': prompt, 'response': response_data})
        cost = tools._charge_once(charge_id, response)
        known_cost += cost
        tools._event(plan, 'llm_result', attempt=attempt, charge_id=charge_id,
                    receipt_id=artifact, known_cost=cost, model=response.model)
        try:
            if getattr(response, 'refusal', None):
                raise ValueError('model refused this research tool proposal')
            action = parse_action(response.text, tools.config, visible_tool_ids=available)
            if action['action'] == 'skip':
                feedback.append({'status': 'skipped', 'reason': action['reason']})
                break
            if action['action'] in ('poll', 'cancel', 'continue'):
                outcome = tools._job_action(plan, action, attempt, io)
                if action['action'] == 'continue':
                    submitted_jobs.append(outcome)
                else:
                    feedback.append(outcome)
                break
            if action['action'] == 'create':
                bundle = tools.library.register(action['name'], action['description'], action['code'],
                    plan['island_id'], parents=action['parents'], dependencies=action['dependencies'],
                    replay_policy=action['replay_policy'])
                ident = bundle['id']
                if ident not in registered:
                    registered.append(ident)
                if ident not in available:
                    available.append(ident)
                tools._event(plan, 'registered', tool_id=ident, name=action['name'],
                    description=action['description'], source_sha256=_sha(action['code']),
                    parent_tool_ids=action['parents'], dependency_tool_ids=action['dependencies'],
                    author_island_id=plan['island_id'], result_schema=RESULT_SCHEMA,
                    replay_policy=action['replay_policy'])
            else:
                ident = action['tool_id']
                if ident not in available:
                    raise ValueError('tool_id is not visible in this frozen episode')
                bundle = tools.library.get(ident)
            inputs = tool_inputs(snapshot, tools.task_identity, action['arguments'])
            if tools.background_enabled:
                inputs['execution_budget'] = deepcopy(action['limits'])
            # Bounded request materialization before entering an external leaf call.
            if len(canonical_bytes(inputs)) > 2 * 1024 * 1024:
                raise ValueError('tool inputs exceed 2 MiB')
            if tools.background_enabled and action['execution'] == 'background':
                submitted_jobs.append(tools._submit_job(plan, bundle, inputs, action['limits'], attempt, io,
                                                      action.get('reason') or plan['question']))
                break
        except (ValueError, TypeError, KeyError, SyntaxError, RecursionError) as error:
            failure = _proposal_failure(error, response, artifact)
            feedback.append(failure)
            tools._event(plan, 'rejected', attempt=attempt, status='proposal_invalid',
                        error=failure['reason'])
            continue
        invocation, observation = tools._run_tool(plan, snapshot, bundle, inputs, attempt, io,
            question=action.get('reason') or plan['question'], limits=action.get('limits'))
        invocations.append(invocation)
        feedback.append(observation)
        if invocation['result']['status'] == 'success':
            published.append(ident)
            break
    episode = tools._finish_episode(plan, registered, invocations, feedback, known_cost, attempt+1,
                                   provider_abandonment=provider_abandonment)
    if submitted_jobs:
        episode['submitted_jobs'] = submitted_jobs
    return episode


def _independent_branch(tools, snapshot, pending, *, job_state=None):
    original = tools._read_artifact(pending['submission_receipt_id'],'invocation')
    new = (getattr(snapshot,'research_search_evidence',None) or {}).get('selected_lineage') or {}
    selected = snapshot.research_targets['selected']
    if new.get('complete') is not True:
        return False
    try:
        origins = tools._method_pipeline(pending,original['cell_id'],job_state=job_state)
    except ValueError:
        return False
    for _, origin in origins:
        old = (origin['inputs'].get('search_evidence') or {}).get('selected_lineage') or {}
        if old.get('complete') is not True:
            return False
        prior_id, current_id = origin['target_binding']['candidate_id'], selected['program_id']
        prior_sha, current_sha = origin['target_binding']['target_source_sha256'], selected['source_sha256']
        for lineage,ident,digest in ((old,prior_id,prior_sha),(new,current_id,current_sha)):
            if (type(lineage.get('program_ids')) is not list or type(lineage.get('source_sha256s')) is not list
                    or ident not in lineage['program_ids'] or digest not in lineage['source_sha256s']):
                return False
        if (prior_id in new['program_ids'] or current_id in old['program_ids']
                or prior_sha in new['source_sha256s'] or current_sha in old['source_sha256s']):
            return False
    return True


def _method_pipeline(tools, pending, cell_id, *, job_state=None):
    """Follow exact stage/job data dependencies, not speculative tool lineage."""
    job_state = tools._state if job_state is None else job_state
    ident, chain, seen = pending['submission_receipt_id'], [], set()
    while ident is not None:
        if ident in seen or len(chain) >= 16:
            raise ValueError('saved method pipeline exceeds 16 stages or contains a cycle')
        seen.add(ident)
        saved = tools._read_artifact(ident,'invocation')
        if saved['cell_id'] != cell_id or saved['result']['status'] != 'success':
            raise ValueError('saved method pipeline contains unavailable or unsuccessful stage evidence')
        chain.append((ident,saved))
        stage = saved['inputs'].get('research_stage') or {}
        ident = stage.get('prior_receipt_id')
        if ident is None and tools.background_enabled:
            job = job_state['jobs'].get(saved['receipt']['invocation_id']) or {}
            parent = job.get('parent_job_id')
            if parent:
                ident = (job_state['collected_jobs'].get(parent) or {}).get('invocation_artifact_id')
                if ident is None:
                    raise ValueError('saved method pipeline parent job has no collected invocation')
    return list(reversed(chain))


def _stage_receipt(tools, ident, plan, state, feedback):
    allowed = {row['receipt_id'] for row in state['steps']
               if row.get('receipt_id') and row.get('status') == 'success'}
    allowed.update(row['receipt_id'] for row in feedback
                   if row.get('receipt_id') and (row.get('result') or {}).get('status') == 'success'
                   and row.get('invocation_id'))
    if ident not in allowed:
        raise ValueError('receipt is not a successful visible invocation of this research cell')
    saved = tools._read_artifact(ident, 'invocation')
    if saved['cell_id'] != plan['cell_id'] or assess_receipt(saved['receipt'])['status'] != 'success':
        raise ValueError('stage input must be a successful saved invocation of this cell')
    return saved
