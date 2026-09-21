"""Bounded, saved scientific decisions separate from tool repair attempts.

The ledger records model claims as claims. Only immutable invocation receipts
establish execution; candidate efficacy comes from the ordinary evaluator.
"""
from copy import deepcopy
import json
import math

from ..methods.models import content_hash, require_hash

SCHEMA = 'sciel-research-state/2'
TEXT_FIELDS = ('question', 'decision_if_supported', 'decision_if_refuted',
               'experiment', 'next_decision')
STATUSES = ('open', 'waiting', 'closed', 'step_limit', 'repair_exhausted', 'provider_abandoned')


def proposal_fields(action):
    invalid = [key for key in TEXT_FIELDS
               if type(action.get(key)) is not str or not action[key].strip()]
    if invalid:
        raise ValueError('; '.join(key + ' must be a nonempty concrete research field'
                                   for key in invalid))
    values = {}
    for key in TEXT_FIELDS:
        text = action.get(key)
        if len(text) > 2000:
            # Length alone must not discard an otherwise usable experiment.
            # The exact full model response remains in its proposal artifact.
            suffix = ' [Excerpt; complete field is in the saved proposal.]'
            text = text[:2000-len(suffix)] + suffix
        values[key] = text
    conclusion = action.get('conclusion', '')
    if conclusion not in ('', 'supported', 'refuted', 'inconclusive'):
        raise ValueError('conclusion must be supported, refuted, inconclusive, or empty')
    if action['action'] == 'finish' and not conclusion:
        raise ValueError('finish requires a supported, refuted, or inconclusive conclusion')
    values['conclusion'] = conclusion or None
    return values


def new_state(previous=None):
    if previous:
        validate_state(previous)
        return deepcopy(previous)
    return {'schema':SCHEMA, 'scope':'cell_cumulative',
        **{key:'' for key in TEXT_FIELDS}, 'current_experiment':'',
        'status':'open', 'conclusion':None, 'scientific_steps':0,
        'known_llm_cost':0.0, 'unknown_llm_calls':0,
        'resources':{'cpu_seconds':0.0, 'wall_seconds':0.0, 'unknown_cpu_invocations':0},
        'steps':[], 'steps_omitted':0, 'pending_method_trial':None,
        'pending_method_trial_backlog':[], 'transfer_review_pending':False}


def pending_method_trials(state):
    """FIFO head plus durable tail; retain the historical head projection."""
    head = (state or {}).get('pending_method_trial')
    return ([head] if head is not None else []) + list((state or {}).get('pending_method_trial_backlog', []))


def enqueue_method_trial(state, trial):
    if any(row['submission_receipt_id'] == trial['submission_receipt_id']
           for row in pending_method_trials(state)):
        return
    if state['pending_method_trial'] is None:
        state['pending_method_trial'] = deepcopy(trial)
    else:
        state['pending_method_trial_backlog'].append(deepcopy(trial))


def complete_method_trial(state, receipt_id):
    trials = pending_method_trials(state)
    remaining = [row for row in trials if row['submission_receipt_id'] != receipt_id]
    if len(remaining) != len(trials) - 1:
        raise ValueError('completed method trial is not a unique pending transfer')
    state['pending_method_trial'] = remaining[0] if remaining else None
    state['pending_method_trial_backlog'] = remaining[1:]


def record_step(state, step, *, cost=0.0, scientific=False):
    """Append an immutable-reference summary; lifetime totals survive trimming."""
    # A parsed proposal can still fail identity/evidence checks. Retain its
    # complete step and costs below without making rejected claims current.
    if step.get('status') != 'proposal_invalid':
        for key in TEXT_FIELDS:
            if step.get(key):
                state[key] = step[key]
        state['current_experiment'] = state['experiment']
        if 'conclusion' in step:
            state['conclusion'] = step['conclusion']
    state['known_llm_cost'] += cost
    state['scientific_steps'] += int(scientific)
    for key in ('cpu_seconds', 'wall_seconds'):
        value = (step.get('resources') or {}).get(key)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            state['resources'][key] += value
        elif key == 'cpu_seconds' and step.get('receipt_id'):
            state['resources']['unknown_cpu_invocations'] += 1
    state['steps'].append(deepcopy(step))
    if len(state['steps']) > 32:
        state['steps_omitted'] += len(state['steps']) - 32
        state['steps'] = state['steps'][-32:]


def finish_state(state, status):
    if status not in STATUSES:
        raise ValueError('unsupported research status')
    state['status'] = status
    validate_state(state)
    return deepcopy(state)


def validate_state(value):
    expected = {'schema','scope',*TEXT_FIELDS,'current_experiment','status','conclusion',
        'scientific_steps','known_llm_cost','unknown_llm_calls','resources','steps','steps_omitted',
        'pending_method_trial','pending_method_trial_backlog','transfer_review_pending'}
    if type(value) is not dict or set(value) not in (expected, expected | {'background_transfer'}) or value.get('schema') != SCHEMA:
        raise ValueError('invalid research decision state')
    if value['scope'] != 'cell_cumulative' or value['status'] not in STATUSES:
        raise ValueError('invalid research decision scope/status')
    for key in (*TEXT_FIELDS,'current_experiment'):
        if type(value[key]) is not str or len(value[key]) > 2000:
            raise ValueError('invalid research decision text')
    if value['conclusion'] not in (None, 'supported', 'refuted', 'inconclusive'):
        raise ValueError('invalid research conclusion')
    for key in ('scientific_steps','unknown_llm_calls','steps_omitted'):
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError('invalid research decision counter')
    if (type(value['known_llm_cost']) not in (int,float)
            or not math.isfinite(value['known_llm_cost']) or value['known_llm_cost'] < 0):
        raise ValueError('invalid research known model cost')
    resources = value['resources']
    if type(resources) is not dict or set(resources) != {'cpu_seconds','wall_seconds','unknown_cpu_invocations'}:
        raise ValueError('invalid research resource totals')
    for number in resources.values():
        if type(number) not in (int,float) or not math.isfinite(number) or number < 0:
            raise ValueError('invalid research resource total')
    if type(value['steps']) is not list or len(value['steps']) > 32:
        raise ValueError('invalid bounded research steps')
    for step in value['steps']:
        if type(step) is not dict or step.get('action') not in (
                'create','use','skip','finish','poll','continue','cancel','submit_candidate',
                'invalid','provider_abandoned','branch_trial','background_result',
                'independent_replay','adopt_trial','use_again'):
            raise ValueError('invalid research step')
        for key in ('proposal_id','receipt_id','tool_id'):
            if step.get(key) is not None:
                require_hash(step[key], key)
    if (type(value['pending_method_trial_backlog']) is not list
            or type(value['transfer_review_pending']) is not bool
            or value['pending_method_trial'] is None and value['pending_method_trial_backlog']):
        raise ValueError('invalid pending method transfer queue')
    trials = pending_method_trials(value)
    for pending in trials:
        fields = {'tool_id','source_program_id','source_sha256','submission_receipt_id'}
        blocked = {'blocked_reason','pipeline_stages','scientific_step_limit'}
        if type(pending) is not dict or set(pending) not in (fields, fields | blocked):
            raise ValueError('invalid pending method branch trial')
        if set(pending) == fields | blocked and (
                pending['blocked_reason'] != 'pipeline_exceeds_scientific_limit'
                or type(pending['pipeline_stages']) is not int
                or type(pending['scientific_step_limit']) is not int
                or not 1 <= pending['scientific_step_limit'] < pending['pipeline_stages'] <= 16):
            raise ValueError('invalid blocked method branch trial')
        for key in ('tool_id','source_sha256','submission_receipt_id'):
            require_hash(pending[key],key)
    if len({row['submission_receipt_id'] for row in trials}) != len(trials):
        raise ValueError('duplicate pending method transfer')
    from .transfer_jobs import validate_progress
    validate_progress(value.get('background_transfer'), value)
    content_hash(value)
    return value


def state_prompt(state, *, step, maximum, feedback_receipts=(), budget=2200, canonical_evaluations=()):
    """Whole JSON projection, retaining actionable receipt IDs ahead of prose."""
    ids = list(dict.fromkeys(row['receipt_id'] for row in state['steps']
        if row.get('receipt_id') and row.get('status') == 'success'))
    ids = list(dict.fromkeys(ids + list(feedback_receipts)))
    value = {'schema':SCHEMA, 'trial_number':step + 1, 'max_trials_this_opportunity':maximum,
        'status':state['status'], 'conclusion':state['conclusion'],
        'available_input_receipt_ids':ids[-16:], 'older_receipts_omitted':max(0,len(ids)-16),
        'scientific_steps_total':state['scientific_steps'], 'known_llm_cost':state['known_llm_cost'],
        'tool_resources_total':state['resources'], 'pending_method_trial':state['pending_method_trial'],
        'canonical_evaluations':[], 'canonical_evaluations_omitted':len(canonical_evaluations),
        'omitted_fields':[]}
    if len(pending_method_trials(state)) > 1:
        value['pending_method_trials_total'] = len(pending_method_trials(state))
    if state['transfer_review_pending']:
        value['transfer_review_pending'] = True
    while len(json.dumps(value,ensure_ascii=True, sort_keys=True)) > budget and len(value['available_input_receipt_ids']) > 1:
        value['available_input_receipt_ids'].pop(0)
        value['older_receipts_omitted'] += 1
    for key in ('pending_method_trial','tool_resources_total','known_llm_cost','scientific_steps_total'):
        if len(json.dumps(value,ensure_ascii=True, sort_keys=True)) > budget:
            value.pop(key)
            value['omitted_fields'].append(key)
    for evaluation in reversed(canonical_evaluations):
        candidate = {**value,'canonical_evaluations':value['canonical_evaluations']+[evaluation],
                     'canonical_evaluations_omitted':value['canonical_evaluations_omitted']-1}
        if not value['canonical_evaluations']:
            # The newest typed result is actionable evidence. Prefer its exact
            # ID/status over redundant accounting and pending-trial metadata,
            # which remain in the immutable checkpoint and receipts.
            for key in ('pending_method_trial','tool_resources_total','known_llm_cost','scientific_steps_total'):
                if len(json.dumps(candidate,ensure_ascii=True, sort_keys=True)) <= budget:
                    break
                if key in candidate:
                    candidate.pop(key)
                    candidate['omitted_fields'] = candidate['omitted_fields'] + [key]
            while len(json.dumps(candidate,ensure_ascii=True, sort_keys=True)) > budget and len(candidate['available_input_receipt_ids']) > 1:
                candidate['available_input_receipt_ids'] = candidate['available_input_receipt_ids'][1:]
                candidate['older_receipts_omitted'] += 1
        if len(json.dumps(candidate,ensure_ascii=True, sort_keys=True)) <= budget:
            value = candidate
    for key in TEXT_FIELDS:
        candidate = {**value,key:state[key]}
        if len(json.dumps(candidate,ensure_ascii=True, sort_keys=True)) <= budget:
            value = candidate
        else:
            value['omitted_fields'].append(key)
    return 'PERSISTENT RESEARCH DECISION JSON:\n' + json.dumps(value,ensure_ascii=True,separators=(',',':'), sort_keys=True)
