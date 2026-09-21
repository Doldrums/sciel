"""Declared, task-independent experiments; observations never prove protocol adherence.

The host enforces finite allocation and explicit uncertainty on known scope repeats.
Algorithms, control construction and measurement code remain model choices.
"""
from copy import deepcopy
import json
import math

from ..methods.models import content_hash, require_hash

SCHEMA = 'sciel-experimental-protocol/1'
TEXT_FIELDS = ('question_key', 'new_uncertainty', 'budget_reason', 'saved_state_plan')
KINDS = ('diagnostic', 'controlled_comparison', 'follow_up', 'closure')
BUDGETS = ('initial', 'maintain', 'reallocate', 'escalate', 'none')
CITATION_GUIDANCE = (
    'prior_receipt_ids: only listed lowercase SHA256 (64 hex) tool receipt_id or '
    'canonical_evaluations[].artifact_id; [] if none. Never use/hash ordinary '
    'evaluation_id (evaluation:...), program IDs or source hashes. '
    'Canonical IDs are evidence, not stage inputs/benefit. ')


def response_property():
    properties = {key:{'type':'string'} for key in (*TEXT_FIELDS, 'conditions_json')}
    properties['conditions_json']['description'] = (
        'JSON object declaring common executed conditions. Each comparison row conditions object is compared '
        'by JSON equality: booleans differ from numbers; finite 1 equals 1.0. Keep unchanged keys; report changes '
        'truthfully, and put extra diagnostics outside conditions.')
    properties.update(kind={'type':'string','enum':list(KINDS)},
        budget_action={'type':'string','enum':list(BUDGETS),
            'description':'Allocation to execute in this action, not advice. reallocate/escalate require '
                          'create/use or an eligible continue with complete executable inputs. '
                          'finish/skip close with kind=closure and budget_action=none.'},
        prior_receipt_ids={'type':'array',
            'items':{'type':'string','pattern':'^[0-9a-f]{64}$'},
            'description':CITATION_GUIDANCE},
        arms={'type':'array','items':{'type':'object','additionalProperties':False,
            'properties':{'id':{'type':'string'},'method':{'type':'string'}},'required':['id','method']}},
        matched_seeds={'type':'array','items':{'type':'integer'}},
        cpu_seconds_per_arm={'type':['number','null'],
            'description':'CPU cap for EACH arm AND matched seed. Total executor allocation must cover '
                          'this value multiplied by the number of arms and the number of matched seeds.'})
    return {'type':'object','additionalProperties':False,'properties':properties,'required':list(properties),
        'description':'Declared experiment design, not measured adherence. Compare >=2 arms on >=2 matched seeds with the same CPU cap and conditions; save best result/state/seed. A known-scope repeat must cite prior receipt and new uncertainty. Selecting another test requires its executable action; closing without another test uses closure/none.'}


def parse_protocol(value):
    fields = {*TEXT_FIELDS,'conditions_json','kind','budget_action','prior_receipt_ids','arms','matched_seeds','cpu_seconds_per_arm'}
    if type(value) is not dict or set(value) != fields:
        raise ValueError('experimental_protocol requires the complete declared experiment design')
    result=deepcopy(value)
    if value['kind'] not in KINDS or value['budget_action'] not in BUDGETS:
        raise ValueError('unsupported experiment kind or budget allocation')
    for key in (*TEXT_FIELDS,'conditions_json'):
        if type(value[key]) is not str or len(value[key])>4000:
            raise ValueError('experimental_protocol.'+key+' must be a bounded string')
    if not value['question_key'].strip():
        raise ValueError('question_key must identify the persistent testable question')
    refs=value['prior_receipt_ids']
    if type(refs) is not list or len(refs)>16 or len(set(refs))!=len(refs):
        raise ValueError('prior_receipt_ids must be at most 16 distinct immutable receipts')
    for ident in refs:
        try:
            require_hash(ident,'prior experiment receipt')
        except ValueError as error:
            raise ValueError(str(error) + '. ' + CITATION_GUIDANCE) from None
    seeds=value['matched_seeds']; arms=value['arms']
    if type(seeds) is not list or len(seeds)>64 or any(type(seed) is not int for seed in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('matched_seeds must be distinct integers, at most 64')
    if type(arms) is not list or len(arms)>16:
        raise ValueError('experiment arms must be a bounded list')
    for arm in arms:
        if type(arm) is not dict or set(arm)!={'id','method'} or any(type(v) is not str or not v.strip() or len(v)>2000 for v in arm.values()):
            raise ValueError('each arm requires a distinct id and concrete method')
    if len({arm['id'] for arm in arms})!=len(arms):
        raise ValueError('experiment arm ids must be unique')
    cpu=value['cpu_seconds_per_arm']
    if cpu is not None and (type(cpu) not in (int,float) or not math.isfinite(cpu) or cpu<=0):
        raise ValueError('declared CPU per arm must be finite and positive')
    def finite_float(text):
        value=float(text)
        if not math.isfinite(value):raise ValueError('nonfinite condition')
        return value
    def unique_pairs(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('duplicate condition key')
            value[key]=item
        return value
    try:
        conditions=json.loads(value['conditions_json'] or '{}',parse_float=finite_float,object_pairs_hook=unique_pairs,
            parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite condition')))
    except (ValueError,TypeError,RecursionError):
        raise ValueError('conditions_json must encode a finite object') from None
    if type(conditions) is not dict:
        raise ValueError('conditions_json must encode an object')
    if value['kind']=='controlled_comparison':
        if len(arms)<2 or len(seeds)<2 or cpu is None or not conditions or not value['saved_state_plan'].strip():
            raise ValueError('controlled comparison requires >=2 arms, >=2 matched seeds, one equal positive CPU cap, explicit common conditions and a saved best/state/seed plan')
    return result


def proposal_protocol(proposal):
    """Read only immutable proposal text; no code import or interpretation."""
    raw=proposal['response']['text'].strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw=raw.split('\n',1)[-1].rsplit('```',1)[0].strip()
    return parse_protocol(json.loads(raw).get('experimental_protocol'))


def normalized_question(text):
    return ' '.join(str(text).casefold().split())


def scope_key(source_sha256, question):
    return content_hash({'candidate_source_sha256':source_sha256,'question':normalized_question(question)})


def validate_dispatch(action, plan, selected_source_sha256, *, default_limits):
    """Validate before registration or execution; no I/O or accounting changes.

    Raises inside the normal charged repair path. Matching is exact normalized
    question/key + source, not a semantic claim about paraphrases or algorithms.
    """
    protocol=parse_protocol(action.get('experimental_protocol'))
    executable = action['action'] in ('create', 'use', 'continue')
    if protocol['budget_action'] in ('reallocate', 'escalate'):
        if not executable:
            raise ValueError('budget_action=' + protocol['budget_action'] +
                ' selects executable research, but action=' + action['action'] +
                ' cannot perform it. Return create/use with the complete next experiment '
                '(or continue for an eligible saved job), its question, arguments and limits. '
                'If no further experiment is selected, finish with kind=closure and '
                'budget_action=none, explaining the evidence or actual blocker. '
                'next_decision text does not schedule work.')
        if not protocol['new_uncertainty'].strip() or not protocol['budget_reason'].strip():
            raise ValueError('reallocate/escalate requires new_uncertainty and budget_reason '
                             'for the experiment being executed now')
    if action['action'] in ('finish', 'skip') and (
            protocol['kind'] != 'closure' or protocol['budget_action'] != 'none'):
        raise ValueError('finish/skip require kind=closure and budget_action=none. '
                         'A selected further experiment must be a complete create/use '
                         'or eligible continue action, not next_decision advice.')
    if executable and protocol['kind'] == 'closure':
        raise ValueError('an executable experiment requires diagnostic, controlled_comparison '
                         'or follow_up kind; closure does not execute a test')
    catalog=(plan.get('shared_observations') or {}).get('check_index',[])
    known={row['receipt_id']:row for row in catalog}
    canonical = {row['artifact_id'] for row in
                 (plan.get('shared_observations') or {}).get('canonical_evaluation_index', [])}
    if not set(protocol['prior_receipt_ids']) <= set(known) | canonical:
        raise ValueError('prior_receipt_ids must name authenticated tool observations or canonical '
                         'candidate evidence available in this research plan/opportunity. ' + CITATION_GUIDANCE)
    if action['action'] in ('finish','skip','poll','cancel','submit_candidate'):
        return {'schema':SCHEMA,'adherence':'not_applicable','repeated_receipt_ids':[],'declared':protocol}
    keys={scope_key(selected_source_sha256,action['question']),scope_key(selected_source_sha256,protocol['question_key'])}
    repeats=[row for row in catalog if row.get('status','success')=='success' and keys.intersection(row['scope_keys'])]
    matching_refs=[{key:row.get(key) for key in ('receipt_id','conclusion')} for row in repeats[:2]]
    known_suffix=' Matching saved receipts: '+json.dumps(matching_refs,separators=(',',':'))
    if repeats:
        prior_ids={row['receipt_id'] for row in repeats}
        if not protocol['new_uncertainty'].strip() or not prior_ids.intersection(protocol['prior_receipt_ids']):
            raise ValueError('known-scope repeated check requires new_uncertainty and matching prior_receipt_id; renaming a tool is insufficient.'+known_suffix)
    refuted=[row for row in repeats if row.get('conclusion')=='refuted']
    if refuted and (protocol['budget_action'] not in ('reallocate','escalate') or not protocol['budget_reason'].strip()):
        raise ValueError('refuted same-scope check cannot retain its allocation: reallocate or justify escalation.'+known_suffix)
    limits=action.get('limits') or default_limits
    declared_cpu=None
    if protocol['kind']=='controlled_comparison':
        declared_cpu=protocol['cpu_seconds_per_arm']*len(protocol['arms'])*len(protocol['matched_seeds'])
        if declared_cpu>limits['cpu_seconds']:
            raise ValueError('requested executor CPU budget is smaller than arms × matched seeds × declared equal CPU cap: '
                f'requested {limits["cpu_seconds"]:g} CPU seconds; required {declared_cpu:g} '
                f'({len(protocol["arms"])} arms × {len(protocol["matched_seeds"])} seeds × '
                f'{protocol["cpu_seconds_per_arm"]:g} CPU seconds per arm/seed). '
                'Correct the declared design or request an allocation that covers it within the available limits.')
    return {'schema':SCHEMA,'adherence':'declared_not_verified','repeated_receipt_ids':[row['receipt_id'] for row in repeats],
        'negative_allocation_change_declared':bool(refuted),'declared_cpu_seconds':declared_cpu,
        'accounting_scope':'Executor total CPU is host-measured; any per-arm CPU is tool-reported and unverified.',
        'declared':protocol}


def _json_equal(left, right):
    """Compare JSON values without Python's bool/number aliasing.

    Finite JSON numbers compare by value (1 equals 1.0, -0.0 equals 0),
    without lossy float conversion. Object order is irrelevant; array order
    is significant. Non-JSON values and nonfinite floats never compare equal.
    """
    lt,rt=type(left),type(right)
    if lt in (int,float) and rt in (int,float):
        return ((lt is int or math.isfinite(left)) and
                (rt is int or math.isfinite(right)) and left==right)
    if lt is not rt:
        return False
    if lt is dict:
        return (all(type(key) is str for key in left) and
                all(type(key) is str for key in right) and left.keys()==right.keys() and
                all(_json_equal(left[key],right[key]) for key in left))
    if lt is list:
        return len(left)==len(right) and all(_json_equal(a,b) for a,b in zip(left,right))
    return lt in (str,bool,type(None)) and left==right


def result_assessment_from_output(output, protocol):
    """Assess even failed raw JSON without altering its immutable receipt.

    The executor retains schema-invalid arrays, strings and scalar outputs.
    They have no observation mapping; a comparison must remain invalid rather
    than raising in the host's repair, collection or restoration path.
    """
    observations = output.get('observations') if type(output) is dict else None
    return result_assessment(observations, protocol)


def result_assessment(observations, protocol):
    """Check saved report shape only; never upgrade model/tool claims to trust."""
    if protocol is None or protocol.get('kind')!='controlled_comparison':
        return None
    expected={(arm['id'],seed) for arm in protocol['arms'] for seed in protocol['matched_seeds']}
    rows=observations.get('comparison_results') if type(observations) is dict else None
    problems=[]; actual=set()
    if type(rows) is not list:
        problems.append('comparison_results missing or not an array')
    else:
        for row in rows:
            if type(row) is not dict or type(row.get('arm_id')) is not str or type(row.get('seed')) is not int:
                problems.append('invalid arm_id/seed');continue
            key=(row['arm_id'],row['seed'])
            if key in actual:problems.append('duplicate arm/seed result')
            actual.add(key)
            cpu=row.get('cpu_seconds')
            if type(cpu) not in (int,float) or not math.isfinite(cpu) or not 0<=cpu<=protocol['cpu_seconds_per_arm']:
                problems.append('reported CPU missing, invalid or above declared cap')
            if not {'best_result','saved_state','conditions'} <= set(row):
                problems.append('best_result/saved_state/conditions missing')
            elif not _json_equal(row['conditions'],json.loads(protocol['conditions_json'] or '{}')):
                problems.append('reported conditions differ')
        if actual!=expected:problems.append('arm/seed coverage differs')
    return {'schema':SCHEMA,'report_shape':'invalid' if problems else 'complete',
        'problems':list(dict.fromkeys(problems)), 'adherence':'tool_reported_unverified',
        'scope':'Shape validation does not establish equal actual CPU, truthful measurements, or canonical task benefit.'}


def guidance(*, compact=False):
    if compact:
        return ('EXPERIMENT PROTOCOL: prior_receipt_ids: listed lowercase SHA256 tool receipt_id/'
            'canonical_evaluations[].artifact_id; [] if none. Never use/hash evaluation_id '
            '(evaluation:...), program IDs/source hashes. Canonical IDs: no stage inputs. '
            'Repeat: receipt+new_uncertainty; refuted:no maintain. '
            'reallocate/escalate: new_uncertainty,budget_reason; execute create/use/eligible continue. '
            'finish/skip:kind=closure,budget_action=none. '
            'Comparison: >=2 arms×>=2 matched seeds; equal CPU/conditions; budget all. '
            'observations.comparison_results: '
            '[{arm_id,seed,cpu_seconds,best_result,saved_state,conditions}]. '
            'conditions=conditions_json, actual changes; diagnostics outside. '
            'CPU incl. overshoot,no tolerance/round/clamp; arms unverified,total host-measured. '
            'Adherence/benefit need independent evidence.\n')
    return ('EXPERIMENT PROTOCOL: declare question_key, prior_receipt_ids, new_uncertainty and budget_reason. '
        + CITATION_GUIDANCE +
        'Known candidate/question repeats need a matching receipt and new uncertainty. '
        'For a refuted-scope repeat: reallocate to another test or justify escalate; maintain is rejected. '
        'reallocate/escalate require new_uncertainty, budget_reason and executable create/use or eligible continue now. '
        'finish/skip: kind=closure, budget_action=none; finish with reallocate/escalate is rejected for repair. Advice never schedules work. '
        'Controlled comparison: >=2 arms × >=2 matched seeds, equal CPU caps/conditions; executor CPU covers every arm/seed. '
        'Save best/state/seed/conditions. observations.comparison_results rows: '
        '{arm_id,seed,cpu_seconds,best_result,saved_state,conditions}. Arm CPU is tool-reported, total is host-measured. '
        'Row conditions must equal parsed conditions_json: retain unchanged keys, report actual changes truthfully; '
        'extra diagnostics go outside conditions. Never echo unexecuted conditions. '
        'Report actual cpu_seconds including soft-deadline overshoot; cap check has no tolerance. '
        'Never round down/clamp CPU. No adherence/benefit claim without independent evidence; choose your algorithms.\n')
