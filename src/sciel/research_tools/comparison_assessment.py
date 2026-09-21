"""Host comparison findings bound to immutable invocation and proposal evidence."""
from copy import deepcopy
import json

from ..methods.models import content_hash, require_hash
from .experiment_protocol import parse_protocol, result_assessment

SCHEMA = 'sciel-research-comparison-assessment/1'
CARD_SCHEMA = 'sciel-research-check/5'
PROBLEMS = ('comparison_results missing or not an array', 'invalid arm_id/seed',
    'duplicate arm/seed result', 'reported CPU missing, invalid or above declared cap',
    'best_result/saved_state/conditions missing', 'reported conditions differ',
    'arm/seed coverage differs')
SCOPE = 'Shape validation does not establish equal actual CPU, truthful measurements, or canonical task benefit.'


def unavailable(receipt_id):
    return dict(schema=SCHEMA, receipt_id=receipt_id, proposal_id=None,
                status='unavailable', assessment=None)


def validate(value, receipt_id):
    """Validate presentation shape; artifact authentication is separate."""
    if type(value) is not dict or set(value) != {'schema', 'receipt_id', 'proposal_id', 'status', 'assessment'}:
        return False
    if value['schema'] != SCHEMA or value['receipt_id'] != receipt_id:
        return False
    try:
        require_hash(receipt_id, 'comparison receipt')
        if value['proposal_id'] is not None:
            require_hash(value['proposal_id'], 'comparison proposal')
    except ValueError:
        return False
    if value['status'] == 'unavailable':
        return value['assessment'] is None
    if value['proposal_id'] is None:
        return False
    if value['status'] == 'not_comparison':
        return value['assessment'] is None
    a = value['assessment']
    if value['status'] != 'assessed' or type(a) is not dict or set(a) != {'schema','report_shape','problems','adherence','scope'}:
        return False
    problems = a['problems']
    return (a['schema'] == 'sciel-experimental-protocol/1'
        and a['adherence'] == 'tool_reported_unverified' and a['scope'] == SCOPE
        and type(problems) is list and all(type(p) is str and p in PROBLEMS for p in problems)
        and len(problems) == len(set(problems))
        and a['report_shape'] == ('invalid' if problems else 'complete'))


def from_record(record, receipt_id, read_proposal):
    """Recompute from exact saved declaration, never model-authored status."""
    value = unavailable(receipt_id)
    ident = record.get('experiment_proposal_id')
    if ident is None:
        return value  # Legacy/automatic stages do not invent a declaration.
    require_hash(ident, 'comparison proposal')
    proposal = read_proposal(ident)
    if any(proposal.get(key) != record.get(key) for key in ('action_id','cell_id','cell_revision_id')):
        raise ValueError('comparison proposal differs from invocation scope')
    raw = proposal['response']['text'].strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = raw.split('\n',1)[-1].rsplit('```',1)[0].strip()
    action = json.loads(raw)
    if action.get('action') not in ('create', 'use', 'continue'):
        raise ValueError('comparison proposal did not request an invocation')
    attempt = proposal.get('attempt')
    if type(attempt) is not int or attempt < 0:
        raise ValueError('comparison proposal has invalid attempt')
    tool_id = record['tool_id']
    if action['action'] == 'use' and action.get('tool_id') != tool_id:
        raise ValueError('comparison proposal refers to another tool')
    foreground = content_hash({'action_id':record['action_id'], 'attempt':attempt, 'tool_id':tool_id})
    job_key = content_hash({'action_id':record['action_id'], 'attempt':attempt, 'tool_id':tool_id,
                           'parent_job_id':action.get('job_id') if action['action'] == 'continue' else None})
    from .jobs import SCHEMA as JOB_SCHEMA
    background = content_hash({'schema':JOB_SCHEMA, 'job_key':job_key})
    if record['receipt']['invocation_id'] not in (foreground, background):
        raise ValueError('comparison proposal differs from exact invocation attempt')
    if action.get('experimental_protocol') is None:
        value['proposal_id'] = ident
        return value
    protocol = parse_protocol(action['experimental_protocol'])
    output = record['receipt'].get('output')
    observations = output.get('observations') if type(output) is dict else None
    assessment = result_assessment(observations, protocol)
    value.update(proposal_id=ident, status='assessed' if assessment is not None else 'not_comparison',
                 assessment=deepcopy(assessment))
    return value
