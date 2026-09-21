"""Typed canonical candidate evidence, never a tool input/comparison receipt."""
from copy import deepcopy
import hashlib
import json

from ..methods.models import require_hash
from .canonical_diagnostics import validate as validate_diagnostics, add_summary


def authenticate(lifecycle, library, ident, *, cell_id=None, run_id=None, task_identity=None):
    saved = lifecycle._artifact(ident, 'candidate_evaluation', library)
    if set(saved) != {'source', 'provenance', 'evaluation'}:
        raise ValueError('invalid canonical candidate evidence')
    provenance, value = saved['provenance'], saved['evaluation']
    source = saved['source']
    if type(provenance) is not dict or type(value) is not dict or type(source) is not str:
        raise ValueError('invalid canonical candidate evidence fields')
    invocation = lifecycle._receipt(provenance['receipt_id'], provenance['tool_id'], library)
    origin_run = invocation['inputs']['research_targets']['run_id']
    origin_task = invocation['inputs']['task']
    digest = hashlib.sha256(source.encode('utf-8', errors='surrogatepass')).hexdigest()
    require_hash(provenance.get('action_id'), 'canonical evaluation action')
    if (type(provenance.get('iteration')) is not int or provenance['iteration'] < 1
            or invocation['result']['status'] != 'success'
            or provenance.get('cell_id') != invocation['cell_id']
            or (cell_id is not None and provenance['cell_id'] != cell_id)
            or provenance.get('run_id') != origin_run
            or provenance.get('task_identity') != origin_task
            or (run_id is not None and origin_run != run_id)
            or (task_identity is not None and origin_task != task_identity)
            or source != invocation['receipt']['output']['observations'].get('candidate_source')
            or provenance.get('candidate_source_sha256') != digest
            or provenance.get('target_binding') != invocation['target_binding']
            or provenance.get('invocation_id') != invocation['receipt']['invocation_id']
            or value.get('status') not in ('rejected', 'evaluated', 'eval_failed', 'output_repeat_rejected')
            or ('source_sha256' in value and value['source_sha256'] != digest)):
        raise ValueError('canonical evidence differs from its run/task/cell/source binding')
    evaluation_id = value.get('evaluation_id')
    if evaluation_id is not None and (type(evaluation_id) is not str
            or not evaluation_id.startswith('evaluation:' + origin_run + ':')):
        raise ValueError('canonical evaluation identity belongs to another run')
    if 'failure_diagnostics' in value:
        if value['status'] != 'eval_failed':
            raise ValueError('canonical failure diagnostics attached to nonfailure')
        validate_diagnostics(value['failure_diagnostics'], source_sha256=digest,
            evaluation_id=evaluation_id, artifacts_sha256=value.get('artifacts_sha256'),
            library=library, run_id=origin_run,
            task_hash=(origin_task or {}).get('task_hash'))
    return saved, invocation


def summary(ident, saved, invocation):
    value, provenance = saved['evaluation'], saved['provenance']
    result = {key: deepcopy(value[key]) for key in (
        'status', 'evaluation_id', 'score', 'reached', 'confirmed', 'timing_seconds',
        'timing_scope', 'efficacy') if key in value}
    result.update(artifact_id=ident, kind='candidate_evaluation', cell_id=provenance['cell_id'],
        island_id=invocation['island_id'], run_id=provenance['run_id'],
        task=deepcopy(provenance['task_identity']), research_target_id=provenance['target_binding']['candidate_id'],
        source_sha256=provenance['candidate_source_sha256'],
        tool_receipt_id=provenance['receipt_id'], action_id=provenance['action_id'],
        iteration=provenance['iteration'],
        scope='Canonical candidate result; citable evidence, not tool stage input or controlled method benefit.')
    if len(json.dumps(value.get('cases'), ensure_ascii=True)) <= 600:
        result['cases'] = deepcopy(value.get('cases'))
    elif 'cases' in value:
        result['cases_omitted'] = True
    return add_summary(value, result)


def cited_ids(text, allowed):
    """Snapshot only typed citations actually present in a model response.

    This does not accept the proposal; parse_action and validate_dispatch still
    validate the complete wire, including unknown or malformed references.
    """
    try:
        raw = text.strip()
        if raw.startswith('```') and raw.endswith('```'):
            raw = raw.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
        refs = json.loads(raw)['experimental_protocol']['prior_receipt_ids']
        if type(refs) is not list:
            return []
        return list(dict.fromkeys(ident for ident in refs if type(ident) is str and ident in allowed))
    except (ValueError, TypeError, KeyError, RecursionError):
        return []


def authenticate_citations(proposal, allowed):
    """A saved metadata array must exactly reflect the model's typed citations."""
    expected = cited_ids(proposal['response']['text'], set(allowed))
    if proposal.get('canonical_evaluation_ids', []) != expected:
        raise ValueError('canonical citation metadata differs from the saved proposal response')
    return expected
