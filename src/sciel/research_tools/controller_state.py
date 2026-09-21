"""Restore research state and authenticate committed followup evidence."""
from __future__ import annotations

from copy import copy, deepcopy
import math
from ..engine.research_targets import input_binding
from ..methods.models import canonical_bytes, content_hash, require_hash
from .config import effective_limits
from .result_contract import assess_receipt
from .followups import build_followup
from .research_steps import pending_method_trials
from .tool_artifacts import _sha
from .tool_protocol import CONTRACT

def load_state(tools, state, *, expected_run_id=None):
    if (type(state) is not dict or set(state) != set(tools._state)
            or state.get('schema') != CONTRACT
            or state.get('identity_sha256') != content_hash(tools._identity)):
        raise ValueError('research tool checkpoint identity or shape differs')
    run_id = state['run_id']
    if (run_id is not None and (type(run_id) is not str or not run_id.strip())):
        raise ValueError('invalid research tool run identity')
    if expected_run_id is not None and run_id not in (None, expected_run_id):
        raise ValueError('research tool checkpoint belongs to another run')
    if state['committed_episodes'] and run_id is None:
        raise ValueError('committed research tool state lacks run identity')
    for key in ('registered', 'published', 'committed_episodes'):
        values = state[key]
        if type(values) is not list or len(values) != len(set(values)):
            raise ValueError('invalid research tool checkpoint ' + key)
        for ident in values:
            require_hash(ident, key)
    if not set(state['published']) <= set(state['registered']):
        raise ValueError('published research tools are absent from registered state')
    for ident in state['registered']:
        tools.library.get(ident)  # Authenticate every visible immutable version.
    if (type(state['opportunities']) is not dict or any(
            type(key) is not str or not key.isdigit() or type(value) is not int or value < 0
            for key, value in state['opportunities'].items())):
        raise ValueError('invalid research tool opportunity cursors')
    anchors = state['last_episode_opportunity']
    if (type(anchors) is not dict or any(
            type(key) is not str or key not in state['opportunities']
            or type(value) is not int or not 0 <= value < state['opportunities'][key]
            for key, value in anchors.items())):
        raise ValueError('invalid research tool episode cadence anchors')
    if type(state['charges']) is not dict:
        raise ValueError('invalid research tool charges')
    for key, value in state['charges'].items():
        require_hash(key, 'charge')
        if type(value) is dict:
            from types import SimpleNamespace
            from ..engine.llm.client import response_accounting
            if (set(value) != {'known_usd', 'unpriced_calls', 'cost_provenance'}
                    or type(value['known_usd']) not in (int, float) or value['known_usd'] != 0
                    or type(value['unpriced_calls']) is not int or value['unpriced_calls'] != 1
                    or response_accounting(SimpleNamespace(cost=None, finish_reason='stop',
                        cost_provenance=value['cost_provenance']))['unpriced_calls'] != 1):
                raise ValueError('invalid research tool unpriced charge')
        elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('invalid research tool known cost')
    proactive = state['last_proactive_iteration']
    if proactive is not None and (type(proactive) is not int or proactive < 0
            or tools.config['proactive_every_attempts'] is None):
        raise ValueError('invalid proactive research reservation cursor')
    if type(state['uses']) is not dict or not set(state['uses']) <= set(state['registered']):
        raise ValueError('invalid research tool usage counters')
    for islands in state['uses'].values():
        if type(islands) is not dict or any(not key.isdigit() or type(n) is not int or n < 0
                                          for key, n in islands.items()):
            raise ValueError('invalid research tool island usage')
    tools._validate_job_state(state)
    canonical_bytes(state)
    restored = copy(tools.lifecycle)
    restored.run_id = run_id
    restored.load_state(state['lifecycle'])
    tools._validate_continuations(state, restored)
    tools._validate_sharing(state, restored)
    from .canonical_evidence import authenticate as authenticate_canonical
    canonical_records = {}
    for cell in restored.cells.all_cells():
        head = cell['head_revision']
        while head is not None:
            revision = restored.cells._read(restored.cells.revisions, head)
            for ref in revision['evidence']:
                if ref['kind'] != 'candidate_evaluation' or ref['id'] in canonical_records:
                    continue
                saved, _ = authenticate_canonical(restored, tools.library, ref['id'],
                    cell_id=cell['id'], run_id=run_id, task_identity=tools.task_identity)
                if saved['provenance']['action_id'] not in state['committed_episodes']:
                    raise ValueError('canonical evidence action is not a committed research episode')
                canonical_records[ref['id']] = saved
            head = revision['previous_revision']
    for cell_id,research_state in state['lifecycle']['research_state'].items():
        tools._validate_transfer_progress(cell_id, research_state, job_state=state)
        for pending_trial in pending_method_trials(research_state):
            if pending_trial.get('blocked_reason'):
                pipeline = tools._method_pipeline(pending_trial, cell_id, job_state=state)
                if (len(pipeline) != pending_trial['pipeline_stages']
                        or pending_trial['scientific_step_limit'] != tools.config['max_scientific_steps']):
                    raise ValueError('blocked method trial differs from its saved pipeline or opportunity cap')
        for step in research_state['steps']:
            ident = step.get('canonical_evaluation_id')
            if ident is None:
                continue
            saved = canonical_records[ident]
            provenance = saved['provenance']
            source = saved['source']
            invocation = tools._read_artifact(provenance['receipt_id'],'invocation')
            if (provenance.get('cell_id') != cell_id
                    or invocation['cell_id'] != cell_id
                    or invocation['result']['status'] != 'success'
                    or saved['source'] != invocation['receipt']['output']['observations'].get('candidate_source')
                    or provenance['candidate_source_sha256'] != _sha(source)
                    or provenance['target_binding'] != invocation['target_binding']
                    or provenance['tool_id'] != invocation['tool_id']
                    or step.get('source_sha256') != _sha(source)
                    or step.get('candidate_receipt_id') != provenance['receipt_id']
                    or step.get('canonical_action_id') != provenance['action_id']
                    or step['status'] != saved['evaluation'].get('status','evaluated')):
                raise ValueError('research candidate decision differs from its immutable source/evaluation')
    methods = state['lifecycle']['methods']
    released = {ident for ident, method in methods.items() if method['stage'] == 'released'}
    if (set(state['published']) != released
            or not set(methods) <= set(state['registered'])):
        raise ValueError('research tool visibility differs from its verified cell lifecycle')
    followups = tools._new_followup_ledger()
    followups.load_state(state['followups'])
    for record in state['followups']['records']:
        if any(card['source']['tool_id'] not in state['registered']
               for card in record['exposures']['cards'] if card['source']['tool_id']):
            raise ValueError('method followup refers to an unregistered version')
    tools.lifecycle = restored
    tools.followups = followups
    tools._state = deepcopy(state)


def _authenticate_method_exposure(tools, card):
    source = card['source']
    saved = tools._read_artifact(source['receipt_id'], 'invocation')
    receipt, inputs = saved['receipt'], saved['inputs']
    bundle = tools.library.get(source['tool_id'])
    # Retained cards carry the post-commit cell revision; the invocation
    # carries the earlier planning revision. Both are immutable cell history.
    revision = tools.lifecycle.cells._read(tools.lifecycle.cells.revisions, card['revision_id'])
    binding = input_binding(inputs)
    expected = {'tool_id': saved['tool_id'], 'invocation_id': receipt['invocation_id'],
                'receipt_id': source['receipt_id'], **binding}
    if (source != expected or saved['target_binding'] != binding
            or card['cell_id'] != saved['cell_id']
            or revision['cell_id'] != card['cell_id']
            or saved['result'] != assess_receipt(receipt)
            or card['observation_status'] != saved['result']['status']
            or receipt['method_id'] != bundle['id']
            or receipt['plan_id'] != saved['action_id']
            or receipt['reservation_id'] != receipt['invocation_id']
            or receipt['source_sha256'] != content_hash(bundle['files'])
            or receipt['runtime_sha256'] != content_hash(tools.library.identity['executor'])
            or receipt['input_sha256'] != content_hash(inputs)
            or effective_limits(tools.config, receipt['limits']) != receipt['limits']
            or receipt['dependency_ids'] != sorted(row['id'] for row in tools.library.dependency_closure(bundle['id']))
            or receipt['receipt_sha256'] != content_hash(
                {key: value for key, value in receipt.items() if key != 'receipt_sha256'})):
        raise ValueError('method exposure differs from its saved invocation')


def _authenticate_followup_outcome(tools, record):
    saved = tools._read_artifact(record['evidence_artifact_id'], 'method_followup_evidence')
    expected = build_followup(saved['research_record'],
        child_program_id=saved['child_program_id'], link_kind=saved['link_kind'],
        observed_record=saved['observed_record'], evidence_artifact_id=record['evidence_artifact_id'])
    if expected != record:
        raise ValueError('method followup differs from its saved research outcome')


def record_followup(tools, record, *, child_program_id=None, link_kind='mutation', observed_record=None):
    # Validate before writing. The detached journal record survives bounded
    # KB retention; the admission binding is saved alongside it, not inferred
    # from a later mutable population or from a method's own output.
    row = build_followup(record, child_program_id=child_program_id,
        link_kind=link_kind, observed_record=observed_record)
    if row is None:
        return False
    for card in row['exposures']['cards']:
        if card['source']['tool_id'] is not None:
            if card['source']['tool_id'] not in tools._state['registered']:
                raise ValueError('method followup refers to an unregistered version')
            tools._authenticate_method_exposure(card)
    artifact_id = tools._artifact('method_followup_evidence', {
        'research_record': deepcopy(record), 'child_program_id': child_program_id,
        'link_kind': link_kind, 'observed_record': deepcopy(observed_record)})
    row = build_followup(record, child_program_id=child_program_id, link_kind=link_kind,
        observed_record=observed_record, evidence_artifact_id=artifact_id)
    added = tools.followups.add(row)
    if added:
        tools.emit(schema='sciel-research-tool-event/1', phase='method_followup',
            experiment_id=row['experiment_id'], island_id=row['island_id'],
            iteration=row['iteration'], followup_sha256=row['sha256'],
            evidence_artifact_id=artifact_id, child_program_id=child_program_id,
            tool_ids=list(dict.fromkeys(card['source']['tool_id'] for card in row['exposures']['cards']
                                       if card['source']['tool_id'])),
            dispatch=row['generation']['dispatch'], outcome_status=row['outcome']['status'],
            relation=row['relation'], model_use='unknown', causal_benefit='unknown')
    return added
