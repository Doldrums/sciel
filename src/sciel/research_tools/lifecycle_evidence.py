"""Authenticate persisted cell, method and invocation history before restoration."""

from copy import copy, deepcopy
import math
from ..methods.models import content_hash, require_hash
from .artifact_reading import read_json
from .config import effective_limits, library_limits
from .library import ToolLibrary
from .result_contract import assess_receipt, replay_evidence
from .replay_feedback import build_feedback, is_projection
from ..engine.research_cells_context import _card
from ..engine.research_targets import input_binding
from .research_steps import validate_state
from .decision_projection import authenticate_projection, NO_MEASUREMENT_QUESTION
from .tool_protocol import LIFECYCLE_SCHEMA as SCHEMA

def _artifact(lifecycle, ident, kind, library):
    """Authenticate immutable evidence without executing its contents."""
    require_hash(ident, 'receipt artifact')
    target = library.path / 'artifacts' / (ident + '.json')
    library._refuse_links(target)
    limit = (max(4 * 1024 * 1024, 6 * (lifecycle.config['max_prompt_chars'] + lifecycle.config['max_response_chars']))
             if kind == 'proposal' else 2 * 1024 * 1024 + 6 * lifecycle.config['limits']['max_output_bytes']
             + 6 * lifecycle.config['limits']['max_stderr_bytes'] + 65536)
    body = read_json(target, limit)
    if (type(body) is not dict or content_hash(body) != ident or body.get('schema') != 'sciel-research-tool-artifact/1'
            or body.get('kind') != kind or type(body.get('payload')) is not dict):
        raise ValueError('invalid immutable invocation artifact')
    return body['payload']


def _receipt(lifecycle, ident, tool_id, library):
    """Authenticate an already saved observation, without invoking anything."""
    payload = lifecycle._artifact(ident, 'invocation', library)
    receipt = payload.get('receipt')
    bundle = library.get(tool_id)
    if (type(receipt) is not dict or payload.get('tool_id') != tool_id
            or type(payload.get('inputs')) is not dict
            or receipt.get('method_id') != tool_id
            or receipt.get('plan_id') != payload.get('action_id')
            or receipt.get('reservation_id') != receipt.get('invocation_id')
            or receipt.get('source_sha256') != content_hash(bundle['files'])
            or receipt.get('input_sha256') != content_hash(payload['inputs'])
            or effective_limits(lifecycle.config, receipt.get('limits')) != receipt.get('limits')
            or receipt.get('dependency_ids') != sorted(row['id'] for row in library.dependency_closure(tool_id))
            or receipt.get('receipt_sha256') != content_hash(
                {key: value for key, value in receipt.items() if key != 'receipt_sha256'})):
        raise ValueError('unbound immutable invocation receipt')
    for key in ('plan_id', 'invocation_id', 'runtime_sha256'):
        require_hash(receipt.get(key), key)
    if type(payload.get('island_id')) is not int or payload['island_id'] < 0:
        raise ValueError('invalid invocation island')
    if receipt.get('status') not in ('ok', 'failed') or receipt.get('dispatch_status') not in ('invoked', 'not_dispatched'):
        raise ValueError('unresolved invocation cannot support committed cell state')
    resources = receipt.get('resources')
    if type(resources) is not dict or any(type(resources.get(key)) not in (int, float)
            or not math.isfinite(resources[key]) or resources[key] < 0
            for key in ('cpu_seconds', 'wall_seconds')):
        raise ValueError('invocation has unresolved resource accounting')
    if receipt['status'] == 'ok' and (receipt['dispatch_status'] != 'invoked'
            or type(receipt.get('output')) is not dict
            or receipt.get('output_sha256') != content_hash(receipt.get('output'))):
        raise ValueError('successful invocation lacks bound output')
    if (payload.get('result') != assess_receipt(receipt)
            or payload.get('replay_policy') != bundle['manifest']['protocol']['replay_policy']):
        raise ValueError('invocation outcome or replay policy differs from its immutable evidence')
    binding = input_binding(payload['inputs'])
    if payload.get('target_binding') != binding or payload.get('input_program_id') != binding['candidate_id']:
        raise ValueError('invocation input source binding differs')
    from .comparison_assessment import from_record
    from_record(payload, ident, lambda proposal_id: lifecycle._artifact(proposal_id, 'proposal', library))
    return payload


def load_state(lifecycle, state):
    if (type(state) is not dict or set(state) != {'schema','affinity','methods','cards','cells','research_state'}
            or state.get('schema') != SCHEMA):
        raise ValueError('invalid cell-method lifecycle state')
    restored = copy(lifecycle.cells)
    restored.load_state(state['cells'])
    cells = {cell['id']: cell for cell in restored.all_cells()}
    known = set(cells)
    if type(state['research_state']) is not dict or not set(state['research_state']) <= known:
        raise ValueError('research decision state has no retained cell')
    for value in state['research_state'].values():
        validate_state(value)
    if type(state['affinity']) is not dict or type(state['methods']) is not dict or type(state['cards']) is not dict:
        raise ValueError('invalid cell-method lifecycle indexes')
    for island, ids in state['affinity'].items():
        if (type(island) is not str or not island.isdigit() or type(ids) is not list
                or any(type(ident) is not str for ident in ids)
                or len(ids) != len(set(ids)) or not set(ids) <= known
                or any(island not in map(str, cells[ident]['affinities']) for ident in ids)):
            raise ValueError('invalid research cell affinity')
    # The immutable source reader does not initialize an executor or Docker.
    library = ToolLibrary(lifecycle.run_dir, {'image_id': lifecycle.config['image_id']}, library_limits(lifecycle.config))
    history, receipts = {}, {}
    for cell_id, cell in cells.items():
        chain, head = {}, cell['head_revision']
        while head is not None:
            revision = restored._read(restored.revisions, head)
            chain[head] = revision
            head = revision['previous_revision']
        history[cell_id] = chain

    def observed(receipt_id, tool_id, cell_id):
        require_hash(receipt_id, 'method receipt')
        key = (receipt_id, tool_id)
        if key not in receipts:
            receipts[key] = lifecycle._receipt(receipt_id, tool_id, library)
        record = receipts[key]
        if (record.get('cell_id') != cell_id
                or record.get('cell_revision_id') not in history[cell_id]
                or record.get('stage') not in ('research', 'independent_replay', 'adopt_trial', 'use_again')
                or type(record.get('question')) is not str or not record['question'].strip()
                or len(record['question']) > 600
                or record.get('input_program_id') != record['target_binding']['candidate_id']):
            raise ValueError('invocation is not bound to its research cell and question')
        if str(record['island_id']) not in map(str, cells[cell_id]['affinities']):
            raise ValueError('method receipt belongs to another cell affinity')
        if not any({'kind': 'tool_invocation', 'id': receipt_id, 'tool_id': tool_id} in row['evidence']
                   for row in history[cell_id].values()):
            raise ValueError('method receipt absent from committed cell history')
        return record

    # Canonical candidate records are typed evidence, not invocation
    # receipts. Only committed revision references grant later visibility.
    from .canonical_evidence import authenticate as authenticate_canonical
    canonical = {}
    for cell_id, chain in history.items():
        for revision in chain.values():
            for ref in revision['evidence']:
                if ref['kind'] != 'candidate_evaluation':
                    continue
                saved, invocation = authenticate_canonical(lifecycle, library, ref['id'],
                    cell_id=cell_id, run_id=lifecycle.run_id)
                observed(saved['provenance']['receipt_id'], invocation['tool_id'], cell_id)
                canonical[ref['id']] = saved

    from .canonical_evidence import authenticate_citations
    for cell_id, chain in history.items():
        for revision in chain.values():
            for ref in revision['evidence']:
                if ref['kind'] != 'research_decision':
                    continue
                proposal = lifecycle._artifact(ref['id'], 'proposal', library)
                if proposal.get('cell_id') != cell_id or proposal.get('cell_revision_id') not in chain:
                    raise ValueError('terminal canonical citation lacks committed proposal provenance')
                authenticate_citations(proposal, canonical)

    for cell_id, research in state['research_state'].items():
        for step in research['steps']:
            if step.get('canonical_evaluation_id'):
                ident = step['canonical_evaluation_id']
                if ident not in canonical or canonical[ident]['provenance']['cell_id'] != cell_id:
                    raise ValueError('canonical research step lacks committed typed evidence')
            if step.get('receipt_id'):
                record = observed(step['receipt_id'],step['tool_id'],cell_id)
                if (step.get('resources') != record['receipt']['resources']
                        or step['status'] != record['result']['status']
                        or step.get('requested_limits') != record['receipt']['limits']
                        or step.get('target_binding') != record['target_binding']):
                    raise ValueError('research decision step differs from its saved invocation')
                if step.get('experimental_protocol') is not None:
                    from .experiment_protocol import proposal_protocol, result_assessment_from_output
                    proposal=lifecycle._artifact(step['proposal_id'],'proposal',library)
                    protocol=proposal_protocol(proposal)
                    if (protocol!=step['experimental_protocol'] or proposal.get('cell_id')!=cell_id
                            or proposal.get('action_id')!=record['action_id']):
                        raise ValueError('research experiment design differs from its saved proposal')
                    assessment=result_assessment_from_output(record['receipt'].get('output'),protocol)
                    for key in ('comparison_report','comparison_report_assessment'):
                        if key in step and step[key]!=assessment:
                            raise ValueError('research comparison assessment differs from its saved result')
        from .research_steps import pending_method_trials
        for pending in pending_method_trials(research):
            record = observed(pending['submission_receipt_id'],pending['tool_id'],cell_id)
            if (record['result']['status'] != 'success'
                    or pending['source_sha256'] != record['target_binding']['target_source_sha256']
                    or pending['source_program_id'] != record['target_binding']['candidate_id']):
                raise ValueError('pending branch trial differs from its candidate-producing invocation')

    fields = {'author_cell_id', 'author_island_id', 'stage', 'local_receipt_id', 'release_record', 'replay_policy',
              'replay_attempts', 'adoption_attempts', 'adoptions', 'use_attempts', 'used_again'}
    for ident, method in state['methods'].items():
        require_hash(ident, 'method')
        if (type(method) is not dict or set(method) != fields or method.get('author_cell_id') not in known
                or method.get('stage') not in ('locally_tested','released')):
            raise ValueError('invalid method lifecycle')
        author = cells[method['author_cell_id']]
        bundle = library.get(ident)
        if (type(method['author_island_id']) is not int
                or method['author_island_id'] != author['origin_island']
                or method['author_island_id'] != bundle['manifest']['protocol']['author_island']
                or method['replay_policy'] != bundle['manifest']['protocol']['replay_policy']):
            raise ValueError('method creator differs from pinned source or cell origin')
        local = observed(method['local_receipt_id'], ident, author['id'])
        if local['island_id'] != method['author_island_id'] or local['result']['status'] != 'success':
            raise ValueError('method lacks successful local creator invocation')
        for key in ('replay_attempts', 'adoption_attempts', 'adoptions', 'use_attempts', 'used_again'):
            if type(method[key]) is not dict or not set(method[key]) <= known:
                raise ValueError('invalid method lifecycle recipients')
            for cell_id, reference in method[key].items():
                require_hash(reference, key)
                if key == 'adoptions':
                    continue
                record = observed(reference, ident, cell_id)
                if record['island_id'] == method['author_island_id']:
                    raise ValueError('recipient/replay observation is not independent of creator')
        release = method['release_record']
        if method['stage'] == 'locally_tested':
            if release is not None or any(method[key] for key in ('adoption_attempts', 'adoptions', 'use_attempts', 'used_again')):
                raise ValueError('unreleased method has release or adoption state')
        else:
            if (type(release) is not dict or set(release) != {'cell_id', 'receipt_id', 'reference_receipt_id',
                    'policy', 'matched', 'reference_observations_sha256', 'replay_observations_sha256',
                    'measurement_reproduced'}
                    or release.get('matched') is not True or release.get('cell_id') not in known
                    or release.get('reference_receipt_id') != method['local_receipt_id']
                    or method['replay_attempts'].get(release['cell_id']) != release.get('receipt_id')):
                raise ValueError('released method lacks bound independent replay evidence')
            replay = observed(release['receipt_id'], ident, release['cell_id'])
            expected = replay_evidence(local['receipt'], replay['receipt'], method['replay_policy'])
            if (replay['island_id'] == method['author_island_id'] or not expected['matched']
                    or any(release[key] != value for key, value in expected.items())):
                raise ValueError('released method replay does not satisfy its immutable policy')
        for cell_id, revision_id in method['adoptions'].items():
            trial_id = method['adoption_attempts'].get(cell_id)
            if trial_id is None or revision_id not in history[cell_id]:
                raise ValueError('adoption revision is not reachable or lacks recipient trial')
            trial = observed(trial_id, ident, cell_id)
            revision = history[cell_id][revision_id]
            if (trial['result']['status'] != 'success' or revision['phase'] != 'adopt'
                    or revision['tool_ids'] != [ident]
                    or {'kind': 'tool_invocation', 'id': trial_id, 'tool_id': ident} not in revision['evidence']
                    or revision['decision'] != {'rule': 'recipient_contract_passed_for_further_trials',
                                                 'benefit': 'not_established'}):
                raise ValueError('adoption revision does not bind its successful recipient trial')
        for cell_id, receipt_id in method['use_attempts'].items():
            if cell_id not in method['adoptions'] or receipt_id == method['adoption_attempts'][cell_id]:
                raise ValueError('later-use attempt lacks an earlier adoption')
            attempt = observed(receipt_id, ident, cell_id)
            adopted = history[cell_id][method['adoptions'][cell_id]]
            if attempt['stage'] != 'use_again' or not any(
                    row['ordinal'] > adopted['ordinal'] and row['phase'] == 'use_again' and
                    {'kind': 'tool_invocation', 'id': receipt_id, 'tool_id': ident} in row['evidence']
                    for row in history[cell_id].values()):
                raise ValueError('later-use attempt is not bound to its post-adoption episode')
        for cell_id, receipt_id in method['used_again'].items():
            if cell_id not in method['adoptions'] or receipt_id == method['adoption_attempts'][cell_id]:
                raise ValueError('later use is absent or reuses the adoption trial')
            later = observed(receipt_id, ident, cell_id)
            adopted = history[cell_id][method['adoptions'][cell_id]]
            if (later['receipt']['dispatch_status'] != 'invoked' or not any(
                    row['ordinal'] > adopted['ordinal'] and
                    {'kind': 'tool_invocation', 'id': receipt_id, 'tool_id': ident} in row['evidence']
                    for row in history[cell_id].values())):
                raise ValueError('later use lacks an actual post-adoption invocation')
    if not set(state['cards']) <= known:
        raise ValueError('research card has no retained cell')
    for cell_id, card in state['cards'].items():
        if (_card(card) is None or card['cell_id'] != cell_id
                or card['revision_id'] not in history[cell_id]
                or len(str(card)) > 16384):
            raise ValueError('invalid or unbound retained research card')
        source = card['source']
        decision = card.get('decision')
        if decision is not None:
            proposal_id = decision['proposal_id']
            proposal = lifecycle._artifact(proposal_id, 'proposal', library)
            authenticate_projection(decision, proposal, proposal_id)
            if (proposal.get('cell_id') != cell_id
                    or proposal.get('cell_revision_id') not in history[cell_id]
                    or not any({'kind':'research_decision', 'id':proposal_id} in row['evidence']
                               for row in history[cell_id].values())):
                raise ValueError('terminal decision lacks committed cell provenance')
            for receipt_id in proposal['evidence_receipt_ids']:
                artifact = lifecycle._artifact(receipt_id, 'invocation', library)
                observed(receipt_id, artifact['tool_id'], cell_id)
            from .canonical_evidence import authenticate_citations
            for ident in authenticate_citations(proposal, canonical):
                if ident not in canonical:
                    raise ValueError('terminal canonical citation lacks committed typed evidence')
            if source['receipt_id'] is None:
                if ({key:source[key] for key in proposal['target_binding']} != proposal['target_binding']
                        or any(source[key] is not None for key in ('tool_id','invocation_id'))
                        or card['question'] != NO_MEASUREMENT_QUESTION
                        or card['observations'] != {'status':'not_invoked_this_episode','output_excerpt':'','resources':{}}):
                    raise ValueError('terminal-only card invented a new tool observation')
        if source['receipt_id'] is not None:
            record = observed(source['receipt_id'], source['tool_id'], cell_id)
            if (source['invocation_id'] != record['receipt']['invocation_id']
                    or {k:source[k] for k in record['target_binding']} != record['target_binding']
                    or card['question'] != record['question']
                    or card['observations']['status'] != record['result']['status']
                    or card['observations']['resources'] != record['receipt']['resources']):
                raise ValueError('research card source differs from immutable receipt')
            if decision is not None and (record['action_id'] != decision['action_id']
                    or source['receipt_id'] not in proposal['evidence_receipt_ids']):
                raise ValueError('terminal decision refers to another observation episode')
            from .comparison_assessment import from_record, CARD_SCHEMA as ASSESSED_CARD_SCHEMA
            expected_assessment = from_record(record, source['receipt_id'],
                lambda ident: lifecycle._artifact(ident, 'proposal', library))
            if (card['schema'] == ASSESSED_CARD_SCHEMA and card['comparison_assessment'] != expected_assessment
                    or 'experiment_proposal_id' in record and card['schema'] != ASSESSED_CARD_SCHEMA):
                raise ValueError('research card comparison assessment differs from immutable evidence')
            comparison = card['replay_comparison']
            if record['stage'] == 'independent_replay':
                method = state['methods'].get(source['tool_id'])
                if method is None:
                    raise ValueError('replay card lacks its retained method')
                reference_id = method['local_receipt_id']
                original = observed(reference_id, source['tool_id'], method['author_cell_id'])
                expected = build_feedback(original, record, tool_id=source['tool_id'],
                    reference_receipt_id=reference_id, replay_receipt_id=source['receipt_id'])
                if not is_projection(comparison, expected):
                    raise ValueError('replay card comparison differs from its saved evidence')
            elif comparison is not None:
                raise ValueError('non-replay card has an unbound replay comparison')
    content_hash(state)
    lifecycle.cells = restored
    lifecycle._state = deepcopy({key:value for key,value in state.items() if key != 'cells'})
    lifecycle._shared_evidence_index = None
