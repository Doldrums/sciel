"""Project and commit completed research episodes and candidate provenance."""
from __future__ import annotations

from copy import deepcopy
from ..methods.models import content_hash
from .result_contract import replay_evidence
from .replay_feedback import build_feedback
from .research_steps import new_state, record_step
from .decision_projection import project_decision, NO_MEASUREMENT_QUESTION
from .observation_cards import measurement_card
from .prompt_sections import _feedback_card
from .tool_artifacts import _sha
from .tool_protocol import ARTIFACT_SCHEMA, CONTRACT

def _card(tools, plan, observation, replay_comparison=None, *, decision=None):
    return measurement_card(tools.config, plan, observation, replay_comparison,
        transform=(lambda card: tools._decision_card(plan, card, decision)) if decision is not None else None)


def _decision_card(tools, plan, card, proposal_id):
    """Keep terminal interpretation beside, never inside, raw observations."""
    proposal = tools._read_artifact(proposal_id, 'proposal')
    if (proposal['action_id'] != plan['action_id'] or proposal['cell_id'] != plan['cell_id']
            or proposal['cell_revision_id'] != plan['cell_revision_id']):
        raise ValueError('terminal decision differs from its frozen research episode')
    if card is None:
        card = {'schema':'sciel-research-check/4', 'cell_id':plan['cell_id'],
            'revision_id':plan['cell_revision_id'],
            'question':NO_MEASUREMENT_QUESTION,
            'source':{'tool_id':None, 'invocation_id':None, 'receipt_id':None,
                      **deepcopy(proposal['target_binding'])},
            'observations':{'status':'not_invoked_this_episode', 'output_excerpt':'', 'resources':{}},
            'interpretation':'', 'next_check':'', 'replay_comparison':None,
            'limitations':['No new tool measurement in this episode; referenced prior evidence remains separate.',
                           'Model interpretation is not canonical task validity, score or benefit.']}
    card['schema'] = ('sciel-research-check/5' if 'comparison_assessment' in card
                      else 'sciel-research-check/4')
    card['interpretation'] = 'Terminal model interpretation is recorded in decision; it is not canonical task validation.'
    card['next_check'] = 'Follow decision.next_decision as a proposed test, not an established improvement.'
    maximum = min(tools.config['max_context_chars'], 10000)
    limits, receipts = [600, 2000, 600], 16
    while True:
        card['decision'] = project_decision(proposal, proposal_id,
            text_limits=tuple(limits), max_receipts=receipts)
        # Reserve room for a whole result projection or a failure excerpt.
        if len(tools.format_context([card])) <= maximum - 400:
            break
        if receipts > 1:
            receipts -= 1
            continue
        # The receipt already preserves the measured question. Prefer the
        # terminal rationale over duplicate question/next-step prose.
        candidates = ([i for i in (0, 2) if limits[i] > 100]
                      or ([1] if limits[1] > 500 else [])
                      or [i for i in (0, 2) if limits[i] > 0]
                      or ([1] if limits[1] > 0 else []))
        if not candidates:
            raise ValueError('terminal research card cannot fit its supported context budget')
        index = max(candidates, key=lambda i: limits[i])
        reduced = max(0, limits[index] - 100)
        # At small budgets, a whole-field omission is safer than a tiny
        # fragment that can hide a negation. Length is checked after ASCII
        # escaping; a character floor cannot safely budget Unicode text.
        limits[index] = reduced if reduced >= (64, 160, 64)[index] else 0
    return card


def _finish_episode(tools, plan, registered, invocations, feedback, known_cost, attempts, *, replay_result=None,
                    provider_abandonment=None, terminal_decision=None):
    actual = [row for row in invocations if row['dispatch_status'] == 'invoked']
    receipt_ids = [row['receipt_id'] for row in actual]
    tool_ids = list(dict.fromkeys(row['tool_id'] for row in actual))
    latest = next((row for row in reversed(feedback) if row.get('invocation_id')), None)
    comparison = None
    if latest is not None and plan['stage'] == 'independent_replay':
        original = tools._read_artifact(plan['reference_receipt_id'], 'invocation')
        repeated = tools._read_artifact(latest['receipt_id'], 'invocation')
        comparison = build_feedback(original, repeated, tool_id=latest['tool_id'],
            reference_receipt_id=plan['reference_receipt_id'], replay_receipt_id=latest['receipt_id'])
        expected = replay_evidence(original['receipt'], repeated['receipt'], comparison['policy'])
        if replay_result != expected:
            raise ValueError('replay comparison differs from exact saved release evidence')
    card = (tools._card(plan, latest, comparison, decision=terminal_decision) if latest is not None else
            tools._decision_card(plan, None, terminal_decision) if terminal_decision is not None else None)
    cards = tools._bounded_cards([card]) if card is not None else []
    context = tools.format_context(cards) if cards else _feedback_card(feedback, tools.config['max_context_chars'])
    tools._event(plan, 'context_prepared', tool_ids=tool_ids, receipt_ids=receipt_ids,
        context_sha256=_sha(context), chars=len(context),
        scope='Prepared research check results, not proof of model delivery or task benefit.')
    episode = {'schema': CONTRACT, 'plan':deepcopy(plan), 'context':context, 'cards':cards,
        'registered_ids':registered, 'publish_ids':[], 'receipt_ids':receipt_ids,
        'tool_ids':tool_ids, 'invocations':invocations, 'known_llm_cost':known_cost,
        'replay_matched':bool(replay_result and replay_result['matched']),
        'replay_evidence':deepcopy(replay_result)}
    if terminal_decision is not None:
        episode['terminal_decision_proposal_id'] = terminal_decision
    if (tools.config['max_scientific_steps'] > 1 and plan['stage'] != 'research'
            and invocations and not plan.get('sharing_work_id')):
        state = new_state(tools.lifecycle._state['research_state'].get(plan['cell_id']))
        for row in invocations:
            saved = tools._read_artifact(row['receipt_id'],'invocation')
            step = {'action':plan['stage'],'status':row['result']['status'],'proposal_id':None,
                'receipt_id':row['receipt_id'],'tool_id':row['tool_id'],'observed_at':None,
                'question':saved['question'],'experiment':saved['question'],
                'requested_limits':deepcopy(saved['receipt']['limits']),
                'resources':deepcopy(row['resources']),'target_binding':deepcopy(row['target_binding']),
                'benefit':'not_established'}
            record_step(state,step,scientific=row['result']['status']=='success')
            tools._event(plan,'research_step',**step)
        state['status'] = 'open'
        episode['research_state'] = state
    if provider_abandonment is not None:
        episode['provider_abandonment'] = deepcopy(provider_abandonment)
    tools._event(plan, 'episode_finished',
        status='provider_abandoned' if provider_abandonment is not None else 'returned', tool_ids=tool_ids,
        receipt_ids=receipt_ids, known_llm_cost=known_cost, attempts=attempts, invocations=len(invocations),
        **({'provider_abandonment':deepcopy(provider_abandonment), 'current_call_cost_unknown':True}
           if provider_abandonment is not None else {}))
    return episode


def candidate_binding(tools, episode, *, source, provenance, evaluation_id, score, cases):
    """Authenticate the measured producer before the engine admits a child.

        An episode may continue using other tools after submitting a candidate.
        Its final observations therefore cannot identify the candidate producer.
        """
    from .canonical_evidence import authenticate
    if (type(episode) is not dict or episode.get('schema') != CONTRACT
            or episode['plan']['identity_sha256'] != content_hash(tools._identity)):
        raise ValueError('invalid research candidate episode identity')
    plan = episode['plan']
    matches = [row for row in episode.get('candidate_submissions', [])
        if row.get('source') == source and row.get('provenance') == provenance
        and row.get('evaluation', {}).get('evaluation_id') == evaluation_id]
    if len(matches) != 1:
        raise ValueError('admitted candidate differs from its episode submission')
    ident = content_hash({'schema': ARTIFACT_SCHEMA, 'kind': 'candidate_evaluation',
                          'payload': matches[0]})
    saved, invocation = authenticate(tools.lifecycle, tools.library, ident,
        cell_id=plan['cell_id'], run_id=tools._state['run_id'], task_identity=tools.task_identity)
    value, origin = saved['evaluation'], saved['provenance']
    if (saved != matches[0] or origin['action_id'] != plan['action_id']
            or origin['iteration'] != plan['iteration']
            or value.get('status') != 'evaluated'
            or value.get('score') != score or value.get('cases') != cases
            or not any(step.get('canonical_evaluation_id') == ident
                and step.get('canonical_action_id') == plan['action_id']
                for step in episode.get('research_state', {}).get('steps', []))):
        raise ValueError('admitted candidate differs from its canonical evaluation')
    return {'link_basis': 'canonical_candidate_source',
        'tool_ids': [origin['tool_id']], 'receipt_ids': [origin['receipt_id']],
        'candidate_source_sha256': origin['candidate_source_sha256'],
        'canonical_evaluation_id': ident, 'evaluation_id': evaluation_id,
        'candidate_invocation_id': invocation['receipt']['invocation_id']}


def _candidate_link(tools, episode, binding, child_source):
    saved = tools._read_artifact(binding.get('canonical_evaluation_id'), 'candidate_evaluation')
    value = saved['evaluation']
    expected = tools.candidate_binding(episode, source=child_source,
        provenance=saved['provenance'], evaluation_id=value.get('evaluation_id'),
        score=value.get('score'), cases=value.get('cases'))
    if binding != expected:
        raise ValueError('candidate child link differs from authenticated producing evidence')
    return expected


def commit(tools, episode, *, child_program_id=None, link_kind='mutation', completed_iteration=None,
           candidate_binding=None, child_source=None):
    if type(episode) is not dict or episode.get('schema') != CONTRACT:
        raise ValueError('invalid research tool episode result')
    if link_kind not in ('mutation', 'remeasurement'):
        raise ValueError('unsupported research tool candidate link kind')
    plan = episode['plan']
    ident = plan['action_id']
    if ident in tools._state['committed_episodes']:
        return
    if plan['identity_sha256'] != content_hash(tools._identity):
        raise ValueError('cannot commit a tool episode from another search identity')
    child_link = None
    if child_program_id is not None:
        if candidate_binding is not None:
            child_link = tools._candidate_link(episode, candidate_binding, child_source)
            disposition = ('was recorded as a remeasurement of an existing program'
                if link_kind == 'remeasurement' else 'was admitted')
            child_link['scope'] = (f'Exact saved candidate source was canonically evaluated and {disposition}; '
                'method benefit and cross-branch transfer are not established by this link.')
        elif episode.get('candidate_submission') or episode.get('candidate_submissions'):
            raise ValueError('canonical candidate admission requires its producing evidence binding')
        elif episode['receipt_ids']:
            child_link = {'link_basis': 'observations_before_generation',
                'tool_ids': episode['tool_ids'], 'receipt_ids': episode['receipt_ids'],
                'scope': 'Candidate generated after these observations; causal benefit is not established.'}
    for tool_id in episode['registered_ids']:
        tools.library.get(tool_id)
        if tool_id not in tools._state['registered']:
            tools._state['registered'].append(tool_id)
    if tools.background_enabled:
        tools._commit_jobs(episode)
    publish_ids = tools.lifecycle.pending(episode) if episode.get('submitted_jobs') or episode.get('deferred') else tools.lifecycle.commit(episode)
    for tool_id in publish_ids:
        if tool_id not in tools._state['registered']:
            raise ValueError('cannot publish an unregistered tool')
        if tool_id not in tools._state['published']:
            tools._state['published'].append(tool_id)
            release = tools.lifecycle.method(tool_id)['release_record']
            tools._event(plan, 'published', tool_id=tool_id, author_island_id=tools._author(tool_id),
                availability='shared' if tools.config['share_between_islands'] else 'owner_only',
                status='independently_replayed', replay_policy=release['policy'],
                measurement_reproduced=release['measurement_reproduced'],
                scope='No claim of task improvement or transfer benefit.')
    for row in episode['invocations']:
        if row['dispatch_status'] == 'invoked':
            uses = tools._state['uses'].setdefault(row['tool_id'], {})
            island = str(plan['island_id'])
            uses[island] = uses.get(island, 0) + 1
    if child_link is not None:
        tools._event(plan, 'child_link', child_program_id=child_program_id,
            link_kind=link_kind, **child_link)
    tools._state['committed_episodes'].append(ident)
    tools._commit_continuation(episode, completed_iteration=completed_iteration)
    tools._commit_sharing(episode, completed_iteration=completed_iteration)
    tools._sync_sharing()
