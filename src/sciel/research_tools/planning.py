"""Reserve research opportunities against frozen search snapshots."""
from __future__ import annotations

from copy import deepcopy
from ..engine.research_targets import validate_targets
from ..methods.models import content_hash
from .research_steps import pending_method_trials
from .tool_protocol import CONTRACT

def _basis(snapshot):
    # Exclude tool_episode_plan itself; it is assigned after this binding.
    return {name: deepcopy(getattr(snapshot, name, None)) for name in (
        'island', 'iteration_no', 'parent_id', 'parent_code', 'parent_score',
        'model', 'op', 'task_prompt', 'artifacts', 'directive', 'kb_lessons', 'research_cells_context',
        'research_targets', 'research_search_evidence')}


def prepare(tools, snapshot, *, signals=None):
    targets = validate_targets(snapshot.research_targets, snapshot=snapshot,
                               task_hash=tools.task_identity.get('task_hash'))
    if tools._state['run_id'] not in (None, targets['run_id']):
        raise ValueError('research snapshot belongs to another run')
    island = str(snapshot.island)
    opportunity = tools._state['opportunities'].get(island, 0)
    job_rows = tools._island_jobs(snapshot.island)
    priority_jobs = ([row for row in job_rows if row.get('collected') and not row.get('reviewed')]
                     + [row for row in job_rows if not row.get('collected')])
    old_jobs = [row for row in job_rows if row.get('collected') and row.get('reviewed')]
    fresh_result = any(row.get('collected') and not row.get('reviewed') for row in job_rows)
    affinity = tools.lifecycle._state['affinity'].get(island, [])
    prior_research = (tools.lifecycle._state['research_state'].get(affinity[-1]) if affinity else None)
    if tools.config['independent_research'] and any(tools._continuation_owns_question(cell_id)
                                                 for cell_id in affinity):
        if not fresh_result:
            tools._proactive_reservation(signals)  # Validate before mutating a cursor.
        tools._state['run_id'] = tools.lifecycle.run_id = targets['run_id']
        tools._state['opportunities'][island] = opportunity + 1
        return None  # Its independent queue owns this enduring question.
    pending_trial = next((row for row in pending_method_trials(prior_research)
        if not row.get('blocked_reason') and tools._independent_branch(snapshot,row)), None)
    branch_due = bool(pending_trial and (prior_research or {}).get('status') not in ('open','step_limit')
                      and not (prior_research or {}).get('transfer_review_pending'))
    fresh_result = fresh_result or branch_due
    proactive = tools._proactive_reservation(signals) if not fresh_result else None
    tools._state['run_id'] = tools.lifecycle.run_id = targets['run_id']
    tools._state['opportunities'][island] = opportunity + 1
    last_episode = tools._state['last_episode_opportunity'].get(island)
    interval_due = last_episode is None or opportunity - last_episode >= tools.config['interval']
    research_due = bool(prior_research and prior_research['status'] in ('open','step_limit') and interval_due)
    if not interval_due and not fresh_result and proactive is None:
        return None
    maintenance = research_due or branch_due or bool(job_rows) and (fresh_result or any(not row.get('collected') for row in job_rows))
    lifecycle_options = {'maintenance': maintenance}
    if tools.config['independent_research']:
        lifecycle_options['renew_closed'] = True
    if proactive is not None:
        lifecycle_options['proactive'] = True
    cell_plan = tools.lifecycle.prepare(snapshot, signals, **lifecycle_options)
    if cell_plan is None:
        return None
    if proactive is not None and cell_plan['stage'] != 'research':
        raise ValueError('proactive opportunity must reserve a research episode')
    own = [ident for ident in reversed(tools._state['registered'])
           if tools._author(ident) == snapshot.island]
    adopted = set(tools.lifecycle.adopted_tool_ids(cell_plan['cell_id']))
    foreign = ([ident for ident in reversed(tools._state['published'])
                if tools._author(ident) != snapshot.island
                and ident in adopted]
               if tools.config['share_between_islands'] else [])
    # Preserve opportunities to repair local failures AND inspect foreign work.
    visible = []
    for offset in range(max(len(own), len(foreign))):
        for group in (own, foreign):
            if offset < len(group):
                visible.append(group[offset])
    visible = visible[:tools.config['max_visible_tools']]
    assigned = cell_plan.get('assigned_tool_id')
    if assigned is not None and assigned not in visible:
        visible.append(assigned)  # Explicit assigned replay/trial, not general foreign access.
    if branch_due and pending_trial['tool_id'] not in visible:
        visible.append(pending_trial['tool_id'])
    plan = {'schema': CONTRACT, 'island_id': snapshot.island,
        'parent_program_id': snapshot.parent_id, 'iteration': snapshot.iteration_no + 1,
        'research_targets_sha256': targets['sha256'],
        'opportunity': opportunity, 'last_episode_opportunity': last_episode,
        'cadence_reason': ('proactive_research' if proactive is not None else
            'first_episode' if last_episode is None else
            'interval_elapsed' if interval_due else 'fresh_result'),
        'snapshot_sha256': content_hash(_basis(snapshot)),
        'visible_tool_ids': visible,
        'published_tool_ids': [ident for ident in visible if ident in tools._state['published']],
        'library_revision': content_hash(tools._state['published']),
        'identity_sha256': content_hash(tools._identity), **cell_plan}
    plan.update(work_kind='method_transfer' if branch_due else 'question',
                method_trial_receipt_id=pending_trial['submission_receipt_id'] if branch_due else None)
    if tools.background_enabled:
        plan['visible_jobs'] = (priority_jobs + list(reversed(old_jobs)))[:tools.config['max_visible_tools']]
    # Freeze authenticated method feedback with the same plan as the source
    # catalog. A sibling's later replay must not change a dispatched prompt.
    plan['author_replay_feedback'] = tools.lifecycle.author_feedback(
        snapshot.island, visible, tools.library)
    plan['method_followups'] = tools.followups.planner_view(visible, max_chars=8000)
    if tools.config['shared_observations']:
        plan['shared_observations'] = tools.lifecycle.shared_observations(
            tools.library, snapshot.island, tools.task_identity)
    if proactive is not None:
        plan['proactive_research'] = proactive
    plan['action_id'] = content_hash(plan)
    # A healthy visit or refused lease/cap must not consume the first chance
    # to respond to stagnation. A frozen plan owns its interval even if the
    # model later chooses skip; wave rollback restores this reservation.
    tools._state['last_episode_opportunity'][island] = opportunity
    if proactive is not None:
        # All siblings in a parallel wave share the same committed clock.
        # Reserve only after a real cell lease/plan; rollback restores this
        # cursor with the wave baseline. Records never reset this clock.
        tools._state['last_proactive_iteration'] = proactive['committed_iteration']
    if tools.config['independent_research']:
        tools.enqueue_continuation(snapshot, plan)
        return None
    return plan


def _proactive_reservation(tools, signals):
    every = tools.config['proactive_every_attempts']
    if every is None:
        return None
    committed = signals.get('committed_iteration') if type(signals) is dict else None
    if type(committed) is not int or committed < 0:
        raise ValueError('proactive research requires the committed ordinary-attempt clock')
    previous = tools._state['last_proactive_iteration']
    if previous is not None and committed < previous:
        raise ValueError('proactive research clock precedes its saved reservation')
    if previous is not None and committed - previous < every:
        return None
    return {'schema': 'sciel-proactive-research/1', 'committed_iteration': committed,
            'previous_committed_iteration': previous, 'every_attempts': every,
            'reason': 'initial' if previous is None else 'interval_elapsed'}
