"""Checkpoint-owned research work, independent of ordinary island visits.

Only immutable snapshots are written before a wave commit. A paused wave restores
its queue baseline and reuses cooperative-I/O receipts, never repeating paid work.
"""
from copy import deepcopy
from dataclasses import asdict

from ..engine.types import Snapshot
from ..engine.research_targets import validate_targets
from ..methods.models import content_hash, require_hash
from .research_steps import pending_method_trials

SCHEMA = 'sciel-research-continuations/2'
SNAPSHOT_SCHEMA = 'sciel-research-continuation-snapshot/1'
STATUSES = {'ready', 'scheduled', 'waiting', 'closed'}


def snapshot_value(snapshot):
    value = asdict(snapshot)
    value['tool_episode_plan'] = None
    value['code_hashes'] = sorted(value['code_hashes'])
    return value


def restore_snapshot(value):
    value = deepcopy(value)
    value['code_hashes'] = set(value['code_hashes'])
    return Snapshot(**value)


class ContinuationMixin:
    def _init_continuations(self):
        self._state['continuations'] = {'schema': SCHEMA, 'next_sequence': 0, 'items': {}}

    def _validate_continuations(self, state, lifecycle):
        queue = state['continuations']
        if (type(queue) is not dict or set(queue) != {'schema', 'next_sequence', 'items'}
                or queue['schema'] != SCHEMA or type(queue['next_sequence']) is not int
                or queue['next_sequence'] < 0 or type(queue['items']) is not dict):
            raise ValueError('invalid research continuation queue')
        cells = lifecycle.cells.state()['cells']
        sequences = []
        fields = {'snapshot_artifact_id', 'status', 'reason', 'queue_sequence',
                  'dispatch_count', 'last_action_id', 'last_completed_iteration', 'last_work_kind'}
        for cell_id, entry in queue['items'].items():
            if (cell_id not in cells or type(entry) is not dict or set(entry) != fields
                    or entry['status'] not in STATUSES or type(entry['reason']) is not str
                    or entry['last_work_kind'] not in (None, 'question', 'method_transfer')
                    or type(entry['dispatch_count']) is not int or entry['dispatch_count'] < 0
                    or type(entry['queue_sequence']) is not int
                    or not 0 <= entry['queue_sequence'] < queue['next_sequence']):
                raise ValueError('invalid research continuation entry')
            if entry['status'] == 'scheduled':
                raise ValueError('a committed checkpoint cannot retain a scheduled research lease')
            sequences.append(entry['queue_sequence'])
            if entry['last_action_id'] is not None:
                require_hash(entry['last_action_id'], 'continuation action')
                if entry['last_action_id'] not in state['committed_episodes']:
                    raise ValueError('research continuation completion lacks a committed episode')
            if bool(entry['dispatch_count']) != bool(entry['last_action_id']):
                raise ValueError('research continuation dispatch history differs')
            if bool(entry['dispatch_count']) != bool(entry['last_work_kind']):
                raise ValueError('research continuation work history differs')
            last = entry['last_completed_iteration']
            if last is not None and (type(last) is not int or last < 0):
                raise ValueError('invalid research continuation completion clock')
            saved = self._continuation_snapshot(entry['snapshot_artifact_id'], cell_id)
            if saved['snapshot']['island'] not in lifecycle.cells.get(cell_id)['affinities']:
                raise ValueError('research continuation snapshot has no cell affinity')
        if len(sequences) != len(set(sequences)):
            raise ValueError('research continuation queue order is not unique')
        if queue['items'] and not self.config['independent_research']:
            raise ValueError('disabled independent research cannot own queued work')

    def _continuation_snapshot(self, ident, cell_id):
        from .planning import _basis
        saved = self._read_artifact(ident, 'continuation_snapshot')
        fields = {'schema', 'identity_sha256', 'task_identity', 'run_id', 'cell_id',
                  'origin_plan', 'snapshot', 'snapshot_sha256'}
        if (type(saved) is not dict or set(saved) != fields
                or saved.get('schema') != SNAPSHOT_SCHEMA or saved.get('cell_id') != cell_id
                or saved.get('identity_sha256') != content_hash(self._identity)
                or saved.get('task_identity') != self.task_identity):
            raise ValueError('research continuation snapshot identity differs')
        snapshot = restore_snapshot(saved['snapshot'])
        validate_targets(snapshot.research_targets, snapshot=snapshot,
                         task_hash=self.task_identity.get('task_hash'))
        run_id = getattr(getattr(self.emit, '__self__', None), 'run_id', None)
        if run_id is not None and (saved['run_id'] != run_id or snapshot.research_targets['run_id'] != run_id):
            raise ValueError('research continuation belongs to another run')
        origin = saved['origin_plan']
        if (origin.get('cell_id') != cell_id
                or origin.get('island_id') != snapshot.island
                or origin.get('identity_sha256') != content_hash(self._identity)
                or origin.get('action_id') != content_hash({k:v for k,v in origin.items() if k != 'action_id'})
                or saved.get('snapshot_sha256') != content_hash(_basis(snapshot))):
            raise ValueError('research continuation snapshot binding differs')
        return saved

    def _save_continuation_snapshot(self, snapshot, origin_plan):
        from .planning import _basis
        return self._artifact('continuation_snapshot', {
            'schema': SNAPSHOT_SCHEMA, 'identity_sha256': content_hash(self._identity),
            'task_identity': deepcopy(self.task_identity),
            'run_id': getattr(getattr(self.emit, '__self__', None), 'run_id', None),
            'cell_id': origin_plan['cell_id'], 'origin_plan': deepcopy(origin_plan),
            'snapshot': snapshot_value(snapshot), 'snapshot_sha256': content_hash(_basis(snapshot))})

    def _queue_sequence(self):
        queue = self._state['continuations']
        sequence = queue['next_sequence']
        queue['next_sequence'] += 1
        return sequence

    def enqueue_continuation(self, snapshot, plan):
        entry = dict(snapshot_artifact_id=self._save_continuation_snapshot(snapshot, plan),
            status='ready', reason='new_question', queue_sequence=self._queue_sequence(),
            dispatch_count=0, last_action_id=None, last_completed_iteration=None, last_work_kind=None)
        self._state['continuations']['items'][plan['cell_id']] = entry
        # The queue owns the future lease. It must not occupy an active-cell slot
        # while waiting behind another independent research job.
        self.lifecycle.cells.deactivate(plan['cell_id'])
        self._event(plan, 'continuation_queued', **entry)

    def _continuation_jobs(self, cell_id):
        return [row for row in self._state.get('jobs', {}).values()
                if row['owner']['cell_id'] == cell_id]

    def _continuation_owns_question(self, cell_id):
        entry = self._state['continuations']['items'].get(cell_id, {})
        state = self.lifecycle._state['research_state'].get(cell_id) or {}
        if (entry.get('reason') == 'pipeline_exceeds_scientific_limit'
                and state.get('status') not in ('open', 'step_limit')):
            return False  # Retain the blocked transfer; a new question may be opened.
        return entry.get('status') in ('ready', 'scheduled', 'waiting')

    def _reserve_questions(self, choose_snapshot, *, max_jobs=None):
        """Freeze a bounded FIFO cohort; blocked cells never block other work.

        choose_snapshot(saved_snapshot, research_state, work_kind=...,
        pending_trial=...) returns a source-bound snapshot. Only a transfer
        requires another branch. Question and transfer leases never overlap.
        """
        from .planning import _basis
        from .tool_protocol import CONTRACT
        if not self.config['independent_research']:
            return []
        queued = self._state['continuations']['items']
        jobs = []
        limit = self.config['research_jobs_per_wave']
        if max_jobs is not None:
            limit = min(limit, max_jobs)
        ready = ((cell, entry) for cell, entry in queued.items() if entry['status'] in ('ready', 'waiting'))
        for cell_id, entry in sorted(ready, key=lambda row: row[1]['queue_sequence']):
            if len(jobs) >= limit:
                break
            if self.lifecycle.cells.get(cell_id)['active']:
                continue
            pending = [row for row in self._continuation_jobs(cell_id)
                       if row['job_id'] not in self._state.get('collected_jobs', {})]
            if pending:
                entry.update(status='waiting', reason='waiting_background')
                continue
            saved = self._continuation_snapshot(entry['snapshot_artifact_id'], cell_id)
            state = self.lifecycle._state['research_state'].get(cell_id)
            question_due = (state is None or state.get('status') in ('open', 'step_limit')
                            or state.get('transfer_review_pending'))
            trials = pending_method_trials(state)
            review_due = bool((state or {}).get('transfer_review_pending'))
            # A completed question gives the oldest eligible transfer one turn.
            # Every transfer is followed by a review on the question's source;
            # missing branches never delay an otherwise due question.
            prefer_question = question_due and (review_due or entry['last_work_kind'] != 'question')
            snapshot, work_kind, trial = None, 'question', None
            progress = (state or {}).get('background_transfer')
            if progress is not None:
                self._validate_transfer_progress(cell_id, state)
                trial = next(row for row in trials if row['submission_receipt_id'] == progress['submission_receipt_id'])
                frozen = self._continuation_snapshot(progress['snapshot_artifact_id'], cell_id)
                snapshot = choose_snapshot(restore_snapshot(frozen['snapshot']), deepcopy(state),
                    work_kind='method_transfer_continuation', pending_trial=deepcopy(trial))
                if snapshot is None or snapshot.research_targets != frozen['snapshot']['research_targets']:
                    raise ValueError('unfinished transfer lost its frozen recipient')
                work_kind = 'method_transfer'
            elif not prefer_question:
                for candidate in trials:
                    if candidate.get('blocked_reason'):
                        continue
                    snapshot = choose_snapshot(restore_snapshot(saved['snapshot']), deepcopy(state),
                        work_kind='method_transfer', pending_trial=deepcopy(candidate))
                    if snapshot is not None:
                        if not self._independent_branch(snapshot, candidate):
                            raise ValueError('method transfer snapshot is not an independent branch')
                        work_kind, trial = 'method_transfer', candidate
                        break
            if snapshot is None and question_due:
                work_kind, trial = 'question', None
                snapshot = choose_snapshot(restore_snapshot(saved['snapshot']), deepcopy(state),
                    work_kind='question', pending_trial=None)
                if snapshot is None:
                    raise ValueError('a due question cannot depend on independent branch availability')
            if snapshot is None:
                blocked = next((row['blocked_reason'] for row in trials if row.get('blocked_reason')), None)
                entry.update(status='waiting' if trials else 'closed',
                    reason=(blocked if trials and all(row.get('blocked_reason') for row in trials)
                            else 'no_independent_branch' if trials else 'question_closed'))
                continue
            if not self.lifecycle.cells.activate(cell_id):
                continue
            cell = self.lifecycle.cells.get(cell_id)
            plan = deepcopy(saved['origin_plan'])
            # Automatic assigned replay retains its first assigned action. All
            # later work continues the same enduring question, not a new genesis.
            if entry['dispatch_count']:
                plan.update(stage='research', assigned_tool_id=None, reference_receipt_id=None)
            plan.update(schema=CONTRACT, cell_revision_id=cell['head_revision'],
                island_id=snapshot.island, parent_program_id=snapshot.parent_id,
                iteration=snapshot.iteration_no + 1, cadence_reason='independent_continuation',
                snapshot_sha256=content_hash(_basis(snapshot)),
                research_targets_sha256=snapshot.research_targets['sha256'],
                question=(state or {}).get('question') or plan['question'])
            plan.update(work_kind=work_kind,
                method_trial_receipt_id=trial['submission_receipt_id'] if trial else None)
            if state is not None:
                plan['research_state'] = deepcopy(state)
            own = [ident for ident in reversed(self._state['registered'])
                   if self._author(ident) == snapshot.island]
            adopted = set(self.lifecycle.adopted_tool_ids(cell_id))
            foreign = [ident for ident in reversed(self._state['published']) if ident in adopted
                       and self._author(ident) != snapshot.island] if self.config['share_between_islands'] else []
            visible = list(dict.fromkeys(own + foreign))[:self.config['max_visible_tools']]
            assigned = plan.get('assigned_tool_id')
            if assigned and assigned not in visible:
                visible.append(assigned)
            plan.update(visible_tool_ids=visible,
                published_tool_ids=[ident for ident in visible if ident in self._state['published']],
                library_revision=content_hash(self._state['published']),
                author_replay_feedback=self.lifecycle.author_feedback(snapshot.island, visible, self.library),
                method_followups=self.followups.planner_view(visible, max_chars=8000))
            if self.background_enabled:
                plan['visible_jobs'] = self._island_jobs(snapshot.island)[-self.config['max_visible_tools']:]
            if self.config['shared_observations']:
                plan['shared_observations'] = self.lifecycle.shared_observations(
                    self.library, snapshot.island, self.task_identity)
            # A temporary transfer target must never replace the enduring
            # question's source or carry foreign computed state into its review.
            if work_kind == 'question':
                entry['snapshot_artifact_id'] = self._save_continuation_snapshot(snapshot, saved['origin_plan'])
            plan.update(continuation_snapshot_id=entry['snapshot_artifact_id'],
                        continuation_sequence=entry['queue_sequence'])
            plan['action_id'] = content_hash({k:v for k,v in plan.items() if k != 'action_id'})
            entry.update(status='scheduled', reason='leased_for_wave',
                         dispatch_count=entry['dispatch_count'] + 1, last_action_id=plan['action_id'])
            self._event(plan, 'continuation_scheduled', **entry)
            snapshot.tool_episode_plan = deepcopy(plan)
            jobs.append((f'research:{entry["queue_sequence"]}', snapshot))
        return jobs

    def _commit_continuation(self, episode, *, completed_iteration=None):
        plan = episode['plan']
        if 'continuation_sequence' not in plan:
            return
        entry = self._state['continuations']['items'][plan['cell_id']]
        if (entry['last_action_id'] != plan['action_id']
                or entry['queue_sequence'] != plan['continuation_sequence']):
            raise ValueError('research continuation commit differs from its queue lease')
        state = self.lifecycle._state['research_state'].get(plan['cell_id']) or {}
        pending = any(row['job_id'] not in self._state.get('collected_jobs', {})
                      for row in self._continuation_jobs(plan['cell_id']))
        if pending:
            status, reason = 'waiting', 'waiting_background'
        elif state.get('transfer_review_pending'):
            status, reason = 'ready', 'transfer_results_require_review'
        elif (pending_method_trials(state)
                and all(row.get('blocked_reason') for row in pending_method_trials(state))
                and state.get('status') not in ('open', 'step_limit')):
            status, reason = 'waiting', pending_method_trials(state)[0]['blocked_reason']
        elif state.get('status') in ('open', 'step_limit') or pending_method_trials(state):
            status, reason = 'ready', 'research_open'
        else:
            status, reason = 'closed', state.get('status') or 'episode_complete'
        entry.update(status=status, reason=reason,
                     last_work_kind=plan['work_kind'],
                     last_completed_iteration=(plan['iteration'] - 1 if completed_iteration is None else completed_iteration),
                     queue_sequence=self._queue_sequence())
        self._event(plan, 'continuation_completed', **entry)
