"""Durable assigned sharing work, separate from enduring scientific questions.

Only existing foreign cells are recipients. Their immutable question snapshots
bind inputs without ordinary parent selection. Question and sharing cohorts take
turns under the same global research-job budget; queued work grants no adoption.
"""
from copy import deepcopy

from ..methods.models import canonical_bytes, content_hash, require_hash
from ..engine.research_targets import tool_inputs
from .continuations import restore_snapshot

SCHEMA = 'sciel-research-sharing/1'
STAGES = ('independent_replay', 'adopt_trial', 'use_again')
FIELDS = {'tool_id', 'stage', 'cell_id', 'snapshot_artifact_id',
          'reference_receipt_id', 'status', 'reason', 'queue_sequence',
          'action_id', 'job_ids', 'receipt_id', 'completed_iteration'}


def work_id(tool_id, stage, cell_id):
    return content_hash(dict(tool_id=tool_id, stage=stage, cell_id=cell_id))


class SharingMixin:
    def _init_sharing(self):
        self._state['sharing'] = dict(schema=SCHEMA, next_sequence=0,
            last_queue=None, waiting_methods={}, items={})

    def _sharing_enabled(self):
        return self.config['independent_research'] and self.config['share_between_islands']

    @staticmethod
    def _sharing_stage(method, cell_id):
        if method['stage'] == 'locally_tested':
            return 'independent_replay' if cell_id not in method['replay_attempts'] else None
        if cell_id not in method['adoption_attempts']:
            return 'adopt_trial'
        if cell_id in method['adoptions'] and cell_id not in method['use_attempts']:
            return 'use_again'
        return None

    def _sync_sharing(self):
        if not self._sharing_enabled():
            return
        queue = self._state['sharing']
        queue['waiting_methods'] = {}
        # Stable enumeration also survives sorted JSON restore. New work joins
        # behind the checkpointed FIFO rather than leapfrogging it each wave.
        for tool_id, method in sorted(self.lifecycle._state['methods'].items()):
            foreign = False
            for cell_id, question in sorted(self._state['continuations']['items'].items()):
                cell = self.lifecycle.cells.get(cell_id)
                if cell['origin_island'] == method['author_island_id']:
                    continue
                foreign = True
                stage = self._sharing_stage(method, cell_id)
                if stage is None:
                    continue
                ident = work_id(tool_id, stage, cell_id)
                if ident in queue['items']:
                    continue  # Failed/unknown attempts are still actual attempts.
                queue['items'][ident] = dict(tool_id=tool_id, stage=stage, cell_id=cell_id,
                    snapshot_artifact_id=question['snapshot_artifact_id'],
                    reference_receipt_id=method['local_receipt_id'], status='ready',
                    reason='eligible_stage', queue_sequence=queue['next_sequence'],
                    action_id=None, job_ids=[], receipt_id=None, completed_iteration=None)
                queue['next_sequence'] += 1
            if not foreign:
                queue['waiting_methods'][tool_id] = 'no_recipient_yet'

    def _validate_sharing(self, state, lifecycle):
        queue = state['sharing']
        if (type(queue) is not dict or set(queue) !=
                {'schema', 'next_sequence', 'last_queue', 'waiting_methods', 'items'}
                or queue['schema'] != SCHEMA or type(queue['next_sequence']) is not int
                or queue['next_sequence'] < 0 or queue['last_queue'] not in (None, 'question', 'sharing')
                or type(queue['items']) is not dict or type(queue['waiting_methods']) is not dict):
            raise ValueError('invalid independent sharing queue')
        methods = lifecycle._state['methods']
        if any(ident not in methods or reason != 'no_recipient_yet'
               for ident, reason in queue['waiting_methods'].items()):
            raise ValueError('invalid sharing recipient wait')
        if not self._sharing_enabled() and (queue['items'] or queue['waiting_methods']):
            raise ValueError('disabled independent sharing has queued work')
        for tool_id in queue['waiting_methods']:
            if any(lifecycle.cells.get(cell)['origin_island'] != methods[tool_id]['author_island_id']
                   for cell in state['continuations']['items']):
                raise ValueError('sharing recipient wait conceals an existing foreign cell')
        sequences = []
        for ident, entry in queue['items'].items():
            if (type(entry) is not dict or set(entry) != FIELDS
                    or entry['stage'] not in STAGES or entry['tool_id'] not in methods
                    or entry['cell_id'] not in state['continuations']['items']
                    or ident != work_id(entry['tool_id'], entry['stage'], entry['cell_id'])
                    or entry['status'] not in ('ready', 'waiting', 'completed')
                    or type(entry['reason']) is not str
                    or type(entry['queue_sequence']) is not int
                    or not 0 <= entry['queue_sequence'] < queue['next_sequence']):
                raise ValueError('invalid or scheduled sharing work in committed checkpoint')
            sequences.append(entry['queue_sequence'])
            method = methods[entry['tool_id']]
            if (lifecycle.cells.get(entry['cell_id'])['origin_island'] == method['author_island_id']
                    or entry['reference_receipt_id'] != method['local_receipt_id']):
                raise ValueError('sharing requires a foreign recipient and exact local receipt')
            saved = self._continuation_snapshot(entry['snapshot_artifact_id'], entry['cell_id'])
            if saved['snapshot']['island'] != lifecycle.cells.get(entry['cell_id'])['origin_island']:
                raise ValueError('sharing snapshot recipient differs')
            if entry['action_id'] is not None:
                require_hash(entry['action_id'], 'sharing action')
                plan = saved['origin_plan']
                if (entry['action_id'] not in state['committed_episodes']
                        or plan.get('sharing_work_id') != ident or plan['action_id'] != entry['action_id']
                        or plan['stage'] != entry['stage'] or plan['assigned_tool_id'] != entry['tool_id']
                        or plan.get('work_kind') != 'sharing'):
                    raise ValueError('sharing completion differs from its immutable plan')
            elif entry['status'] != 'ready' and entry['reason'] != 'stage_superseded':
                raise ValueError('sharing completion lacks an assigned action')
            if type(entry['job_ids']) is not list or len(entry['job_ids']) > 1:
                raise ValueError('invalid sharing background stages')
            for job_id in entry['job_ids']:
                row = state['jobs'].get(job_id)
                if (row is None or row['plan'].get('sharing_work_id') != ident
                        or row['plan']['action_id'] != entry['action_id']):
                    raise ValueError('sharing job differs from its work item')
            if entry['status'] == 'waiting' and (len(entry['job_ids']) != 1
                    or entry['job_ids'][0] in state['collected_jobs']):
                raise ValueError('sharing wait lacks an outstanding job')
            receipt_id = entry['receipt_id']
            if entry['status'] == 'ready':
                if (entry['job_ids'] or receipt_id is not None
                        or entry['reason'] not in ('eligible_stage', 'submission_deferred')
                        or (entry['action_id'] is None) != (entry['reason'] == 'eligible_stage')):
                    raise ValueError('invalid ready sharing disposition')
            elif entry['status'] == 'waiting':
                if entry['reason'] != 'waiting_background' or receipt_id is not None:
                    raise ValueError('invalid pending sharing disposition')
            elif entry['job_ids']:
                done = state['collected_jobs'].get(entry['job_ids'][0])
                if (done is None or done['invocation_artifact_id'] != receipt_id
                        or done['tool_result_status'] != entry['reason']
                        or not state['jobs'][entry['job_ids'][0]]['reviewed']):
                    raise ValueError('sharing completion differs from collected job')
            elif entry['reason'] == 'stage_superseded':
                if (entry['action_id'] is not None or receipt_id is not None
                        or entry['stage'] != 'independent_replay' or method['stage'] != 'released'):
                    raise ValueError('unproven sharing stage supersession')
            elif entry['reason'] == 'input_admission_limit':
                if receipt_id is not None or entry['action_id'] is None:
                    raise ValueError('invalid undispatched sharing disposition')
                original = self._read_artifact(entry['reference_receipt_id'], 'invocation')
                inputs = (original['inputs'] if entry['stage'] == 'independent_replay' else
                    tool_inputs(restore_snapshot(saved['snapshot']), self.task_identity,
                                original['inputs'].get('arguments', {})))
                if len(canonical_bytes(inputs)) <= 2 * 1024 * 1024:
                    raise ValueError('sharing input admission limit is not established')
            elif entry['reason'] != 'attempt_completed' or receipt_id is None:
                raise ValueError('sharing attempt lacks its actual receipt')
            if receipt_id is not None:
                row = self._read_artifact(receipt_id, 'invocation')
                if (row['action_id'] != entry['action_id'] or row['cell_id'] != entry['cell_id']
                        or row['stage'] != entry['stage'] or row['tool_id'] != entry['tool_id']):
                    raise ValueError('sharing receipt differs from assigned work')
                field = {'independent_replay': 'replay_attempts', 'adopt_trial': 'adoption_attempts',
                         'use_again': 'use_attempts'}[entry['stage']]
                if method[field].get(entry['cell_id']) != receipt_id:
                    raise ValueError('sharing receipt differs from verified lifecycle')
            clock = entry['completed_iteration']
            if clock is not None and (type(clock) is not int or clock < 0):
                raise ValueError('invalid sharing completion clock')
            if (entry['status'] == 'completed') != (clock is not None):
                raise ValueError('sharing terminal status differs from completion clock')
        if len(sequences) != len(set(sequences)):
            raise ValueError('sharing FIFO order is not unique')

    def _reserve_sharing(self, choose_snapshot):
        from .tool_protocol import CONTRACT
        from .planning import _basis
        for ident, entry in sorted(self._state['sharing']['items'].items(),
                                   key=lambda row: row[1]['queue_sequence']):
            if entry['status'] != 'ready':
                continue
            cell_id = entry['cell_id']
            method = self.lifecycle.method(entry['tool_id'])
            if self._sharing_stage(method, cell_id) != entry['stage']:
                release = self._read_artifact(method['release_record']['receipt_id'], 'invocation')
                entry.update(status='completed', reason='stage_superseded',
                             completed_iteration=release['iteration'] - 1)
                continue
            if (self.lifecycle.cells.get(cell_id)['active']
                    or any(row['job_id'] not in self._state['collected_jobs']
                           for row in self._continuation_jobs(cell_id))):
                continue
            saved = self._continuation_snapshot(entry['snapshot_artifact_id'], cell_id)
            snapshot = choose_snapshot(restore_snapshot(saved['snapshot']), None, work_kind='sharing', pending_trial=None)
            if snapshot is None:
                continue
            if (snapshot.island != saved['snapshot']['island']
                    or snapshot.parent_id != saved['snapshot']['parent_id']
                    or snapshot.parent_code != saved['snapshot']['parent_code']):
                raise ValueError('sharing refresh changed its frozen recipient source')
            if not self.lifecycle.cells.activate(cell_id):
                continue
            plan = deepcopy(saved['origin_plan'])
            for key in ('action_id', 'research_state', 'proactive', 'proactive_research',
                        'continuation_sequence', 'continuation_snapshot_id'):
                plan.pop(key, None)
            plan.update(schema=CONTRACT, cell_revision_id=self.lifecycle.cells.get(cell_id)['head_revision'],
                stage=entry['stage'], assigned_tool_id=entry['tool_id'],
                reference_receipt_id=entry['reference_receipt_id'] if entry['stage']=='independent_replay' else None,
                iteration=snapshot.iteration_no+1, cadence_reason='independent_sharing',
                work_kind='sharing', method_trial_receipt_id=None, sharing_work_id=ident,
                visible_tool_ids=[entry['tool_id']], published_tool_ids=[entry['tool_id']]
                    if entry['tool_id'] in self._state['published'] else [],
                library_revision=content_hash(self._state['published']),
                visible_jobs=[], trigger_sources=['independent_sharing'],
                stagnation_signals={}, snapshot_sha256=content_hash(_basis(snapshot)),
                research_targets_sha256=snapshot.research_targets['sha256'])
            plan['action_id'] = content_hash(plan)
            entry.update(status='scheduled', reason='leased_for_wave', action_id=plan['action_id'],
                snapshot_artifact_id=self._save_continuation_snapshot(snapshot, plan))
            snapshot.tool_episode_plan = deepcopy(plan)
            self._event(plan, 'sharing_scheduled', sharing_work_id=ident, tool_id=entry['tool_id'],
                        sharing_stage=entry['stage'], queue_sequence=entry['queue_sequence'])
            return [(f'sharing:{ident}', snapshot)]
        return []

    def reserve_continuations(self, choose_snapshot, *, max_jobs=None):
        if not self._sharing_enabled():
            return self._reserve_questions(choose_snapshot, max_jobs=max_jobs)
        self._sync_sharing()
        limit = self.config['research_jobs_per_wave']
        if max_jobs is not None:
            limit = min(limit, max_jobs)
        result = []
        for _ in range(limit):
            order = ('sharing', 'question') if self._state['sharing']['last_queue']=='question' else ('question', 'sharing')
            for kind in order:
                jobs = (self._reserve_sharing(choose_snapshot) if kind=='sharing' else
                        self._reserve_questions(choose_snapshot, max_jobs=1))
                if jobs:
                    self._state['sharing']['last_queue'] = kind
                    result.extend(jobs)
                    break
            else:
                break
        return result

    def _commit_sharing(self, episode, *, completed_iteration=None):
        plan = episode['plan']
        ident = plan.get('sharing_work_id')
        if ident is None:
            return
        entry = self._state['sharing']['items'][ident]
        if entry['status'] != 'scheduled' or entry['action_id'] != plan['action_id']:
            raise ValueError('sharing commit differs from its queue lease')
        jobs = [row['job_id'] for row in episode.get('submitted_jobs', [])]
        receipt = next((row['receipt_id'] for row in episode['invocations']), None)
        if episode.get('sharing_deferred_reason') == 'submission_deferred':
            queue = self._state['sharing']
            entry.update(status='ready', reason='submission_deferred',
                queue_sequence=queue['next_sequence'], job_ids=[], receipt_id=None, completed_iteration=None)
            queue['next_sequence'] += 1
            self._event(plan, 'sharing_deferred', sharing_work_id=ident, tool_id=entry['tool_id'])
            return
        entry.update(status='waiting' if jobs else 'completed',
            reason='waiting_background' if jobs else episode.get('sharing_deferred_reason', 'attempt_completed'), job_ids=jobs,
            receipt_id=receipt, completed_iteration=None if jobs else
                (plan['iteration']-1 if completed_iteration is None else completed_iteration))
        self._event(plan, 'sharing_completed', sharing_work_id=ident, sharing_stage=entry['stage'],
                    tool_id=entry['tool_id'], status=entry['status'], receipt_id=receipt,
                    scope='Operational sharing only; no task benefit established.')

    def _collect_sharing(self, plan, job_id, done):
        ident = plan.get('sharing_work_id')
        if ident is None:
            return
        entry = self._state['sharing']['items'][ident]
        if (entry['status'] != 'waiting' or entry['action_id'] != plan['action_id']
                or entry['job_ids'] != [job_id]):
            raise ValueError('sharing collection differs from its submitted job')
        entry.update(status='completed', reason=done['tool_result_status'],
            receipt_id=done['invocation_artifact_id'], completed_iteration=plan['iteration']-1)
        self._state['jobs'][job_id]['reviewed'] = True
        self.lifecycle.cells.deactivate(entry['cell_id'])
        self._event(plan, 'sharing_completed', sharing_work_id=ident, sharing_stage=entry['stage'],
            tool_id=entry['tool_id'], status='completed', receipt_id=entry['receipt_id'],
            scope='Operational sharing only; no task benefit established.')
