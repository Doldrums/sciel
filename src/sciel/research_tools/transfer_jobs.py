"""Durable, stage-at-a-time background execution of an immutable saved method.

The recipient is frozen once. Queue opportunities only submit the next stage or
consume the terminal chain; they never wait for numerical work or ask a planner
between stages. These trials certify execution, not method benefit.
"""
from copy import deepcopy

from ..engine.research_targets import tool_inputs
from ..methods.models import canonical_bytes, require_hash
from .config import effective_limits
from .continuations import restore_snapshot
from .research_steps import new_state, pending_method_trials, record_step
from .stage_data import attach_stage_input

SCHEMA = 'sciel-background-method-transfer/1'


def validate_progress(value, state):
    if value is None:
        return
    fields = {'schema','submission_receipt_id','snapshot_artifact_id',
              'reference_receipt_ids','job_ids','stage_results'}
    if type(value) is not dict or set(value) != fields or value['schema'] != SCHEMA:
        raise ValueError('invalid background method transfer')
    for key in ('submission_receipt_id','snapshot_artifact_id'):
        require_hash(value[key], key)
    refs, jobs, results = (value[key] for key in ('reference_receipt_ids','job_ids','stage_results'))
    if (type(refs) is not list or not 1 <= len(refs) <= 16
            or type(jobs) is not list or not 1 <= len(jobs) <= len(refs)
            or type(results) is not list or not len(jobs)-1 <= len(results) <= len(jobs)):
        raise ValueError('invalid background method stage cursor')
    for values in (refs, jobs):
        if len(values) != len(set(values)):
            raise ValueError('duplicate background method stage')
        for ident in values:
            require_hash(ident, 'background method stage')
    for index, result in enumerate(results):
        if (type(result) is not dict or set(result) != {'job_id','receipt_id','status'}
                or result['job_id'] != jobs[index]
                or result['status'] not in ('success','tool_error','invalid_output','host_failed','unknown')
                or index < len(jobs)-1 and result['status'] != 'success'):
            raise ValueError('invalid background method result')
        if result['receipt_id'] is not None:
            require_hash(result['receipt_id'], 'background method receipt')
        elif result['status'] != 'unknown':
            raise ValueError('known background method result lacks invocation')
    if (state['transfer_review_pending'] or not any(
            trial['submission_receipt_id'] == value['submission_receipt_id']
            for trial in pending_method_trials(state))):
        raise ValueError('background method no longer owns its pending transfer')


class TransferJobsMixin:
    def _transfer_inputs(self, snapshot, original, previous_id=None):
        inputs = tool_inputs(snapshot, self.task_identity, original['inputs'].get('arguments', {}))
        if previous_id is not None:
            inputs = attach_stage_input(inputs, self._read_artifact(previous_id, 'invocation'),
                                        receipt_id=previous_id)
        limits = effective_limits(self.config, original['receipt']['limits'])
        inputs['execution_budget'] = deepcopy(limits)
        if len(canonical_bytes(inputs)) > 2*1024*1024:
            raise ValueError('branch trial tool inputs exceed 2 MiB')
        return inputs, limits

    def _background_transfer(self, plan, snapshot, state, pending, pipeline, io):
        """Return (submitted job, terminal receipt, complete); no numerical calls."""
        progress = state.get('background_transfer')
        if progress is None:
            progress = dict(schema=SCHEMA, submission_receipt_id=pending['submission_receipt_id'],
                snapshot_artifact_id=self._save_continuation_snapshot(snapshot, plan),
                reference_receipt_ids=[ident for ident, _ in pipeline], job_ids=[], stage_results=[])
        else:
            self._validate_transfer_progress(plan['cell_id'], state)
            frozen = self._continuation_snapshot(progress['snapshot_artifact_id'], plan['cell_id'])
            if snapshot.research_targets != frozen['snapshot']['research_targets']:
                raise ValueError('background transfer recipient changed between stages')
            snapshot = restore_snapshot(frozen['snapshot'])
        results = progress['stage_results']
        if len(results) != len(progress['job_ids']):
            raise ValueError('background transfer is still awaiting its submitted stage')
        if results and (results[-1]['status'] != 'success' or len(results) == len(pipeline)):
            return None, results[-1]['receipt_id'] if results[-1]['status'] == 'success' else None, True
        index = len(results)
        previous = results[-1]['receipt_id'] if results else None
        inputs, limits = self._transfer_inputs(snapshot, pipeline[index][1], previous)
        owner = dict(submission_receipt_id=progress['submission_receipt_id'],
            snapshot_artifact_id=progress['snapshot_artifact_id'],
            reference_receipt_ids=deepcopy(progress['reference_receipt_ids']), stage_index=index,
            prior_receipt_id=previous)
        job = self._submit_job(plan, self.library.get(pipeline[index][1]['tool_id']), inputs, limits,
            -1-index, io, 'Does stage '+str(index+1)+' of the saved method execute on this frozen recipient?',
            progress['job_ids'][-1] if progress['job_ids'] else None, method_transfer=owner)
        progress['job_ids'].append(job['job_id'])
        state['background_transfer'] = progress
        state['status'] = 'waiting'
        self._event(plan, 'method_transfer_submitted', job_id=job['job_id'],
            reference_submission_receipt_id=progress['submission_receipt_id'],
            recipient_snapshot_id=progress['snapshot_artifact_id'],
            pipeline_stage=index+1, pipeline_stages=len(pipeline), benefit='not_established')
        return job, None, False

    def _collect_transfer(self, row, done, *, invocation=None, observation=None):
        """Update the stage cursor exactly once, inside the wave's collection commit."""
        plan, owner = row['plan'], row['owner']['method_transfer']
        current = new_state(self.lifecycle._state['research_state'].get(plan['cell_id']))
        progress = current.get('background_transfer')
        index = owner['stage_index']
        if (progress is None or len(progress['stage_results']) != index
                or progress['job_ids'][-1] != row['job_id']
                or owner['snapshot_artifact_id'] != progress['snapshot_artifact_id']
                or owner['submission_receipt_id'] != progress['submission_receipt_id']):
            raise ValueError('collected background method differs from its stage lease')
        status = invocation['result']['status'] if invocation is not None else 'unknown'
        receipt_id = invocation['receipt_id'] if invocation is not None else None
        progress['stage_results'].append(dict(job_id=row['job_id'],receipt_id=receipt_id,status=status))
        step = dict(action='branch_trial', status=status, proposal_id=None,
            receipt_id=receipt_id, tool_id=row['tool_id'], observed_at=None,
            requested_limits=deepcopy(row['requested_limits']), resources=deepcopy(done['resources']),
            job_id=row['job_id'], benefit='not_established',
            reference_receipt_id=progress['reference_receipt_ids'][index],
            reference_submission_receipt_id=progress['submission_receipt_id'],
            recipient_snapshot_id=progress['snapshot_artifact_id'],
            branch_independence='complete_lineages_no_ancestor_relation',
            pipeline_stage=index+1, pipeline_stages=len(progress['reference_receipt_ids']),
            whole_method_completed=status=='success' and index+1==len(progress['reference_receipt_ids']))
        if observation is not None:
            step['target_binding'] = deepcopy(observation['target_binding'])
        record_step(current, step, scientific=status=='success')
        if invocation is None and done['resources'].get('cpu_seconds') is None:
            current['resources']['unknown_cpu_invocations'] += 1
        current['status'] = 'waiting'  # Queue owns completion/next stage, never the planner.
        self._event(plan, 'research_step', **step)
        return current

    def _validate_transfer_progress(self, cell_id, research_state, *, job_state=None):
        progress = research_state.get('background_transfer')
        validate_progress(progress, research_state)
        state = self._state if job_state is None else job_state
        pending_ids = {row['submission_receipt_id'] for row in pending_method_trials(research_state)}
        owned = {ident: row for ident, row in state.get('jobs', {}).items()
            if row['owner']['cell_id'] == cell_id
            and (row['owner'].get('method_transfer') or {}).get('submission_receipt_id') in pending_ids}
        if progress is None:
            if owned:
                raise ValueError('pending background transfer lost its durable stage cursor')
            return
        if set(owned) != set(progress['job_ids']):
            raise ValueError('background transfer omits a committed stage job')
        if not self.background_enabled or not self.config['independent_research']:
            raise ValueError('background transfer exists while jobs are disabled')
        pending = next(row for row in pending_method_trials(research_state)
                       if row['submission_receipt_id'] == progress['submission_receipt_id'])
        pipeline = self._method_pipeline(pending, cell_id, job_state=state)
        if [ident for ident, _ in pipeline] != progress['reference_receipt_ids']:
            raise ValueError('background transfer differs from its immutable method')
        saved = self._continuation_snapshot(progress['snapshot_artifact_id'], cell_id)
        snapshot = restore_snapshot(saved['snapshot'])
        if (saved['origin_plan'].get('work_kind') != 'method_transfer'
                or saved['origin_plan'].get('method_trial_receipt_id') != pending['submission_receipt_id']
                or state.get('run_id') not in (None, snapshot.research_targets['run_id'])
                or not self._independent_branch(snapshot, pending, job_state=state)):
            raise ValueError('background transfer recipient is not bound to this trial/run')
        previous = None
        for index, job_id in enumerate(progress['job_ids']):
            row = state.get('jobs', {}).get(job_id)
            if row is None:
                raise ValueError('background transfer lacks its committed job')
            spec = self.jobs.spec(job_id)['request']
            owner = dict(submission_receipt_id=pending['submission_receipt_id'],
                snapshot_artifact_id=progress['snapshot_artifact_id'],
                reference_receipt_ids=progress['reference_receipt_ids'], stage_index=index,
                prior_receipt_id=previous)
            inputs, limits = self._transfer_inputs(snapshot, pipeline[index][1], previous)
            if (row['owner'].get('method_transfer') != owner or row['owner']['cell_id'] != cell_id
                    or row['plan'].get('work_kind') != 'method_transfer'
                    or row['plan'].get('method_trial_receipt_id') != pending['submission_receipt_id']
                    or spec['inputs'] != inputs or spec['limits'] != limits
                    or spec['tool_id'] != pipeline[index][1]['tool_id']
                    or spec['parent_job_id'] != (progress['job_ids'][index-1] if index else None)):
                raise ValueError('background transfer request differs from its frozen method/recipient')
            result = progress['stage_results'][index] if index < len(progress['stage_results']) else None
            collected = state.get('collected_jobs', {}).get(job_id)
            if (result is None) != (collected is None):
                raise ValueError('background transfer collection cursor differs')
            if result is not None:
                if (result['receipt_id'] != collected['invocation_artifact_id']
                        or result['status'] != collected['tool_result_status']):
                    raise ValueError('background transfer result differs from its collected receipt')
                previous = result['receipt_id']
