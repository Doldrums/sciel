"""Wave-owned visibility and evidence for detached research computations."""
from copy import deepcopy
import json

from ..campaign.budget import CampaignStopped
from ..methods.executor import ExecutionRefused
from ..engine.research_targets import input_binding
from ..methods.models import content_hash, require_hash
from .config import effective_limits
from .jobs import BackgroundJobs, TERMINAL
from .result_contract import RESULT_SCHEMA, assess_receipt, parse_result, replay_evidence
from .research_steps import new_state, record_step
from .stage_data import attach_stage_input
from .experiment_protocol import parse_protocol, result_assessment_from_output


class JobRefused(ValueError):
    """Proven rejection before a job was accepted; safe to repair a proposal."""


class BackgroundToolsMixin:
    @property
    def background_enabled(self):
        return self.config['jobs']['enabled'] is True

    def _init_jobs(self, run_dir, capacity, jobs):
        self.jobs = None
        if not self.background_enabled:
            return
        self._state.update(jobs={}, collected_jobs={}, job_observations={})
        config = deepcopy(self.config['jobs'])
        if capacity:
            config['capacity'] = {key: capacity[key] for key in
                ('capacity_dir', 'host_llm_slots', 'host_evaluator_slots', 'host_research_slots') if key in capacity}
        self.jobs = jobs or BackgroundJobs(run_dir, self.library, config)

    def _validate_job_state(self, state):
        if not self.background_enabled:
            return
        for key in ('jobs', 'collected_jobs', 'job_observations'):
            if type(state[key]) is not dict:
                raise ValueError('invalid background research index ' + key)
        if not set(state['collected_jobs']) <= set(state['jobs']) or not set(state['job_observations']) <= set(state['jobs']):
            raise ValueError('background result has no committed job')
        for ident, row in state['jobs'].items():
            require_hash(ident, 'job ID')
            spec = self.jobs.spec(ident)
            input_binding(spec['request']['inputs'])
            if (row.get('job_id') != ident or row.get('request_hash') != spec['request_hash']
                    or row.get('tool_id') != spec['request']['tool_id']
                    or row.get('owner') != spec['request']['owner']
                    or row['plan']['action_id'] != row['owner']['action_id']):
                raise ValueError('committed job differs from immutable spec')
            protocol = self._background_protocol(row['owner'])
            plan = row['plan']
            if (plan['action_id'] != content_hash({k:v for k,v in plan.items() if k != 'action_id'})
                    or plan['identity_sha256'] != content_hash(self._identity)):
                raise ValueError('job plan is not bound to this search')
            if ident in state['collected_jobs']:
                done = state['collected_jobs'][ident]
                receipt = self.jobs.result(ident)
                if done.get('receipt_id') != content_hash(receipt) or done.get('resources') != receipt['resources']:
                    raise ValueError('collected job differs from saved terminal receipt')
                result = assess_receipt(receipt)
                continuation = (result['status']=='success'
                    and 'continuation_state' in parse_result(receipt['output'])['observations'])
                if (done.get('result_schema') != RESULT_SCHEMA
                        or done.get('tool_result_status') != result['status']
                        or done.get('tool_result_error') != result['error']
                        or done.get('continuation_available') is not continuation):
                    raise ValueError('collected job outcome differs from its immutable result')
                expected_assessment = (result_assessment_from_output(receipt.get('output'), protocol)
                    if receipt['status'] != 'unknown' else None)
                if done.get('comparison_report') != expected_assessment:
                    raise ValueError('collected comparison assessment differs from saved result')
                if done.get('invocation_artifact_id'):
                    record = self._read_artifact(done['invocation_artifact_id'], 'invocation')
                    if (record['receipt'] != receipt or record.get('result') != result
                            or record.get('target_binding') != input_binding(record['inputs'])
                            or record.get('input_program_id') != record['target_binding']['candidate_id']
                            or record.get('replay_policy') != self.library.get(row['tool_id'])['manifest']['protocol']['replay_policy']):
                        raise ValueError('collected invocation differs from job receipt')

    def _background_protocol(self, owner):
        protocol = owner.get('experimental_protocol')
        if protocol is None:
            return None
        ident = owner.get('experiment_proposal_id')
        require_hash(ident, 'background experiment proposal')
        proposal = self._read_artifact(ident, 'proposal')
        raw = proposal['response']['text'].strip()
        if raw.startswith('```') and raw.endswith('```'):
            raw = raw.split('\n',1)[-1].rsplit('```',1)[0].strip()
        declared = parse_protocol(json.loads(raw).get('experimental_protocol'))
        if (declared != protocol or proposal.get('cell_id') != owner['cell_id']
                or proposal.get('action_id') != owner['action_id']):
            raise ValueError('background experiment protocol differs from its immutable proposal')
        return declared

    def _island_jobs(self, island):
        if not self.background_enabled:
            return []
        result = []
        for ident, row in self._state['jobs'].items():
            if row['owner']['island_id'] != island:
                continue
            observed = self._state['job_observations'].get(ident, {})
            done = self._state['collected_jobs'].get(ident)
            result.append(dict(job_id=ident, tool_id=row['tool_id'],
                status=observed.get('status', 'submitted'),
                resources=observed.get('resources'), requested_limits=row['requested_limits'],
                parent_job_id=row.get('parent_job_id'), collected=done is not None,
                reviewed=row.get('reviewed', False),
                continuation_available=bool(done and done.get('continuation_available')),
                error=(done.get('tool_result_error') if done else observed.get('error')),
                tool_result_status=done.get('tool_result_status') if done else None))
        return result

    def _job_instructions(self, plan, *, compact=False):
        # Kept in the actual bounded prompt, including sequential research. Jobs
        # themselves are rendered separately with authenticated omission counts.
        if compact:
            return """COMPUTE JOB CONTROLS:
null=defaults, not targets; both modes share maxima. Long stages: background.
Measure throughput/progress-vs-CPU. A pilot only bounds tested effort. Warranted 120-180 CPU-s TOTAL later;
not minimum; no zero-gain escalation.
Return continuation_state before expiry: incumbent/RNG/counters/conditions/trace; kill loses unreturned state.
Own job_id: poll/cancel/continue. payload['execution_budget']; pending isn't evidence.
Task metric+validity; normalization isn't a bound; respect explicit task bounds.
""" + ('CPU/WALL seconds: default={cpu:g}/{wall:g}; maximum={max_cpu:g}/{max_wall:g}.').format(
                cpu=self.config['limits']['cpu_seconds'], wall=self.config['limits']['wall_seconds'],
                max_cpu=self.config['jobs']['max_cpu_seconds'], max_wall=self.config['jobs']['max_wall_seconds'])
        return """COMPUTE JOB CONTROLS:
null=defaults, not targets; CPU != wall. Both modes share maxima. Long stages: execution=background; foreground waits.
Budget by throughput/progress-vs-CPU. Tiny pilots don't refute methods; when justified, later 120-180 CPU-s TOTAL segments, not a minimum. Don't escalate zero-gain polishing; no-need/negative closure valid.
Before expiry return observations.continuation_state: incumbent/RNG/counters/conditions/trace; hard kill loses unreturned state. Own job_id: poll/cancel/continue(same payload/state; arguments replace). payload['execution_budget']; Next-wave results; pending is not evidence.
Task metric+validity is objective. Normalization alone proves no bound; respect explicit task bounds.
DEFAULT LIMITS: """ + json.dumps(self.config['limits'], sort_keys=True, separators=(',', ':')) + (
            '\nMAXIMUM PER INVOCATION: ' + json.dumps(self.config['jobs'], sort_keys=True, separators=(',', ':')))

    def _submit_job(self, plan, bundle, inputs, limits, attempt, io, question, parent_job_id=None, *, experimental_protocol=None, proposal_id=None, method_transfer=None):
        from .observation_cards import _clip, _clip_json
        limits = effective_limits(self.config, limits)
        input_binding(inputs)
        inputs = deepcopy(inputs)
        inputs['execution_budget'] = deepcopy(limits)
        question = _clip_json(_clip(question, 600), min(620, self.config['max_context_chars']//12))
        key = content_hash(dict(action_id=plan['action_id'], attempt=attempt,
                                tool_id=bundle['id'], parent_job_id=parent_job_id))
        owner = {name:plan[name] for name in ('cell_id','island_id','parent_program_id',
                                             'stage','action_id','cell_revision_id')}
        owner['question'] = question
        if method_transfer is not None:
            owner['method_transfer'] = deepcopy(method_transfer)
        if experimental_protocol is not None:
            require_hash(proposal_id, 'background experiment proposal')
            owner['experimental_protocol'] = parse_protocol(experimental_protocol)
            owner['experiment_proposal_id'] = proposal_id
        request = dict(job_key=key, tool_id=bundle['id'], inputs=inputs, limits=limits,
                       owner=owner, parent_job_id=parent_job_id, library_identity=self.library.identity)
        def submit():
            try:
                return self.jobs.submit(key, bundle, inputs, limits, owner, parent_job_id)
            except ExecutionRefused as error:
                return {'accepted':False, 'reason':str(error)}
        ack = io.call('tool_job_submit', request, submit)
        if ack.get('accepted') is False:
            raise JobRefused('background job refused before dispatch: ' + str(ack.get('reason')))
        if ack.get('accepted') is not True:
            raise CampaignStopped('background submission has no conclusive acknowledgement')
        try:
            spec = self.jobs.spec(ack['job_id'])
            expected = dict(tool_id=bundle['id'], inputs=inputs, limits=limits, owner=owner,
                            parent_job_id=parent_job_id, library_identity=content_hash(self.library.identity),
                            source_sha256=content_hash(bundle['files']),
                            runtime_sha256=content_hash(self.library.identity['executor']),
                            dependency_ids=sorted(dep['id'] for dep in self.library.dependency_closure(bundle['id'])))
            if ack['request_hash'] != spec['request_hash'] or spec['request'] != expected:
                raise ValueError('accepted request differs')
        except (KeyError, ValueError, OSError) as error:
            raise CampaignStopped('accepted background submission could not be authenticated') from error
        row = dict(ack, tool_id=bundle['id'], owner=owner, plan=deepcopy(plan),
                   requested_limits=limits, parent_job_id=parent_job_id, reviewed=False)
        self._event(plan, 'job_submission_returned', job_id=ack['job_id'], tool_id=bundle['id'],
                    requested_limits=limits, parent_job_id=parent_job_id)
        return row

    def _job_action(self, plan, action, attempt, io, *, proposal_id=None):
        ident = action['job_id']
        visible = {row['job_id']:row for row in plan.get('visible_jobs', [])}
        if ident not in visible:
            raise ValueError('job_id is not visible in this frozen episode')
        if action['action'] == 'continue':
            if not visible[ident]['continuation_available']:
                raise ValueError('job lacks a collected successful continuation_state')
            prior = self.jobs.continuation_inputs(ident)
            inputs = prior['inputs']
            inputs['continuation_state'] = prior['continuation_state']
            if self.config['max_scientific_steps'] > 1:
                receipt_id = (self._state['collected_jobs'].get(ident) or {}).get('invocation_artifact_id')
                if receipt_id is None:
                    raise ValueError('continued job lacks a collected immutable invocation artifact')
                inputs = attach_stage_input(inputs,self._read_artifact(receipt_id,'invocation'),
                                            receipt_id=receipt_id)
            # parse_action always supplies an object: explicit {} means no
            # arguments, not reuse of the preceding invocation's arguments.
            inputs['arguments'] = deepcopy(action['arguments'])
            return self._submit_job(plan, prior['bundle'], inputs, action['limits'], attempt, io,
                                    action['reason'] or plan['question'], ident,
                                    experimental_protocol=action.get('experimental_protocol'),proposal_id=proposal_id)
        kind = 'tool_job_' + action['action']
        request = {'job_id':ident, 'action_id':plan['action_id']}
        result = io.call(kind, request, lambda: getattr(self.jobs, action['action'])(ident))
        artifact = self._artifact('job_control', dict(action_id=plan['action_id'],
                                                    action=action['action'], result=result))
        self._event(plan, 'job_polled' if action['action']=='poll' else 'job_cancel_requested',
                    job_id=ident, tool_id=visible[ident]['tool_id'], receipt_id=artifact,
                    status=result.get('status'), resources=result.get('resources'))
        return {'status':result.get('status', 'cancel_requested'), 'output':result,
                'reason':'Job control only; no scientific benefit inferred.'}

    def _commit_jobs(self, episode):
        for visible in episode['plan'].get('visible_jobs', []):
            if visible.get('collected'):
                self._state['jobs'][visible['job_id']]['reviewed'] = True
        for row in episode.get('submitted_jobs', []):
            ident = row['job_id']
            if ident in self._state['jobs'] and self._state['jobs'][ident] != row:
                raise ValueError('committed background job identity changed')
            self._state['jobs'][ident] = deepcopy(row)
            self._event(episode['plan'], 'job_submitted', job_id=ident, tool_id=row['tool_id'],
                        requested_limits=row['requested_limits'], parent_job_id=row['parent_job_id'])

    def collect_background(self, io):
        """Called inside the wave transaction before *any* candidate plan is frozen."""
        if not self.background_enabled:
            return
        for ident in sorted(self._state['jobs']):
            if ident in self._state['collected_jobs']:
                continue
            row = self._state['jobs'][ident]
            observed = io.call('tool_job_poll', {'job_id':ident, 'purpose':'wave_collection'},
                               lambda ident=ident: self.jobs.poll(ident))
            if observed.get('job_id') != ident or observed.get('request_hash') != row['request_hash']:
                raise CampaignStopped('background poll is not bound to its committed request')
            self._state['job_observations'][ident] = deepcopy(observed)
            if observed['status'] not in TERMINAL:
                continue
            receipt = self.jobs.result(ident)
            if content_hash(receipt) != observed.get('receipt_id'):
                raise CampaignStopped('background result differs from durable terminal poll')
            done = dict(receipt_id=observed['receipt_id'], invocation_artifact_id=None,
                        status=observed['status'], resources=deepcopy(receipt['resources']),
                        continuation_available=False,
                        result_schema=RESULT_SCHEMA,
                        tool_result_status=assess_receipt(receipt)['status'],
                        tool_result_error=assess_receipt(receipt)['error'])
            plan = row['plan']
            if receipt['status'] != 'unknown':
                spec = self.jobs.spec(ident)['request']
                invocation, observation = self._record_invocation(plan, self.library.get(row['tool_id']),
                    spec['inputs'], ident, receipt, spec['limits'], row['owner']['question'],
                    proposal_id=row['owner'].get('experiment_proposal_id'))
                replay = None
                if plan['stage'] == 'independent_replay':
                    reference = self._read_artifact(plan['reference_receipt_id'], 'invocation')['receipt']
                    replay = replay_evidence(reference, receipt, invocation['replay_policy'])
                protocol = self._background_protocol(row['owner'])
                assessment = result_assessment_from_output(receipt.get('output'), protocol)
                if assessment is not None:
                    done['comparison_report'] = assessment
                    self._event(plan, 'comparison_report', job_id=ident,
                        receipt_id=invocation['receipt_id'], assessment=deepcopy(assessment))
                episode = self._finish_episode(plan, [], [invocation], [observation], 0.0, 0, replay_result=replay)
                if row['owner'].get('method_transfer') is not None:
                    episode['research_state'] = self._collect_transfer(row, done, invocation=invocation, observation=observation)
                elif self.config['max_scientific_steps'] > 1 and plan['stage'] == 'research':
                    current = new_state(self.lifecycle._state['research_state'].get(plan['cell_id']))
                    step = {'action':'background_result','status':invocation['result']['status'],
                        'proposal_id':row['owner'].get('experiment_proposal_id'),'receipt_id':invocation['receipt_id'],'tool_id':row['tool_id'],
                        'observed_at':None,'requested_limits':deepcopy(spec['limits']),
                        'resources':deepcopy(receipt['resources']),'target_binding':deepcopy(observation['target_binding']),
                        'question':row['owner']['question'],'job_id':ident}
                    if protocol is not None:
                        step['experimental_protocol'] = deepcopy(protocol)
                    if assessment is not None:
                        step['comparison_report'] = deepcopy(assessment)
                    record_step(current,step,scientific=invocation['result']['status']=='success')
                    current['status'] = 'open'
                    episode['research_state'] = current
                    self._event(plan,'research_step',**step)
                for tool_id in self.lifecycle.commit(episode):
                    if tool_id not in self._state['published']:
                        self._state['published'].append(tool_id)
                        release = self.lifecycle.method(tool_id)['release_record']
                        self._event(plan, 'published', tool_id=tool_id, author_island_id=self._author(tool_id),
                                    availability='shared', status='independently_replayed', scope='No proof of benefit.',
                                    replay_policy=release['policy'],
                                    measurement_reproduced=release['measurement_reproduced'])
                if invocation['dispatch_status'] == 'invoked':
                    uses = self._state['uses'].setdefault(row['tool_id'], {})
                    island = str(plan['island_id'])
                    uses[island] = uses.get(island, 0) + 1
                done['invocation_artifact_id'] = invocation['receipt_id']
                done['continuation_available'] = (invocation['result']['status']=='success'
                    and 'continuation_state' in parse_result(receipt['output'])['observations'])
            elif row['owner'].get('method_transfer') is not None:
                episode = self._finish_episode(plan, [], [], [], 0.0, 0)
                episode['research_state'] = self._collect_transfer(row, done)
                self.lifecycle.commit(episode)
            # Unknown outcomes remain visible and never become lessons or zero-cost success.
            self._state['collected_jobs'][ident] = done
            self._collect_sharing(plan, ident, done)
            self._sync_sharing()
            self._event(plan, 'job_collected', job_id=ident, tool_id=row['tool_id'], **done)
