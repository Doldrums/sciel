"""Record sequential research evidence and admit exact saved candidates."""
from copy import deepcopy

from ..campaign.budget import CampaignStopped
from ..methods.models import canonical_bytes
from .canonical_evidence import authenticate as authenticate_canonical, summary as canonical_summary
from .research_steps import enqueue_method_trial, record_step
from .result_contract import RESULT_SCHEMA
from .tool_artifacts import _sha


class ResearchEpisodeEvidence:
    """Transient evidence for one durable, frozen research opportunity."""

    def __init__(self, tools, plan, snapshot, state, feedback, candidate_evaluator, pending):
        self.tools = tools
        self.plan = plan
        self.snapshot = snapshot
        self.state = state
        self.feedback = feedback
        self.candidate_evaluator = candidate_evaluator
        self.pending = pending
        # Current-opportunity evidence is separate from the frozen committed index.
        self.dispatch_plan = deepcopy(plan)
        self.candidate_submission = None
        self.candidate_submissions = []
        self.candidate_started = False
        for step in state['steps']:
            if step.get('canonical_evaluation_id'):
                self.add_canonical(step['canonical_evaluation_id'])

    def add_canonical(self, ident):
        saved, invocation = authenticate_canonical(self.tools.lifecycle, self.tools.library, ident,
            cell_id=self.plan['cell_id'], run_id=self.snapshot.research_targets['run_id'],
            task_identity=self.tools.task_identity)
        index = self.dispatch_plan.setdefault('shared_observations', {}).setdefault('canonical_evaluation_index', [])
        if not any(row['artifact_id'] == ident for row in index):
            index.append(canonical_summary(ident, saved, invocation))

    def append_step(self, action, status, *, proposal_id=None, observation=None, cost=0., scientific=False, **extra):
        step = {'action':action.get('action','invalid'), 'status':status,
            'proposal_id':proposal_id, 'receipt_id':None, 'tool_id':action.get('tool_id'),
            # Journal event timestamps record wall time; the resumable state
            # itself stays deterministic under durable wave replay.
            'observed_at':None,
            'requested_limits':deepcopy(action.get('limits')), 'resources':{},
            **{key:deepcopy(action[key]) for key in (
                'question','decision_if_supported','decision_if_refuted','experiment','next_decision','conclusion',
                'experimental_protocol')
                if key in action}, **extra}
        if observation is not None:
            step.update({key:deepcopy(observation.get(key)) for key in ('receipt_id','tool_id','resources')})
            step['target_binding'] = deepcopy(observation.get('target_binding'))
            step['requested_limits'] = deepcopy(self.tools._read_artifact(observation['receipt_id'],'invocation')['receipt']['limits'])
        record_step(self.state,step,cost=cost,scientific=scientific)
        self.tools._event(self.plan,'research_step',**step)
        if observation is not None and self.tools.config.get('experimental_protocol'):
            from .experiment_protocol import scope_key
            protocol = action.get('experimental_protocol') or {}
            source = observation['target_binding']['target_source_sha256']
            index = self.dispatch_plan.setdefault('shared_observations', {}).setdefault('check_index', [])
            index.append({'receipt_id':observation['receipt_id'],
                'status':observation['result']['status'],
                'scope_keys':list(dict.fromkeys(scope_key(source, value) for value in
                    (action.get('question'), protocol.get('question_key')) if value)),
                'conclusion':None})

    def submit_saved(self, action, *, proposal_id=None, cost=0.0, branch_trial=False):
        """One exact saved program through the canonical, durable callback.

            A replay of a method on another branch needs this same gate; merely
            executing its Python does not establish the resulting task score.
            """
        if self.candidate_submission is not None and self.candidate_submission['evaluation'].get('status') != 'rejected':
            raise ValueError('only one canonical candidate submission is allowed per opportunity')
        if len(self.candidate_submissions) >= 3:
            raise ValueError('candidate novelty refusal permits at most two additional submissions')
        if self.candidate_evaluator is None:
            raise ValueError('canonical candidate evaluator is unavailable in this execution context')
        saved = self.tools._stage_receipt(action['candidate_receipt_id'],self.plan,self.state,self.feedback)
        source = saved['receipt']['output']['observations'].get('candidate_source')
        if type(source) is not str or not source.strip() or len(source) > 256000:
            raise ValueError('the saved successful tool must return a complete observations.candidate_source string, at most 256000 characters')
        provenance = {'schema':'sciel-research-candidate-submission/1','cell_id':self.plan['cell_id'],
            'action_id':self.plan['action_id'],'receipt_id':action['candidate_receipt_id'],
            'iteration':self.plan['iteration'],
            'run_id':self.snapshot.research_targets['run_id'],'task_identity':deepcopy(self.tools.task_identity),
            'invocation_id':saved['receipt']['invocation_id'],'tool_id':saved['tool_id'],
            'candidate_source_sha256':_sha(source),'target_binding':deepcopy(saved['target_binding']),
            'discovery_resources':deepcopy(self.state['resources']),
            'known_llm_cost':self.state['known_llm_cost']+cost,'candidate_origin':'research_tool'}
        if branch_trial:
            provenance['independent_branch_trial'] = True
            provenance['reference_submission_receipt_id'] = self.pending['submission_receipt_id']
        stage_input = saved['inputs'].get('research_stage')
        if stage_input:
            provenance['stage_input_binding'] = {key:deepcopy(stage_input[key]) for key in
                ('prior_receipt_id','prior_input_sha256','prior_output_sha256','target_binding')}
        try:
            self.candidate_started = True
            evaluation = self.candidate_evaluator(source,deepcopy(provenance))
            canonical_bytes(evaluation)
        except Exception as error:
            # A possibly dispatched evaluation is never a planner repair.
            raise CampaignStopped('canonical research candidate submission could not be reconciled') from error
        self.candidate_submission = {'source':source,'provenance':provenance,'evaluation':deepcopy(evaluation)}
        self.candidate_submissions.append(self.candidate_submission)
        evidence_id = self.tools._artifact('candidate_evaluation',self.candidate_submission)
        self.add_canonical(evidence_id)
        self.feedback.append({'status':'canonical_evaluation','output':{
            'schema':RESULT_SCHEMA,'status':'success','observations':{'canonical_evaluation':evaluation},
            'diagnostics':{},'error':None},'result':{'status':'success','host_status':'canonical_evaluator'},
            'receipt_id':evidence_id,'reason':'Canonical task evaluation; method benefit requires a matched control.'})
        if not branch_trial and evaluation.get('status') == 'evaluated':
            enqueue_method_trial(self.state, {'tool_id':saved['tool_id'],
                'source_program_id':saved['target_binding']['candidate_id'],
                'source_sha256':saved['target_binding']['target_source_sha256'],
                'submission_receipt_id':action['candidate_receipt_id']})
        self.append_step(action,evaluation.get('status','evaluated'),proposal_id=proposal_id,cost=cost,
            canonical_evaluation_id=evidence_id,source_sha256=_sha(source),
            canonical_action_id=self.plan['action_id'],
            candidate_source_sha256=_sha(source),candidate_receipt_id=action['candidate_receipt_id'],
            candidate_tool_id=saved['tool_id'],candidate_invocation_id=saved['receipt']['invocation_id'],
            canonical_timing_seconds=evaluation.get('timing_seconds'),
            canonical_timing_scope=evaluation.get('timing_scope',
                'Returned evaluation-stage wall time including evaluator overhead; '
                'reported trial count is in cases; CPU unavailable'),
            **({'independent_branch_trial':True,'benefit':'not_established',
                'reference_submission_receipt_id':self.pending['submission_receipt_id']} if branch_trial else {}))
