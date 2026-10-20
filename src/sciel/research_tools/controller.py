"""Research-tool facade over planning, execution and durable cell history."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from ..methods.models import content_hash
from .config import validate_config, library_limits
from .controller_jobs import BackgroundToolsMixin
from .continuations import ContinuationMixin
from .sharing import SharingMixin
from .transfer_jobs import TransferJobsMixin
from .library import ToolLibrary, data_executor_factory
from .lifecycle import CellLifecycle
from .followups import FollowupLedger
from .observation_cards import format_context
from . import (
    controller_state,
    planning,
    tool_artifacts,
    prompt_sections,
    prompt_builder,
    episode_results,
    episode_execution,
    sequential_research,
)
from .tool_actions import _finite_json, _proposal_failure, response_format, parse_action
from .prompt_sections import _research_results_prompt, _feedback_card, _compact_tool_instructions
from .planning import _basis
from .tool_protocol import (
    CONTRACT,
    PROMPT_MARKER,
    ARTIFACT_SCHEMA,
    TARGET_VIEW_SCHEMA,
    TARGET_VIEW_MARKER,
    RESULTS_VIEW_MARKER,
)
from .observation_cards import _head_tail, _clip, _clip_json

class ResearchTools(SharingMixin, TransferJobsMixin, ContinuationMixin, BackgroundToolsMixin):
    def __init__(self, config, run_dir, *, task_identity, emit, charge, library=None, capacity_config=None, jobs=None):
        self.config = validate_config(config)
        self.task_identity = deepcopy(task_identity)
        self.emit, self.charge = emit, charge
        self.culture = None
        self.library = library or ToolLibrary(run_dir,
            {'image_id': self.config['image_id']}, library_limits(self.config),
            executor_factory=data_executor_factory(self.config['data_env'], self.config['task_root']))
        self.path = self.library.path / 'artifacts'
        self.path.mkdir(parents=True, exist_ok=True)
        from ..engine.source_identity import package_source_hashes
        implementation = {
            name.removeprefix('research_tools/'): digest
            for name, digest in package_source_hashes(packages=('research_tools',)).items()
        }
        integration = {
            name.removeprefix('engine/'): digest
            for name, digest in package_source_hashes(packages=('engine',)).items()
        }
        self._identity = {'contract': CONTRACT, 'config': self.config,
            'task': self.task_identity, 'library': self.library.identity,
            'implementation': implementation, 'integration': integration}
        data = self._identity['library']['executor'].get('data')
        self.data_note = '' if data is None else (
            f"DATA: read-only {data['target']} holds {data['files']} files "
            f"({data['bytes'] / 2 ** 20:.0f} MiB) of the task's development data; scored data is absent. "
            f"Read {data['target']}/README.txt first; load arrays with numpy mmap_mode='r'.\n")
        self._state = {'schema': CONTRACT, 'identity_sha256': content_hash(self._identity),
            'opportunities': {}, 'last_episode_opportunity': {}, 'registered': [], 'published': [],
            'committed_episodes': [], 'charges': {}, 'uses': {}, 'lifecycle': {}, 'followups': {},
            'last_proactive_iteration': None, 'run_id': None}
        self._init_continuations()
        self._init_sharing()
        self.followups = self._new_followup_ledger()
        self._authors = {}
        self.lifecycle = CellLifecycle(run_dir, self.config, emit)
        self._init_jobs(run_dir, capacity_config, jobs)

    @property
    def identity(self):
        return deepcopy(self._identity)

    def state(self):
        return {**deepcopy(self._state), 'lifecycle': self.lifecycle.state(),
                'followups': self.followups.state()}

    def load_state(self, state, *, expected_run_id=None):
        return controller_state.load_state(self, state, expected_run_id=expected_run_id)

    def _new_followup_ledger(self):
        return FollowupLedger(content_hash(self._identity),
            authenticate_exposure=self._authenticate_method_exposure,
            authenticate_outcome=self._authenticate_followup_outcome)

    def _authenticate_method_exposure(self, card):
        return controller_state._authenticate_method_exposure(self, card)

    def _authenticate_followup_outcome(self, record):
        return controller_state._authenticate_followup_outcome(self, record)

    def record_followup(self, record, *, child_program_id=None, link_kind='mutation', observed_record=None):
        return controller_state.record_followup(self, record, child_program_id=child_program_id, link_kind=link_kind, observed_record=observed_record)

    def _author(self, ident):
        if ident not in self._authors:
            self._authors[ident] = self.library.get(ident)['manifest']['protocol']['author_island']
        return self._authors[ident]

    def prepare(self, snapshot, *, signals=None):
        return planning.prepare(self, snapshot, signals=signals)

    def _proactive_reservation(self, signals):
        return planning._proactive_reservation(self, signals)

    def context(self, snapshot):
        return self.context_view(snapshot)['cards']

    def context_view(self, snapshot):
        return self.lifecycle.context_view(snapshot)

    def _bounded_cards(self, cards):
        budget, result = min(self.config['max_context_chars'], 10000), []
        for card in cards:
            if len(self.format_context(result + [card])) <= budget:
                result.append(deepcopy(card))
        return result

    @staticmethod
    def format_context(cards):
        return format_context(cards)

    def _event(self, plan, phase, **fields):
        self.emit(schema='sciel-research-tool-event/1', phase=phase,
            action_id=plan['action_id'], island_id=plan['island_id'],
            parent_program_id=plan['parent_program_id'], iteration=plan['iteration'],
            cell_id=plan.get('cell_id'), cell_revision_id=plan.get('cell_revision_id'),
            stage=plan.get('stage'), **fields)

    def _artifact(self, kind, payload):
        return tool_artifacts._artifact(self, kind, payload)

    def _read_artifact(self, ident, kind):
        return tool_artifacts._read_artifact(self, ident, kind)

    @staticmethod
    def _target_prompt(targets, budget, *, compact=False):
        return prompt_sections._target_prompt(targets, budget, compact=compact)

    def _prompt(self, plan, snapshot, available, feedback, attempt, *, research_state=None, scientific_step=0):
        return prompt_builder._prompt(self, plan, snapshot, available, feedback, attempt, research_state=research_state, scientific_step=scientific_step)

    @staticmethod
    def _followups_prompt(plan, budget):
        return prompt_sections._followups_prompt(plan, budget)

    def _known_jobs_prompt(self, plan, budget):
        return prompt_sections._known_jobs_prompt(self, plan, budget)

    def _replay_feedback_prompt(self, plan, budget=None):
        return prompt_sections._replay_feedback_prompt(self, plan, budget)

    def _charge_once(self, charge_id, response):
        return tool_artifacts._charge_once(self, charge_id, response)

    def _invoke(self, bundle, inputs, invocation_id, action_id, limits=None):
        return tool_artifacts._invoke(self, bundle, inputs, invocation_id, action_id, limits)

    def _run_tool(self, plan, snapshot, bundle, inputs, attempt, io, *, question=None, limits=None, proposal_id=None):
        return tool_artifacts._run_tool(self, plan, snapshot, bundle, inputs, attempt, io, question=question, limits=limits, proposal_id=proposal_id)

    def _record_invocation(self, plan, bundle, inputs, invocation_id, receipt, limits, question, *, proposal_id=None):
        return tool_artifacts._record_invocation(self, plan, bundle, inputs, invocation_id, receipt, limits, question, proposal_id=proposal_id)

    def _card(self, plan, observation, replay_comparison=None, *, decision=None):
        return episode_results._card(self, plan, observation, replay_comparison, decision=decision)

    def _decision_card(self, plan, card, proposal_id):
        return episode_results._decision_card(self, plan, card, proposal_id)

    def _assigned_episode(self, plan, snapshot, io):
        return episode_execution._assigned_episode(self, plan, snapshot, io)

    def _finish_episode(self, plan, registered, invocations, feedback, known_cost, attempts, *, replay_result=None, provider_abandonment=None, terminal_decision=None):
        return episode_results._finish_episode(self, plan, registered, invocations, feedback, known_cost, attempts, replay_result=replay_result, provider_abandonment=provider_abandonment, terminal_decision=terminal_decision)

    def execute(self, plan, snapshot, llm, *, io, candidate_evaluator=None):
        if self.config['max_scientific_steps'] == 1:
            return self._execute_single(plan, snapshot, llm, io=io)
        return self._execute_sequential(plan, snapshot, llm, io=io,
                                        candidate_evaluator=candidate_evaluator)

    def _execute_single(self, plan, snapshot, llm, *, io):
        return episode_execution._execute_single(self, plan, snapshot, llm, io=io)

    def _independent_branch(self, snapshot, pending, *, job_state=None):
        return episode_execution._independent_branch(self, snapshot, pending, job_state=job_state)

    def _method_pipeline(self, pending, cell_id, *, job_state=None):
        return episode_execution._method_pipeline(self, pending, cell_id, job_state=job_state)

    def _stage_receipt(self, ident, plan, state, feedback):
        return episode_execution._stage_receipt(self, ident, plan, state, feedback)

    def _execute_sequential(self, plan, snapshot, llm, *, io, candidate_evaluator):
        return sequential_research._execute_sequential(self, plan, snapshot, llm, io=io, candidate_evaluator=candidate_evaluator)

    def candidate_binding(self, episode, *, source, provenance, evaluation_id, score, cases):
        return episode_results.candidate_binding(self, episode, source=source, provenance=provenance, evaluation_id=evaluation_id, score=score, cases=cases)

    def _candidate_link(self, episode, binding, child_source):
        return episode_results._candidate_link(self, episode, binding, child_source)

    def commit(self, episode, *, child_program_id=None, link_kind='mutation', completed_iteration=None, candidate_binding=None, child_source=None):
        return episode_results.commit(self, episode, child_program_id=child_program_id, link_kind=link_kind, completed_iteration=completed_iteration, candidate_binding=candidate_binding, child_source=child_source)
