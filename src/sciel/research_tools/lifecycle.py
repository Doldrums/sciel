"""Cell transitions and the separate inheritance history of research methods."""

from copy import deepcopy
from pathlib import Path
from ..methods.models import require_hash
from .cells import CellRegistry
from .result_contract import RESULT_SCHEMA
from .research_steps import validate_state
from .tool_protocol import LIFECYCLE_SCHEMA as SCHEMA

from . import lifecycle_evidence, lifecycle_delivery


class CellLifecycle:
    def __init__(self, run_dir, config, emit):
        self.config, self.emit = deepcopy(config), emit
        self.run_dir = Path(run_dir)
        self.cells = CellRegistry(run_dir, max_active=config['max_active_cells'])
        self._state = {'schema': SCHEMA, 'affinity': {}, 'methods': {}, 'cards': {}, 'research_state':{}}
        self._shared_evidence_index = None
        self.run_id = None  # Bound by ResearchTools to the frozen run / checkpoint provenance.

    def state(self):
        return {**deepcopy(self._state), 'cells': self.cells.state()}

    def _artifact(self, ident, kind, library):
        return lifecycle_evidence._artifact(self, ident, kind, library)

    def _receipt(self, ident, tool_id, library):
        return lifecycle_evidence._receipt(self, ident, tool_id, library)

    def load_state(self, state):
        return lifecycle_evidence.load_state(self, state)

    def _emit(self, phase, cell, **fields):
        self.emit(schema='sciel-research-tool-event/1', phase=phase,
            cell_id=cell['id'], cell_revision_id=cell['head_revision'],
            origin_island_id=cell['origin_island'], question=cell['question'], **fields)

    def _active_signals(self, signals):
        if type(signals) is not dict or signals.get('schema') != 'sciel-stagnation-signals/1':
            return []
        names = []
        for name in self.config['trigger_sources']:
            value = signals.get(name) or {}
            if name in ('population_stall', 'search_attempt_stall'):
                detected = value.get('threshold_reached') is True
            elif name == 'controller_plateau':
                detected = value.get('enabled') is True and value.get('active') is True
            else:
                detected = (value.get('enabled') is True and value.get('current_phase') is True
                    and value.get('detected') is True
                    and (value.get('mode') != 'shadow' or self.config['allow_shadow_triggers']))
            if detected:
                names.append(name)
        if type(signals.get('open_claim')) is str:
            names.append('open_claim')
        return names

    def prepare(self, snapshot, signals, *, maintenance=False, proactive=False, renew_closed=False):
        if type(proactive) is not bool:
            raise ValueError('proactive research reservation must be boolean')
        proactive_trigger = None
        if proactive:
            clock = signals.get('committed_iteration') if type(signals) is dict else None
            every = self.config.get('proactive_every_attempts')
            if (type(clock) is not int or clock < 0
                    or type(every) is not int or every < 1):
                raise ValueError('proactive reservation requires a committed clock and configured cadence')
            proactive_trigger = {'kind': 'proactive', 'sources': ['proactive'],
                'evidence': {'committed_iteration': clock, 'every_attempts': every,
                    'scope': 'Scheduled research opportunity; no stagnation is asserted.'}}
        active = self._active_signals(signals)
        island = str(snapshot.island)
        affinity = self._state['affinity'].get(island, [])
        cell = self.cells.get(affinity[-1]) if affinity else None
        previous_closed = None
        prior_state = self._state['research_state'].get(cell['id']) if cell else None
        if renew_closed and prior_state and prior_state['status'] not in ('open', 'step_limit'):
            previous_closed, cell = cell, None
        # One frozen episode owns a cell at a time. Wave restore starts from
        # the pre-prepare checkpoint, so this lease cannot strand a replay.
        if cell is not None and cell['active']:
            return None
        if proactive:
            # ensure_cell activates while creating. Check admission first so a
            # refused proactive reservation creates no dormant cell/artifact.
            if self.cells.active_count >= self.cells.max_active:
                return None
            if cell is None:
                question = ('Which reusable check or alternative approach could resolve an uncertainty '
                            'about the supplied candidate under this task and inform a future search decision? '
                            'Develop or retain evidence-backed options before a slowdown; '
                            'keep untested alternatives distinct from validated findings. '
                            f'Scheduling context: proactive research for island {snapshot.island}.')
                if previous_closed is not None:
                    question += (' New question after closed cell ' + previous_closed['id'][:12]
                                 + '; allocate a different uncertainty using its retained evidence.')
                cell = self.cells.ensure_cell(snapshot.island, question, proactive_trigger)
                cell = self.cells.revise(cell['id'], phase='propose',
                    summary='Research question opened by a proactive opportunity, without a stagnation claim.')
                self._emit('cell_created', cell, island_id=snapshot.island,
                    iteration=snapshot.iteration_no + 1, trigger=deepcopy(proactive_trigger))
                self._state['affinity'].setdefault(island, []).append(cell['id'])
            # Reuse the latest existing cell, including its private versions and
            # pending work. New population records do not fork proactive cells.
        # Pending work belongs to the enduring question that submitted it.
        # Maintenance alone cannot create a new research cell.
        elif active and (not maintenance or cell is None):
            anchor = (signals.get('population_stall') or {}).get('last_record_B_evals')
            question = ('Which uncertainty about the supplied candidate\'s behavior under this task '
                        'would affect the next search decision, and what reusable check can investigate it? '
                        f'Trigger context: island {snapshot.island}, record evaluation {anchor}; '
                        'this explains scheduling only.')
            if type(signals.get('open_claim')) is str:
                question += ' Open claim of the search theory worth resolving: ' + signals['open_claim']
            if previous_closed is not None:
                question += (' New question after closed cell ' + previous_closed['id'][:12]
                             + '; allocate a different uncertainty using its retained evidence.')
            previous_count = self.cells.cell_count
            cell = self.cells.ensure_cell(snapshot.island, question,
                {'kind':'stagnation', 'sources':active, 'evidence':deepcopy(signals)})
            if self.cells.cell_count > previous_count:
                cell = self.cells.revise(cell['id'], phase='propose',
                    summary='Research question opened from recorded search stagnation.')
                self._emit('cell_created', cell, island_id=snapshot.island,
                    iteration=snapshot.iteration_no+1, trigger=deepcopy(signals))
            ids = self._state['affinity'].setdefault(island, [])
            if cell['id'] not in ids:
                ids.append(cell['id'])
        if cell is None:
            return None
        stage, target, reference = None, None, None
        methods = list(reversed(self._state['methods'].items()))
        if (self.config['share_between_islands'] and not self.config['independent_research']
                and not maintenance and not proactive):
            for ident, method in methods:
                if (method['stage'] == 'locally_tested' and method['author_island_id'] != snapshot.island
                        and cell['id'] not in method['replay_attempts']):
                    stage, target, reference = 'independent_replay', ident, method['local_receipt_id']
                    break
            if stage is None:
                for ident, method in methods:
                    if (method['stage'] == 'released' and method['author_island_id'] != snapshot.island
                            and cell['id'] not in method['adoption_attempts']):
                        stage, target = 'adopt_trial', ident
                        break
            if stage is None:
                for ident, method in methods:
                    if (cell['id'] in method['adoptions'] and cell['id'] not in method['used_again']
                            and cell['id'] not in method['use_attempts']):
                        stage, target = 'use_again', ident
                        break
        if stage is None and (active or maintenance or proactive):
            stage = 'research'
        if stage is None or not self.cells.activate(cell['id']):
            return None
        cell = self.cells.get(cell['id'])
        result = dict(cell_id=cell['id'], cell_revision_id=cell['head_revision'],
            stage=stage, question=cell['question'], trigger_sources=(['proactive'] + active if proactive else active),
            stagnation_signals=deepcopy(signals), assigned_tool_id=target,
            reference_receipt_id=reference)
        if cell['id'] in self._state['research_state']:
            result['research_state'] = deepcopy(self._state['research_state'][cell['id']])
            if result['research_state']['question']:
                result['question'] = result['research_state']['question']
        if proactive:
            result['proactive'] = True
        self._emit('cell_activated', cell, island_id=snapshot.island,
            iteration=snapshot.iteration_no+1, stage=stage)
        return result

    def method(self, tool_id):
        """Detached single-method lifecycle; never copies the complete registry."""
        require_hash(tool_id, 'tool id')
        return deepcopy(self._state['methods'][tool_id])

    def adopted_tool_ids(self, cell_id):
        self.cells.get(cell_id)  # An orphan artifact does not grant adoption.
        return [ident for ident, method in self._state['methods'].items()
                if cell_id in method['adoptions']]

    def author_feedback(self, island_id, visible, library):
        return lifecycle_delivery.author_feedback(self, island_id, visible, library)

    def shared_observations(self, library, island_id, task_identity):
        return lifecycle_delivery.shared_observations(self, library, island_id, task_identity)

    def context(self, snapshot):
        return self.context_view(snapshot)['cards']

    def context_view(self, snapshot):
        return lifecycle_delivery.context_view(self, snapshot)

    def _delivery_measurement(self, card, episode=None):
        return lifecycle_delivery._delivery_measurement(self, card, episode)

    def delivery_context(self, context, *, recipient_cell_id=None, episode=None, max_cells=None):
        return lifecycle_delivery.delivery_context(self, context, recipient_cell_id=recipient_cell_id, episode=episode, max_cells=max_cells)

    def _commit_research_state(self, episode):
        value = episode.get('research_state')
        if episode['plan'].get('sharing_work_id') and value is not None:
            raise ValueError('independent sharing cannot overwrite a scientific question')
        if value is None:
            return
        validate_state(value)
        plan = episode['plan']
        self._state['research_state'][plan['cell_id']] = deepcopy(value)
        self.emit(schema='sciel-research-tool-event/1', phase='research_state',
            cell_id=plan['cell_id'], action_id=plan['action_id'], island_id=plan['island_id'],
            research_state={key:deepcopy(item) for key,item in value.items() if key != 'steps'})

    def pending(self, episode):
        """Commit submission provenance without claiming an executed observation."""
        plan = episode['plan']
        cell_id = plan['cell_id']
        # Earlier foreground stages remain committed evidence even when the
        # final stage starts a detached job. Submission must not erase them.
        published = self.commit(episode) if episode.get('invocations') else []
        self._commit_research_state(episode)
        evidence = [{'kind': 'tool_job', 'id': job['job_id'], 'tool_id': job['tool_id']}
                    for job in episode.get('submitted_jobs', [])]
        deferred = episode.get('deferred') is True and not evidence
        if not evidence and not deferred:
            raise ValueError('pending research episode has no submitted jobs')
        phase = 'background_deferred' if deferred else 'background_submitted'
        summary = ('Background research was not submitted; execution remains deferred.' if deferred else
                   'Background research submitted; execution and results are not yet established.')
        cell = self.cells.revise(cell_id, phase=phase, evidence=evidence, summary=summary)
        self._emit('cell_revision', cell, island_id=plan['island_id'],
            stage=phase, tool_ids=cell['tool_ids'], evidence=evidence)
        self.cells.deactivate(cell_id)
        return published

    def commit(self, episode):
        plan = episode['plan']
        cell_id, stage = plan['cell_id'], plan['stage']
        self._commit_research_state(episode)
        evidence = [{'kind':'tool_invocation','id':row['receipt_id'],'tool_id':row['tool_id']}
                    for row in episode['invocations']]
        evidence.extend({'kind':'candidate_evaluation','id':ident} for ident in dict.fromkeys(
            step['canonical_evaluation_id'] for step in (episode.get('research_state') or {}).get('steps', [])
            if step.get('canonical_evaluation_id')))
        if self.config.get('experimental_protocol'):
            current_receipts = {row['receipt_id'] for row in episode['invocations']}
            experiment_ids = {step['proposal_id'] for step in (episode.get('research_state') or {}).get('steps', [])
                if step.get('receipt_id') in current_receipts and step.get('proposal_id')
                and step.get('status') != 'proposal_invalid'}
            evidence.extend({'kind':'research_experiment','id':ident} for ident in sorted(experiment_ids))
        if episode.get('terminal_decision_proposal_id') is not None:
            evidence.append({'kind':'research_decision','id':episode['terminal_decision_proposal_id']})
        success = [row for row in episode['invocations']
                   if row['status']=='ok' and row['dispatch_status']=='invoked'
                   and row['result']['status']=='success']
        published = []
        if stage == 'research':
            for row in success:
                ident = row['tool_id']
                if ident not in self._state['methods']:
                    self._state['methods'][ident] = dict(author_cell_id=cell_id,
                        author_island_id=row['author_island_id'], stage='locally_tested',
                        replay_policy=row['replay_policy'],
                        local_receipt_id=row['receipt_id'], release_record=None, replay_attempts={},
                        adoption_attempts={}, adoptions={}, use_attempts={}, used_again={})
            for row in episode['invocations']:
                ident = row['tool_id']
                method = self._state['methods'].get(ident)
                if (method is not None and cell_id in method['adoptions']
                        and row['dispatch_status'] == 'invoked'):
                    method['used_again'][cell_id] = row['receipt_id']
                    self._emit('post_adoption_use', self.cells.get(cell_id),
                        tool_id=ident, receipt_id=row['receipt_id'], invocation_id=row['invocation_id'],
                        island_id=plan['island_id'], status=row['status'],
                        result_schema=RESULT_SCHEMA, tool_result_status=row['result']['status'],
                        tool_result_error=row['result']['error'],
                        adoption_revision_id=method['adoptions'][cell_id])
        else:
            ident = plan['assigned_tool_id']
            method = self._state['methods'][ident]
            row = episode['invocations'][-1]
            if stage == 'independent_replay':
                method['replay_attempts'][cell_id] = row['receipt_id']
                replay = episode.get('replay_evidence')
                if replay is None or replay['policy'] != method['replay_policy']:
                    raise ValueError('replay episode lacks its immutable policy evidence')
                matched = bool(success) and replay['matched'] is True
                if matched:
                    method['stage'] = 'released'
                    method['release_record'] = {'cell_id': cell_id, 'receipt_id': row['receipt_id'],
                        'reference_receipt_id': method['local_receipt_id'], **deepcopy(replay)}
                    published.append(ident)
                author = self.cells.assign_role(method['author_cell_id'], 'replayer', plan['island_id'])
                self._emit('cell_revision', author, island_id=plan['island_id'],
                    stage='replay_role_assignment', tool_ids=author['tool_ids'], evidence=[])
                self._emit('replay_result', self.cells.get(cell_id), tool_id=ident,
                    receipt_id=row['receipt_id'], reference_receipt_id=method['local_receipt_id'],
                    island_id=plan['island_id'], matched=matched,
                    status=('reproduced' if replay['measurement_reproduced'] else 'independently_executed')
                        if matched else 'not_reproduced',
                    replay_policy=replay['policy'], measurement_reproduced=replay['measurement_reproduced'],
                    reference_observations_sha256=replay['reference_observations_sha256'],
                    replay_observations_sha256=replay['replay_observations_sha256'],
                    comparison=('Exact declared observations; diagnostics and host resources are retained separately.'
                        if replay['policy']=='exact_observations' else
                        'Successful independent execution on the pinned input/runtime; measurement reproduction and benefit are not established.'))
            elif stage == 'adopt_trial':
                method['adoption_attempts'][cell_id] = row['receipt_id']
                if success:
                    revision = self.cells.revise(cell_id, phase='adopt', tool_ids=[ident], evidence=evidence,
                        summary='Recipient input passed execution contract; adopted for further research trials.',
                        decision={'rule':'recipient_contract_passed_for_further_trials', 'benefit':'not_established'})
                    method['adoptions'][cell_id] = revision['head_revision']
                    author = self.cells.assign_role(method['author_cell_id'], 'recipient', plan['island_id'])
                    self._emit('cell_revision', author, island_id=plan['island_id'],
                        stage='recipient_role_assignment', tool_ids=author['tool_ids'], evidence=[])
                    self._emit('adopted', revision, tool_id=ident, receipt_id=row['receipt_id'],
                        island_id=plan['island_id'], status='adopted_for_trials', benefit='not_established')
            elif stage == 'use_again':
                method['use_attempts'][cell_id] = row['receipt_id']
                self._emit('post_adoption_attempt', self.cells.get(cell_id), tool_id=ident,
                    receipt_id=row['receipt_id'], invocation_id=row['invocation_id'],
                    island_id=plan['island_id'], status=row['status'], dispatch_status=row['dispatch_status'],
                    result_schema=RESULT_SCHEMA, tool_result_status=row['result']['status'],
                    tool_result_error=row['result']['error'],
                    adoption_revision_id=method['adoptions'][cell_id])
                if row['dispatch_status']=='invoked':
                    method['used_again'][cell_id] = row['receipt_id']
                    self._emit('post_adoption_use', self.cells.get(cell_id),
                        tool_id=ident, receipt_id=row['receipt_id'], invocation_id=row['invocation_id'],
                        island_id=plan['island_id'], status=row['status'],
                        result_schema=RESULT_SCHEMA, tool_result_status=row['result']['status'],
                        tool_result_error=row['result']['error'],
                        adoption_revision_id=method['adoptions'][cell_id])
        phase = {'research':'execute_and_repair', 'independent_replay':'independent_replay',
                 'adopt_trial':'recipient_trial', 'use_again':'use_again'}[stage]
        if not (stage == 'adopt_trial' and success):
            current_tools = [row['tool_id'] for row in success][-self.config['max_visible_tools']:] or None
            self.cells.revise(cell_id, phase=phase, tool_ids=list(dict.fromkeys(current_tools)) if current_tools else None, evidence=evidence,
                summary='Recorded research episode; task benefit requires subsequent controlled evidence.')
        cell = self.cells.get(cell_id)
        if episode.get('cards') and not plan.get('sharing_work_id'):
            card = deepcopy(episode['cards'][-1])
            card['revision_id'] = cell['head_revision']
            self._state['cards'][cell_id] = card
        self._emit('cell_revision', cell, island_id=plan['island_id'],
            stage=stage, tool_ids=cell['tool_ids'], evidence=evidence)
        self.cells.deactivate(cell_id)
        return published
