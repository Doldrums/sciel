"""Circle structured candidate selection and execution."""
from __future__ import annotations

from ..eval.facade import EvaluationInfrastructureError
from .llm.client import response_accounting, response_cost_provenance
import copy
from pathlib import Path
from ..campaign.budget import CampaignStopped, RequestNotDispatched
from ..hypothesis import circle_restart_body as builder
from ..hypothesis import circle_baseline_equivalence
from ..hypothesis.measurement import evaluation_observation
from .db import normalized_hash
from .prompt.sources import source_digest
from .types import NoChild, Result
from .validation import TRACE_SCHEMA, trace_attempt, validate_candidate
from .circle_generation_contracts import (CONTEXT_SCOPE, _finite, _json)


class CircleGenerationRun:

    @staticmethod
    def _group_counts(group):
        entries = group['entries']
        selected = group['selected_index'] is not None
        return {'returned_groups': 1, 'entries_present': len(entries),
                'forecast_valid_entries': sum(e['hypothesis'] is not None for e in entries),
                'body_valid_entries': sum(e['body'] is not None for e in entries),
                'eligible_entries': sum(e['eligible'] for e in entries),
                'selected_entries': int(selected), 'no_eligible_requests': int(not selected)}

    @staticmethod
    def _group_history(group):
        from .circle_pair_selection import digest
        return {'experiment_id': group['request_id'], 'group_sha256': digest(group),
                'response_sha256': group['response_sha256'], 'selected_index': group['selected_index'],
                'selected_source_sha256': group['selected_source_sha256']}

    def _source_history(self, proofs, code_hashes=()):
        """Retain exact and normalized consumed sources independently of DB admission.

        Only committed source proofs contribute: a merely proposed, unselected
        alternative is still available. Rebuild both sides of each validated
        proof without importing or executing its source.
        """
        from .external_research import generation_records
        external = generation_records(self.research)
        from .generated_paired_import import check as check_component
        from ..knowledge_base.generated_paired_history import proposal_records
        capsule = check_component(self.research, require_present=True)
        generated = proposal_records(capsule) if capsule is not None else {}
        consumed, normalized = set(), set(code_hashes)
        for value in [*proofs, *(row['proof'] for row in external.values()),
                      *(row['proof'] for row in generated.values())]:
            proof = self._proof(value)
            consumed.update(proof[k] for k in ('parent_source_sha256', 'child_source_sha256'))
            normalized.update(normalized_hash(self.builder.source_for(proof[k]))
                              for k in ('parent_body', 'target_body'))
        from .evaluated_study_import import check as check_study
        study = check_study(self.research, require_present=True)
        from .evaluated_review_import import check as check_study_review
        check_study_review(self.research, require_present=True)
        if study is not None:
            # check_study authenticated the complete retained dependency chain.
            # Ancestor sources remain consumed even if the current study's
            # editable registry or rendered memory no longer contains them.
            from ..knowledge_base.evaluated_detached_study import _retained_ancestry
            for source_sha, source in _retained_ancestry(study)['source_registry'].items():
                consumed.add(source_sha)
                normalized.add(normalized_hash(source))
        return consumed, normalized

    def _select_pair(self, snapshot, response, decoded, ident, novelty_check):
        from ..hypothesis.circle_proposal_group import build_group
        from . import circle_pair_selection as selection
        consumed, code_hashes = self._source_history(self.seen_proofs, snapshot.code_hashes)
        frozen = selection.snapshot(snapshot.parent_code, code_hashes, consumed,
            snapshot.required_entrypoints, novelty_enabled=novelty_check is not None,
            restart_predictions=self.research.config.get('restart_predictions', False))
        assessed = selection.assess(response.text, frozen,
            provider_error=decoded['provider_error'], novelty_check=novelty_check)
        selected = assessed['selected']
        group = build_group(request_id=ident, raw_response=response.text,
            parent_source_sha256=source_digest(snapshot.parent_code), gates=assessed['gates'],
            provider_error=decoded['provider_error'],
            selected_proof=selected['built']['proof'] if selected else None,
            **({'execution_profile': self.execution_profile} if self.execution_profile else {}))
        if group['assessment_complete'] is not True or group['selected_index'] != (selected['index'] if selected else None):
            raise ValueError('Circle pair group selection differs from assessment')
        self.research.bind_pair_group(ident, response, group)
        self._emit('proposal_group', group=group, selection_snapshot=frozen,
                   gate_evidence=assessed['evidence'], quality_measured=False)
        for key, value in self._group_counts(group).items():
            self.pair_counts[key] += value
        self.pair_history.append(self._group_history(group))
        self.pending['proposal_group_sha256'] = selection.digest(group)
        return selected, group

    def _finish(self, status, meta):
        meta['circle_generation'] = {**copy.deepcopy(self.pending), 'status': status}
        self._emit('finish', status=status, known_cost=self.pending['cost'])
        self.finished += 1
        self.pending = None

    def run(self, snapshot, llm, evaluate_fn, *, base_temp=.7, novelty_check=None):
        from .worker import _bounded_failure, _evaluation_failure
        if self.stopped_reason or self.research.stopped_reason:
            raise CampaignStopped(self.stopped_reason or self.research.stopped_reason)
        if self.pending is not None:
            raise ValueError('unfinished Circle generation cannot be replayed')
        prompt = self.prepare_prompt(snapshot)
        self.attempts += 1
        self.pending = {'research_id': None, 'parent_source_sha256': source_digest(snapshot.parent_code),
            'generation_dispatch': 'not_dispatched', 'cost': 0.0, 'cost_known': False,
            'proof': None, 'checks': None}
        meta = {'op_used': 'circle_generate', 'dup_attempts': 0, 'escalated': False,
            'llm_retries': 0, 'grafted': [], 'empty_responses': 0, 'gate_rejections': 0,
            'novelty_checks': [], 'validation_trace': {'schema': TRACE_SCHEMA, 'profile': 'worker', 'attempts': []}}

        def reject(reason, *, status='no_child', diagnostic=None):
            record = self.research.resolve(ident, status=status,
                failure=reason if diagnostic is None else diagnostic)
            meta.update(hypothesis=record['hypothesis'], gate_rejections=1)
            self._finish(reason, meta)
            return NoChild(snapshot.island, reason, meta, cost)

        ident, source, cost = None, '', 0.0
        evaluation_started = False
        try:
            if self.pair_mode:
                from ..hypothesis import circle_proposal_pair as wire
            else:
                wire = builder
            wire_format = wire.CONTRACT
            instruction = (wire.response_instruction(interface=self.builder.interface_description())
                if self.pair_mode and self.execution_profile is not None else
                wire.response_instruction() if self.pair_mode else self.builder.response_instruction())
            if self.research.config.get('restart_predictions', False):
                from ..hypothesis.restart_predictions import instruction as prediction_instruction
                instruction = instruction.replace('five-field hypothesis', 'six-field hypothesis')
                instruction += '\n' + prediction_instruction()
            ident, prompt = self.research.begin(snapshot, 0, 'circle_generate', prompt,
                response_instruction=instruction, context_scope=CONTEXT_SCOPE)
            self.pending['research_id'] = ident
            meta['research_id'] = ident
            self._emit('request_start', prompt=prompt, prompt_sha256=source_digest(prompt),
                model=snapshot.model, temperature=base_temp, response_format=self._response_format(),
                parent_source_sha256=source_digest(snapshot.parent_code))
            self.calls_started += 1
            self.pending['generation_dispatch'] = 'unknown'
            response = llm.generate(prompt, snapshot.model, base_temp, response_format=self._response_format())
            self.calls_returned += 1
            self.pending['generation_dispatch'] = 'returned'
            record = self.research.pending[ident]
            record['generation_dispatch'] = 'returned'
            try:
                accounting = response_accounting(response)
                returned_cost = accounting['known_usd']
            except ValueError:
                accounting, returned_cost = None, None
            if _finite(returned_cost):
                cost = float(returned_cost)
                self.known_cost += cost
                self.pending.update(cost=cost, cost_known=True)
                record['generation_cost'] = cost
            text = getattr(response, 'text', None)
            tokens = {k: getattr(response, k, None) for k in ('prompt_tokens', 'completion_tokens')}
            self._emit('request_result', response=text if type(text) is str else None,
                response_sha256=source_digest(text) if type(text) is str else None,
                response_truncated=False,
                cost=getattr(response, 'cost', None) if _finite(getattr(response, 'cost', None)) else None,
                **({'cost_accounting': accounting, 'cost_provenance': response_cost_provenance(response)}
                   if accounting is not None and accounting['unpriced_calls'] else {}),
                finish_reason=getattr(response, 'finish_reason', None), refusal=getattr(response, 'refusal', None),
                **{k:v if type(v) is int and v >= 0 else None for k,v in tokens.items()})
            if not self.pending['cost_known'] or type(text) is not str:
                raise ValueError('Circle generation response text or cost is unknown')
            if type(getattr(response, 'model', None)) is not str or any(type(v) is not int or v < 0 for v in tokens.values()):
                raise ValueError('Circle generation response metadata is invalid')
            decoded = self.research.record_response(ident, response, response_format=wire_format)
            checks = {key: decoded.get(key) for key in ('error', 'body_error', 'provider_error')}
            self.pending['checks'] = checks
            self._emit('response_checks', checks=checks, quality_measured=False)
            if self.pair_mode:
                selected, group = self._select_pair(snapshot, response, decoded, ident, novelty_check)
                if selected is None:
                    invalid = checks['error'] is not None or checks['provider_error'] is not None
                    self.invalid_responses += int(invalid)
                    reason = 'generation_invalid_response' if invalid else 'generation_pair_no_eligible'
                    return reject(reason, diagnostic=_bounded_failure(reason,
                        'Both alternatives are unmeasured; see the proposal group for each rejection.'))
                source, built, validation = selected['source'], selected['built'], selected['validation']
            else:
                proposal = decoded.get('proposal')
                if proposal is None or checks['provider_error'] is not None:
                    self.invalid_responses += 1
                    return reject('generation_invalid_response',
                        status='hypothesis_invalid' if decoded.get('hypothesis') is None else 'no_child',
                        diagnostic=_bounded_failure('generation_invalid_response', _json(checks)))
                source = self.builder.source_for(proposal.body)
                consumed, code_hashes = self._source_history(self.seen_proofs, snapshot.code_hashes)
                source_sha = source_digest(source)
                normalized_sha = normalized_hash(source)
                match_kind = ('current_parent' if source == snapshot.parent_code else
                    'previous_source' if source_sha in consumed else
                    'known_normalized_source' if normalized_sha in code_hashes else None)
                initializer_proof = None
                if match_kind is None and self.execution_profile is None:
                    equivalence = circle_baseline_equivalence.verify(proposal.body, snapshot.parent_code,
                        builder_source=Path(builder.__file__).read_bytes(),
                        proposal_source=Path(builder.proposals.__file__).read_bytes())
                    # Unknown syntax/dataflow is not evidence of equivalence. The
                    # narrow proof applies only to the exact pinned default parent,
                    # and concerns initializer values, not runtime or quality.
                    if equivalence['verdict'] == 'equivalent':
                        match_kind = 'baseline_initializer'
                        initializer_proof = equivalence['proof']
                if match_kind is not None:
                    # A duplicate has a known proposed fingerprint, but no new
                    # measurement or committed source proof. Keep that distinction
                    # in KB feedback as well as the low-level journal.
                    repetition = {'proposed_source_sha256': source_sha, 'match_kind': match_kind,
                        'matched_fingerprint': (source_digest(snapshot.parent_code) if initializer_proof else
                            normalized_sha if match_kind == 'known_normalized_source' else source_sha),
                        'quality_measured': False}
                    if initializer_proof is not None:
                        repetition.update(proof_sha256=initializer_proof['sha256'],
                            scope='Same initializer values as current control; runtime and quality unmeasured.')
                    self.repeated_proposals += 1
                    self._emit('plan_rejected',
                        reason='baseline_initializer_equivalence' if initializer_proof else 'repeated_source',
                        **repetition, **({'initializer_equivalence_proof': initializer_proof} if initializer_proof else {}))
                    diagnostic = _bounded_failure('generation_repeated', _json({**repetition,
                        'next_action': ('Change initializer values relative to the current parent; a baseline rewrite is not new.'
                            if initializer_proof else 'Change the executable body; revising the forecast alone is not a new candidate.')}))
                    return reject('generation_repeated', diagnostic=diagnostic)
                built = self.builder.apply(snapshot.parent_code, proposal.body)
                if built['code'] != source:
                    raise ValueError('Circle generation source assembly differs')
            proof = built['proof']
            self.pending['proof'] = copy.deepcopy(proof)
            self.seen_proofs.append(copy.deepcopy(proof))
            self._emit('source_binding', proof=proof, source_origin='deterministic_body_compiler',
                semantic_or_causal_claim=False)
            if not self.pair_mode:
                validation = validate_candidate(source, parent_code=snapshot.parent_code,
                    required_entrypoints=snapshot.required_entrypoints, code_hashes=code_hashes,
                    novelty_check=novelty_check)
            if validation.code != source or validation.grafted:
                raise ValueError('Circle generated source cannot be repaired or grafted')
            trace = trace_attempt(validation, extracted_code=source, response=response,
                attempt=1, operator='circle_generate', temperature=base_temp, cost=cost)
            trace['source_origin'] = 'deterministic_body_compiler'
            meta['validation_trace']['attempts'].append(trace)
            if validation.novelty is not None:
                meta['novelty_checks'].append(validation.novelty)
            if not self.research.proposal(ident, response, source, validation,
                    controlled_change=proof, response_format=wire_format):
                raise ValueError('accepted structured forecast changed')
            if validation.violation is not None:
                violation = validation.violation
                return reject('gate_rejected:' + _bounded_failure(violation.kind, violation.detail))
            self.evaluations += 1
            if self.pair_mode:
                self.pair_counts['evaluated_entries'] += 1
            evaluation_started = True
            outcome = evaluate_fn(source)
            result = outcome.result
            record = self.research.resolve(ident, status='evaluated' if result.ok else 'eval_failed',
                score=result.score if result.ok else None, source=source,
                evaluation=evaluation_observation(outcome),
                failure=None if result.ok else _evaluation_failure(result.artifacts))
            meta['hypothesis'] = record['hypothesis']
            self._finish('evaluated' if result.ok else 'eval_failed', meta)
            return Result(source, result.score, result.cases, result.ok, snapshot.island,
                snapshot.parent_id, 'gen', None, meta, outcome.raw_first_score, outcome.reached,
                outcome.confirmed, prompt, response.text, cost, result.timing, result.artifacts)
        except Exception as exc:
            dispatch = self.pending['generation_dispatch'] if self.pending else 'unknown'
            if dispatch != 'returned' and isinstance(exc, RequestNotDispatched):
                dispatch = 'not_dispatched'
            if self.pending is not None:
                self.pending['generation_dispatch'] = dispatch
            reason = 'Circle generation stopped: ' + type(exc).__name__
            self.stopped_reason = self.research.stopped_reason = reason
            try:
                if evaluation_started and not isinstance(exc, EvaluationInfrastructureError):
                    try:
                        if ident in self.research.pending:
                            self.research.resolve(ident, status='eval_failed', source=source, failure=type(exc).__name__)
                    finally:
                        self.research.charge(cost)
                else:
                    self.research.abort(ident, reason, cost, generation_dispatch=dispatch)
            except Exception as cleanup_exc:
                reason += '; research bookkeeping failed: ' + type(cleanup_exc).__name__
            self.stopped_reason = self.research.stopped_reason = reason
            try:
                self._emit('stopped', error_type=type(exc).__name__, known_cost=cost,
                    generation_dispatch=dispatch, current_call_cost_unknown=dispatch != 'not_dispatched'
                    and not bool(self.pending and self.pending['cost_known']))
            except Exception:
                pass
            raise CampaignStopped(reason) from (exc if isinstance(exc, EvaluationInfrastructureError) else None)
