"""Research hypotheses, paired evidence and measured resolution."""
from __future__ import annotations

import copy
import difflib
import hashlib
import math
from .llm.client import accounted_cost, response_resolved_model
from ..hypothesis import parse_hypothesis
from ..evidence.measurement import validate_evaluation
from ..hypothesis.controlled_changes import validate_controlled_change
from ..knowledge_base.catalogs import canonical_json, canonical_sha
from .research_protocol import (_OUTCOME_CAPTURES, digest)


class ResearchEvidence:

    def record_response(self, ident, response, *, response_format):
        """Persist a native structured reply before body/source validation.

        This creates no proposed source hash and never rewrites a response into
        the old marker/fence transport. A valid nested forecast can survive an
        invalid body as an unmeasured, rejected proposal.
        """
        pair_format = response_format == 'circle_proposal_pair/1'
        if response_format not in ('circle_restart_body/1', 'circle_proposal_pair/1'):
            raise ValueError('unknown structured research response format')
        record = self.pending[ident]
        if (record['operator'] != 'circle_generate' or ident in self._structured_responses
                or record['hypothesis'] is not None
                or record['proposed_sha256'] is not None or record['source_sha256'] is not None):
            raise ValueError('structured response is already recorded or belongs to another operator')
        self.record_generation_response(ident, response)
        cost = accounted_cost(response)
        previous_cost = record['generation_cost']
        if previous_cost is not None and (type(previous_cost) not in (int, float)
                                          or previous_cost != cost):
            raise ValueError('structured response differs from the recorded charge')
        raw = getattr(response, 'text', None)
        finish, refusal = getattr(response, 'finish_reason', None), getattr(response, 'refusal', None)
        provider_error = (None if finish == 'stop' and refusal in (None, '')
                          else 'structured_response_not_complete_or_refused')
        record.update(generation_cost=float(cost), generation_dispatch='returned')
        receipt = {'format': response_format, 'text': raw, 'cost': float(cost),
                   'finish_reason': finish, 'refusal': refusal, 'provider_error': provider_error}
        self._structured_responses[ident] = receipt
        # Known charge and raw transport are durable before even the pure parser.
        self.emit('structured_response', experiment_id=ident, response_format=response_format,
            response=(raw if pair_format else raw[:262144]) if type(raw) is str else None,
            response_sha256=digest(raw) if type(raw) is str else None,
            response_truncated=not pair_format and type(raw) is str and len(raw) > 262144,
            response_type=type(raw).__name__, generation_cost=float(cost),
            finish_reason=finish if type(finish) in (str, type(None)) else None,
            refusal=refusal if type(refusal) in (str, type(None)) else None,
            provider_error=provider_error,
            prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
            **({'resolved_model': response_resolved_model(response)}
               if response_resolved_model(response) is not None else {}))
        if pair_format:
            from ..hypothesis import circle_proposal_pair as pair
            decoded = pair.decode_response(raw)
            # Parsing alternatives does not select a forecast or create two
            # experiments/exposures. bind_pair_group owns that one-time choice.
            self.emit('structured_pair', experiment_id=ident, response_format=response_format,
                entries=[{'index': entry['index'],
                          'hypothesis': entry['hypothesis'].to_dict() if entry['hypothesis'] else None,
                          'error': entry['error'], 'body_error': entry['body_error']}
                         for entry in decoded['entries']],
                error=decoded['error'], provider_error=provider_error)
            return {**decoded, 'provider_error': provider_error}
        from ..hypothesis import circle_restart_body as body
        decoded = body.decode_response(raw)
        hyp = decoded['hypothesis']
        if (self.config.get('restart_predictions', False) and hyp is not None
                and hyp.restart_predictions is None):
            decoded.update(proposal=None, error='missing_restart_predictions')
        record['hypothesis'] = hyp.to_dict() if hyp is not None else None
        if hyp is not None:
            self.valid_hypotheses += 1
        self.emit('structured_hypothesis', experiment_id=ident, response_format=response_format,
            hypothesis=copy.deepcopy(record['hypothesis']), error=decoded['error'],
            provider_error=provider_error)
        return {**decoded, 'provider_error': provider_error}

    def _validated_pair_group(self, ident, group, response=None):
        """Check a terminal choice against this attempt's original paid reply."""
        from ..hypothesis import circle_proposal_pair as pair
        from ..hypothesis.circle_proposal_group import validate_group
        record = self.pending[ident]
        receipt = self._structured_responses.get(ident)
        if (receipt is None or receipt['format'] != pair.CONTRACT
                or record['operator'] != 'circle_generate'):
            raise ValueError('pair group requires its recorded native response')
        if (record['generation_dispatch'] != 'returned'
                or type(record['generation_cost']) not in (int, float)
                or record['generation_cost'] != receipt['cost']):
            raise ValueError('pair response charge or dispatch changed')
        if response is not None and (
                receipt['text'] != response.text or receipt['cost'] != accounted_cost(response)
                or receipt['finish_reason'] != getattr(response, 'finish_reason', None)
                or receipt['refusal'] != getattr(response, 'refusal', None)):
            raise ValueError('pair group differs from its durable raw response')
        checked = validate_group(group, raw_response=receipt['text'])
        from ..hypothesis.circle_source_family import builder_for_source
        source_builder = builder_for_source(self._parents[ident])
        if checked.get('execution_profile') != getattr(source_builder, 'EXECUTION_PROFILE', None):
            raise ValueError('pair group execution profile differs from measured parent')
        if (checked['request_id'] != ident
                or checked['parent_source_sha256'] != record['parent_source_sha256']
                or digest(self._parents[ident]) != record['parent_source_sha256']
                or checked['provider_error'] != receipt['provider_error']):
            raise ValueError('pair group differs from request, parent or provider metadata')
        if checked['assessment_complete'] is not True:
            raise ValueError('pair group still has pending admission gates')
        index = checked['selected_index']
        if index is not None:
            decoded = pair.decode_response(receipt['text'])
            entries = decoded['entries']
            if (receipt['provider_error'] is not None or type(index) is not int
                    or index not in (0, 1) or len(entries) != 2):
                raise ValueError('pair group cannot select this response')
            selected = entries[index]['proposal']
            entry = checked['entries'][index]
            if (selected is None or selected.body != entry['body']
                    or canonical_json(selected.hypothesis.to_dict()) != canonical_json(entry['hypothesis'])):
                raise ValueError('pair group selected body or forecast differs from raw slot')
            if (self.config.get('restart_predictions', False)
                    and selected.hypothesis.restart_predictions is None):
                raise ValueError('pair group selected a missing restart prediction')
            built = source_builder.apply(self._parents[ident], selected.body)
            if (digest(built['code']) != checked['selected_source_sha256']
                    or hashlib.sha256(canonical_json(built['proof']).encode()).hexdigest()
                       != checked['selected_proof_sha256']):
                raise ValueError('pair group selected source or proof differs from original parent')
        return checked

    def bind_pair_group(self, ident, response, group):
        """Bind at most one selected forecast; alternatives are never outcomes.

        This is a one-time durable commitment after all local gate assessments.
        The original pair response remains the only response; no single-entry
        surrogate is constructed. A no-selection terminal group is committed
        too, retaining its independent unmeasured forecasts and diagnostics.
        """
        record = self.pending[ident]
        receipt = self._structured_responses.get(ident)
        if ('proposal_group' in record or (receipt and 'proposal_group' in receipt)
                or record['hypothesis'] is not None
                or record['proposed_sha256'] is not None or record['source_sha256'] is not None):
            raise ValueError('pair group or research proposal is already committed')
        checked = self._validated_pair_group(ident, group, response)
        index = checked['selected_index']
        hypothesis = checked['entries'][index]['hypothesis'] if index is not None else None
        record['proposal_group'] = copy.deepcopy(checked)
        record['hypothesis'] = copy.deepcopy(hypothesis)
        # Keep a detached commitment outside mutable pending evidence so later
        # source commitment/resolve cannot silently substitute an alternative.
        receipt['proposal_group'] = copy.deepcopy(checked)
        if hypothesis is not None:
            self.valid_hypotheses += 1
        self.emit('proposal_group', experiment_id=ident, response_format=receipt['format'],
                  proposal_group=copy.deepcopy(checked), hypothesis=copy.deepcopy(hypothesis))
        return copy.deepcopy(checked)

    def _bound_pair_group(self, ident, response=None):
        record = self.pending[ident]
        receipt = self._structured_responses.get(ident)
        if receipt is None or 'proposal_group' not in receipt:
            raise ValueError('pair outcome requires a terminal assessed proposal group')
        group = self._validated_pair_group(ident, receipt['proposal_group'], response)
        if canonical_json(record.get('proposal_group')) != canonical_json(group):
            raise ValueError('committed pair group changed')
        index = group['selected_index']
        expected = group['entries'][index]['hypothesis'] if index is not None else None
        if canonical_json(record['hypothesis']) != canonical_json(expected):
            raise ValueError('selected pair forecast changed')
        if (record['generation_dispatch'] != 'returned'
                or type(record['generation_cost']) not in (int, float)
                or record['generation_cost'] != receipt['cost']):
            raise ValueError('pair response charge or dispatch changed')
        return group

    def proposal(self, ident, response, proposed, validation, *, controlled_change=None,
                 response_format=None):
        record = self.pending[ident]
        if record['proposed_sha256'] is not None or record['source_sha256'] is not None:
            raise ValueError('research proposal is already committed')
        structured = response_format is not None
        if structured:
            if response_format not in ('circle_restart_body/1', 'circle_proposal_pair/1'):
                raise ValueError('unknown structured research response format')
            receipt = self._structured_responses.get(ident)
            if (receipt is None or receipt['format'] != response_format
                    or receipt['provider_error'] is not None
                    or receipt['text'] != response.text or receipt['cost'] != accounted_cost(response)
                    or receipt['finish_reason'] != getattr(response, 'finish_reason', None)
                    or receipt['refusal'] != getattr(response, 'refusal', None)):
                raise ValueError('structured proposal differs from its durable raw response')
            from ..hypothesis import circle_restart_body as body
            from ..hypothesis.circle_source_family import builder_for_source
            source_builder = builder_for_source(self._parents[ident])
            if response_format == 'circle_proposal_pair/1':
                from ..hypothesis import circle_proposal_pair as pair
                group = self._bound_pair_group(ident, response)
                index = group['selected_index']
                if index is None:
                    raise ValueError('pair group has no selected proposal')
                decoded = pair.decode_response(response.text)['entries'][index]['proposal']
                if (digest(proposed) != group['selected_source_sha256']
                        or digest(validation.code) != group['selected_source_sha256']
                        or hashlib.sha256(canonical_json(controlled_change).encode()).hexdigest()
                           != group['selected_proof_sha256']):
                    raise ValueError('selected pair source or proof differs from committed group')
            else:
                decoded = body.parse_response(response.text)
            if (decoded is None or controlled_change is None
                    or controlled_change.get('operation') != source_builder.OPERATION
                    or controlled_change.get('target_body') != decoded.body
                    or canonical_json(record['hypothesis']) != canonical_json(decoded.hypothesis.to_dict())):
                raise ValueError('structured body or forecast differs from compiled proposal')
            hyp = decoded.hypothesis
        else:
            if ident in self._structured_responses:
                raise ValueError('structured response requires its explicit transport')
            hyp = parse_hypothesis(response.text)
        proof = None
        if controlled_change is not None:
            proof = validate_controlled_change(controlled_change,
                parent_source_sha256=record['parent_source_sha256'],
                proposed_sha256=digest(proposed), source_sha256=digest(validation.code),
                source_repaired=proposed != validation.code,
                parent_code=self._parents[ident], proposed_code=proposed, source_code=validation.code)
        cost = accounted_cost(response)
        self.record_generation_response(ident, response)
        record.update(hypothesis=hyp.to_dict() if hyp else None,
            proposed_sha256=digest(proposed), source_sha256=digest(validation.code),
            source_repaired=proposed != validation.code, generation_cost=cost,
            generation_dispatch='returned')
        if hyp and hyp.restart_predictions is not None:
            from ..knowledge_base.catalogs import canonical_sha
            record['restart_prediction_binding'] = {
                'schema': 'sciel-restart-prediction-binding/1',
                'spec_sha256': canonical_sha(hyp.restart_predictions),
                'response_sha256': digest(response.text),
                'proposed_sha256': digest(proposed), 'source_sha256': digest(validation.code)}
        if proof is not None:
            record['controlled_change'] = proof
        if hyp and not structured:
            self.valid_hypotheses += 1
        # This durable record precedes every evaluator invocation.
        self.emit('hypothesis', experiment_id=ident, evidence=copy.deepcopy(record),
                  response=response.text[:65536], response_sha256=digest(response.text),
                  response_truncated=len(response.text) > 65536,
                  prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens,
                  **({'resolved_model': response_resolved_model(response)}
                     if response_resolved_model(response) is not None else {}),
                  **({'response_format': response_format} if structured else {}))
        return hyp is not None

    def resolve(self, ident, *, status, score=None, source='', failure=None, evaluation=None,
                output_novelty=None):
        record = self.pending[ident]
        receipt = self._structured_responses.get(ident)
        if receipt is not None and receipt['format'] == 'circle_proposal_pair/1':
            if 'proposal_group' in receipt or 'proposal_group' in record:
                group = self._bound_pair_group(ident)
                if record['source_sha256'] is not None and (
                        record['source_sha256'] != group['selected_source_sha256']
                        or record['proposed_sha256'] != group['selected_source_sha256']
                        or hashlib.sha256(canonical_json(record.get('controlled_change')).encode()).hexdigest()
                           != group['selected_proof_sha256']):
                    raise ValueError('resolved pair source or proof differs from selected group')
            elif status != 'generation_stopped':
                raise ValueError('pair outcome requires a terminal assessed proposal group')
        if 'controlled_change' in record and source:
            validate_controlled_change(record['controlled_change'],
                parent_source_sha256=record['parent_source_sha256'],
                proposed_sha256=record['proposed_sha256'], source_sha256=record['source_sha256'],
                source_repaired=record['source_repaired'], parent_code=self._parents[ident],
                source_code=source)
        observation = validate_evaluation(evaluation)
        if observation is not None and "native_geometry" in observation:
            from ..eval.native_geometry import validate_binding
            if record['source_sha256'] is None:
                raise ValueError('native geometry requires a committed experiment source')
            validate_binding(observation['native_geometry'], observation,
                             source_sha256=record['source_sha256'], metric=record['metric'])
        if observation is not None:
            if status == 'evaluated':
                if not observation['ok'] or observation['score'] != score:
                    raise ValueError('returned evaluation differs from measured research score')
            elif status != 'eval_failed' or observation['ok']:
                raise ValueError('returned evaluation conflicts with research outcome status')
        repeated_output = False
        if output_novelty is not None:
            from .output_novelty import is_repeat
            from ..evidence.eligibility import validate_output_observation
            output_novelty = validate_output_observation(output_novelty,
                source_sha256=record['source_sha256'], metric=record.get('metric'),
                island=record.get('island'), evaluation=observation, status=status)
            repeated_output = is_repeat(output_novelty)
            record['output_novelty'] = output_novelty
        if status == 'evaluated':
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score)):
                raise ValueError('evaluated research result requires a finite score')
            if digest(source) != record['source_sha256']:
                raise ValueError('research evaluated source differs from committed source')
            delta = score - record['parent_score']
            if not math.isfinite(delta):
                raise ValueError('nonfinite research delta')
            record.update(score=score, actual_delta=delta)
            hyp = record['hypothesis']
            if hyp and not record['source_repaired']:
                record['forecast_verdict'] = ('CONFIRMED' if
                    hyp['predicted_low'] <= delta <= hyp['predicted_high'] else 'REFUTED')
            # Full source, not an unbound diff. Keep exceptionally large sources
            # out of the reusable fragment library while preserving their hashes.
            if not repeated_output and delta > 0 and len(source) <= 12000:
                record['code_fragment'] = source
        parent = self._parents[ident]
        if source and record['hypothesis'] and status in ('evaluated', 'eval_failed') and not repeated_output:
            if digest(parent) != record['parent_source_sha256'] or digest(source) != record['source_sha256']:
                raise ValueError('conclusion source differs from resolved source identities')
        record.update(status=status, failure=failure, evaluation=observation)
        record['change_excerpt'] = ''.join(difflib.unified_diff(
            parent.splitlines(keepends=True), source.splitlines(keepends=True),
            fromfile='evaluated_parent', tofile='candidate', n=2))[:6000] if source else ''
        record['change_excerpt_may_be_truncated'] = True
        self.knowledge.record(record)
        # Retire reviews with their ordinary evidence immediately. Waiting for
        # complete_step leaves an inconsistent window during worker retries.
        self._retire_conclusions()
        del self.pending[ident]
        del self._parents[ident]
        self._structured_responses.pop(ident, None)
        self.outcomes += 1
        self.emit('outcome', experiment_id=ident, evidence=record)
        for controller, captured in _OUTCOME_CAPTURES.get():
            if controller is self:
                captured['ids'].append(ident)
                if 'method_exposures' in record:
                    # Retention can evict an earlier retry before the parallel
                    # coordinator commits this step. Preserve the exact outcome
                    # just validated by KnowledgeBase and emitted above.
                    captured['method_followup_records'].append(copy.deepcopy(record))
        if record['hypothesis'] and status in ('evaluated', 'eval_failed') and not repeated_output:
            if source:
                self._conclusion_sources[ident] = (parent, source)
            self.queue.append(copy.deepcopy(record))
        return record

    def abort(self, ident, reason, spent=0.0, *, generation_dispatch='unknown'):
        if generation_dispatch not in ('unknown', 'not_dispatched', 'returned'):
            raise ValueError('invalid aborted generation dispatch state')
        try:
            if ident in self.pending:
                if self.pending[ident].get('generation_dispatch') != 'returned':
                    self.pending[ident]['generation_dispatch'] = generation_dispatch
                self.resolve(ident, status='generation_stopped', failure=reason)
        finally:
            # Even broken evidence persistence cannot erase prior known charges.
            # Unfinished state remains non-resumable and no call is replayed.
            self.stopped_reason = reason
            self._stopped_cause = None
            self.charge(spent)
