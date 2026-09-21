"""Durable auxiliary research request preparation and response accounting."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from ..campaign.budget import CampaignStopped, RequestNotDispatched
from .llm.client import (
    llm_error_metadata,
    response_accounting,
    response_cost_provenance,
    response_resolved_model,
)
from .research_protocol import (_FrozenPayload, digest)


class ResearchAuxiliary:

    def _aux_schema_identity(self):
        from ..hypothesis.auxiliary_schema import identity
        value = identity(comparison_reviews=self.config.get('comparison_reviews', False),
            comparison_review_version=self.config.get('comparison_review_version', 1))
        if self.config.get('conclusion_source_refs', False):
            from ..hypothesis.conclusion_refs import identity as refs_identity
            value['conclusion_source_refs'] = refs_identity()
        return value

    def prepare_aux(self, kind, prompt, reference, *, conclusion_evidence=None, repair_evidence=None):
        """Reserve one C/R call on the coordinator without invoking transport.

        The returned Mapping is immutable, including nested options. A caller
        must dispatch it at most once and pass its response/error to consume_aux.
        Preparing a request commits its call index and aux_start journal entry.
        """
        if kind not in ('conclusion', 'reflection') or type(prompt) is not str:
            raise ValueError('invalid auxiliary request')
        if repair_evidence is not None and (kind != 'conclusion'
                or not self.config.get('conclusion_source_refs', False)
                or type(repair_evidence) is not dict or repair_evidence.get('id') != reference):
            raise ValueError('conclusion repair evidence must match this reference')
        if conclusion_evidence is not None and (repair_evidence is not None
                or kind != 'conclusion' or not self.config.get('conclusion_source_refs', False)
                or type(conclusion_evidence) is not dict or conclusion_evidence.get('id') != reference):
            raise ValueError('conclusion evidence must match this reference without repair evidence')
        if (kind == 'conclusion' and self.config.get('conclusion_source_refs', False)
                and conclusion_evidence is None and repair_evidence is None):
            raise ValueError('source-reference conclusion requires its saved evidence')
        from .generated_paired_import import check as check_component
        check_component(self, require_present=True)
        from .evaluated_study_import import check as check_study
        check_study(self, require_present=True)
        from .evaluated_review_import import check as check_study_review
        check_study_review(self, require_present=True)
        if self.stopped_reason:
            raise CampaignStopped(self.stopped_reason) from self._stopped_cause
        max_aux_calls = self.config['max_aux_calls']
        if max_aux_calls is not None and self.aux_calls >= max_aux_calls:
            self.emit('aux_skipped', kind=kind, reference=reference, reason='aux_call_limit')
            return None
        model = self.config['model'] or self.llm.models[-1]
        if model not in self.llm.models:
            raise ValueError('research model must be in the configured ensemble')
        options = {}
        if self.config.get('structured_auxiliary', False):
            from ..hypothesis.auxiliary_schema import response_format
            review_mode = self.config.get('comparison_reviews', False)
            review_version = self.config.get('comparison_review_version', 1)
            if self._aux_schema_identity() != self._auxiliary_schema_identity:
                raise ValueError('structured auxiliary schema changed')
            options['response_format'] = response_format(kind, comparison_reviews=review_mode,
                comparison_review_version=review_version)
            if kind == 'conclusion' and self.config.get('conclusion_source_refs', False):
                from ..hypothesis.conclusion_refs import evidence_response_format, repair_response_format
                options['response_format'] = (evidence_response_format(conclusion_evidence, model=model)
                    if repair_evidence is None else
                    repair_response_format(repair_evidence, model=model))
        # Omitted/null preserves the LLM's configured default. Only C (including
        # its one permitted format repair) gets this optional output budget.
        if kind == 'conclusion' and self.config.get('conclusion_max_tokens') is not None:
            options['max_tokens'] = self.config['conclusion_max_tokens']
        self.aux_calls += 1
        self.emit('aux_start', kind=kind, reference=reference, model=model,
                  call_index=self.aux_calls, prompt_sha256=digest(prompt),
                  **({'max_tokens': options['max_tokens']} if 'max_tokens' in options else {}),
                  **({'response_format': copy.deepcopy(options['response_format']),
                      'auxiliary_schema': copy.deepcopy(self._auxiliary_schema_identity)}
                     if 'response_format' in options else {}))
        request = _FrozenPayload.from_dict(dict(kind=kind, prompt=prompt,
            reference=reference, model=model, temperature=0.2, options=options,
            call_index=self.aux_calls, prompt_sha256=digest(prompt)))
        self._aux_requests[self.aux_calls] = request
        return request

    def consume_aux(self, request, response=None, error=None):
        """Consume a single issued receipt and account it exactly once.

        Already submitted siblings must still be consumed after another request
        stops the controller. Their known charges cannot disappear at a barrier.
        Scientific decoding/admission is a separate coordinator operation.
        """
        if error is not None and (not isinstance(error, Exception) or response is not None):
            raise ValueError('auxiliary receipt must contain response or error')
        if not isinstance(request, Mapping):
            raise ValueError('auxiliary request is not an issued mapping')
        call_index = request.get('call_index')
        issued = self._aux_requests.get(call_index) if type(call_index) is int else None
        if issued is None or dict(issued) != dict(request):
            raise ValueError('unknown, changed, or already consumed auxiliary request')
        # Remove before any fallible accounting/persistence, preventing duplicate
        # receipt delivery from ever invoking charge a second time.
        del self._aux_requests[call_index]
        kind, reference, options = issued['kind'], issued['reference'], issued['options']
        structured = 'response_format' in options
        if error is not None:
            from .provider_abandonment import AbandonedProviderRequest
            if isinstance(error, AbandonedProviderRequest):
                # Only an authenticated, explicitly abandoned physical request
                # may continue without a response. Unknown is not a zero charge
                # or a successful/incomplete model answer. The original receipt
                # remains durable, and this request cannot create a conclusion.
                fields = dict(kind=kind, reference=reference, call_index=call_index,
                    stage='request', generation_returned=False,
                    generation_dispatch='unknown', cost=None,
                    current_call_cost_unknown=True,
                    provider_abandonment=copy.deepcopy(error.receipt),
                    llm_error=llm_error_metadata(error))
                self.emit('aux_abandoned', **fields)
                self.emit('aux_skipped', reason='provider_request_abandoned', **fields)
                return None
            exc = error
            # No retry after an unknown outcome. The BudgetedLLM ledger, when
            # present, retains an unresolved reservation for reconciliation.
            self.stopped_reason = ('research auxiliary request stopped: ' + type(exc).__name__)
            self._stopped_cause = exc
            not_dispatched = isinstance(exc, RequestNotDispatched)
            try:
                self.emit('aux_stopped', kind=kind, reference=reference,
                          call_index=call_index, stage='request', generation_returned=False,
                          cost=0.0 if not_dispatched else None,
                          current_call_cost_unknown=not not_dispatched,
                          generation_dispatch='not_dispatched' if not_dispatched else 'unknown',
                          error_type=type(exc).__name__, llm_error=llm_error_metadata(exc))
            except Exception as persistence_exc:
                self.stopped_reason += '; stop persistence failed: ' + type(persistence_exc).__name__
                # The active exception can be a normal ParallelPause. Preserve
                # this independent persistence failure as a detached fatal cause,
                # so implicit exception context cannot relabel it as a pause.
                self._stopped_cause = CampaignStopped(
                    'Auxiliary stop persistence failed: ' + type(persistence_exc).__name__)
                raise CampaignStopped(self.stopped_reason) from self._stopped_cause
            if isinstance(exc, CampaignStopped):
                raise exc
            raise CampaignStopped(self.stopped_reason) from None
        cost = None
        try:
            accounting = response_accounting(response)
            cost = accounting['known_usd']
            # A returned paid response remains a cost even when its text or
            # metadata is unusable, or its journal write subsequently fails.
            self.aux_cost += cost
            self.charge(cost)
            text, returned_model = response.text, response.model
            if type(text) is not str or type(returned_model) is not str or not returned_model.strip():
                raise ValueError('malformed auxiliary response text/model')
            response_error = getattr(response, 'response_error', None)
            if response_error is not None:
                # Only the explicit client representation of a returned invalid
                # body may relax missing optional token counts. A malformed
                # receipt/invariant must still stop rather than be called a skip.
                from .llm.client import _ERROR_CAUSES
                if (type(response_error) is not dict
                        or set(response_error) != {'stage', 'reason', 'cause_type'}
                        or response_error['stage'] != 'response'
                        or response_error['reason'] not in ('empty_choices', 'malformed_response')
                        or type(response_error['cause_type']) is not str
                        or response_error['cause_type'] not in _ERROR_CAUSES | {'OtherError'}
                        or text != ''):
                    raise ValueError('malformed auxiliary response-error receipt')
            tokens = {}
            for key in ('prompt_tokens', 'completion_tokens'):
                value = getattr(response, key, None)
                if response_error is not None and value is None:
                    tokens[key] = None
                    continue
                if type(value) is not int or not 0 <= value <= 10**9:
                    raise ValueError('malformed auxiliary token metadata')
                tokens[key] = value
            metadata = ({'response_error': copy.deepcopy(response_error)}
                        if response_error is not None else {})
            resolved = response_resolved_model(response)
            if resolved is not None:
                metadata['resolved_model'] = resolved
            if accounting['unpriced_calls']:
                metadata.update(known_usd=cost, unpriced_calls=1,
                                cost_provenance=response_cost_provenance(response))
            if structured:
                try:
                    finish = getattr(response, 'finish_reason', None)
                    refusal = getattr(response, 'refusal', None)
                except Exception:
                    finish, refusal = None, '[unavailable refusal metadata]'
                usable = (response_error is None and type(finish) is str and finish == 'stop' and
                    (refusal is None or (type(refusal) is str and refusal == '')))
                metadata.update(finish_reason=finish if type(finish) is str and len(finish) <= 128 else None,
                    refusal=refusal[:2048] if type(refusal) is str else None,
                    refusal_truncated=type(refusal) is str and len(refusal) > 2048,
                    structured_response_usable=usable)
            self.emit('aux_result', kind=kind, reference=reference, model=returned_model,
                      call_index=call_index, cost=response.cost,
                      response=text[:32768], response_sha256=digest(text),
                      response_truncated=len(text) > 32768, **tokens, **metadata)
            if response_error is not None or (structured and not metadata['structured_response_usable']):
                terminal_error = (response_error is not None or metadata.get('finish_reason') == 'error')
                no_refusal = (not structured or refusal is None or (type(refusal) is str and refusal == ''))
                if (self.config.get('incomplete_auxiliary', 'stop') == 'skip'
                        and (terminal_error or metadata.get('finish_reason') == 'length')
                        and no_refusal):
                    # A known, already charged terminal response may be skipped
                    # by explicit run policy. Never retry it or admit partial text.
                    self.emit('aux_skipped', kind=kind, reference=reference,
                              call_index=call_index,
                              reason='response_error' if terminal_error else 'response_length',
                              cost=cost, generation_returned=True,
                              current_call_cost_unknown=False)
                    return None
                raise ValueError('structured auxiliary response incomplete or refused')
            return text
        except Exception as exc:
            self.stopped_reason = 'research auxiliary response stopped: ' + type(exc).__name__
            self._stopped_cause = exc
            try:
                self.emit('aux_stopped', kind=kind, reference=reference,
                          call_index=call_index, stage='response', generation_returned=True,
                          cost=cost, current_call_cost_unknown=cost is None,
                          error_type=type(exc).__name__)
            except Exception as persistence_exc:
                self.stopped_reason += '; stop persistence failed: ' + type(persistence_exc).__name__
                # The active exception can be a normal ParallelPause. Preserve
                # this independent persistence failure as a detached fatal cause,
                # so implicit exception context cannot relabel it as a pause.
                self._stopped_cause = CampaignStopped(
                    'Auxiliary stop persistence failed: ' + type(persistence_exc).__name__)
                raise CampaignStopped(self.stopped_reason) from self._stopped_cause
            raise CampaignStopped(self.stopped_reason) from None

    def _aux(self, kind, prompt, reference, *, conclusion_evidence=None, repair_evidence=None):
        """Serial adapter over the same coordinator/transport receipt boundary."""
        request = self.prepare_aux(kind, prompt, reference,
            conclusion_evidence=conclusion_evidence, repair_evidence=repair_evidence)
        if request is None:
            return None
        try:
            response = self.llm.generate(request['prompt'], request['model'],
                temperature=request['temperature'], **request['options'])
        except Exception as exc:
            return self.consume_aux(request, error=exc)
        return self.consume_aux(request, response=response)
