"""Snapshot-based mutation, duplicate checks, escalation and NoChild outcomes."""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy

from .llm.client import LLMError, accounted_cost, llm_error_metadata
from .provider_abandonment import AbandonedProviderRequest
from .prompt.build import EDIT_OPS, build, build_fix
from .prompt.parse import extract_code, has_code_block, parse_bet
from .prompt.patch import apply_edits, parse_edits
from .types import NoChild, Result
from .operators import usable_inspirations
from .validation import TRACE_SCHEMA, ValidationResult, _decision, trace_attempt, validate_candidate, source_sha256
from .validation import candidate_attempt_limit, candidate_registry_enabled
from .validation import graft_entrypoints as _graft_entrypoints  # noqa: F401 (back compatibility)
from ..campaign.budget import CampaignStopped, RequestNotDispatched
from ..eval.facade import EvaluationInfrastructureError
from ..evidence.measurement import evaluation_observation
from .viability import Violation, _docstring_repair, pre_eval_gate  # noqa: F401 (back compatibility)


def run_worker(op: str, snapshot, llm, evaluate_fn, dup_cap: int = 4,
               base_temp: float = 0.7, novelty_check=None, research=None, idea=None,
               intervention=None, circle_generation=None) -> Result | NoChild:
    options = dict(dup_cap=dup_cap, base_temp=base_temp, novelty_check=novelty_check,
                   research=research, idea=idea, intervention=intervention,
                   circle_generation=circle_generation)
    if research is None:
        return _run_worker(op, snapshot, llm, evaluate_fn, **options)
    with research.capture_outcomes() as captured:
        result = _run_worker(op, snapshot, llm, evaluate_fn, **options)
    result.meta['research_outcome_ids'] = list(captured['ids'])
    if captured['method_followup_records']:
        result.meta['method_followup_records'] = captured['method_followup_records']
    return result


def _run_worker(op: str, snapshot, llm, evaluate_fn, dup_cap: int = 4,
                base_temp: float = 0.7, novelty_check=None, research=None, idea=None,
                intervention=None, circle_generation=None) -> Result | NoChild:
    decision = deepcopy(getattr(snapshot, 'operator_decision', None)) or dict(
        requested=op, effective=op, selection='standalone', fallback_reason=None, on_policy=False)
    if decision['effective'] != op:
        raise ValueError('worker operator differs from its frozen decision')
    if op == 'cross':
        donors = usable_inspirations(snapshot.parent_id, snapshot.parent_code, snapshot.inspirations)
        snapshot = deepcopy(snapshot)
        snapshot.inspirations = donors
        if not donors:
            op = 'full_rewrite'
            decision.update(effective=op, fallback_reason='no_usable_crossover_donor', on_policy=False)
            snapshot.op = op
        snapshot.operator_decision = decision
    initial_op = op

    def operator_metadata():
        return dict(op_requested=decision['requested'], op_initial=initial_op,
                    op_fallback_reason=decision['fallback_reason'],
                    op_on_policy=bool(decision['on_policy'] and not escalated),
                    operator_selection=decision['selection'])

    if op == 'circle_generate':
        if circle_generation is None or research is None or circle_generation.research is not research:
            raise ValueError('circle generation must share the worker research controller')
        return circle_generation.run(snapshot, llm, evaluate_fn, base_temp=base_temp,
                                     novelty_check=novelty_check)
    if op == 'intervention':
        if intervention is None or research is None or intervention.research is not research:
            raise ValueError('intervention must share the worker research controller')
        return intervention.run(snapshot, llm, evaluate_fn, base_temp=base_temp,
                                novelty_check=novelty_check)
    if idea is not None and op == "idea":
        if research is None or idea.research is not research:
            raise ValueError("two-stage idea must share the worker research controller")
        return idea.run(snapshot, llm, evaluate_fn, base_temp=base_temp, novelty_check=novelty_check)
    registry_enabled = candidate_registry_enabled(novelty_check)
    dup_cap = candidate_attempt_limit(novelty_check, dup_cap)
    op_cur, temp = op, base_temp
    escalated = False
    llm_retries = 0
    n_empty = 0
    n_gate = 0
    novelty_checks = []
    evaluations = []
    measured_repeat = None
    n_output_repeats = 0
    validation_trace = {"schema": TRACE_SCHEMA, "profile": "worker", "attempts": []}
    spent = 0.0          # every call costs, not only the one that produced a child
    rejected_total, recent_rejections = 0, []
    previous_response = previous_validation = previous_reason = None
    edit_mode = research is None
    edit_base, fix_error = snapshot.parent_code, None

    def stop_unknown(exc):
        stopped = unknown_request_stop(exc, spent)
        validation_trace['attempts'].append({
            'attempt': attempt + 1, 'operator': op_cur, 'temperature': temp,
            'disposition': 'llm_failed', 'generation_returned': False,
            'cost': None, 'llm_error': stopped.llm_error, 'decisions': []})
        stopped.validation_trace = validation_trace
        return stopped

    def rejected(ident, status, category, diagnostic, trace):
        # These proposals never reached evaluation. Keep this dispatch's latest
        # bounded diagnostics separate from measured branch history and baseline.
        nonlocal rejected_total
        rejected_total += 1
        recent_rejections.append({
            "experiment_id": ident, "status": status, "category": category,
            "diagnostic": diagnostic,
            "proposed_sha256": trace["extracted_code_sha256"],
            "source_sha256": trace["candidate_code_sha256"],
            "quality_measured": False, "score": None, "actual_delta": None})
        del recent_rejections[:-3]

    for attempt in range(dup_cap):
        research_id = None
        try:
            prompt = (build_fix(edit_base, *fix_error) if op_cur == "fix" else
                      build(op_cur, snapshot, edits=edit_mode and op_cur in EDIT_OPS))
            if registry_enabled and previous_response is not None and op_cur != "fix":
                prompt = candidate_retry_prompt(prompt, previous_response, previous_validation,
                                                reason=previous_reason, measured_repeat=measured_repeat)
            if research is not None:
                options = {}
                if recent_rejections:
                    options["retry_feedback"] = {
                        "schema": "sciel-research-retry-feedback/1",
                        "total_rejections": rejected_total,
                        "omitted_count": rejected_total - len(recent_rejections),
                        "rejections": [dict(row) for row in recent_rejections]}
                research_id, prompt = research.begin(snapshot, attempt, op_cur, prompt, **options)
        except Exception as exc:
            if spent:
                # Preparation makes no provider request and never charges the
                # worker's earlier responses. Preserve those known costs for
                # Engine.run to book once while retaining the original error.
                exc.unbooked_known_cost = spent
                exc.known_cost = spent
                exc.current_call_cost_unknown = False
                exc.generation_returned = False
                exc.generation_dispatch = 'not_dispatched'
                exc.stage = 'generation_preparation'
                exc.validation_trace = validation_trace
            raise
        try:
            resp = llm.generate(prompt, snapshot.model, temp)
        except AbandonedProviderRequest as exc:
            # Recovery has authenticated and preserved the failed physical
            # request. Resolve this attempt without pretending it returned a
            # response, creating a hypothesis, charging an unknown amount or
            # retrying it. The caller books only earlier known charges once.
            abandonment = deepcopy(exc.receipt)
            diagnostic = llm_error_metadata(exc)
            validation_trace['attempts'].append({
                'attempt': attempt + 1, 'operator': op_cur, 'temperature': temp,
                'disposition': 'llm_failed', 'generation_returned': False,
                'generation_dispatch': 'unknown', 'cost': None,
                'current_call_cost_unknown': True,
                'provider_abandonment': abandonment, 'llm_error': diagnostic,
                'decisions': []})
            if research is not None:
                research.emit('generation_abandoned', experiment_id=research_id,
                    generation_returned=False, generation_dispatch='unknown',
                    cost=None, current_call_cost_unknown=True,
                    provider_abandonment=deepcopy(abandonment), llm_error=diagnostic)
                research.resolve(research_id, status='generation_stopped',
                                 failure='Provider request explicitly abandoned; outcome and cost unknown')
            return NoChild(island=snapshot.island, reason='llm_failed', cost=spent,
                meta={**operator_metadata(), **evaluation_metadata(evaluations), 'op_used': op_cur, 'dup_attempts': attempt,
                      'escalated': escalated, 'llm_retries': llm_retries,
                      'generation_returned': False, 'generation_dispatch': 'unknown',
                      'current_call_cost_unknown': True, 'provider_abandonment': abandonment,
                      'llm_error': diagnostic, 'novelty_checks': novelty_checks,
                      'validation_trace': validation_trace})
        except CampaignStopped as exc:
            if research is not None:
                research.abort(research_id, 'generation campaign stopped', spent,
                    generation_dispatch=('not_dispatched' if isinstance(exc, RequestNotDispatched) else 'unknown'))
            elif isinstance(exc, RequestNotDispatched):
                exc.unbooked_known_cost = spent
                exc.known_cost = spent
                exc.current_call_cost_unknown = False
                exc.generation_returned = False
                exc.generation_dispatch = 'not_dispatched'
                exc.stage = 'generation_dispatch'
                exc.validation_trace = validation_trace
            else:
                raise stop_unknown(exc) from None
            raise
        except LLMError as e:
            if research is not None:
                research.emit('generation_error', experiment_id=research_id,
                              generation_returned=False, generation_dispatch='unknown',
                              cost=None, current_call_cost_unknown=True,
                              llm_error=llm_error_metadata(e))
                research.abort(research_id, 'generation outcome unknown', spent)
                raise CampaignStopped('research generation outcome unknown') from None
            raise stop_unknown(e) from None
        except Exception as exc:
            if research is None:
                raise stop_unknown(exc) from None
            # Custom providers may raise outside LLMError. Prior returned calls
            # still cost money; this unknown dispatch must never be retried.
            reason = 'research generation outcome unknown: ' + type(exc).__name__
            try:
                research.abort(research_id, reason, spent)
            except Exception as cleanup_exc:
                reason += '; research bookkeeping failed: ' + type(cleanup_exc).__name__
                research.stopped_reason = reason
            raise CampaignStopped(reason) from None
        response_cost, child = None, None
        remeasurement = False
        evaluation_started = evaluation_returned = False
        try:
            try:
                response_cost = accounted_cost(resp)
            except ValueError:
                if research is not None:
                    research.record_generation_response(research_id, resp)
                    raise
                stopped = CampaignStopped("generation response cost is unknown or invalid")
                stopped.known_cost = spent
                stopped.current_call_cost_unknown = True
                stopped.stage = "generation"
                raise stopped from None
            spent += response_cost
            if research is not None:
                # Preserve known charges before optional response binding can
                # reject inconsistent evidence. This precedes all parsing.
                research.record_generation_response(research_id, resp)
            if getattr(resp, 'response_error', None) is not None:
                # The physical request returned with a known charge, but without
                # usable content. Book it once as a failed generation; no code or
                # efficacy evidence is inferred and no hidden request is made.
                validation_trace['attempts'].append({
                    'attempt': attempt + 1, 'operator': op_cur, 'temperature': temp,
                    'disposition': 'llm_failed', 'generation_returned': True,
                    'generation_dispatch': 'returned', 'cost': response_cost,
                    'response_error': resp.response_error, 'decisions': []})
                if research is not None:
                    pending = research.pending[research_id]
                    pending['generation_dispatch'] = 'returned'
                    pending['generation_cost'] = response_cost
                    research.emit('generation_error', experiment_id=research_id,
                        generation_returned=True, generation_dispatch='returned',
                        cost=response_cost, current_call_cost_unknown=False,
                        llm_error=resp.response_error)
                    research.resolve(research_id, status='generation_stopped',
                                     failure='Malformed returned provider response')
                return NoChild(island=snapshot.island, reason='llm_failed', cost=spent,
                    meta={**operator_metadata(), **evaluation_metadata(evaluations), 'op_used': op_cur, 'dup_attempts': attempt,
                          'escalated': escalated, 'llm_retries': llm_retries,
                          'response_error': resp.response_error,
                          'novelty_checks': novelty_checks, 'validation_trace': validation_trace})
            patch = None
            if edit_mode and (op_cur in EDIT_OPS or op_cur == "fix"):
                edits = parse_edits(resp.text)
                patch = apply_edits(edit_base, edits) if edits else None
            if patch is not None and not patch.applied:
                child = edit_base
                violation = Violation("patch", "no SEARCH block matched the edit base")
                validation = ValidationResult(child, violation, [], None, [_decision(
                    "patch", "resample", child, child, violation=violation, **patch.summary())])
            else:
                child = (patch.code if patch is not None else
                         "" if edit_mode and not has_code_block(resp.text) else extract_code(resp.text))
                validation = validate_candidate(
                    child, parent_code=snapshot.parent_code,
                    required_entrypoints=snapshot.required_entrypoints,
                    code_hashes=snapshot.code_hashes, novelty_check=novelty_check)
            attempt_trace = trace_attempt(
                validation, extracted_code=child, response=resp, attempt=attempt + 1,
                operator=op_cur, temperature=temp, cost=response_cost)
            if patch is not None:
                attempt_trace["patch"] = patch.summary()
            validation_trace["attempts"].append(attempt_trace)
            if registry_enabled:
                previous_response, previous_validation, previous_reason = resp.text, validation, None
                measured_repeat = None
            if research is not None and not research.proposal(research_id, resp, child, validation):
                failure = _bounded_failure("hypothesis_invalid", "missing or malformed pre-evaluation hypothesis")
                research.resolve(research_id, status='hypothesis_invalid',
                                 failure='missing or malformed pre-evaluation hypothesis')
                rejected(research_id, "hypothesis_invalid", "hypothesis_invalid", failure, attempt_trace)
                n_gate += 1
                previous_reason = failure
                attempt_trace['research'] = {'id': research_id, 'status': 'hypothesis_invalid'}
                if attempt + 1 < dup_cap:
                    op_cur, temp = _escalate(op_cur, temp)
                    attempt_trace['resample'] = {'operator': op_cur, 'temperature': temp}
                    escalated = True
                continue
            if validation.novelty is not None:
                novelty_checks.append(validation.novelty)
            if validation.violation is not None:
                if research is not None:
                    failure = _bounded_failure(validation.violation.kind, validation.violation.detail)
                    research.resolve(research_id, status='no_child',
                                     failure=failure)
                    rejected(research_id, "no_child", validation.violation.kind, failure, attempt_trace)
                if validation.violation.kind == "empty":
                    n_empty += 1
                elif validation.violation.kind != "duplicate":
                    n_gate += 1
                # Report the operator actually dispatched, not an escalation that
                # will never run after the final allowed rejection.
                if attempt + 1 < dup_cap:
                    if edit_mode and validation.violation.kind == "syntax" and op_cur != "fix":
                        line = re.search(r"\(line (\d+)\)", validation.violation.detail)
                        edit_base = validation.code
                        fix_error = (validation.violation.detail, int(line.group(1)) if line else None)
                        op_cur = "fix"
                    else:
                        op_cur, temp = _escalate(op_cur, temp)
                    attempt_trace["resample"] = {"operator": op_cur, "temperature": temp}
                    escalated = True
                continue
            child, grafted = validation.code, validation.grafted
            remeasurement = bool(registry_enabled and validation.novelty.get('remeasurement'))
            bet = parse_bet(resp.text)
            evaluation_started = True
            outcome = evaluate_fn(child)
            evaluation_returned = True
            res = outcome.result
            measured = evaluation_attempt(outcome, child)
            evaluations.append(measured)
            from .output_novelty import is_repeat
            repeated_output = measured['output_novelty'] is not None and is_repeat(measured['output_novelty']) and not remeasurement
            research_record = None
            if research is not None:
                if remeasurement:
                    research_record = research.resolve(research_id, status='no_child', source=child,
                        failure='Remeasurement of known source; no new mutation or efficacy claim. '
                                'The evaluator receipt records this measurement separately.')
                else:
                    research_record = research.resolve(research_id, status='evaluated' if res.ok else 'eval_failed',
                                     score=res.score if res.ok else None, source=child,
                                     evaluation=evaluation_observation(outcome),
                                     failure=None if res.ok else _evaluation_failure(res.artifacts),
                                     **({'output_novelty':measured['output_novelty']} if measured['output_novelty'] is not None else {}))
            if repeated_output:
                n_output_repeats += 1
                measured['admission'] = 'output_repeat_rejected'
                measured_repeat = {**deepcopy(measured), 'quality_measured': True, 'admission': 'output_repeat_rejected'}
                attempt_trace.update(disposition='output_repeat_rejected', evaluation=deepcopy(measured))
                previous_response, previous_validation, previous_reason = resp.text, None, None
                if attempt + 1 < dup_cap:
                    op_cur, temp = _escalate(op_cur, temp)
                    attempt_trace['resample'] = {'operator':op_cur, 'temperature':temp}
                    escalated = True
                continue
            meta = {**operator_metadata(), **evaluation_metadata(evaluations), "op_used": op_cur, "dup_attempts": attempt,
                    "escalated": escalated, "llm_retries": llm_retries,
                    "grafted": grafted, "empty_responses": n_empty,
                    "gate_rejections": n_gate, "novelty_checks": novelty_checks,
                    "validation_trace": validation_trace}
            if registry_enabled:
                meta.update(novelty=validation.novelty, remeasurement=remeasurement)
            if getattr(outcome, 'evaluation_id', None) is not None:
                meta['evaluation_id'] = outcome.evaluation_id
            if research_id is not None:
                meta['research_id'] = research_id
                meta['hypothesis'] = research_record['hypothesis']
            return Result(
                child_code=child, score=res.score, cases=res.cases, ok=res.ok,
                island=snapshot.island, parent_id=snapshot.parent_id,
                origin="gen", bet=bet, meta=meta,
                raw_first_score=outcome.raw_first_score, reached=outcome.reached,
                confirmed=outcome.confirmed, prompt=prompt, response=resp.text,
                cost=spent, timing=res.timing, artifacts=res.artifacts)
        except Exception as exc:
            if research is None:
                if not registry_enabled:
                    if isinstance(exc, EvaluationInfrastructureError):
                        exc.known_cost = exc.unbooked_known_cost = spent
                        exc.current_call_cost_unknown = response_cost is None
                        exc.generation_returned = True
                        exc.generation_dispatch = 'returned'
                        exc.stage = 'evaluation_infrastructure'
                        exc.validation_trace = validation_trace
                    raise
                if isinstance(exc, CampaignStopped) and hasattr(exc, 'known_cost'):
                    raise
                stopped = CampaignStopped('generation processing stopped: ' + type(exc).__name__)
                stopped.known_cost = spent
                stopped.unbooked_known_cost = spent
                stopped.current_call_cost_unknown = response_cost is None
                stopped.generation_returned = True
                stopped.generation_dispatch = 'returned'
                stopped.stage = 'evaluation' if evaluation_started else 'response_processing'
                stopped.validation_trace = validation_trace
                raise stopped from (exc if isinstance(exc, EvaluationInfrastructureError) else None)
            # A paid response cannot disappear when parsing, validation, novelty
            # or returned-result processing fails. Never retry an internal error.
            prefix = ('research evaluator raised: ' if evaluation_started and not evaluation_returned
                      else 'research response processing stopped: ')
            reason = prefix + type(exc).__name__
            research.stopped_reason = reason
            try:
                if research_id in research.pending:
                    pending = research.pending[research_id]
                    pending['generation_dispatch'] = 'returned'
                    if response_cost is not None:
                        pending['generation_cost'] = response_cost
                    research.resolve(research_id,
                        status=('generation_stopped' if isinstance(exc, EvaluationInfrastructureError) else
                                'no_child' if remeasurement else
                                'eval_failed' if evaluation_started and not evaluation_returned
                                else 'generation_stopped'),
                        source=child if isinstance(child, str) else '', failure=type(exc).__name__)
            except Exception as cleanup_exc:
                # Preserve pending work if its evidence cannot be safely closed;
                # checkpoint replay must fail closed, but known costs still count.
                reason += '; research bookkeeping failed: ' + type(cleanup_exc).__name__
                research.stopped_reason = reason
            finally:
                research.charge(spent)
            raise CampaignStopped(reason) from (exc if isinstance(exc, EvaluationInfrastructureError) else None)
    n_dup = dup_cap - n_empty - n_gate - n_output_repeats
    meta = {**operator_metadata(), **evaluation_metadata(evaluations), "op_used": op_cur, "dup_attempts": dup_cap, "escalated": escalated,
            "llm_retries": llm_retries, "empty_responses": n_empty,
            "gate_rejections": n_gate, "duplicates": n_dup,
            "output_repeat_rejections": n_output_repeats,
            "novelty_checks": novelty_checks, "validation_trace": validation_trace}
    # Empty completions and gate rejections are LLM problems, not island saturation:
    # when they dominate the exhausted attempts, do not let the scheduler read the
    # batch as dup_cap (which rests the island). Ties go to the LLM-problem reason, a
    # false saturation mark being worse than an undercounted one -- but the reason
    # must name what actually happened. The old test compared each count against the
    # duplicate count with `>=`, so four gate rejections and zero empty responses
    # reported "empty_response", which is the most common case of all.
    if measured_repeat is not None:
        reason = 'output_repeat_cap'
    elif n_dup > max(n_empty, n_gate):
        reason = "dup_cap"
    elif n_empty >= n_gate:
        reason = "empty_response"
    else:
        reason = "gate_rejected"
    return NoChild(island=snapshot.island, reason=reason, meta=meta, cost=spent)

from .worker_feedback import (
    _bounded_failure,
    _escalate,
    _evaluation_failure,
    candidate_retry_prompt,
    evaluation_attempt,
    evaluation_metadata,
    unknown_request_stop,
)
