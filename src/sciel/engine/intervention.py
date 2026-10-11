"""One hypothesis-only request, followed by an exact, source-bound Circle edit.

The model chooses a fixed operation; it never supplies candidate code. A receipt
proves which assignment the builder changed, not why a score changed. There is
no retry, implicit alternative, evaluator invocation at import, or search RNG.
"""
from __future__ import annotations

from ..eval.facade import EvaluationInfrastructureError

from .llm.client import response_accounting, response_cost_provenance

import copy
import hashlib
import json
import math

from ..campaign.budget import CampaignStopped, RequestNotDispatched
from ..hypothesis import circle_interventions as edits
from ..evidence.measurement import evaluation_observation
from ..hypothesis.models import CONTRACT as RESEARCH_CONTRACT
from ..hypothesis.parsing import parse_hypothesis
from .db import normalized_hash
from .prompt.sources import source_bundle, source_catalog_block, source_digest, source_reference
from .types import NoChild, Result
from .validation import TRACE_SCHEMA, ValidationResult, trace_attempt, validate_candidate

CONTRACT = 'sciel-circle-intervention-worker/2'
DEFAULT_OPERATION = 'circle_restart_reference/1'
_COUNTERS = ('attempts', 'finished', 'calls_started', 'calls_returned',
             'invalid_plans', 'repeated_plans', 'exhausted', 'evaluations')


def response_instruction():
    return (f'Research protocol {RESEARCH_CONTRACT}; controlled Circle intervention. '
        'Return exactly HYPOTHESIS_JSON: followed by one unfenced JSON object, and nothing else. '
        'Use exactly hypothesis, predicted_low, predicted_high, reasoning, confidence. '
        'hypothesis must be one exact allowed label from ALLOWED_INTERVENTIONS_JSON. '
        'reasoning explains your uncertain forecast, not a measured or causal result. '
        'predicted_low/high bound score(child) minus the measured parent score, in the '
        'task-owned metric units, not raw sum_radii; low <= high and confidence in [0,1]. '
        'All numbers must be finite, not booleans. Do not return code, a code fence, '
        'a patch, another JSON block or surrounding prose. The engine builds the exact '
        'source edit after accepting your plan.\n')


def builder_for(operation=DEFAULT_OPERATION):
    """Resolve only the two explicit pure source builders; never import a solver."""
    if operation == DEFAULT_OPERATION:
        return edits
    if operation == 'circle_restart_scale/1':
        from ..hypothesis import circle_restart_scales
        return circle_restart_scales
    raise ValueError('unknown Circle intervention operation')


def parse_target(text, operation=DEFAULT_OPERATION):
    builder = builder_for(operation)
    # The shared parser supplies the strict five-field/numeric/duplicate-key
    # validation. This extra envelope guard forbids its ordinary code transport.
    if type(text) is not str or not text.strip().startswith('HYPOTHESIS_JSON:'):
        return None
    hypothesis = parse_hypothesis(text)
    if hypothesis is None:
        return None
    try:
        raw = json.loads(text.strip()[len('HYPOTHESIS_JSON:'):])
    except (ValueError, RecursionError):
        return None
    for policy in builder.POLICIES:
        if raw['hypothesis'] == f'{builder.OPERATION}:{policy}':
            return policy
    return None


class CircleIntervention:
    def __init__(self, research, operation=DEFAULT_OPERATION):
        if research is None:
            raise ValueError('Circle intervention requires enabled research')
        self.research = research
        self.edits = builder_for(operation)
        self._family = {policy: self.edits.source_for(policy) for policy in self.edits.POLICIES}
        self._identity = {'schema': CONTRACT, 'builder_contract': self.edits.CONTRACT,
            'operation': self.edits.OPERATION, 'kernel_source_sha256': self.edits.PINNED_SOLVER_SHA256,
            'family_source_sha256': {p: source_digest(c) for p, c in self._family.items()},
            'wire': 'hypothesis-only/1'}
        for name in _COUNTERS:
            setattr(self, name, 0)
        self.known_cost = 0.0
        self.seen_plans = []
        self.pending = None
        self.stopped_reason = None

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def _emit(self, action, **fields):
        self.research.emit('intervention', intervention_contract=CONTRACT, action=action,
            experiment_id=self.pending.get('research_id') if self.pending else None, **fields)

    def _finish(self, status, meta):
        meta['intervention'] = {**copy.deepcopy(self.pending), 'status': status}
        self._emit('finish', status=status, known_cost=self.pending['cost'])
        self.finished += 1
        self.pending = None

    def _prompt(self, snapshot, allowed, proofs):
        bundle = source_bundle(snapshot, include_inspirations=False)
        # These receipts were derived from the exact verified measured source,
        # not a program's model-written card or an assumed initial-grid baseline.
        first = proofs[allowed[0]]
        descriptor = self.edits.policy_descriptor(first['parent_policy'])
        current = {'policy': first['parent_policy'],
            'source_sha256': first['parent_source_sha256'],
            'assignment': copy.deepcopy(first['before']),
            'description': descriptor['change'],
            'uncertain_effect': descriptor['uncertain_effect']}
        policies = [{**self.edits.policy_descriptor(p), 'hypothesis': f'{self.edits.OPERATION}:{p}',
                     'from_policy': proofs[p]['parent_policy'], 'to_policy': p,
                     'before_assignment': copy.deepcopy(proofs[p]['before']),
                     'after_assignment': copy.deepcopy(proofs[p]['after'])} for p in allowed]
        intro = ('Choose one of these untried controlled restart-reference interventions. '
            'Only the assignment reference changes; random draw, clamps, solver, incumbent '
            'retention and 27-second / 10000-restart / 15000-iteration limits remain fixed. '
            'These are a finite hand-authored search space, not newly discovered algorithm families.\n')
        baseline_warning = 'the parent always uses the original-grid policy.\n'
        if self.edits.OPERATION == 'circle_restart_scale/1':
            intro = ('Choose one of these untried controlled restart-scale interventions. '
                'Only the symmetric uniform noise bounds change; this perturbs both centers and radii. '
                'The incumbent reference, random draw shape, clamps, solver, incumbent retention '
                'and 27-second / 10000-restart / 15000-iteration limits remain fixed. '
                'These are a finite hand-authored search space, not newly discovered algorithm families.\n')
            baseline_warning = 'the parent always uses the original 0.06 noise bound.\n'
        return (intro +
            'CURRENT_PARENT_POLICY_JSON identifies the measured parent from exact verified source bytes. '
            'Its policy, source hash, assignment and description are source facts. All uncertain_effect '
            'fields below are unverified interpretations, not measured benefits. Forecast the transition '
            'from this current parent, using its score and cases in RESEARCH_CONTEXT_JSON; do not assume '
            + baseline_warning + 'CURRENT_PARENT_POLICY_JSON:\n' + json.dumps(current, ensure_ascii=True) + '\n'
            'ALLOWED_INTERVENTIONS_JSON:\n' + json.dumps(policies, ensure_ascii=True) + '\n'
            f'TASK:\n{snapshot.task_prompt}\nDIRECTIVE:\n{snapshot.directive}\n'
            f'PARENT SOURCE:\n{source_reference(bundle["target_ref"])}\n'
            'LESSONS:\n' + '\n'.join(snapshot.kb_lessons) + '\n' + source_catalog_block(bundle))

    def prepare_prompt(self, snapshot, allowed_targets=None):
        """Build the exact base prompt without dispatch, mutation, or evaluation.

        A supplied target list only narrows currently eligible choices. It does
        not consume omitted policies or bypass observed-source deduplication.
        ``prompt`` is None for an unknown parent or an exhausted choice set.
        """
        ctx = snapshot.research_context
        if (type(ctx) is not dict or ctx.get('parent_code') != snapshot.parent_code
                or ctx.get('parent_source_sha256') != source_digest(snapshot.parent_code)
                or ctx.get('parent_id') != snapshot.parent_id
                or ctx.get('parent_score') != snapshot.parent_score):
            raise ValueError('Circle intervention must use the exact measured parent')
        # Recheck the pinned kernel at dispatch, even if the object predates an
        # on-disk edit. Unknown parents are expected rejection; kernel drift is not.
        family = {p: self.edits.source_for(p) for p in self.edits.POLICIES}
        if family != self._family:
            raise ValueError('Circle intervention source identity changed')
        parent_policy = next((p for p, code in family.items() if code == snapshot.parent_code), None)
        # A different directed edit can produce exactly the same source. Active
        # island membership alone forgets evicted parents and failed children.
        # Derive the consumed-source closure from the existing journal-bound
        # proofs: their parents were measured, their children already proposed.
        known_proofs = {proof['plan_key']: proof
            for p, code in family.items() for target in family if target != p
            for proof in [self.edits.apply(code, target)['proof']]}
        consumed_sources = set()
        for key in self.seen_plans:
            if key not in known_proofs:
                raise ValueError('unknown consumed Circle intervention plan')
            consumed_sources.update(known_proofs[key][field]
                for field in ('parent_source_sha256', 'child_source_sha256'))
        allowed, proofs = [], {}
        if parent_policy is not None:
            for policy in family:
                if policy == parent_policy:
                    continue
                proof = self.edits.apply(snapshot.parent_code, policy)['proof']
                proofs[policy] = proof
                if (proof['plan_key'] not in self.seen_plans
                        and proof['child_source_sha256'] not in consumed_sources
                        and normalized_hash(family[policy]) not in snapshot.code_hashes):
                    allowed.append(policy)
        if allowed_targets is not None:
            if (type(allowed_targets) is not list
                    or any(type(p) is not str or p not in allowed for p in allowed_targets)
                    or len(set(allowed_targets)) != len(allowed_targets)):
                raise ValueError('allowed targets must be unique eligible Circle policies')
            # Keep builder order, independent of the caller's container order.
            allowed = [p for p in allowed if p in allowed_targets]
        return {'prompt': self._prompt(snapshot, allowed, proofs) if allowed else None,
                'proofs': copy.deepcopy(proofs), 'allowed': allowed,
                'parent_policy': parent_policy}

    def run(self, snapshot, llm, evaluate_fn, *, base_temp=.7, novelty_check=None):
        from .worker import _bounded_failure, _evaluation_failure
        if self.stopped_reason or self.research.stopped_reason:
            raise CampaignStopped(self.stopped_reason or self.research.stopped_reason)
        if self.pending is not None:
            raise ValueError('unfinished Circle intervention cannot be replayed')
        prepared = self.prepare_prompt(snapshot)
        allowed, proofs, parent_policy = (prepared[k] for k in ('allowed', 'proofs', 'parent_policy'))
        ctx = snapshot.research_context
        self.attempts += 1
        self.pending = {'research_id': None, 'parent_source_sha256': source_digest(snapshot.parent_code),
            'parent_policy': parent_policy, 'allowed_targets': allowed, 'cost': 0.0,
            'cost_known': False, 'generation_dispatch': 'not_dispatched',
            'proof': None, 'plan_key': None}
        meta = {'op_used': 'intervention', 'dup_attempts': 0, 'escalated': False,
            'llm_retries': 0, 'grafted': [], 'empty_responses': 0,
            'gate_rejections': 0, 'novelty_checks': [],
            'validation_trace': {'schema': TRACE_SCHEMA, 'profile': 'worker', 'attempts': []}}

        def no_child(reason):
            cost = self.pending['cost']
            self._finish(reason, meta)
            return NoChild(snapshot.island, reason, meta, cost)

        if parent_policy is None:
            return no_child('intervention_parent_unknown')
        if not allowed:
            self.exhausted += 1
            return no_child('intervention_exhausted')
        ident, source = None, ''
        evaluation_started = False
        try:
            ident, prompt = self.research.begin(snapshot, 0, 'intervention', prepared['prompt'],
                                                response_instruction=response_instruction())
            self.pending['research_id'] = ident
            meta['research_id'] = ident
            self._emit('request_start', prompt=prompt, prompt_sha256=source_digest(prompt),
                model=snapshot.model, temperature=base_temp, parent_id=snapshot.parent_id,
                parent_source_sha256=source_digest(snapshot.parent_code), island=snapshot.island,
                iteration=snapshot.iteration_no + 1, allowed_targets=allowed,
                injected_ids=copy.deepcopy(ctx['injected_ids']))
            self.calls_started += 1
            self.pending['generation_dispatch'] = 'unknown'
            response = llm.generate(prompt, snapshot.model, base_temp)
            self.calls_returned += 1
            self.pending['generation_dispatch'] = 'returned'
            record = self.research.pending[ident]
            record['generation_dispatch'] = 'returned'
            try:
                accounting = response_accounting(response)
                cost, cost_known = accounting['known_usd'], True
            except ValueError:
                cost, cost_known, accounting = None, False, None
            if cost_known:
                self.known_cost += cost
                self.pending['cost'] = cost
                self.pending['cost_known'] = True
                record['generation_cost'] = cost
            text = getattr(response, 'text', None)
            tokens = {k: getattr(response, k, None) for k in ('prompt_tokens', 'completion_tokens')}
            self._emit('request_result', generation_returned=True, cost=getattr(response, 'cost', None),
                **({'cost_accounting': accounting, 'cost_provenance': response_cost_provenance(response)}
                   if accounting is not None and accounting['unpriced_calls'] else {}),
                response=text[:65536] if type(text) is str else None,
                response_sha256=source_digest(text) if type(text) is str else None,
                response_truncated=type(text) is str and len(text) > 65536,
                **{k: v if type(v) is int and v >= 0 else None for k, v in tokens.items()})
            if not cost_known:
                raise ValueError('intervention response cost is unknown or invalid')
            if type(text) is not str:
                raise ValueError('intervention response is not text')
            # Bad metadata is an infrastructure failure after a paid response,
            # not a reason to regenerate or invent a quality observation.
            if (type(getattr(response, 'model', None)) is not str
                    or any(type(v) is not int or v < 0 for v in tokens.values())):
                raise ValueError('intervention response metadata is invalid')
            target = parse_target(text, self.edits.OPERATION)
            if target not in allowed:
                repeated = target is not None and target != parent_policy
                reason = 'intervention_plan_repeated' if repeated else 'intervention_plan_invalid'
                if repeated:
                    self.repeated_plans += 1
                else:
                    self.invalid_plans += 1
                empty = ValidationResult('', None, [], None, [], 'worker')
                self.research.proposal(ident, response, '', empty)
                self._emit('plan_rejected', reason=reason, target_policy=target,
                           quality_measured=False)
                record = self.research.resolve(ident, status='hypothesis_invalid', failure=reason)
                meta.update(hypothesis=record['hypothesis'], gate_rejections=1)
                return no_child(reason)
            built = self.edits.apply(snapshot.parent_code, target)
            source, proof = built['code'], built['proof']
            if proof != proofs[target] or source != self._family[target]:
                raise ValueError('Circle intervention proof changed during dispatch')
            self.pending.update(proof=proof, plan_key=proof['plan_key'])
            self.seen_plans.append(proof['plan_key'])
            # Persist the actual builder operation before the forecast/source
            # commitment or any candidate evaluation. No model code is extracted.
            self._emit('source_binding', proof=proof, source_origin='deterministic_builder',
                       semantic_or_causal_claim=False)
            validation = validate_candidate(source, parent_code=snapshot.parent_code,
                required_entrypoints=snapshot.required_entrypoints, code_hashes=snapshot.code_hashes,
                novelty_check=novelty_check)
            if validation.code != source or validation.grafted:
                raise ValueError('Circle intervention cannot repair or graft its controlled source')
            trace = trace_attempt(validation, extracted_code=source, response=response,
                attempt=1, operator='intervention', temperature=base_temp, cost=cost)
            # This source came from the builder, never an extracted Python fence.
            trace['source_origin'] = 'deterministic_builder'
            meta['validation_trace']['attempts'].append(trace)
            if validation.novelty is not None:
                meta['novelty_checks'].append(validation.novelty)
            if not self.research.proposal(ident, response, source, validation, controlled_change=proof):
                raise ValueError('accepted intervention forecast no longer parses')
            if validation.violation is not None:
                violation = validation.violation
                record = self.research.resolve(ident, status='no_child',
                    failure=_bounded_failure(violation.kind, violation.detail))
                meta.update(hypothesis=record['hypothesis'], gate_rejections=1)
                return no_child('dup_cap' if violation.kind == 'duplicate' else 'gate_rejected')
            self.evaluations += 1
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
            spent = self.pending['cost'] if self.pending else 0.0
            reason = 'Circle intervention stopped: ' + type(exc).__name__
            self.stopped_reason = self.research.stopped_reason = reason
            try:
                if evaluation_started and not isinstance(exc, EvaluationInfrastructureError):
                    try:
                        if ident in self.research.pending:
                            self.research.resolve(ident, status='eval_failed', source=source,
                                                  failure=type(exc).__name__)
                    finally:
                        self.research.charge(spent)
                else:
                    self.research.abort(ident, reason, spent, generation_dispatch=dispatch)
            except Exception as cleanup_exc:
                reason += '; research bookkeeping failed: ' + type(cleanup_exc).__name__
            self.stopped_reason = self.research.stopped_reason = reason
            try:
                self._emit('stopped', error_type=type(exc).__name__, known_cost=spent,
                           generation_dispatch=dispatch,
                           current_call_cost_unknown=dispatch != 'not_dispatched' and
                           not bool(self.pending and self.pending['cost_known']))
            except Exception:
                pass  # Original stop and pending guard survive journal failure.
            raise CampaignStopped(reason) from (exc if isinstance(exc, EvaluationInfrastructureError) else None)

    def summary(self):
        return {'identity': self.identity, **{name: getattr(self, name) for name in _COUNTERS},
                'known_cost': self.known_cost, 'stopped_reason': self.stopped_reason,
                'seen_plan_count': len(self.seen_plans)}

    def state(self):
        return {'schema': CONTRACT, **self.summary(), 'seen_plans': self.seen_plans[:],
                'pending': copy.deepcopy(self.pending)}

    def assert_journal(self, log_path, anchor):
        """Bind consumed plans to the exact already checkpointed journal prefix.

        The enclosing research loader checks unknown paid work in the tail. This
        additional check prevents replacing a consumed plan with another valid
        family key while retaining superficially consistent state counters.
        """
        if (type(anchor) is not dict or type(anchor.get('byte_offset')) is not int
                or anchor['byte_offset'] < 1):
            raise ValueError('Circle intervention requires a journal anchor')
        with open(log_path, 'rb') as source:
            prefix = source.read(anchor['byte_offset'])
        if (len(prefix) != anchor['byte_offset'] or not prefix.endswith(b'\n')
                or hashlib.sha256(prefix).hexdigest() != anchor.get('prefix_sha256')):
            raise ValueError('Circle intervention journal prefix differs')

        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError('duplicate journal object key')
                value[key] = item
            return value

        keys = []
        for line in prefix.splitlines():
            try:
                event = json.loads(line, object_pairs_hook=unique)
            except (ValueError, UnicodeError, RecursionError):
                raise ValueError('invalid Circle intervention journal') from None
            if type(event) is not dict:
                raise ValueError('invalid Circle intervention journal event')
            if (event.get('event'), event.get('phase'), event.get('action')) != (
                    'research', 'intervention', 'source_binding'):
                continue
            proof = event.get('proof')
            if type(proof) is not dict or proof.get('parent_policy') not in self._family:
                raise ValueError('invalid journal Circle intervention proof')
            expected = self.edits.apply(self._family[proof['parent_policy']], proof.get('target_policy'))['proof']
            if (json.dumps(proof, sort_keys=True, allow_nan=False) !=
                    json.dumps(expected, sort_keys=True, allow_nan=False)
                    or event.get('intervention_contract') != CONTRACT
                    or event.get('source_origin') != 'deterministic_builder'
                    or event.get('semantic_or_causal_claim') is not False
                    or proof['plan_key'] in keys):
                raise ValueError('Circle intervention journal proof differs or repeats')
            keys.append(proof['plan_key'])
        if keys != self.seen_plans:
            raise ValueError('Circle intervention seen plans differ from journal proofs')

    def load_state(self, state):
        if (type(state) is not dict or set(state) != set(self.state())
                or state['schema'] != CONTRACT or state['identity'] != self.identity
                or state['pending'] is not None):
            raise ValueError('invalid or unfinished Circle intervention checkpoint')
        if any(type(state[k]) is not int or state[k] < 0 for k in (*_COUNTERS, 'seen_plan_count')):
            raise ValueError('invalid Circle intervention counters')
        possible = {self.edits.apply(code, target)['proof']['plan_key']
                    for parent, code in self._family.items() for target in self._family if target != parent}
        seen, cost = state['seen_plans'], state['known_cost']
        if (type(seen) is not list or any(type(key) is not str or key not in possible for key in seen)
                or len(set(seen)) != len(seen) or state['seen_plan_count'] != len(seen)
                or type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
                or state['attempts'] != state['finished']
                or not state['evaluations'] <= len(seen) <= state['calls_returned'] <= state['calls_started'] <= state['attempts']
                or state['calls_started'] != state['calls_returned']
                or state['invalid_plans'] + state['repeated_plans'] + len(seen) != state['calls_returned']
                or state['exhausted'] > state['attempts'] - state['calls_started']
                or (not state['calls_returned'] and cost != 0)
                or (state['stopped_reason'] is not None and
                    (type(state['stopped_reason']) is not str or not state['stopped_reason']))):
            raise ValueError('inconsistent Circle intervention checkpoint')
        for key in _COUNTERS:
            setattr(self, key, state[key])
        self.known_cost, self.seen_plans = float(cost), seen[:]
        self.stopped_reason = state['stopped_reason']
