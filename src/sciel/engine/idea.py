"""Opt-in card/code mutation with bounded code retries and durable provenance."""
from __future__ import annotations

from ..eval.facade import EvaluationInfrastructureError

import copy
import hashlib
import json
import math

from ..campaign.budget import CampaignStopped, RequestNotDispatched
from ..evidence.measurement import evaluation_observation
from .prompt.parse import extract_code, parse_bet
from .types import NoChild, Result
from .provider_abandonment import AbandonedProviderRequest
from .llm.client import response_accounting, response_cost_provenance, response_resolved_model
from .validation import TRACE_SCHEMA, trace_attempt, validate_candidate
from .validation import candidate_attempt_limit, candidate_registry_enabled

CONTRACT = "sciel-two-stage-idea/1"
FIELDS = {"family": 128, "strategy": 512, "mechanism": 1024}


def digest(text):
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate card key")
        result[key] = value
    return result


def validate_card(text):
    """Return the unchanged accepted card or precise, source-free diagnostics."""
    if type(text) is not str:
        return None, [{'kind': 'not_text', 'expected': 'JSON text'}]
    if len(text) > 8192:
        return None, [{'kind': 'response_too_long', 'length': len(text), 'maximum': 8192}]
    body = text.strip()
    if body.startswith("IDEA_CARD_JSON:"):
        body = body[len("IDEA_CARD_JSON:"):].strip()
    try:
        value = json.loads(body, object_pairs_hook=_unique)
    except json.JSONDecodeError as exc:
        return None, [{'kind': 'invalid_json', 'line': exc.lineno, 'column': exc.colno,
                       'diagnostic': exc.msg}]
    except (ValueError, RecursionError):
        return None, [{'kind': 'invalid_json', 'diagnostic': 'Duplicate keys or excessive nesting.'}]
    if type(value) is not dict:
        return None, [{'kind': 'not_object', 'expected': 'family, strategy, mechanism object'}]
    if set(value) != set(FIELDS):
        return None, [{'kind': 'wrong_fields', 'required': list(FIELDS),
                       'missing': sorted(set(FIELDS) - set(value)),
                       'unexpected_count': len(set(value) - set(FIELDS))}]
    errors = []
    for field, limit in FIELDS.items():
        item = value[field]
        if type(item) is not str:
            errors.append({'kind': 'not_string', 'field': field})
            continue
        if not item.strip():
            errors.append({'kind': 'empty', 'field': field})
        if len(item) > limit:
            errors.append({'kind': 'field_too_long', 'field': field, 'length': len(item), 'maximum': limit})
        if '```' in item:
            errors.append({'kind': 'code_fence', 'field': field})
        if any(ord(c) < 32 and not c.isspace() for c in item):
            errors.append({'kind': 'control_character', 'field': field})
    if errors:
        return None, errors
    # Interior whitespace may be part of a quoted literal in the proposed plan.
    # JSON layout/key order is irrelevant; the actual string content is retained.
    return {key: value[key].strip() for key in FIELDS}, []


def parse_card(text):
    """One bounded JSON card; preserve operators, case and word order."""
    return validate_card(text)[0]


def card_text(card):
    text = json.dumps(card, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if parse_card(text) != card:
        raise ValueError("invalid normalized idea card")
    return text


def card_prompt(snapshot, *, compact=False):
    from .prompt.sources import source_bundle, source_catalog_block, source_reference
    references = [{"program_id": row["program_id"], "card": row["card"]}
                  for row in snapshot.idea_cards[:20]]
    context = snapshot.research_context or {}
    measurement = {key: context.get(key) for key in ('metric', 'parent_score', 'parent_cases')}
    bundle = (source_bundle(snapshot, include_inspirations=False, include_fragments=False)
              if snapshot.research_context is not None else None)
    parent = source_reference(bundle['target_ref']) if bundle is not None else snapshot.parent_code
    prompt = ("IDEA STAGE 1: mutate the strategy card, without generating code.\n"
        "Return only IDEA_CARD_JSON: followed by one JSON object with exactly family, strategy, mechanism.\n"
        "Each value is a nonempty string: family <=128 characters, strategy <=512, mechanism <=1024.\n"
        "Propose a materially different strategy; do not merely paraphrase an existing card.\n"
        "This is a proposal, not a claim of measured benefit. No code fences or forecast are requested here.\n"
        "The task-owned metric describes score units; parent_cases are separate returned measurements. "
        "Do not equate a normalized score with a raw physical quantity; null units remain unknown.\n"
        "RESEARCH_CONTEXT_JSON:\n" + json.dumps(measurement, ensure_ascii=True, allow_nan=False) + "\n"
        f"ITERATION: {snapshot.iteration_no + 1}\n"
        f"TASK:\n{snapshot.task_prompt}\nDIRECTIVE:\n{snapshot.directive}\n"
        f"PARENT CARD (may be legacy prose or absent):\n{snapshot.parent_card or '(none)'}\n"
        f"PARENT REFERENCE CODE:\n{parent}\n"
        "ACTIVE CARD EXAMPLES (at most 20; dedup checks all active structured cards):\n"
        + json.dumps(references, ensure_ascii=False) + "\nLESSONS:\n"
        + "\n".join(snapshot.kb_lessons))
    if compact:
        prompt += ('\nFORMAT TARGET: keep family <=80 characters, strategy <=320 characters, '
                   'and mechanism <=640 characters. These shorter targets leave room below the hard limits. '
                   'Use concise sentences; escape embedded double quotes as required by JSON.\n')
    return prompt + '\n' + source_catalog_block(bundle) if bundle is not None else prompt


def card_repair_prompt(prompt, response, diagnostics):
    """Ask for one full corrected card; never truncate or repair semantic text locally."""
    previous = response if type(response) is str else None
    return (prompt + '\nIDEA CARD FORMAT REPAIR: the previous card was rejected before code generation. '
            'Correct only its JSON format and concision while retaining the proposed strategy. '
            'Return the complete three-field card, with no commentary or code. '
            'This is the only additional card request.\n'
            'CARD_VALIDATION_ERRORS_JSON:\n' + json.dumps(diagnostics, ensure_ascii=True) + '\n'
            'PREVIOUS_CARD_RESPONSE_JSON (exact previous text; null if no text was returned):\n'
            + json.dumps(previous, ensure_ascii=True) + '\n')


def code_prompt(snapshot, card):
    from .prompt.sources import source_bundle, source_catalog_block, source_reference
    serialized = card_text(card)
    bundle = (source_bundle(snapshot, include_inspirations=False)
              if snapshot.research_context is not None else None)
    parent = (source_reference(bundle['target_ref']) if bundle is not None else
              f'```python\n{snapshot.parent_code}\n```')
    prompt = ("IDEA STAGE 2: implement this fixed proposed card as a complete Python program.\n"
        "The card is bound by the engine; do not replace it with another strategy.\n"
        "The binding records which plan was requested, not proof that the implementation follows it.\n"
        f"IDEA_CARD_SHA256: {digest(serialized)}\nIDEA_CARD:\n{serialized}\n"
        f"TASK:\n{snapshot.task_prompt}\nDIRECTIVE:\n{snapshot.directive}\n"
        f"REQUIRED ENTRYPOINTS: {json.dumps(snapshot.required_entrypoints)}\n"
        f"PARENT REFERENCE (preserve its interface):\n{parent}\n"
        "LESSONS:\n" + "\n".join(snapshot.kb_lessons) + "\n"
        "Follow the research forecast format above, then output the complete program in one Python fence.\n")
    return prompt + '\n' + source_catalog_block(bundle) if bundle is not None else prompt


class TwoStageIdea:
    """One fixed card per dispatch; research durably logs each bounded code request."""
    def __init__(self, research, *, card_repair_attempts=0):
        if research is None:
            raise ValueError("two-stage idea requires enabled research")
        if type(card_repair_attempts) is not int or not 0 <= card_repair_attempts <= 1:
            raise ValueError('card_repair_attempts must be an integer from 0 to 1')
        self.research = research
        self.card_repair_attempts = card_repair_attempts
        self.attempts = self.finished = self.calls_started = self.calls_returned = 0
        self.card_invalid = self.card_duplicates = self.evaluations = 0
        self.known_cost = 0.0
        self.code_retries = 0
        self.card_retries = 0
        self.pending = None
        self.stopped_reason = None

    def emit(self, action, **fields):
        self.research.emit("idea", idea_contract=CONTRACT, action=action,
                           idea_id=self.pending["id"] if self.pending else None, **fields)

    def _request(self, stage, prompt, snapshot, llm, temperature):
        self.pending["stage"] = stage
        self.pending["current_call_cost_unknown"] = True
        self.pending["current_call_returned"] = False
        self.calls_started += 1
        self.emit("request_start", stage=stage, model=snapshot.model, temperature=temperature,
            prompt=prompt[:65536], prompt_sha256=digest(prompt), prompt_truncated=len(prompt) > 65536,
            parent_id=snapshot.parent_id, parent_source_sha256=digest(snapshot.parent_code),
            island=snapshot.island, iteration=snapshot.iteration_no + 1,
            card_sha256=self.pending.get("card_sha256"), experiment_id=self.pending.get("research_id"),
            injected_ids=(snapshot.research_context or {}).get("injected_ids", []))
        response = llm.generate(prompt, snapshot.model, temperature)
        self.calls_returned += 1
        self.pending["current_call_returned"] = True
        try:
            accounting = response_accounting(response)
            cost, known = accounting['known_usd'], True
        except ValueError:
            cost, known, accounting = None, False, None
        if known:
            self.known_cost += float(cost)
            self.pending["cost"] += float(cost)
            self.pending["current_call_cost_unknown"] = False
        text = getattr(response, "text", None)
        fields = {"stage": stage, "generation_returned": True, "cost": getattr(response, "cost", None),
            "response_sha256": digest(text) if type(text) is str else None,
            "response_is_text": type(text) is str, "card_sha256": self.pending.get("card_sha256"),
            "experiment_id": self.pending.get("research_id"),
            "returned_model": getattr(response, "model", None)}
        resolved = response_resolved_model(response)
        if resolved is not None:
            fields['resolved_model'] = resolved
        if accounting is not None and accounting['unpriced_calls']:
            fields.update(known_usd=cost, unpriced_calls=1,
                          cost_provenance=response_cost_provenance(response))
        if type(fields['returned_model']) is not str:
            fields['returned_model'] = None
        for key in ('prompt_tokens', 'completion_tokens'):
            value = getattr(response, key, None)
            fields[key] = value if type(value) is int and 0 <= value <= 10**9 else None
        if stage == "card":
            fields.update(response=text[:8192] if type(text) is str else None,
                          response_truncated=type(text) is str and len(text) > 8192)
        self.pending["calls"].append(copy.deepcopy(fields))
        self.emit("request_result", **fields)
        if not known:
            raise CampaignStopped("two-stage idea response cost is unknown or invalid")
        return response

    def _finish(self, status, meta):
        meta["idea"] = copy.deepcopy(self.pending)
        meta["idea"]["status"] = status
        self.emit("finish", status=status, known_cost=self.pending["cost"],
                  card_sha256=self.pending.get("card_sha256"),
                  experiment_id=self.pending.get("research_id"))
        self.finished += 1
        self.pending = None

    def run(self, snapshot, llm, evaluate_fn, *, base_temp=.7, novelty_check=None):
        from .worker import (_bounded_failure, _evaluation_failure, candidate_retry_prompt,
                             evaluation_attempt, evaluation_metadata)
        if self.stopped_reason or self.research.stopped_reason:
            raise CampaignStopped(self.stopped_reason or self.research.stopped_reason)
        if self.pending is not None:
            raise ValueError("unfinished idea request cannot be replayed")
        self.attempts += 1
        self.pending = {"id": (getattr(self, 'id_prefix', '') + ':' if getattr(self, 'id_prefix', '') else '')
                             + f"idea{self.attempts:06d}", "stage": "card", "cost": 0.0,
                        "calls": [], "parent_id": snapshot.parent_id,
                        "parent_source_sha256": digest(snapshot.parent_code)}
        meta = {"op_used": "idea", "dup_attempts": 0, "escalated": False, "llm_retries": 0,
                "grafted": [], "empty_responses": 0, "gate_rejections": 0, "novelty_checks": [],
                "validation_trace": {"schema": TRACE_SCHEMA, "profile": "worker", "attempts": []}}

        evaluations = []

        def no_child(reason):
            cost = self.pending["cost"]
            meta.update(evaluation_metadata(evaluations))
            self._finish(reason, meta)
            return NoChild(snapshot.island, reason, meta, cost)

        try:
            prepared_card = card_prompt(snapshot, compact=bool(self.card_repair_attempts))
            prompt = prepared_card
            for card_attempt in range(1 + self.card_repair_attempts):
                if card_attempt:
                    self.card_retries += 1
                first = self._request("card", prompt, snapshot, llm, base_temp)
                raw_card = getattr(first, 'text', None)
                card, diagnostics = validate_card(raw_card)
                if card is not None:
                    break
                retry = card_attempt < self.card_repair_attempts
                fields = (dict(diagnostics=diagnostics, card_attempt=card_attempt + 1,
                               response_sha256=digest(raw_card) if type(raw_card) is str else None,
                               retry_scheduled=retry) if self.card_repair_attempts else {})
                self.emit("card_rejected", reason="invalid_card_schema", **fields)
                if not retry:
                    self.card_invalid += 1
                    if self.card_repair_attempts:
                        meta['card_validation_errors'] = diagnostics
                    return no_child('idea_card_repair_exhausted' if self.card_repair_attempts else 'idea_card_invalid')
                prompt = card_repair_prompt(prepared_card, raw_card, diagnostics)
            serialized = card_text(card)
            card_hash = digest(serialized)
            self.pending.update(card=card, card_sha256=card_hash)
            matches = []
            for reference in snapshot.idea_cards:
                existing = parse_card(reference["card"])
                if existing is not None and card_text(existing) == serialized:
                    matches.append({"program_id": reference["program_id"],
                                    "source_sha256": reference["source_sha256"]})
            self.emit("card_decision", card=card, card_sha256=card_hash, matches=matches,
                      accepted=not matches, scope="active_island_structured_cards")
            if matches:
                self.card_duplicates += 1
                return no_child("idea_card_duplicate")
            limit = candidate_attempt_limit(novelty_check, 1)
            previous_response = previous_validation = previous_reason = measured_repeat = None
            for code_attempt in range(limit):
                prepared = code_prompt(snapshot, card)
                if previous_response is not None:
                    prepared = candidate_retry_prompt(prepared, previous_response, previous_validation,
                                                      reason=previous_reason, measured_repeat=measured_repeat)
                ident, prompt = self.research.begin(snapshot, code_attempt, "idea", prepared)
                if code_attempt:
                    self.code_retries += 1
                self.pending["research_id"] = ident
                response = self._request("code", prompt, snapshot, llm, base_temp)
                # Receipt provenance exists before parsing or candidate validation.
                # Malformed text is still a returned, known-cost code response.
                record = self.research.pending[ident]
                record['generation_dispatch'] = 'returned'
                record['generation_cost'] = self.pending['calls'][-1].get('known_usd', self.pending['calls'][-1]['cost'])
                if type(getattr(response, "text", None)) is not str:
                    self.research.resolve(ident, status="no_child", failure="code response is not text")
                    return no_child("idea_code_invalid")
                child = extract_code(response.text)
                validation = validate_candidate(child, parent_code=snapshot.parent_code,
                    required_entrypoints=snapshot.required_entrypoints, code_hashes=snapshot.code_hashes,
                    novelty_check=novelty_check)
                trace = trace_attempt(validation, extracted_code=child, response=response, attempt=code_attempt + 1,
                                      operator="idea", temperature=base_temp, cost=response.cost)
                meta["validation_trace"]["attempts"].append(trace)
                if validation.novelty is not None:
                    meta["novelty_checks"].append(validation.novelty)
                hyp_ok = self.research.proposal(ident, response, child, validation)
                self.pending['candidate_source_sha256'] = digest(validation.code)
                self.emit("source_binding", experiment_id=ident, card_sha256=card_hash,
                    prompt_sha256=digest(prompt), response_sha256=digest(response.text),
                    proposed_source_sha256=digest(child), candidate_source_sha256=digest(validation.code),
                    source_repaired=child != validation.code, hypothesis_valid=hyp_ok,
                    binding_scope="requested card; semantic implementation adherence is not verified")
                previous_response, previous_validation = response.text, validation
                previous_reason = measured_repeat = None
                if not hyp_ok:
                    self.research.resolve(ident, status="hypothesis_invalid", failure="missing or malformed hypothesis")
                    meta["gate_rejections"] += 1
                    previous_reason = "Missing or malformed pre-evaluation hypothesis."
                    if code_attempt + 1 < limit:
                        continue
                    return no_child("gate_rejected")
                if validation.violation is not None:
                    violation = validation.violation
                    self.research.resolve(ident, status="no_child", failure=_bounded_failure(violation.kind, violation.detail))
                    if violation.kind == "duplicate":
                        meta["dup_attempts"] = code_attempt + 1
                        meta["duplicates"] = meta.get("duplicates", 0) + 1
                        reason = "dup_cap"
                    elif violation.kind == "empty":
                        meta["empty_responses"] += 1
                        reason = "empty_response"
                    else:
                        meta["gate_rejections"] += 1
                        reason = "gate_rejected"
                    if code_attempt + 1 < limit:
                        continue
                    return no_child(reason)
                self.evaluations += 1
                remeasurement = bool(candidate_registry_enabled(novelty_check)
                                     and validation.novelty.get('remeasurement'))
                try:
                    outcome = evaluate_fn(validation.code)
                except Exception as exc:
                    self.research.resolve(ident, status=('generation_stopped' if isinstance(exc, EvaluationInfrastructureError)
                        else 'no_child' if remeasurement else 'eval_failed'), source=validation.code,
                                          failure=type(exc).__name__)
                    raise
                result = outcome.result
                measured = evaluation_attempt(outcome, validation.code)
                evaluations.append(measured)
                from .output_novelty import is_repeat
                repeated_output = is_repeat(measured["output_novelty"]) and not remeasurement
                if remeasurement:
                    record = self.research.resolve(ident, status='no_child', source=validation.code,
                        failure='Remeasurement of known source; no new mutation or efficacy claim. '
                                'The evaluator receipt records this measurement separately.')
                else:
                    record = self.research.resolve(ident, status="evaluated" if result.ok else "eval_failed",
                        score=result.score if result.ok else None, source=validation.code,
                        evaluation=evaluation_observation(outcome),
                        failure=None if result.ok else _evaluation_failure(result.artifacts),
                        **({"output_novelty": measured["output_novelty"]} if measured["output_novelty"] is not None else {}))
                if repeated_output:
                    measured['admission'] = 'output_repeat_rejected'
                    measured_repeat = {**copy.deepcopy(measured), 'quality_measured': True}
                    trace.update(disposition='output_repeat_rejected', evaluation=copy.deepcopy(measured))
                    previous_response, previous_validation, previous_reason = response.text, None, None
                    if code_attempt + 1 < limit:
                        continue
                    return no_child('output_repeat_cap')
                meta.update(evaluation_metadata(evaluations))
                meta.update(research_id=ident, hypothesis=record["hypothesis"], idea_card=card,
                            grafted=validation.grafted)
                if candidate_registry_enabled(novelty_check):
                    meta.update(novelty=validation.novelty, remeasurement=remeasurement)
                if getattr(outcome, 'evaluation_id', None) is not None:
                    meta['evaluation_id'] = outcome.evaluation_id
                cost = self.pending["cost"]
                self._finish("evaluated" if result.ok else "eval_failed", meta)
                return Result(child_code=validation.code, score=result.score, cases=result.cases, ok=result.ok,
                    island=snapshot.island, parent_id=snapshot.parent_id, origin="gen", bet=parse_bet(response.text),
                    meta=meta, raw_first_score=outcome.raw_first_score, reached=outcome.reached,
                    confirmed=outcome.confirmed, prompt=prompt, response=response.text, cost=cost,
                    timing=result.timing, artifacts=result.artifacts)
        except AbandonedProviderRequest as exc:
            # This request has no returned card/code and is not retried. Keep
            # earlier known charges on NoChild for the ordinary commit path,
            # while the authenticated recovery record retains the unknown cost.
            receipt = copy.deepcopy(exc.receipt)
            meta['provider_abandonment'] = receipt
            meta.update(generation_returned=False, generation_dispatch='unknown',
                        current_call_cost_unknown=True, llm_error=copy.deepcopy(exc._metadata))
            self.pending['provider_abandonment'] = copy.deepcopy(receipt)
            self.emit('provider_abandoned', stage=self.pending['stage'],
                experiment_id=self.pending.get('research_id'), provider_abandonment=receipt,
                llm_error=copy.deepcopy(exc._metadata), generation_returned=False,
                generation_dispatch='unknown', cost=None, current_call_cost_unknown=True,
                known_cost=self.pending['cost'])
            ident = self.pending.get('research_id')
            if ident is not None and ident in self.research.pending:
                self.research.pending[ident]['generation_dispatch'] = 'unknown'
                self.research.resolve(ident, status='generation_stopped',
                    failure='Provider request explicitly abandoned; outcome and cost remain unknown.')
            return no_child('provider_request_abandoned')
        except BaseException as exc:
            from .llm.client import llm_error_metadata
            # Never replay an unknown request or evaluation. Known earlier costs
            # would otherwise be lost because no Result reaches Engine._on_result.
            reason = "two-stage idea stopped: " + type(exc).__name__
            self.stopped_reason = reason
            pending = self.pending
            dispatch = ('returned' if pending.get('current_call_returned') else
                        'not_dispatched' if isinstance(exc, RequestNotDispatched) else 'unknown')
            try:
                self.emit("stopped", stage=pending["stage"], error_type=type(exc).__name__,
                          experiment_id=pending.get('research_id'), llm_error=llm_error_metadata(exc),
                          known_cost=pending["cost"], generation_returned=pending.get('current_call_returned', False),
                          generation_dispatch=dispatch,
                          current_call_cost_unknown=(
                              dispatch != 'not_dispatched' and pending.get('current_call_cost_unknown', False)))
            finally:
                # A research experiment belongs to the code request. Card-stage
                # advice exposure, if any, remains in its separate idea journal.
                self.research.abort(pending.get("research_id"), reason, pending["cost"],
                                    generation_dispatch=dispatch)
            self.pending = None
            self.finished += 1
            raise CampaignStopped(reason) from (exc if isinstance(exc, EvaluationInfrastructureError) else None)

    def summary(self):
        return {key: getattr(self, key) for key in ("attempts", "finished", "calls_started",
            "calls_returned", "known_cost", "card_invalid", "card_duplicates", "evaluations", "stopped_reason",
            *(('card_repair_attempts',) if self.card_repair_attempts else ()),
            *(('card_retries',) if self.card_retries else ()),
            *(('code_retries',) if self.code_retries else ()))}

    def state(self):
        return {"schema": CONTRACT, **self.summary(), "pending": copy.deepcopy(self.pending)}

    def load_state(self, state):
        optional = {'code_retries', 'card_retries'}
        fields = set(self.state()) - optional
        if (type(state) is not dict or set(state) - optional != fields or state["schema"] != CONTRACT
                or state["pending"] is not None
                or type(state.get('card_repair_attempts', 0)) is not int
                or state.get('card_repair_attempts', 0) != self.card_repair_attempts):
            raise ValueError("invalid or unfinished two-stage idea checkpoint")
        counts = ("attempts", "finished", "calls_started", "calls_returned", "card_invalid", "card_duplicates", "evaluations")
        if any(type(state[k]) is not int or state[k] < 0 for k in counts):
            raise ValueError("invalid two-stage idea counters")
        cost = state["known_cost"]
        retries = state.get('code_retries', 0)
        card_retries = state.get('card_retries', 0)
        if (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
                or type(retries) is not int or not 0 <= retries <= 2 * state['attempts']
                or type(card_retries) is not int or not 0 <= card_retries <= self.card_repair_attempts * state['attempts']
                or state["finished"] != state["attempts"]
                or not state["attempts"] + retries + card_retries <= state["calls_started"] <= 2 * state["attempts"] + retries + card_retries
                or state["calls_returned"] > state["calls_started"]
                or sum(state[k] for k in ("card_invalid", "card_duplicates", "evaluations")) > state["attempts"] + retries
                or (state["stopped_reason"] is not None and type(state["stopped_reason"]) is not str)):
            raise ValueError("inconsistent two-stage idea checkpoint")
        for key in self.summary():
            if key not in optional:
                setattr(self, key, copy.deepcopy(state[key]))
        self.code_retries = retries
        self.card_retries = card_retries
