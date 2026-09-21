"""Detached, pure generation from an accepted component-scoped target.

No Program, ordinary parent score, scheduler, provider or evaluator is created.
Recorded SDK objects establish byte consistency, not proof of dispatch/payment.
The caller owns durable one-shot intent, spending limits and unknown-call holds.
Results are NOT generated-paired history/1 records and cannot enter that history
without a future explicit versioned evaluation/admission contract.
"""
from __future__ import annotations

import copy
import json
import math

from .external_reflection import _digest, _event, _hash, _id, _journal, _RESPONSE_FIELDS
from .external_research_v2 import policy_identity, _review_text_policy
from .generated_paired_reflection import validate_sidecar, SCOPED_SIDECAR_SCHEMA
from .db import normalized_hash
from .validation import validate_candidate
from ..hypothesis import circle_optimizer_traced_body as builder
from ..hypothesis import component_generation_hypothesis as wire
from ..hypothesis import component_targets, restart_predictions
from ..hypothesis.comparison_targets import instruction as target_instruction
from ..hypothesis.review_eligibility import decision_for
from ..hypothesis.scoped_paired_review import CARD_SCHEMA
from ..knowledge_base.catalogs import canonical_json, canonical_sha
from ..knowledge_base.generated_paired_history import validate_capsule, _bounded
from ..knowledge_base.generated_paired_context import build_context, block

SCHEMA = 'sciel-component-target-generation-result/1'
PLAN_SCHEMA = 'sciel-component-target-generation-plan/1'
JOURNAL_CONTRACT = 'sciel-component-target-generation/1'
MAX_PROMPT_BYTES = 96000
MAX_TASK_CHARS = 16000
MAX_CAPTURE_BYTES = 512000
REQUIRED_ENTRYPOINTS = ['run_packing']


def _require(ok, message):
    if not ok:
        raise ValueError('component target generation: ' + message)


def _same(a, b):
    return canonical_json(a) == canonical_json(b)


def _finite(value):
    return (type(value) is int and 0 <= value <= 10**100
            or type(value) is float and math.isfinite(value) and value >= 0)


def _transport(value):
    _bounded(value)
    _require(type(value) is dict and set(value) == {'reasoning', 'provider'}
        and value['reasoning'] == {'effort': 'none'}, 'explicit transport reasoning required')
    provider = value['provider']
    _require(type(provider) is dict and set(provider) == {'allow_fallbacks', 'require_parameters', 'max_price'}
        and provider['allow_fallbacks'] is False and provider['require_parameters'] is True,
        'explicit bounded provider routing required')
    prices = provider['max_price']
    _require(type(prices) is dict and set(prices) == {'prompt', 'completion', 'request'}
        and all(_finite(prices[key]) and prices[key] <= 10**6 for key in ('prompt', 'completion'))
        and _finite(prices['request']) and prices['request'] <= 10**6,
        'finite frozen token and per-request prices required')
    result = copy.deepcopy(value)
    # BudgetedLLM emits float price caps even when the frozen input used ints.
    # These are host transport scalars, never authored model content.
    result['provider']['max_price'] = {key: float(prices[key]) for key in prices}
    return result


def _sources(component):
    sources = {}
    for proposal in component['proposals']:
        for source in (proposal['parent_source'], proposal['source']):
            body = builder.body_of(source)
            _require(builder.source_for(body) == source, 'source cannot be reconstructed exactly')
            sources[_digest(source)] = dict(source=source, body=body, body_sha256=_digest(body),
                                            normalized_hash=normalized_hash(source))
    return dict(sorted(sources.items()))


def _append(rows, plan, phase, **fields):
    _event(rows, plan['run_id'], plan['segment_id'], 'component_target_generation',
           contract=JOURNAL_CONTRACT, phase=phase, request_key=plan['request_key'], **fields)


class ComponentTargetGenerationRejected(ValueError):
    """An unbounded/non-JSON capture cannot enter a portable result; keep a safe journal."""
    def __init__(self, journal):
        super().__init__('component target generation: capture is not bounded literal JSON')
        self.journal = journal
        self.journal_sha256 = _digest(journal)


class ComponentTargetGenerationJob:
    # Evidence-specific subclasses prepare their own plans and author bindings.
    # Raw transport decoding and exact body admission are shared unchanged.
    RESULT_SCHEMA = SCHEMA
    GENERATION_ID_PREFIX = 'component-generated:'

    def _append(self, rows, plan, phase, **fields):
        _append(rows, plan, phase, **fields)

    def _historical_provider_ids(self):
        return {p['provider_response']['id'] for p in self._component['proposals']}

    def _consumed_sources(self):
        return self._sources

    def __init__(self, component, review_sidecar, *, expected_component_sha256,
                 expected_sidecar_sha256, comparison_id, expected_card_sha256, config, task_prompt):
        for value in (component, review_sidecar, config):
            _bounded(value)
        _require(type(config) is dict and all(_hash(value) for value in
            (expected_component_sha256, expected_sidecar_sha256, expected_card_sha256)),
            'explicit input digests required')
        _require(config.get('external_component_sha256') == expected_component_sha256
            and config.get('external_component_review_sha256') == expected_sidecar_sha256
            and 'external_history_sha256' not in config, 'exact standalone component/review config required')
        _require(type(config.get('generated_paired_review_version')) is int
            and config['generated_paired_review_version'] == 2
            and type(config.get('comparison_review_version')) is int and config['comparison_review_version'] == 3,
            'explicit scoped review version required')
        _require(all(config.get(key) is True for key in ('comparison_review_text', 'restart_predictions',
            'inject_lessons', 'structured_auxiliary', 'comparison_reviews', 'batch_only_lessons')),
            'enabled structured review delivery required')
        model = config.get('model')
        _require(type(model) is str and 0 < len(model) <= 256 and bool(model.strip()), 'explicit model required')
        _require(type(task_prompt) is str and 0 < len(task_prompt) <= MAX_TASK_CHARS
            and bool(task_prompt.strip()), 'bounded task prompt required')
        task_prompt.encode('utf-8')
        policy = policy_identity()
        validated = validate_capsule(component, expected_sha256=expected_component_sha256,
                                     metric=component['metric'])
        sidecar = validate_sidecar(review_sidecar, component=validated, expected_sha256=expected_sidecar_sha256)
        _require(sidecar['schema'] == SCOPED_SIDECAR_SCHEMA, 'scoped raw-authenticated review required')
        card = sidecar['review_cards'].get(comparison_id)
        _require(type(card) is dict and card.get('schema') == CARD_SCHEMA
            and canonical_sha(card) == expected_card_sha256, 'exact selected scoped card required')
        text_policy = _review_text_policy(config)
        _require(text_policy is not None, 'explicit host eligibility registry required')
        eligibility = decision_for(card, text_policy, policy_sha256=canonical_sha(policy))
        _require(eligibility['status'] == 'accepted', 'exact current-policy host acceptance required')
        target = card['unverified_interpretation']['next_hypothesis']
        _require(target is not None, 'selected review has no prospective target')
        target = component_targets.validate_target(target, component=validated, comparison_id=comparison_id)
        sources = _sources(validated)
        _require(all(target[key] in sources for key in ('base_source_sha256', 'comparator_source_sha256')),
                 'declared source unavailable')
        self._component, self._sidecar, self._config = validated, sidecar, copy.deepcopy(config)
        self._task_prompt, self._card, self._target = task_prompt, copy.deepcopy(card), target
        self._sources, self._policy = sources, policy
        self._bindings = dict(component_sha256=expected_component_sha256,
            sidecar_sha256=expected_sidecar_sha256, card_sha256=expected_card_sha256,
            comparison_id=comparison_id, raw_reflection_sha256=card['writer']['raw_response_sha256'],
            host_registry_sha256=canonical_sha(text_policy), host_decision_sha256=eligibility['decision_sha256'],
            policy_sha256=canonical_sha(policy), target_sha256=canonical_sha(target),
            source_registry_sha256=canonical_sha(sources),
            config_sha256=canonical_sha(config), task_sha256=_digest(task_prompt))

    def _unchanged(self):
        _require(_same(policy_identity(), self._policy), 'production policy changed since preparation')
        _require(canonical_sha(self._component) == self._bindings['component_sha256']
            and canonical_sha(self._sidecar) == self._bindings['sidecar_sha256']
            and canonical_sha(self._config) == self._bindings['config_sha256']
            and canonical_sha(self._card) == self._bindings['card_sha256']
            and canonical_sha(self._target) == self._bindings['target_sha256']
            and canonical_sha(self._sources) == self._bindings['source_registry_sha256']
            and _digest(self._task_prompt) == self._bindings['task_sha256'], 'frozen input changed')

    def prepare(self, *, run_id, segment_id, transport_settings, request_key='target_g1', max_tokens=4096):
        self._unchanged()
        _require(all(_id(value) for value in (run_id, segment_id, request_key)), 'bounded request identities required')
        old_runs = {p['provenance']['run_id'] for p in self._component['proposals']}
        old_runs.add(self._sidecar['plan']['run_id'])
        _require(run_id not in old_runs, 'fresh detached generation run required')
        _require(type(max_tokens) is int and 1 <= max_tokens <= 8192, 'bounded generation token cap required')
        transport = _transport(transport_settings)
        target = self._target
        projection = build_context([self._component], branch=self._component['root_parent']['id'],
                                   max_chars=16000, max_items=1)
        _require(projection is not None and projection['presented_count'] == 1,
                 'whole historical study does not fit generation memory')
        source_bodies = {role:dict(source_sha256=target[key], body=self._sources[target[key]]['body'],
            body_sha256=self._sources[target[key]]['body_sha256']) for role, key in
            (('edit_base', 'base_source_sha256'), ('paired_comparator', 'comparator_source_sha256'))}
        native = dict(status='UNMEASURED', score=None, forecast=None, ordinary_origin=None,
                      reason='Detached generation has no native measurement or ordinary parent.')
        intent = dict(bindings=self._bindings, target=target,
            receiving_root=self._component['root_parent'], selected_review=self._card,
            source_bodies=source_bodies, builder_identity=builder.identity(),
            interface=builder.interface_description(), native_observation=native,
            source_scope='Complete editable bodies plus exact shared scaffold identity; no source executed.')
        quality_rules = target_instruction()
        quality_rules = quality_rules[quality_rules.index('The caller must verify'):]
        restart_rules = restart_predictions.instruction().replace(
            'These forecasts do not change the native score forecast or quality/lesson gates.',
            'This detached job has no native score forecast; restart forecasts do not change quality or lesson gates.')
        prompt = (wire.INSTRUCTION + '\n' + quality_rules + '\n' + restart_rules
            + '\n\nTASK:\n' + self._task_prompt + '\n\nCOMPONENT_TARGET_CONTEXT_JSON:\n'
            + canonical_json(intent) + '\n\n' + block(projection))
        _require(len(prompt.encode('utf-8')) <= MAX_PROMPT_BYTES, 'complete required prompt exceeds byte cap')
        response_format = wire.response_format(self._bindings['target_sha256'])
        request = dict(model=self._config['model'], messages=[dict(role='user', content=prompt)],
            temperature=.2, max_tokens=max_tokens, response_format=response_format, extra_body=transport)
        plan = dict(schema=PLAN_SCHEMA, **copy.deepcopy(self._bindings), run_id=run_id,
            segment_id=segment_id, request_key=request_key, max_tokens=max_tokens,
            transport_settings=transport, request=request, request_sha256=canonical_sha(request),
            prompt=prompt, prompt_sha256=_digest(prompt), prompt_bytes=len(prompt.encode('utf-8')),
            response_format=response_format, builder_identity=builder.identity(),
            execution_profile=builder.EXECUTION_PROFILE, target=copy.deepcopy(target),
            source_registry=copy.deepcopy(self._sources), native_observation=native,
            new_ordinary_outcomes=0, new_search_attempts=0)
        rows = []
        _append(rows, plan, 'prepared', bindings=copy.deepcopy(self._bindings),
            request_sha256=plan['request_sha256'], prompt_sha256=plan['prompt_sha256'],
            response_format=response_format, base_source_sha256=target['base_source_sha256'],
            comparator_source_sha256=target['comparator_source_sha256'])
        plan.update(journal_prefix=_journal(rows), journal_prefix_sha256=_digest(_journal(rows)))
        return plan

    def admit(self, plan, response, *, manifest_sha256, journal_prefix, provider_request, provider_response):
        """Rebuild one recorded result. This is re-entrant, not a dispatch permit."""
        _bounded(plan)
        _require(type(plan) is dict and _hash(manifest_sha256), 'plan and external manifest digest required')
        expected = self.prepare(run_id=plan.get('run_id'), segment_id=plan.get('segment_id'),
            request_key=plan.get('request_key'), max_tokens=plan.get('max_tokens'),
            transport_settings=plan.get('transport_settings'))
        _require(_same(plan, expected), 'prepared request changed')
        _require(type(journal_prefix) is str and journal_prefix == expected['journal_prefix'],
                 'persisted request prefix changed')
        rows = [json.loads(line) for line in journal_prefix.splitlines()]
        raw = response.get('text') if type(response) is dict else None
        raw_sha = None
        try:
            raw_sha = _digest(raw) if type(raw) is str else None
            for value in (response, provider_request, provider_response):
                _bounded(value)
                _require(len(canonical_json(value).encode('utf-8')) <= MAX_CAPTURE_BYTES, 'capture byte limit')
        except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
            self._append(rows, expected, 'invalid_capture', response_sha256=raw_sha,
                response=raw[:wire.MAX_RESPONSE_BYTES] if type(raw) is str and raw_sha is not None else None,
                response_truncated=True, error=dict(code='capture_bound', path='response'))
            raise ComponentTargetGenerationRejected(_journal(rows)) from None
        self._append(rows, expected, 'returned', response_sha256=raw_sha,
            response=raw[:wire.MAX_RESPONSE_BYTES] if type(raw) is str else None,
            response_truncated=type(raw) is str and len(raw) > wire.MAX_RESPONSE_BYTES,
            response_metadata_sha256=canonical_sha(response), provider_request_sha256=canonical_sha(provider_request),
            provider_response_sha256=canonical_sha(provider_response))
        result = dict(schema=self.RESULT_SCHEMA, status='invalid_response', eligible=False,
            plan=copy.deepcopy(expected), config=copy.deepcopy(self._config), task_prompt=self._task_prompt,
            manifest_sha256=manifest_sha256, response=copy.deepcopy(response),
            provider_request=copy.deepcopy(provider_request), provider_response=copy.deepcopy(provider_response),
            response_sha256=raw_sha, generation_id=None, hypothesis=None, abstention_reason=None,
            body=None, code=None, source_sha256=None, normalized_source_hash=None, proof=None,
            native_observation=copy.deepcopy(expected['native_observation']),
            paired_observation=dict(status='UNMEASURED', target_sha256=self._bindings['target_sha256'],
                quality_verdict='INCONCLUSIVE', restart_verdicts=None, reason='No paired draw executed.'),
            error=None, provider_dispatch_verified=False, execution_performed=False,
            one_shot_enforcement='external durable transport owner',
            new_ordinary_outcomes=0, new_search_attempts=0, new_lesson_ids=[], retired_lesson_ids=[])
        error = _receipt_error(response, provider_request, provider_response, expected)
        if error is None:
            # Prevent an old provider response being relabelled as this fresh request.
            old_ids = self._historical_provider_ids()
            if provider_response['id'] in old_ids:
                error = dict(code='historical_provider_replay', path='provider_response.id')
        if error is None:
            decoded = wire.decode_response(raw, target_sha256=self._bindings['target_sha256'])
            error = decoded['error']
            if decoded['status'] == 'abstained':
                result.update(status='abstained', abstention_reason=decoded['abstention_reason'])
            elif decoded['status'] == 'proposed':
                body = decoded['body']
                try:
                    builder.validate_body(body)
                    code = builder.source_for(body)
                except (ValueError, TypeError, SyntaxError):
                    error = dict(code='invalid_body', path='body')
                else:
                    digest, normalized = _digest(code), normalized_hash(code)
                    consumed = self._consumed_sources()
                    result['hypothesis'] = decoded['hypothesis'].to_dict()
                    if digest in consumed or normalized in {s['normalized_hash'] for s in consumed.values()}:
                        result['status'] = 'duplicate'
                        error = dict(code='historical_duplicate', path='body')
                    else:
                        base = self._sources[self._target['base_source_sha256']]['source']
                        built = builder.apply(base, body)
                        validation = validate_candidate(code, parent_code=base,
                            required_entrypoints=REQUIRED_ENTRYPOINTS,
                            code_hashes={s['normalized_hash'] for s in consumed.values()})
                        if validation.code != code or validation.grafted or validation.violation is not None:
                            result['status'] = 'validation_rejected'
                            error = dict(code='repair_graft_or_static_rejection', path='body')
                        else:
                            _require(built['code'] == code and built['proof']['parent_source_sha256'] ==
                                self._target['base_source_sha256'], 'declared base derivation differs')
                            result.update(status='eligible', eligible=True, body=body, code=code,
                                source_sha256=digest, normalized_source_hash=normalized, proof=built['proof'])
        if raw_sha is not None:
            result['generation_id'] = self.GENERATION_ID_PREFIX + canonical_sha(dict(
                manifest_sha256=manifest_sha256, run_id=expected['run_id'], request_key=expected['request_key'],
                request_sha256=expected['request_sha256'],
                response_sha256=raw_sha, target_sha256=self._bindings['target_sha256']))
        result['error'] = error
        self._append(rows, expected, 'admission', status=result['status'], eligible=result['eligible'], error=error,
            generation_id=result['generation_id'], source_sha256=result['source_sha256'],
            proof_sha256=canonical_sha(result['proof']) if result['proof'] else None,
            native_status='UNMEASURED', paired_status='UNMEASURED', new_ordinary_outcomes=0, new_search_attempts=0)
        result.update(journal=_journal(rows), journal_sha256=_digest(_journal(rows)))
        _bounded(result)
        return result


def _receipt_error(response, request, provider, plan):
    """Exact captured transport consistency only; no provider authenticity claim."""
    def fail(code, path):
        return dict(code=code, path=path)
    if not _same(request, plan['request']):
        return fail('request_binding', 'provider_request')
    if (type(response) is not dict or set(response) != _RESPONSE_FIELDS
        or type(response.get('text')) is not str or response.get('model') != request['model']
        or not _finite(response.get('cost'))
        or any(type(response.get(key)) is not int or not 0 <= response[key] <= 10**9
               for key in ('prompt_tokens', 'completion_tokens'))
        or response.get('finish_reason') != 'stop' or response.get('refusal') not in (None, '')):
        return fail('response_metadata', 'response')
    if (type(provider) is not dict or type(provider.get('id')) is not str
        or not 0 < len(provider['id']) <= 256 or not provider['id'].strip()
        or provider.get('model') != request['model'] or type(provider.get('choices')) is not list
        or len(provider['choices']) != 1):
        return fail('provider_envelope', 'provider_response')
    choice, usage = provider['choices'][0], provider.get('usage')
    if (type(choice) is not dict or type(choice.get('index')) is not int or choice['index'] != 0
        or choice.get('finish_reason') != response['finish_reason'] or type(choice.get('message')) is not dict):
        return fail('provider_choice', 'provider_response.choices')
    message = choice['message']
    if (message.get('role') != 'assistant' or message.get('content') != response['text']
        or message.get('refusal') != response['refusal'] or type(usage) is not dict
        or any(not _same(usage.get(key), response[key]) for key in ('prompt_tokens', 'completion_tokens'))
        or not _finite(usage.get('cost')) or float(usage['cost']) != float(response['cost'])):
        return fail('provider_return_binding', 'provider_response')
    return None


def validate_result(result, *, component, review_sidecar, expected_sha256, config):
    """Authenticate recorded inputs and rebuild raw decoding, edit proof and journal.

    The external receiving config supplies approval authority. Rehashing this
    snapshot cannot create an approval or convert its unmeasured status to data.
    """
    _bounded(result)
    _require(_hash(expected_sha256) and canonical_sha(result) == expected_sha256
        and type(result) is dict and result.get('schema') == SCHEMA, 'declared result digest/schema differs')
    plan = result.get('plan')
    _require(type(plan) is dict and _same(result.get('config'), config), 'receiving configuration differs')
    job = ComponentTargetGenerationJob(component, review_sidecar,
        expected_component_sha256=config.get('external_component_sha256'),
        expected_sidecar_sha256=config.get('external_component_review_sha256'),
        comparison_id=plan.get('comparison_id'), expected_card_sha256=plan.get('card_sha256'),
        config=config, task_prompt=result.get('task_prompt'))
    rebuilt = job.admit(plan, result.get('response'), manifest_sha256=result.get('manifest_sha256'),
        journal_prefix=plan.get('journal_prefix'), provider_request=result.get('provider_request'),
        provider_response=result.get('provider_response'))
    _require(_same(rebuilt, result), 'result differs from raw request, response, source or journal')
    return copy.deepcopy(rebuilt)
