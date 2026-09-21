"""Generate from a host-accepted review of one evaluated detached study.

The evidence family, review and compiled target keep their own schemas. Only
transport/body admission is shared with the older component generator. This
pure API creates neither a Program nor a native score, provider call or draw.
"""
from __future__ import annotations

import copy

from .component_target_generation import (ComponentTargetGenerationJob,
    ComponentTargetGenerationRejected, _transport, _require, _same,
    MAX_PROMPT_BYTES, MAX_TASK_CHARS)
from .external_reflection import _digest, _event, _hash, _id, _journal
from .external_research_v2 import policy_identity, _review_text_policy
from .evaluated_study_import import validate_bundle
from .db import normalized_hash
from ..hypothesis import circle_optimizer_traced_body as builder
from ..hypothesis import component_generation_hypothesis as wire
from ..hypothesis import restart_predictions
from ..hypothesis.comparison_targets import instruction as target_instruction
from ..hypothesis.review_eligibility import decision_for
from ..knowledge_base.catalogs import canonical_json, canonical_sha
from ..knowledge_base.generated_paired_history import _bounded
from ..knowledge_base.evaluated_study_context import build_context, block

SCHEMA = 'sciel-evaluated-target-generation-result/1'
PLAN_SCHEMA = 'sciel-evaluated-target-generation-plan/1'
JOURNAL_CONTRACT = 'sciel-evaluated-target-generation/1'


def _sources(bundle):
    sources = {}
    for digest, source in bundle['capsule']['source_registry'].items():
        body = builder.body_of(source)
        _require(builder.source_for(body) == source and _digest(source) == digest,
                 'evaluated study source reconstruction differs')
        sources[digest] = dict(source=source, body=body, body_sha256=_digest(body),
                              normalized_hash=normalized_hash(source))
    return dict(sorted(sources.items()))


class EvaluatedTargetGenerationJob(ComponentTargetGenerationJob):
    RESULT_SCHEMA = SCHEMA
    GENERATION_ID_PREFIX = 'evaluated-generated:'

    def __init__(self, bundle, review_sidecar, *, reflection_config,
                 expected_sha256, expected_authority_sha256, expected_sidecar_sha256,
                 expected_card_sha256, config, task_prompt):
        _bounded([bundle, review_sidecar, reflection_config, config])
        _require(type(bundle) is dict and type(bundle.get('capsule')) is dict
            and type(reflection_config) is dict and type(review_sidecar) is dict,
            'literal evaluated study and reflection dependencies required')
        _require(type(config) is dict and all(_hash(value) for value in
            (expected_sha256, expected_authority_sha256, expected_sidecar_sha256, expected_card_sha256)),
            'explicit evaluated study/review/card digests required')
        _require(config.get('external_evaluated_study_sha256') == expected_sha256
            and config.get('external_evaluated_study_authority_sha256') == expected_authority_sha256
            and config.get('external_evaluated_review_sha256') == expected_sidecar_sha256
            and config.get('source_reflection_config_sha256') == canonical_sha(reflection_config),
            'independently configured evaluated study/review/reflection authority differs')
        _require(not any(key in config for key in ('external_history_sha256', 'external_component_sha256',
            'external_component_review_sha256', 'generated_paired_review_version')),
            'generation requires one evaluated study scope')
        _require(all(config.get(key) is True for key in
            ('comparison_review_text', 'restart_predictions', 'inject_lessons', 'structured_auxiliary',
             'comparison_reviews', 'batch_only_lessons')), 'explicit review delivery required')
        model = config.get('model')
        _require(type(model) is str and 0 < len(model) <= 256 and bool(model.strip()), 'explicit model required')
        _require(type(task_prompt) is str and 0 < len(task_prompt) <= MAX_TASK_CHARS
            and bool(task_prompt.strip()), 'bounded task prompt required')
        task_prompt.encode('utf-8')
        validated = validate_bundle(bundle, config, metric=bundle['capsule']['metric'])
        from .evaluated_study_reflection import validate_sidecar
        from ..hypothesis.evaluated_study_review import CARD_SCHEMA
        from ..hypothesis.evaluated_study_targets import validate_target
        sidecar = validate_sidecar(review_sidecar, bundle=validated,
            expected_sha256=expected_sidecar_sha256, config=reflection_config)
        card = sidecar.get('review_card')
        _require(type(card) is dict and card.get('schema') == CARD_SCHEMA
            and canonical_sha(card) == expected_card_sha256, 'exact selected evaluated study card required')
        policy = policy_identity()
        text_policy = _review_text_policy(config)
        _require(text_policy is not None, 'explicit host eligibility registry required')
        eligibility = decision_for(card, text_policy, policy_sha256=canonical_sha(policy))
        _require(eligibility['status'] == 'accepted', 'exact current-policy host acceptance required')
        target = card.get('resolved_target')
        _require(target is not None, 'selected review has no prospective target')
        target = validate_target(target, bundle=validated)
        self._bundle, self._sidecar = validated, sidecar
        self._reflection_config, self._config = copy.deepcopy(reflection_config), copy.deepcopy(config)
        self._card, self._target, self._policy = copy.deepcopy(card), target, policy
        self._task_prompt, self._sources = task_prompt, _sources(validated)
        self._bindings = dict(study_sha256=expected_sha256, author_authority_sha256=expected_authority_sha256,
            bundle_sha256=canonical_sha(validated), sidecar_sha256=expected_sidecar_sha256,
            reflection_config_sha256=canonical_sha(reflection_config), card_sha256=expected_card_sha256,
            comparison_id=card['comparison_id'], raw_reflection_sha256=card['writer']['raw_response_sha256'],
            host_registry_sha256=canonical_sha(text_policy), host_decision_sha256=eligibility['decision_sha256'],
            policy_sha256=canonical_sha(policy), target_sha256=canonical_sha(target),
            source_registry_sha256=canonical_sha(self._sources), config_sha256=canonical_sha(config),
            task_sha256=_digest(task_prompt))

    def _unchanged(self):
        _require(_same(policy_identity(), self._policy), 'production policy changed since preparation')
        pairs = ((self._bundle, 'bundle_sha256'), (self._sidecar, 'sidecar_sha256'),
                 (self._reflection_config, 'reflection_config_sha256'), (self._config, 'config_sha256'),
                 (self._card, 'card_sha256'), (self._target, 'target_sha256'),
                 (self._sources, 'source_registry_sha256'))
        _require(all(canonical_sha(value) == self._bindings[key] for value, key in pairs)
            and _digest(self._task_prompt) == self._bindings['task_sha256'], 'frozen input changed')

    def _append(self, rows, plan, phase, **fields):
        _event(rows, plan['run_id'], plan['segment_id'], 'evaluated_target_generation',
            contract=JOURNAL_CONTRACT, phase=phase, request_key=plan['request_key'], **fields)

    def _historical_provider_ids(self):
        # Only IDs actually retained in authenticated dependencies are available.
        # This consistency check is not a claim about provider-side freshness.
        from ..knowledge_base.evaluated_detached_study import _retained_ancestry
        return {self._sidecar['provider_response']['id'],
                *_retained_ancestry(self._bundle)['provider_response_ids']}

    def _consumed_sources(self):
        # __init__ authenticates the complete bundle; _unchanged guards it on
        # admission. Ancestors are a replay set, not additional editable targets.
        from ..knowledge_base.evaluated_detached_study import _retained_ancestry
        return {digest: dict(normalized_hash=normalized_hash(source)) for digest, source
                in _retained_ancestry(self._bundle)['source_registry'].items()}

    def prepare(self, *, run_id, segment_id, transport_settings,
                request_key='evaluated_target_g1', max_tokens=4096):
        self._unchanged()
        _require(all(_id(value) for value in (run_id, segment_id, request_key)), 'bounded request identities required')
        _require(run_id != self._sidecar['plan']['run_id'], 'fresh generation run required')
        _require(type(max_tokens) is int and 1 <= max_tokens <= 8192, 'bounded generation token cap required')
        transport = _transport(transport_settings)
        capsule, target = self._bundle['capsule'], self._target
        projection = build_context(capsule, author_dependencies=self._bundle['author_dependencies'],
            author_authority=self._bundle['author_authority'], branch=capsule['receiving_root']['id'],
            max_chars=16000, max_items=1)
        _require(projection is not None and projection['presented_count'] == 1,
                 'whole evaluated study does not fit generation memory')
        bodies = {role: dict(source_sha256=target[key], body=self._sources[target[key]]['body'],
            body_sha256=self._sources[target[key]]['body_sha256']) for role,key in
            (('edit_base','base_source_sha256'), ('paired_comparator','comparator_source_sha256'))}
        native = dict(status='UNMEASURED', score=None, forecast=None, ordinary_origin=None,
            reason='Detached generation has no native measurement or ordinary parent.')
        from ..hypothesis.evaluated_study_review import _source_block
        # Preserve every fact in lossless tables, without repeating the full map
        # inside the selected card. This delivery object is not a stored card.
        selected_review = dict(schema='sciel-evaluated-study-review-delivery/1',
            card_sha256=canonical_sha(self._card),
            card_fields={key: copy.deepcopy(value) for key, value in self._card.items()
                         if key != 'host_facts'},
            host_facts_catalog_sha256=canonical_sha(self._card['host_facts']),
            host_facts_delivery='Complete lossless typed tables in the following study source block.')
        fact_block = _source_block(self._card['host_facts'])
        intent = dict(bindings=self._bindings, target=target, receiving_root=capsule['receiving_root'],
            selected_review=selected_review, source_bodies=bodies, builder_identity=builder.identity(),
            interface=builder.interface_description(), native_observation=native,
            source_scope='Complete editable bodies and exact shared scaffold identity. Model target choices '
                'are preserved in selected_review.card_fields.unverified_interpretation; host binding fields '
                'are deterministic resolution, not newly authored decisions or measurements.')
        quality = target_instruction()
        quality = quality[quality.index('The caller must verify'):]
        restart = restart_predictions.instruction().replace(
            'These forecasts do not change the native score forecast or quality/lesson gates.',
            'This detached job has no native score forecast; restart forecasts do not change quality or lesson gates.')
        prompt = (wire.INSTRUCTION + '\n' + quality + '\n' + restart + '\n\nTASK:\n' + self._task_prompt
            + '\n\nEVALUATED_TARGET_CONTEXT_JSON:\n' + canonical_json(intent) + '\n\n'
            + fact_block + '\n' + block(projection))
        _require(len(prompt.encode('utf-8')) <= MAX_PROMPT_BYTES, 'complete required prompt exceeds byte cap')
        response_format = wire.response_format(self._bindings['target_sha256'])
        request = dict(model=self._config['model'], messages=[dict(role='user',content=prompt)],
            temperature=.2, max_tokens=max_tokens, response_format=response_format, extra_body=transport)
        plan = dict(schema=PLAN_SCHEMA, **copy.deepcopy(self._bindings), run_id=run_id, segment_id=segment_id,
            request_key=request_key, max_tokens=max_tokens, transport_settings=transport,
            request=request, request_sha256=canonical_sha(request), prompt=prompt, prompt_sha256=_digest(prompt),
            prompt_bytes=len(prompt.encode('utf-8')), response_format=response_format,
            builder_identity=builder.identity(), execution_profile=builder.EXECUTION_PROFILE,
            target=copy.deepcopy(target), source_registry=copy.deepcopy(self._sources),
            native_observation=native, new_ordinary_outcomes=0, new_search_attempts=0)
        rows = []
        self._append(rows, plan, 'prepared', bindings=copy.deepcopy(self._bindings),
            request_sha256=plan['request_sha256'], prompt_sha256=plan['prompt_sha256'],
            response_format=response_format, base_source_sha256=target['base_source_sha256'],
            comparator_source_sha256=target['comparator_source_sha256'])
        plan.update(journal_prefix=_journal(rows), journal_prefix_sha256=_digest(_journal(rows)))
        return plan


def validate_result(result, *, bundle, review_sidecar, reflection_config, expected_sha256, config):
    """Rebuild from separately pinned study, raw review and receiving host decision."""
    _bounded(result)
    _require(_hash(expected_sha256) and canonical_sha(result) == expected_sha256
        and type(result) is dict and result.get('schema') == SCHEMA, 'declared evaluated result differs')
    plan = result.get('plan')
    _require(type(plan) is dict and _same(result.get('config'), config), 'receiving configuration differs')
    job = EvaluatedTargetGenerationJob(bundle, review_sidecar, reflection_config=reflection_config,
        expected_sha256=config.get('external_evaluated_study_sha256'),
        expected_authority_sha256=config.get('external_evaluated_study_authority_sha256'),
        expected_sidecar_sha256=config.get('external_evaluated_review_sha256'),
        expected_card_sha256=plan.get('card_sha256'), config=config, task_prompt=result.get('task_prompt'))
    rebuilt = job.admit(plan, result.get('response'), manifest_sha256=result.get('manifest_sha256'),
        journal_prefix=plan.get('journal_prefix'), provider_request=result.get('provider_request'),
        provider_response=result.get('provider_response'))
    _require(_same(rebuilt, result), 'evaluated generation differs from bound raw author, source or journal')
    return copy.deepcopy(rebuilt)
