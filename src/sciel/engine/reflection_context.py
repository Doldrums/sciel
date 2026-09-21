"""Pure reflection preparation shared by live scheduling and imported history.

Preparing a prompt does not schedule a request, increment a watermark or turn
historical observations into outcomes of a new run.
"""
from __future__ import annotations

import copy
import hashlib

from ..hypothesis.conclusion_feedback import (
    reflection_review_context, REFLECTION_REVIEW_MAX_CHARS,
    REFLECTION_REVIEW_PREFIX, REFLECTION_REVIEW_RULES)
from ..hypothesis.prompts import reflection_prompt, initializer_block, restart_block
from ..hypothesis.rendering import MAX_CONTEXT_CHARS, serialize_reflection
from ..knowledge_base.catalogs import canonical_json, canonical_sha
from ..knowledge_base.paired_comparisons import SCHEMA as PAIRED_SCHEMA


ALLOCATION_POLICY = "sciel-reflection-anchor-allocation/1"

def _digest(text):
    return hashlib.sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest()


def initializer_context(knowledge, config, ids, available):
    if not config.get('initializer_witness', False):
        return None
    from .initializer_research import MAX_CONTEXT_CHARS as maximum
    from ..knowledge_base.presentation import INITIALIZER_CONTEXT_PREFIX, INITIALIZER_CONTEXT_RULES
    allowance = max(0, min(maximum, available - len(INITIALIZER_CONTEXT_PREFIX)
                           - len(INITIALIZER_CONTEXT_RULES) - 1))
    return knowledge.initializer_context(ids, max_items=3, max_chars=allowance)


def restart_context(groups, available):
    from ..knowledge_base.restart_context import build_context, RULES, PREFIX
    selected = [g for g in groups if any(
        type(p.get(role)) is dict and 'restart_trace' in p[role]
        for p in g['pairs'] for role in ('comparator', 'challenger'))]
    if not selected:
        return None, []
    allowance = max(0, min(12000, available - len(RULES) - len(PREFIX) - 1))
    return build_context(selected, max_chars=allowance), [g['id'] for g in selected]


def prediction_context(knowledge, config, groups, available):
    if not config.get('restart_predictions', False):
        return None
    from ..knowledge_base.restart_prediction_context import build_context, MAX_CHARS, RULES, PREFIX
    records = knowledge._evidence()
    records.update((r['id'], r) for r in knowledge.experiments)
    allowance = max(0, min(MAX_CHARS, available - len(RULES) - len(PREFIX) - 1))
    return build_context(groups, records, max_chars=allowance)


REVIEW_ONLY_INSTRUCTION = (
    'This reflection stage is review-only: return lessons=[] and retire_ids=[]. '
    'Any proposed lesson or retirement rejects the whole response before memory changes. '
    'Comparison reviews are stored for audit; a separate host decision controls whether '
    'their complete unverified text may be shown in a later generation. Structural '
    'acceptance grants neither semantic validity nor text delivery authority.\n\n'
)


def _component_review_context(knowledge, config, component):
    """Validate the immutable R sidecar before considering text eligibility."""
    from .generated_paired_review_import import configured_digest
    expected = configured_digest(config)
    sidecar = knowledge.generated_paired_review
    if expected is None and sidecar is None:
        return None
    if expected is None:
        raise ValueError('generated paired reflection requires configured review digest')
    if component is None or sidecar is None:
        raise ValueError('generated paired reflection requires imported component and review')
    from .external_research_v2 import _review_text_policy, policy_identity
    policy = _review_text_policy(config)
    if policy is None:
        raise ValueError('generated paired reflection review requires explicit host text policy')
    from .generated_paired_reflection import validate_sidecar
    checked = validate_sidecar(sidecar, component=component, expected_sha256=expected)
    from ..knowledge_base.generated_paired_review_context import build_context
    return build_context(component, checked['review_cards'], branch=component['root_parent']['id'],
        max_chars=8000, text_policy=policy, policy_sha256=canonical_sha(policy_identity()),
        text_enabled=config['comparison_review_text'], enabled=config.get('inject_lessons', True))


def _evaluated_study_context(knowledge, config):
    from .evaluated_study_import import configured_digests, validate_bundle
    expected, _ = configured_digests(config)
    bundle = knowledge.evaluated_detached_study
    if expected is None and bundle is None:
        return None
    if bundle is None:
        raise ValueError('evaluated reflection requires imported study')
    validate_bundle(bundle, config, metric=bundle['capsule']['metric'])
    from ..knowledge_base.evaluated_study_context import build_context
    return build_context(bundle['capsule'], author_dependencies=bundle['author_dependencies'],
        author_authority=bundle['author_authority'], branch=bundle['capsule']['receiving_root']['id'],
        max_chars=16000, max_items=1, enabled=config.get('inject_lessons', True))


def _evaluated_review_context(knowledge, config, study):
    from .evaluated_review_import import configured_digests, validate_retained
    expected, _ = configured_digests(config)
    retained = knowledge.evaluated_study_review
    if expected is None and retained is None:
        return None
    if retained is None or knowledge.evaluated_detached_study is None:
        raise ValueError('evaluated reflection requires imported study and review')
    checked = validate_retained(retained, knowledge.evaluated_detached_study, config)
    from .external_research_v2 import _review_text_policy, policy_identity
    from ..knowledge_base.evaluated_review_context import build_context
    bundle = knowledge.evaluated_detached_study
    return build_context(bundle, checked['sidecar']['review_card'],
        branch=bundle['capsule']['receiving_root']['id'], max_chars=8000,
        text_policy=_review_text_policy(config), policy_sha256=canonical_sha(policy_identity()),
        text_enabled=config['comparison_review_text'], enabled=config.get('inject_lessons', True),
        source_facts_presented=study is not None and study['presented_count'] == 1)


def prepare_reflection_context(knowledge, conclusions, *, config, interval, through,
                               selection_epoch=None):
    """Return detached prompt and journal projections for one reflection.

    ``through`` identifies the caller's evidence watermark. External jobs must
    label it as historical; this function never owns local scheduler counters.
    Live callers supply the persisted R attempt ordinal as selection_epoch.
    Without it, preserve the unversioned archived recent-first allocation.
    """
    if type(interval) is not int or interval < 1:
        raise ValueError('reflection interval must be a positive integer')
    if type(through) is not int or through < 1:
        raise ValueError('reflection watermark must be a positive integer')
    if selection_epoch is not None and (type(selection_epoch) is not int or selection_epoch < 0):
        raise ValueError('reflection selection epoch must be a nonnegative integer')
    from .external_research_v2 import _review_text_policy
    review_only = _review_text_policy(config) is not None
    effects_instruction = REVIEW_ONLY_INSTRUCTION if review_only else ''
    from .generated_paired_import import configured_digest
    from ..knowledge_base.generated_paired_history import validate_capsule
    from ..knowledge_base.generated_paired_context import build_context as component_context, block as component_block
    capsule = knowledge.generated_paired_history
    component = None
    expected_component = configured_digest(config)
    if capsule is not None:
        if expected_component is None:
            raise ValueError('generated paired reflection requires configured evidence digest')
        validate_capsule(capsule, expected_sha256=expected_component, metric=capsule['metric'])
        component = component_context([capsule], branch=capsule['root_parent']['id'],
            max_chars=16000, max_items=1, enabled=config.get('inject_lessons', True))
    elif expected_component is not None:
        raise ValueError('generated paired reflection requires imported evidence')
    component_text = component_block(component)
    study = _evaluated_study_context(knowledge, config)
    from ..knowledge_base.evaluated_study_context import block as study_block
    study_text = study_block(study)
    study_review = _evaluated_review_context(knowledge, config, study)
    from ..knowledge_base.evaluated_review_context import block as study_review_block
    study_review_text = study_review_block(study_review)
    study_review_reserve = study_review['reserved_chars'] if study_review is not None else 0
    component_reviews = _component_review_context(knowledge, config, capsule)
    from ..knowledge_base.generated_paired_review_context import block as component_review_block
    component_review_text = component_review_block(component_reviews)
    component_review_reserve = component_reviews['reserved_chars'] if component_reviews is not None else 0
    selector = (knowledge.legacy_reflection_context if selection_epoch is None
                else knowledge.reflection_context)
    records, lessons = selector(interval,
        **({"selection_epoch": selection_epoch} if selection_epoch is not None else {}),
        max_chars=90000 - len(component_text) - component_review_reserve - len(study_text) - study_review_reserve)
    data = serialize_reflection(records, lessons)
    comparison_review_groups, review_source_chars = None, 0
    if config.get('comparison_reviews', False):
        from ..hypothesis.prompts import comparison_review_source_block
        retained = knowledge._evidence()
        comparison_review_groups = [retained[r['id']] for r in records
            if r['id'] in retained and retained[r['id']].get('schema') == PAIRED_SCHEMA][-4:]
        review_source_chars = len(comparison_review_source_block(comparison_review_groups,
            version=config.get('comparison_review_version', 1), records=retained, seen=knowledge._seen))
    review_source_chars += len(effects_instruction) + len(component_text) + component_review_reserve + len(study_text) + study_review_reserve
    ordinary = {ident: record for ident, record in knowledge._evidence().items()
                if record.get('schema') != PAIRED_SCHEMA}
    ordinary.update((record['id'], record) for record in knowledge.experiments)
    selected_ids = [r['id'] for r in records if r['id'] in ordinary]
    allowance = max(0, min(REFLECTION_REVIEW_MAX_CHARS,
        MAX_CONTEXT_CHARS - len(data) - len(REFLECTION_REVIEW_RULES)
        - len(REFLECTION_REVIEW_PREFIX) - 1 - review_source_chars))
    review_context = reflection_review_context(conclusions, ordinary,
        selected_ids, max_chars=allowance,
        batch_only_lessons=config.get('batch_only_lessons', False))
    review_chars = (len(REFLECTION_REVIEW_RULES) + len(REFLECTION_REVIEW_PREFIX)
        + len(canonical_json(review_context)) + 1 if review_context is not None else 0)
    initializer = initializer_context(knowledge, config, selected_ids,
        MAX_CONTEXT_CHARS - len(data) - review_chars - review_source_chars)
    retained = knowledge._evidence()
    trace_groups = [retained[r['id']] for r in records
        if r['id'] in retained and retained[r['id']].get('schema') == PAIRED_SCHEMA]
    restart, restart_ids = restart_context(trace_groups,
        MAX_CONTEXT_CHARS - len(data) - review_chars - review_source_chars - len(initializer_block(initializer)))
    predictions = prediction_context(knowledge, config, trace_groups,
        MAX_CONTEXT_CHARS - len(data) - review_chars - review_source_chars - len(initializer_block(initializer))
        - len(restart_block(restart)))
    prompt = reflection_prompt(records, lessons, review_context=review_context,
        initializer_context=initializer, restart_context=restart, prediction_context=predictions,
        structured=config.get('structured_auxiliary', False),
        comparison_reviews=config.get('comparison_reviews', False),
        comparison_review_groups=comparison_review_groups,
        comparison_review_version=config.get('comparison_review_version', 1),
        comparison_review_records=knowledge._evidence() if config.get('comparison_reviews', False) else None,
        comparison_review_seen=knowledge._seen if config.get('comparison_reviews', False) else None)
    prompt = effects_instruction + study_text + study_review_text + component_text + component_review_text + prompt
    if (component_reviews is not None or study is not None or study_review is not None) and (
            len(prompt) + component_review_reserve - len(component_review_text)
                + study_review_reserve - len(study_review_text) > MAX_CONTEXT_CHARS):
        raise ValueError('generated paired reflection review exceeds reserved prompt budget')
    presentation_events = []
    if config.get('initializer_witness', False):
        presentation_events.append(dict(phase='initializer_witness_presentation', role='reflection',
            selected_experiment_ids=selected_ids, prompt_sha256=_digest(prompt), experiment_id=None,
            context=initializer, context_sha256=_digest(canonical_json(initializer)) if initializer else None,
            quality_changed=False, delivery='prompt_prepared_not_model_use'))
    if restart_ids:
        presentation_events.append(dict(phase='restart_observation_presentation', role='reflection',
            selected_comparison_ids=restart_ids, experiment_id=None, context=restart,
            context_sha256=_digest(canonical_json(restart)) if restart else None,
            omission='presentation_budget' if restart is None else None,
            prompt_sha256=_digest(prompt), quality_changed=False, delivery='prompt_prepared_not_model_use'))
    context_event = dict(through=through,
        **({"allocation_policy": ALLOCATION_POLICY, "selection_epoch": selection_epoch}
           if selection_epoch is not None else {}),
        **({'evaluated_detached_study': copy.deepcopy(study)} if study is not None else {}),
        **({'evaluated_study_review': copy.deepcopy(study_review)} if study_review is not None else {}),
        **({'generated_paired_history': copy.deepcopy(component)} if component is not None else {}),
        **({'generated_paired_reviews': copy.deepcopy(component_reviews)} if component_reviews is not None else {}),
        **({'reflection_effects_policy': 'review_only/1'} if review_only else {}),
        experiment_ids=[r['id'] for r in records], lesson_ids=[lesson['id'] for lesson in lessons],
        omitted_lesson_count=len(knowledge.lessons) - len(lessons), context_sha256=_digest(data),
        review_context_sha256=(_digest(canonical_json(review_context)) if review_context is not None else None),
        review_selected_experiment_ids=selected_ids, review_context_max_chars=allowance,
        review_context_omission=('no_selected_ordinary' if not selected_ids else
                                'memory_char_limit' if review_context is None else None),
        review_presented_count=(review_context['presented_count'] if review_context else 0),
        review_omitted_count=(review_context['omitted_count'] if review_context else len(selected_ids)),
        **({'comparison_review_source_ids': [g['id'] for g in comparison_review_groups],
            'comparison_review_omitted_ids': [g['id'] for g in trace_groups
                if g['id'] not in {r['id'] for r in comparison_review_groups}]}
           if comparison_review_groups is not None else {}),
        **({'restart_predictions': copy.deepcopy(predictions)} if predictions is not None else {}))
    wire, schema = None, None
    if config.get('structured_auxiliary', False):
        from ..hypothesis.auxiliary_schema import identity, response_format
        wire = response_format('reflection', comparison_reviews=config.get('comparison_reviews', False),
            comparison_review_version=config.get('comparison_review_version', 1))
        schema = identity(comparison_reviews=config.get('comparison_reviews', False),
            comparison_review_version=config.get('comparison_review_version', 1))
    return copy.deepcopy(dict(prompt=prompt, response_format=wire, auxiliary_schema=schema,
        context_event=context_event, presentation_events=presentation_events,
        records=records, lessons=lessons, comparison_review_groups=comparison_review_groups))
