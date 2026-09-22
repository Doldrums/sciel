"""Host-authored delivery receipts, not a claim of complete learned ancestry.

The first slice records ordinary G/repair inputs at their final render boundary.
Research projections and learned/program ancestry remain explicitly unresolved.
No registry lookup, source-similarity inference or provider request happens here.
"""
from __future__ import annotations

from copy import deepcopy

from .types import InferenceSpec, content_hash, sha256_bytes

SCHEMA = 'sciel-adaptation-visibility/1'
STANDARD_OPS = frozenset({'full_rewrite', 'diff', 'cross', 'error_repair'})


def _sha(text):
    return sha256_bytes(text.encode('utf-8'))


def freeze_context(engine, snapshot, parent):
    """Freeze identities from authoritative native state, before job intent."""
    if getattr(engine.llm, 'local_inference', False) is not True:
        return
    same = snapshot.parent_code == parent.code
    public = same and parent.origin == 'seed' and not engine.cfg.get('warm_start')
    target = dict(kind='public_source' if public else 'program' if same else 'failed_source',
                  role='target', program_id=parent.id, source_sha256=_sha(snapshot.parent_code))
    observed = engine.observed_changes.block(snapshot.parent_code)
    snapshot.adaptation_context = dict(run_id=engine.log.run_id, task_hash=engine.task.task_hash,
        target=target, measured_parent=dict(program_id=parent.id,
            source_sha256=_sha(parent.code), score=parent.score),
        observed_changes=(dict(text_sha256=_sha(observed), chars=len(observed)) if observed else None))


def card_refs(context):
    """Only cards retained by the final renderer; available inventory is absent."""
    if context is None:
        return []
    from ..research_tools.followups import capture_exposures
    # Reuse the strict native card validator. The real prompt hash is sealed below.
    capture_exposures(context, prompt_sha256='0' * 64)
    return [dict(kind='research_card', role='context', card_sha256=content_hash(card),
        cell_id=card['cell_id'], revision_id=card['revision_id'], source=deepcopy(card['source']),
        **({'decision_proposal_id': card['decision']['proposal_id']}
           if (card.get('decision') or {}).get('proposal_id') is not None else {}))
        for card in context['cards']]


def capture_research_delivery(snapshot, prompt, context, rendered_context):
    """Called after BOTH research memory budgets, immediately before dispatch."""
    if getattr(snapshot, 'inference_spec', None) is None:
        return
    refs = [dict(kind='research_context', role='context',
                 text_sha256=_sha(rendered_context), chars=len(rendered_context))]
    refs.extend(card_refs(context.get('research_cells_context')))
    snapshot.adaptation_delivery = dict(prompt_sha256=_sha(prompt), delivered_refs=refs)


def _seal(prompt, spec, refs, reasons):
    value = dict(schema=SCHEMA, prompt_sha256=_sha(prompt), inference_spec=deepcopy(spec),
        delivered_refs=deepcopy(refs), ancestry=dict(status='unknown' if reasons else 'complete',
            reasons=sorted(set(reasons))))
    return {**value, 'sha256': content_hash(value)}


def unknown_capsule(prompt, spec):
    return _seal(prompt, spec, [], ['generation_render_not_bound'])


def build_capsule(snapshot, prompt, spec, *, op, previous_response=None, previous_call_id=None):
    InferenceSpec.from_dict(spec)
    refs, reasons = [], []
    frozen = snapshot.adaptation_context
    from ..engine.prompt.build import build, _load_template
    if op not in STANDARD_OPS:
        reasons.append('specialized_generation_render_not_bound')
    elif build(op, snapshot) not in prompt:
        reasons.append('standard_generation_render_not_bound')
    else:
        from ..engine.prompt.build import source_bundle_for_prompt
        template = _load_template(op)
        bundle = source_bundle_for_prompt(op, snapshot)
        target = deepcopy((frozen or {}).get('target'))
        if target is None:
            target = dict(kind='program', role='target', program_id=snapshot.parent_id,
                          source_sha256=_sha(snapshot.parent_code))
            reasons.append('native_source_identity_not_frozen')
        if (target['source_sha256'] != bundle['target_ref']
                or target['program_id'] != snapshot.parent_id):
            raise ValueError('visibility target differs from frozen source')
        refs.append(target)
        if target['kind'] != 'public_source':
            reasons.append('program_ancestry_unresolved')
        if snapshot.task_prompt and '{task_prompt}' in template:
            refs.append(dict(kind='public_task', role='task', text_sha256=_sha(snapshot.task_prompt),
                task_hash=(frozen or {}).get('task_hash')))
        for row in bundle['inspirations']:
            refs.append(dict(kind='program', role='inspiration', program_id=row.get('program_id'),
                source_sha256=row['source_ref']))
            reasons.append('program_ancestry_unresolved')
        # Fragments are interned into the actual research source catalog.
        if snapshot.research_context is not None:
            for row in bundle['fragments']:
                refs.append(dict(kind='research_fragment', role='fragment',
                    experiment_id=row.get('experiment_id'), source_sha256=row['source_ref']))
        if snapshot.kb_lessons and '{lessons}' in template:
            refs.extend(dict(kind='lesson_text', role='lesson', text_sha256=_sha(text))
                        for text in snapshot.kb_lessons)
            reasons.append('lesson_ancestry_unresolved')
        if snapshot.artifacts and '{artifacts}' in template:
            refs.append(dict(kind='failure_context', role='diagnostic',
                             content_sha256=content_hash(snapshot.artifacts)))
            reasons.append('failure_ancestry_unresolved')
        if frozen and frozen.get('observed_changes'):
            refs.append(dict(kind='observed_changes', role='directive', **frozen['observed_changes']))
            reasons.append('observed_history_ancestry_unresolved')
    if snapshot.research_context is not None:
        delivered = snapshot.adaptation_delivery
        if delivered is None or delivered['prompt_sha256'] != _sha(prompt):
            reasons.append('final_research_delivery_not_bound')
        else:
            refs.extend(deepcopy(delivered['delivered_refs']))
        reasons.append('research_ancestry_unresolved')
    elif snapshot.research_cells_context is not None:
        # The no-research path binds the formatter's final output in _execute_step.
        delivered = snapshot.adaptation_delivery
        if delivered and delivered.get('text') and delivered['text'] in prompt:
            refs.extend(deepcopy(delivered['delivered_refs']))
        elif snapshot.research_cells_context.get('cards'):
            reasons.append('final_card_delivery_not_bound')
        if snapshot.research_cells_context.get('cards'):
            reasons.append('research_ancestry_unresolved')
    if previous_response is not None:
        refs.append(dict(kind='llm_response', role='previous_response', call_id=previous_call_id,
                         response_sha256=_sha(previous_response)))
        reasons.append('retry_ancestry_unresolved')
    if spec.get('adapter_id') is not None:
        reasons.append('adapter_training_ancestry_unresolved')
    return _seal(prompt, spec, refs, reasons)


def validate_capsule(value, *, prompt, inference_spec):
    if (not isinstance(value, dict) or set(value) != {'schema', 'prompt_sha256', 'inference_spec',
            'delivered_refs', 'ancestry', 'sha256'} or value.get('schema') != SCHEMA):
        raise ValueError('invalid adaptation visibility capsule')
    if value['sha256'] != content_hash({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('adaptation visibility hash differs')
    if value['prompt_sha256'] != _sha(prompt) or value['inference_spec'] != inference_spec:
        raise ValueError('adaptation visibility differs from final prompt/inference')
    InferenceSpec.from_dict(inference_spec)
    ancestry = value['ancestry']
    if (not isinstance(ancestry, dict) or set(ancestry) != {'status', 'reasons'}
            or ancestry['status'] not in ('complete', 'unknown')
            or not isinstance(ancestry['reasons'], list)
            or any(not isinstance(r, str) or not r for r in ancestry['reasons'])
            or (ancestry['status'] == 'complete') != (not ancestry['reasons'])
            or not isinstance(value['delivered_refs'], list)
            or any(not isinstance(r, dict) or not r.get('kind') or not r.get('role')
                   for r in value['delivered_refs'])):
        raise ValueError('invalid adaptation visibility resolution')
    if ancestry['status'] == 'complete':
        # This slice can certify only public inputs. Program/research/learned
        # history requires the future authenticated ancestry resolver.
        targets = [r for r in value['delivered_refs'] if r['role'] == 'target']
        if (inference_spec.get('adapter_id') is not None or len(targets) != 1
                or targets[0]['kind'] != 'public_source'
                or any(r['kind'] not in ('public_source', 'public_task') for r in value['delivered_refs'])):
            raise ValueError('complete visibility requires exclusively public inputs')
    return deepcopy(value)
