"""Evaluate an exact tool-produced program through the ordinary search gates.

No model rewriting or silent graft changes the submitted artifact. This is an
unpaired candidate observation, not an efficacy claim about its producing tool.
"""
from copy import deepcopy

from ..methods.models import content_hash
from .types import NoChild, Result
from .validation import source_sha256, validate_candidate
from .worker import evaluation_attempt, evaluation_metadata
from .output_novelty import is_repeat
from ..research_tools.canonical_diagnostics import capture as capture_diagnostics


def candidate_binding(tools, episode, result):
    """Bind successful research results before any admission/accounting mutation."""
    if not getattr(result, 'ok', False):
        return None
    provenance = result.meta.get('research_candidate')
    if provenance is None and result.origin != 'research_tool':
        return None
    if type(provenance) is not dict or not provenance:
        raise ValueError('successful research candidate is missing its producing provenance')
    return tools.candidate_binding(episode, source=result.child_code,
        provenance=provenance, evaluation_id=result.meta.get('evaluation_id'),
        score=result.score, cases=result.cases)


def evaluate_submission(source, provenance, snapshot, evaluate_fn, novelty_check=None):
    digest = source_sha256(source)
    if provenance.get('candidate_source_sha256') != digest:
        raise ValueError('research candidate source differs from its immutable receipt')
    target = provenance.get('target_binding') or {}
    parent_id = target.get('ancestry_parent_id') or snapshot.parent_id
    validation = validate_candidate(source, parent_code=snapshot.parent_code,
        required_entrypoints=snapshot.required_entrypoints,
        code_hashes=snapshot.code_hashes, novelty_check=novelty_check)
    meta = dict(op_used='research_tool', op_on_policy=False, op_requested=snapshot.op,
        op_initial='research_tool', op_fallback_reason='artifact_candidate_submission',
        operator_selection='research_cell', escalated=False, dup_attempts=0,
        research_candidate=deepcopy(provenance), novelty=validation.novelty,
        novelty_checks=[validation.novelty] if validation.novelty else [],
        validation_trace=dict(schema='sciel-validate-repair/1', profile='research_artifact',
            attempts=[dict(attempt=1, operator='research_tool',
                candidate_code_sha256=digest, decisions=validation.decisions)]),
        **evaluation_metadata([]))
    feedback = dict(schema='sciel-research-candidate-evaluation/1', source_sha256=digest,
        status='rejected', evaluation_id=None, score=None, efficacy='unknown',
        scope='One ordinary unpaired evaluation does not establish method benefit or cross-branch transfer.')
    if validation.code != source or validation.violation is not None:
        reason = ('Exact saved source requires repair/graft; revise the tool and save a new candidate artifact.'
                  if validation.code != source else validation.violation.detail or validation.violation.kind)
        feedback.update(reason=reason, novelty=validation.novelty)
        return NoChild(snapshot.island, 'research_candidate_rejected', meta), feedback
    outcome = evaluate_fn(source)
    measured = evaluation_attempt(outcome, source)
    remeasurement = bool(validation.novelty and validation.novelty.get('remeasurement'))
    repeated = measured['output_novelty'] is not None and is_repeat(measured['output_novelty']) and not remeasurement
    measured['admission'] = 'output_repeat_rejected' if repeated else 'candidate' if outcome.result.ok else 'eval_failed'
    meta.update(evaluation_metadata([measured]), evaluation_id=getattr(outcome, 'evaluation_id', None),
        remeasurement=remeasurement)
    result = outcome.result
    feedback.update(status='output_repeat_rejected' if repeated else 'evaluated' if result.ok else 'eval_failed',
        evaluation_id=getattr(outcome, 'evaluation_id', None), score=result.score if result.ok else None,
        cases=deepcopy(result.cases), reached=outcome.reached, confirmed=outcome.confirmed,
        timing_seconds=result.timing, timing_scope='Returned evaluation-stage wall time including evaluator overhead; '
            'reported trial count is in cases; CPU time unavailable',
        artifacts_sha256=content_hash(result.artifacts or {}),
        output_novelty=deepcopy(measured['output_novelty']))
    if getattr(outcome, 'execution', None) is not None:
        feedback['execution'] = deepcopy(outcome.execution)
        feedback['timing_scope'] = 'Returned stage host wall time; all-stage candidate/scorer container CPU is separately bound in execution.'
    if not result.ok:
        feedback['failure_diagnostics'] = capture_diagnostics(result,
            source_sha256=digest, evaluation_id=getattr(outcome, 'evaluation_id', None),
            record_reference=getattr(outcome, 'saved_evaluation_reference', None))
    if repeated:
        return NoChild(snapshot.island, 'output_repeat_cap', meta), feedback
    return Result(child_code=source, score=result.score, cases=result.cases, ok=result.ok,
        island=snapshot.island, parent_id=parent_id, origin='research_tool',
        bet=None, meta=meta, raw_first_score=outcome.raw_first_score,
        reached=outcome.reached, confirmed=outcome.confirmed,
        prompt='', response='', cost=0.0, timing=result.timing, artifacts=result.artifacts), feedback
