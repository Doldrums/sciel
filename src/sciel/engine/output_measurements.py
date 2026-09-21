"""Restore near-output evidence from immutable evaluator receipts, never scores alone."""
import hashlib
from pathlib import Path
from ..eval.cascade import CascadeOutcome
from ..eval.facade import EvalResult
from ..eval.output_identity import adapt_output, canonical
from ..eval.output_identity import adapt_near_output
from .research_artifacts import saved_evaluation_by_hash


def validate_task_resolution(task, policy):
    """Fail before dispatch when a trusted opt-in has no matching task adapter."""
    from ..eval.near_output import validate_policy
    from ..eval.output_identity import validate_output_identity_config
    policy = validate_policy(policy)
    if policy is None:
        return None
    evaluator = hashlib.sha256(Path(task.evaluator_path).read_bytes()).hexdigest()
    config = validate_output_identity_config(getattr(task, 'output_identity', None))
    if config is not None:
        adapter = config['adapter']
    else:
        from .repeat_support import NATIVE_GEOMETRY_TASK_EVALUATORS
        adapter = ('native-circle-geometry/1'
                   if NATIVE_GEOMETRY_TASK_EVALUATORS.get(task.task_hash) == evaluator else None)
    if (policy['task_hash'] != task.task_hash or policy['evaluator_sha256'] != evaluator
            or policy['adapter'] != adapter):
        raise ValueError('output resolution differs from configured task/evaluator/adapter')
    return policy


def restore_measurement(engine, evaluation_id, identity, binding, *, run_id):
    saved = saved_evaluation_by_hash(engine, binding['source_sha256'], evaluation_id,
                                    expected_run_id=run_id)
    if saved['status'] != 'available' or saved['reference']['sha256'] != binding.get('receipt_sha256'):
        raise ValueError('near-output history lacks its bound immutable evaluation receipt')
    record = saved['record']
    outcome = CascadeOutcome(EvalResult.from_json(record['result']), record['raw_first_score'],
                             record['reached'], record['confirmed'])
    rebuilt = adapt_output(engine.task, binding['source_sha256'], outcome, binding)
    if canonical(rebuilt) != canonical(identity):
        raise ValueError('output identity differs from immutable evaluation receipt')
    return adapt_near_output(engine.task, binding['source_sha256'], outcome, binding,
        identity=rebuilt, policy=engine.cfg['candidate_novelty'].get('output_resolution'))
