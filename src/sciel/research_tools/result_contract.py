"""Declared research outcomes, separate from immutable executor receipts.

Only new tool versions pin this contract. Historical raw JSON must not be
retroactively classified as a declared outcome. No arbitrary output keys or
task scores are interpreted here.
"""
from ..methods.models import content_hash, detached

RESULT_SCHEMA = 'sciel-tool-result/1'
ASSESSMENT_SCHEMA = 'sciel-tool-result-assessment/1'
REPLAY_POLICIES = ('exact_observations', 'independent_execution')


def validate_policy(policy):
    if type(policy) is not str or policy not in REPLAY_POLICIES:
        raise ValueError('unsupported immutable research replay policy')
    return policy


def parse_result(output):
    if (type(output) is not dict
            or set(output) != {'schema', 'status', 'observations', 'diagnostics', 'error'}
            or output.get('schema') != RESULT_SCHEMA
            or output.get('status') not in ('success', 'tool_error')
            or type(output.get('observations')) is not dict
            or type(output.get('diagnostics')) is not dict):
        raise ValueError('tool must return a sciel-tool-result/1 envelope with status, observations, diagnostics and error')
    error = output['error']
    if output['status'] == 'success':
        if error is not None:
            raise ValueError('successful tool result requires error=null')
    elif type(error) is not str or not error.strip():
        raise ValueError('tool_error requires a nonempty error string')
    return detached(output)


def assess_receipt(receipt):
    """Project a new-contract receipt without changing host evidence or cost."""
    host = receipt.get('status')
    result = dict(schema=ASSESSMENT_SCHEMA, host_status=host,
                  status='unknown' if host == 'unknown' else 'host_failed',
                  error=receipt.get('reason'), observations_sha256=None,
                  diagnostics_sha256=None)
    if host != 'ok' or receipt.get('dispatch_status') != 'invoked':
        return result
    try:
        output = parse_result(receipt.get('output'))
    except (ValueError, TypeError, RecursionError) as error:
        result.update(status='invalid_output', error=str(error))
        return result
    result.update(status=output['status'], error=output['error'],
                  observations_sha256=content_hash(output['observations']),
                  diagnostics_sha256=content_hash(output['diagnostics']))
    return result


def replay_evidence(reference, replay, policy):
    """Evaluate the policy fixed at creation, retaining both observation hashes.

    Independent execution establishes contract-valid execution on the pinned
    input and runtime, never measurement reproduction or scientific benefit.
    """
    validate_policy(policy)
    original, actual = assess_receipt(reference), assess_receipt(replay)
    pinned = all(reference.get(key) == replay.get(key) and reference.get(key) is not None
                 for key in ('method_id', 'input_sha256', 'source_sha256', 'runtime_sha256', 'limits'))
    matched = (original['status'] == actual['status'] == 'success' and pinned
               and (policy == 'independent_execution'
                    or original['observations_sha256'] == actual['observations_sha256']))
    return dict(policy=policy, matched=matched,
                reference_observations_sha256=original['observations_sha256'],
                replay_observations_sha256=actual['observations_sha256'],
                measurement_reproduced=matched and policy == 'exact_observations')
