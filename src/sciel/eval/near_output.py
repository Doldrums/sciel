"""Task-bound numeric output resolution, independent of admission and scores.

This compares complete, ordered JSON observations. It does not establish program
equivalence. The caller authenticates the evaluator receipt; a seal alone is not
producer authentication. No task or numeric resolution is inferred here.
"""
from copy import deepcopy
from fractions import Fraction
import math

from .output_identity import (MAX_WITNESS_BYTES, canonical, check_seal, digest,
                              require, require_hash, validate_identity)

POLICY_SCHEMA = 'sciel-near-output-policy/1'
MEASUREMENT_SCHEMA = 'sciel-near-output-measurement/1'
ASSESSMENT_SCHEMA = 'sciel-near-output-assessment/1'


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def validate_policy(value):
    """Explicit trusted run policy. Uniform tolerance uses the declared units."""
    if value is None:
        return None
    require(type(value) is dict and set(value) == {'schema', 'task_hash',
        'evaluator_sha256', 'adapter', 'units', 'numeric_abs_tolerance'}
        and value['schema'] == POLICY_SCHEMA, 'invalid near-output policy fields/schema')
    for key in ('task_hash', 'evaluator_sha256'):
        require_hash(value[key], 'near-output ' + key)
    require(value['adapter'] in ('canonical-json/1', 'native-circle-geometry/1'),
            'unsupported near-output adapter')
    require(type(value['units']) is str and 0 < len(value['units']) <= 256
            and value['units'].strip() == value['units'] and '\x00' not in value['units'],
            'invalid near-output units')
    require(finite_number(value['numeric_abs_tolerance'])
            and value['numeric_abs_tolerance'] > 0, 'invalid near-output numeric tolerance')
    return deepcopy(value)


def policy_matches_identity(policy, identity):
    return all(policy[key] == identity['scope'][key]
               for key in ('task_hash', 'evaluator_sha256', 'adapter'))


def validate_measurement(value, *, identity, binding, policy):
    """Check joins and complete payload; caller must authenticate the receipt."""
    check_seal(value)
    require(type(value) is dict and set(value) == {'schema', 'policy_sha256',
        'identity_sha256', 'receipt_sha256', 'score', 'output', 'sha256'}
        and value['schema'] == MEASUREMENT_SCHEMA, 'invalid near-output measurement fields/schema')
    policy = validate_policy(policy)
    identity = validate_identity(identity)
    require(policy is not None and policy_matches_identity(policy, identity),
            'near-output policy differs from output scope')
    require_hash(binding.get('receipt_sha256'), 'near-output receipt')
    require(identity['evidence']['binding_sha256'] == digest(binding)
            and identity['evidence']['source_sha256'] == binding['source_sha256']
            and identity['scope']['task_hash'] == binding['task_hash'],
            'near-output evaluation binding differs')
    require(value['policy_sha256'] == digest(policy)
            and value['identity_sha256'] == identity['sha256']
            and value['receipt_sha256'] == binding['receipt_sha256'],
            'near-output measurement provenance differs')
    require(finite_number(value['score']), 'near-output score must be finite')
    output = canonical(value['output'])
    require(len(output.encode()) <= MAX_WITNESS_BYTES, 'near-output payload exceeds limit')
    require(digest(value['output']) == identity['output_sha256'],
            'near-output payload differs from exact output identity')
    return deepcopy(value)


def distance(left, right, policy):
    """Maximum absolute leaf delta, or None for different shape/outside radius.

    JSON node types, keys, list lengths/order and nonnumeric values must agree.
    In particular bool/int and int/float are distinct representations. Exact
    rational subtraction avoids overflow or rounding across the policy boundary.
    """
    policy = validate_policy(policy)
    require(policy is not None, 'near-output comparison requires a policy')
    canonical(left)
    canonical(right)
    limit = Fraction(policy['numeric_abs_tolerance'])
    maximum = Fraction(0)
    stack = [(left, right)]
    while stack:
        a, b = stack.pop()
        if type(a) is not type(b):
            return None
        if type(a) is dict:
            if a.keys() != b.keys():
                return None
            stack.extend((a[key], b[key]) for key in sorted(a))
        elif type(a) is list:
            if len(a) != len(b):
                return None
            stack.extend(zip(a, b))
        elif type(a) in (int, float):
            delta = abs(Fraction(a) - Fraction(b))
            if delta > limit:
                return None
            maximum = max(maximum, delta)
        elif a != b:
            return None
    return float(maximum)


def assessment(measurement, policy, match=None):
    if measurement is None:
        return None
    return dict(schema=ASSESSMENT_SCHEMA, policy=deepcopy(policy),
        measurement_sha256=measurement['sha256'], score=measurement['score'],
        match=deepcopy(match))


def validate_assessment(value, *, identity, status):
    if value is None:
        require(status != 'non_improving_near_repeat', 'near repeat requires measured assessment')
        return None
    require(type(value) is dict and set(value) == {'schema', 'policy',
        'measurement_sha256', 'score', 'match'} and value['schema'] == ASSESSMENT_SCHEMA,
        'invalid near-output assessment fields/schema')
    policy = validate_policy(value['policy'])
    require(identity is not None and policy is not None
            and policy_matches_identity(policy, identity), 'near-output assessment scope differs')
    require_hash(value['measurement_sha256'], 'near-output measurement')
    require(finite_number(value['score']), 'near-output assessment score must be finite')
    match = value['match']
    if status == 'non_improving_near_repeat':
        require(type(match) is dict and set(match) == {'measurement_sha256', 'score', 'max_abs_delta'},
                'invalid near-output matched measurement')
        require_hash(match['measurement_sha256'], 'matched near-output measurement')
        require(finite_number(match['score']) and match['score'] >= value['score'],
                'near-output match does not dominate candidate score')
        require(finite_number(match['max_abs_delta'])
                and 0 <= match['max_abs_delta'] <= policy['numeric_abs_tolerance'],
                'near-output distance exceeds declared resolution')
    else:
        require(match is None, 'non-near verdict cannot claim a nearby match')
    return deepcopy(value)
