"""Bounded model forecasts of retained restart observations, not quality gates.

The caller binds the original hypothesis/source and durable receipt before using
this pure projection. Shape validation and hashes provide transport consistency,
not authentication, preregistration, model use or additional lesson evidence.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json

from . import restart_motion_observations as observations

SCHEMA = 'sciel-restart-predictions/1'
RESULT_SCHEMA = 'sciel-restart-prediction-result/1'
EPSILON = 1e-6
NAMES = ('effective_locations_changed', 'result_changed', 'accepted_interval_gain')
COLUMNS = ['ordinal', 'restart_index', 'helper_accepted', *NAMES]
POLICY = {
    'scope': 'Each planned challenger draw separately; existential over accepted calls within '
             'the fixed retained scheduled-call prefix (at most the first 16 scheduled calls).',
    'effective_locations_changed': 'Exact numeric best/effective center-pair multiset differs; no operator formula claim.',
    'result_changed': 'Captured result is tolerance-valid and has a unique nearest-center bijection to best; '
                      'matched center OR radius maximum is strictly above 1e-6.',
    'accepted_interval_gain': 'Terminal incumbent sum minus optimizer-entered incumbent sum is strictly positive; '
                              'exact represented binary64 difference, no causal attribution.',
    'unknown': 'A true witness wins; otherwise any relevant unavailable or ambiguous input is unknown; '
               'all known false is false; no accepted retained calls is unknown. Omitted calls are outside scope.',
    'quality': 'Forecast agreement is separate from native score forecast, packing quality and lesson admission.',
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode('ascii')).hexdigest()


def validate_spec(value):
    """Return a detached exact schema, with no implicit defaults or coercion."""
    if (type(value) is not dict or set(value) != {'schema', 'expected'}
            or value['schema'] != SCHEMA or type(value['expected']) is not dict
            or set(value['expected']) != set(NAMES)
            or any(type(value['expected'][name]) is not bool for name in NAMES)):
        raise ValueError('restart predictions: invalid specification')
    return {'schema': SCHEMA, 'expected': {name: value['expected'][name] for name in NAMES}}


def json_schema():
    """Strict provider schema for the optional hypothesis field itself."""
    return {'type': 'object', 'additionalProperties': False,
            'required': ['schema', 'expected'], 'properties': {
                'schema': {'type': 'string', 'enum': [SCHEMA]},
                'expected': {'type': 'object', 'additionalProperties': False,
                             'required': list(NAMES),
                             'properties': {name: {'type': 'boolean'} for name in NAMES}}}}


def instruction():
    return (
        'Include restart_predictions with schema="' + SCHEMA + '" and expected with exactly '
        'three booleans: effective_locations_changed, result_changed, accepted_interval_gain. '
        'Each predicts existence separately for EACH planned paired challenger draw, among accepted calls '
        'within its fixed retained scheduled-helper-call prefix (at most the first 16 scheduled calls). '
        'effective_locations_changed means the exact numeric center-location multiset differs best to effective, '
        'not that a particular formula ran. result_changed requires independently tolerance-valid result, '
        'a unique nearest-center bijection to best and matched center OR radius residual >1e-6. '
        'accepted_interval_gain means strictly positive recorded terminal-minus-optimizer-entered incumbent sum; '
        'this interval gain is not causal credit. A known true witness wins; otherwise missing/ambiguous evidence '
        'is unknown, all known false is false, and no accepted retained calls is unknown. '
        'Omitted calls are outside scope. These forecasts do not change the native score forecast or quality/lesson gates. '
        'Declare expected values before evaluation; do not report observations as forecasts.'
    )


def _exists(values):
    if any(value is True for value in values):
        return True
    if not values or any(value is None for value in values):
        return None
    return False


def _facts(case):
    if case['helper_accepted'] is None:
        return [None, None, None]
    changed = case['effective_locations_changed']
    valid = case.get('result_tolerance_valid')
    if valid is False:
        result = False
    elif valid is not True or case['match_status'] != 'matched':
        result = None
    else:
        result = case['match_center_max'] > EPSILON or case['match_radius_max'] > EPSILON
    before, after = (case['phase_sums'][i] for i in (2, 4))
    gain = (None if before is None or after is None else
            Fraction.from_float(after) - Fraction.from_float(before) > 0)
    return [changed, result, gain]


def evaluate(spec, observation):
    """Evaluate one shape-validated motion observation; execute no candidate.

    Expected source/seed/runtime/receipt bindings and preregistration are the
    caller's responsibility. Recomputing this result's digest cannot establish
    any of those external facts. Unknown acceptance remains a possible witness.
    """
    spec = validate_spec(spec)
    observation = observations.validate_shape(observation)
    cases = observations.observation_cases(observation)
    rows = [[case['ordinal'], case['restart_index'], case['helper_accepted'], *_facts(case)]
            for case in cases if case['helper_accepted'] is not False]
    observed = {name: _exists([row[3+i] for row in rows]) for i, name in enumerate(NAMES)}
    available = observation['status'] == 'available'
    prefix = observation['prefix']
    value = dict(schema=RESULT_SCHEMA, spec_sha256=_sha(spec),
        observation_sha256=observation['observation_sha256'], receipt_sha256=observation['receipt_sha256'],
        status=observation['status'], reason=observation['reason'],
        expected=spec['expected'], observed=observed,
        verdicts={name: ('INCONCLUSIVE' if observed[name] is None else
                         'CONFIRMED' if observed[name] is spec['expected'][name] else 'REFUTED')
                  for name in NAMES},
        coverage={'retained': len(cases) if available else None,
                  'accepted': sum(row[2] is True for row in rows) if available else None,
                  'unknown_acceptance': sum(row[2] is None for row in rows) if available else None,
                  'omitted': prefix['omitted_count'] if prefix is not None else None},
        columns=list(COLUMNS), cases=rows)
    value['result_sha256'] = _sha(value)
    return json.loads(_canonical(value))
