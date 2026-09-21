"""Deterministic controller-score comparisons, independent of model prose.

No physical validity, causal effect or future quality is inferred. Incomplete
standalone/legacy records produce explicit unavailable facts rather than zeros.
Input records are never normalized or changed in place.
"""
from __future__ import annotations

import math
import re


SCHEMA = 'sciel-observed-comparison/1'
_HASH = re.compile(r'[0-9a-f]{64}')


def _number(value):
    if type(value) not in (int, float):
        return None
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


def _interval_sign(low, high):
    if low > 0:
        return 'positive'
    if high < 0:
        return 'negative'
    if low == high == 0:
        return 'zero'
    if low == 0:
        return 'nonnegative'
    if high == 0:
        return 'nonpositive'
    return 'spans_zero'


def observed_comparison(record) -> dict:
    """Return fresh finite JSON facts about a maximized controller score.

    Measurement requires status=evaluated and the *stored* exact subtraction.
    A missing/None evaluation is allowed for legacy records, not fabricated as
    ok=true. If an evaluation object is present, its ok and score must agree.
    Cases, metric names, verdict prose and validity sentinels never substitute
    for these controller fields. All numeric values are retained without
    rounding when finite; absent/invalid values are represented as None.

    Forecast position additionally requires an ordered finite interval, explicit
    source_repaired=False, and equal known SHA256 bindings. Its sign describes
    the interval independently of measurement/comparability. The supplied
    verdict is preserved separately, never rewritten from the computed position.
    """
    valid_record = type(record) is dict
    row = record if valid_record else {}
    numbers = {key: _number(row.get(key)) for key in ('parent_score', 'score', 'actual_delta')}
    measurement = {'eligible': False, 'reason': None, **numbers, 'direction': 'unmeasured'}
    if not valid_record:
        reason = 'record_not_object'
    elif row.get('status') != 'evaluated':
        reason = 'status_not_evaluated'
    else:
        reason = next((('missing_' if row.get(key) is None else 'invalid_') + key
                       for key, value in numbers.items() if value is None), None)
        if reason is None:
            difference = numbers['score'] - numbers['parent_score']
            if _number(difference) is None:
                reason = 'nonfinite_score_difference'
            elif difference != numbers['actual_delta']:
                reason = 'inconsistent_actual_delta'
        evaluation = row.get('evaluation')
        if reason is None and evaluation is not None:
            if type(evaluation) is not dict:
                reason = 'invalid_evaluation'
            elif evaluation.get('ok') is not True:
                reason = 'evaluation_not_ok'
            elif _number(evaluation.get('score')) is None:
                reason = 'missing_or_invalid_evaluation_score'
            elif evaluation['score'] != numbers['score']:
                reason = 'evaluation_score_mismatch'
    measurement['reason'] = reason
    if reason is None:
        delta = numbers['actual_delta']
        measurement.update(eligible=True, direction='improved' if delta > 0 else 'worsened' if delta < 0 else 'unchanged')

    hypothesis = row.get('hypothesis')
    h = hypothesis if type(hypothesis) is dict else {}
    low, high = _number(h.get('predicted_low')), _number(h.get('predicted_high'))
    verdict = row.get('forecast_verdict')
    forecast = {'eligible': False, 'reason': None, 'predicted_low': low, 'predicted_high': high,
        'interval_sign': 'unavailable', 'position': 'unavailable',
        'supplied_verdict': verdict if type(verdict) is str else None,
        'supplied_verdict_reason': None if type(verdict) is str else 'missing' if verdict is None else 'invalid_type'}
    if hypothesis is None:
        reason = 'missing_hypothesis'
    elif type(hypothesis) is not dict:
        reason = 'invalid_hypothesis'
    elif low is None or high is None:
        reason = 'missing_or_invalid_interval'
    elif low > high:
        reason = 'reversed_interval'
    else:
        forecast['interval_sign'] = _interval_sign(low, high)
        proposed, source = row.get('proposed_sha256'), row.get('source_sha256')
        bound = (type(proposed) is str and _HASH.fullmatch(proposed) is not None
                 and type(source) is str and _HASH.fullmatch(source) is not None)
        if not measurement['eligible']:
            reason = 'unmeasured_outcome'
        elif row.get('source_repaired') is True:
            reason = 'source_repaired'
        elif bound and proposed != source:
            reason = 'source_mismatch'
        elif row.get('source_repaired') is not False or not bound:
            reason = 'unknown_source_binding'
        else:
            reason = None
            delta = numbers['actual_delta']
            forecast.update(eligible=True, position='below' if delta < low else 'above' if delta > high else 'inside')
    forecast['reason'] = reason
    return {'schema': SCHEMA, 'scope': 'maximize_controller_score_only',
            'measurement': measurement, 'forecast': forecast}
