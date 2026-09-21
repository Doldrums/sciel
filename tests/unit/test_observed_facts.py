"""Score direction and interval position are independent observed facts."""
import copy
import json
import math

import pytest

from sciel.hypothesis.observed_facts import SCHEMA, observed_comparison


def record(delta=.25, low=-.5, high=.5, **changes):
    parent, score = 1., 1. + delta
    value = {'status': 'evaluated', 'parent_score': parent, 'score': score,
        'actual_delta': score - parent, 'source_sha256': 'a' * 64, 'proposed_sha256': 'a' * 64,
        'source_repaired': False, 'hypothesis': {'predicted_low': low, 'predicted_high': high},
        'forecast_verdict': 'CONFIRMED', 'evaluation': {'ok': True, 'score': score, 'cases': {}}}
    value.update(changes)
    return value


@pytest.mark.parametrize('delta,direction', [(-.5, 'worsened'), (0., 'unchanged'), (.5, 'improved')])
@pytest.mark.parametrize('position,low_shift,high_shift', [('below', .25, .5), ('inside', -.25, .25), ('above', -.5, -.25)])
def test_direction_cross_product_interval_position(delta, direction, position, low_shift, high_shift):
    facts = observed_comparison(record(delta, delta + low_shift, delta + high_shift))
    assert facts['schema'] == SCHEMA and facts['scope'] == 'maximize_controller_score_only'
    assert facts['measurement']['eligible'] and facts['measurement']['direction'] == direction
    assert facts['forecast']['eligible'] and facts['forecast']['position'] == position


@pytest.mark.parametrize('low,high,sign', [(1., 2., 'positive'), (-2., -1., 'negative'),
    (0., 0., 'zero'), (-0., 1., 'nonnegative'), (-1., -0., 'nonpositive'), (-1., 1., 'spans_zero')])
def test_interval_sign_is_separate_from_actual_direction(low, high, sign):
    facts = observed_comparison(record(.25, low, high))
    assert facts['forecast']['interval_sign'] == sign
    assert facts['measurement']['direction'] == 'improved'


@pytest.mark.parametrize('delta', [-.5, .5])
def test_closed_interval_includes_both_endpoints(delta):
    assert observed_comparison(record(delta))['forecast']['position'] == 'inside'


def test_negative_confirmed_forecast_is_not_a_positive_quality_outcome():
    facts = observed_comparison(record(-.25, -.5, -.125))
    assert facts['measurement']['direction'] == 'worsened'
    assert facts['forecast']['position'] == 'inside'
    assert facts['forecast']['interval_sign'] == 'negative'
    assert facts['forecast']['supplied_verdict'] == 'CONFIRMED'


def test_stored_verdict_is_never_replaced_by_interval_recomputation():
    facts = observed_comparison(record(.25, -.5, .5, forecast_verdict='REFUTED'))
    assert facts['forecast']['position'] == 'inside'
    assert facts['forecast']['supplied_verdict'] == 'REFUTED'
    assert observed_comparison(record(forecast_verdict='unrecognized prose'))['forecast']['supplied_verdict'] == 'unrecognized prose'


def test_portable_pilot_delta_exact_numbers():
    parent = 2.6310941739724454 / 2.635
    score = 2.6359839924231934 / 2.635
    delta = score - parent
    assert delta == 0.001855718577133958
    row = record(low=-.0005, high=.0005, parent_score=parent, score=score, actual_delta=delta,
                 evaluation={'ok': True, 'score': score}, forecast_verdict='REFUTED')
    before = copy.deepcopy(row)
    facts = observed_comparison(row)
    assert facts['measurement']['direction'] == 'improved'
    assert facts['measurement']['actual_delta'] == 0.001855718577133958
    assert (facts['forecast']['predicted_low'], facts['forecast']['predicted_high']) == (-.0005, .0005)
    assert facts['forecast']['interval_sign'] == 'spans_zero'
    assert facts['forecast']['position'] == 'above'
    assert facts['forecast']['supplied_verdict'] == 'REFUTED'
    assert row == before


@pytest.mark.parametrize('status', ['eval_failed', 'no_child', 'hypothesis_invalid', 'generation_stopped', None])
def test_failure_sentinels_never_become_measured_quality(status):
    facts = observed_comparison(record(-1., status=status))
    assert not facts['measurement']['eligible'] and facts['measurement']['direction'] == 'unmeasured'
    assert facts['measurement']['reason'] == 'status_not_evaluated'
    assert facts['forecast']['position'] == 'unavailable'
    assert facts['forecast']['interval_sign'] == 'spans_zero'


def test_invalid_geometry_zero_is_only_a_measured_controller_penalty():
    row = record(-1., evaluation={'ok': True, 'score': 0., 'cases': {'validity': 0., 'sum_radii': 0.}})
    facts = observed_comparison(row)
    assert facts['measurement']['eligible'] and facts['measurement']['direction'] == 'worsened'
    assert facts['measurement']['score'] == 0.
    assert facts['scope'] == 'maximize_controller_score_only'
    assert 'validity' not in facts and 'sum_radii' not in facts


@pytest.mark.parametrize('evaluation,reason', [({}, 'evaluation_not_ok'), (False, 'invalid_evaluation'),
    ({'ok': 1, 'score': 1.25}, 'evaluation_not_ok'), ({'ok': False, 'score': 1.25}, 'evaluation_not_ok'),
    ({'ok': True}, 'missing_or_invalid_evaluation_score'), ({'ok': True, 'score': True}, 'missing_or_invalid_evaluation_score'),
    ({'ok': True, 'score': math.nan}, 'missing_or_invalid_evaluation_score'),
    ({'ok': True, 'score': 8.}, 'evaluation_score_mismatch')])
def test_contradictory_returned_evaluation_is_unavailable(evaluation, reason):
    facts = observed_comparison(record(evaluation=evaluation))
    assert facts['measurement']['direction'] == 'unmeasured' and facts['measurement']['reason'] == reason


def test_legacy_missing_evaluation_and_unrelated_case_names_do_not_invent_units():
    row = record(); del row['evaluation']
    expected = observed_comparison(row)
    row.update(evaluation=None, parent_cases={'score': 9e9, 'combined_score': False})
    assert observed_comparison(row) == expected
    row['evaluation'] = {'ok': True, 'score': row['score'], 'cases': {'combined_score': -1e12, 'sum_radii': 987.}}
    assert observed_comparison(row) == expected


@pytest.mark.parametrize('field', ['parent_score', 'score', 'actual_delta'])
@pytest.mark.parametrize('bad', [None, True, '1', math.nan, math.inf, -math.inf, 10**500])
def test_missing_bool_nonfinite_and_overflow_numbers_are_not_zero(field, bad):
    facts = observed_comparison(record(**{field: bad}))
    assert not facts['measurement']['eligible'] and facts['measurement'][field] is None
    assert facts['measurement']['direction'] == 'unmeasured'
    json.dumps(facts, allow_nan=False)


def test_exact_inconsistency_and_subtraction_overflow_fail_conservatively():
    assert observed_comparison(record(actual_delta=.25 + 1e-15))['measurement']['reason'] == 'inconsistent_actual_delta'
    facts = observed_comparison(record(parent_score=-1e308, score=1e308, actual_delta=1e308))
    assert facts['measurement']['reason'] == 'nonfinite_score_difference'
    assert facts['measurement']['actual_delta'] == 1e308


@pytest.mark.parametrize('change,reason', [({'source_repaired': True}, 'source_repaired'),
    ({'source_sha256': 'b' * 64}, 'source_mismatch'), ({'source_repaired': None}, 'unknown_source_binding'),
    ({'source_repaired': 0}, 'unknown_source_binding'), ({'source_sha256': None}, 'unknown_source_binding'),
    ({'proposed_sha256': 'not-a-hash'}, 'unknown_source_binding')])
def test_source_incomparability_does_not_erase_observed_direction(change, reason):
    facts = observed_comparison(record(**change))
    assert facts['measurement']['eligible'] and facts['measurement']['direction'] == 'improved'
    assert not facts['forecast']['eligible'] and facts['forecast']['reason'] == reason
    assert facts['forecast']['position'] == 'unavailable' and facts['forecast']['interval_sign'] == 'spans_zero'


def test_missing_source_repair_flag_is_not_a_claim_of_matching_source():
    row = record(); del row['source_repaired']
    assert observed_comparison(row)['forecast']['reason'] == 'unknown_source_binding'


@pytest.mark.parametrize('hypothesis,reason', [(None, 'missing_hypothesis'), ([], 'invalid_hypothesis'),
    ({}, 'missing_or_invalid_interval'), ({'predicted_low': True, 'predicted_high': 1.}, 'missing_or_invalid_interval'),
    ({'predicted_low': -1., 'predicted_high': math.inf}, 'missing_or_invalid_interval'),
    ({'predicted_low': 2., 'predicted_high': 1.}, 'reversed_interval')])
def test_bad_hypothesis_interval_keeps_actual_measurement(hypothesis, reason):
    facts = observed_comparison(record(hypothesis=hypothesis))
    assert facts['measurement']['direction'] == 'improved'
    assert facts['forecast']['reason'] == reason and facts['forecast']['interval_sign'] == 'unavailable'


@pytest.mark.parametrize('value', [None, [], True, 'record', 1])
def test_nonobject_standalone_input_is_finite_unknown(value):
    facts = observed_comparison(value)
    assert facts['measurement']['reason'] == 'record_not_object'
    assert facts['forecast']['position'] == 'unavailable'
    json.dumps(facts, allow_nan=False)


def test_fresh_nested_output_and_input_nonmutation_invalid_verdict():
    row = record(forecast_verdict={'invalid': math.inf})
    before = copy.deepcopy(row)
    first, second = observed_comparison(row), observed_comparison(row)
    first['measurement']['score'] = -99
    first['forecast']['reason'] = 'changed'
    assert second == observed_comparison(row) and row == before
    assert second['forecast']['supplied_verdict'] is None
    assert second['forecast']['supplied_verdict_reason'] == 'invalid_type'
    json.dumps(second, allow_nan=False)
