"""Saved-number/trace targets only; no candidate, helper, model or solver runs."""
import copy
from fractions import Fraction
import math

import pytest

from sciel.hypothesis import comparison_targets as targets
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.paired_comparisons import build_comparison, UNKNOWN_RECEIPT
from tests.unit.test_paired_comparisons import inputs, receipt, SOURCE_A, SOURCE_B
from tests.unit.test_restart_observations import synthetic_adapter, synthetic_receipt, seal_trace


def target(*diagnostics, base=SOURCE_A, comparator=SOURCE_A):
    return dict(schema=targets.SCHEMA, base_source_sha256=base,
        comparator_source_sha256=comparator, change='Change one declared reinsertion decision.',
        quality_gate=targets.QUALITY_GATE,
        diagnostics=[dict(observable=name, direction=direction) for name, direction in diagnostics])


def evaluate(value=None, data=None):
    return targets.evaluate_target(target() if value is None else value,
        build_comparison(**(inputs() if data is None else data)), candidate_source_sha256=SOURCE_B)


def timing_data(old=(1., 1., 1.), new=(.5, .5, .5)):
    data = inputs()
    for pair, a, b in zip(data['pairs'], old, new):
        pair['comparator']['candidate_elapsed_seconds'] = a
        pair['challenger']['candidate_elapsed_seconds'] = b
    return data


def traced_data(*, comparator_calls=2, candidate_calls=1):
    data = inputs()
    adapter = synthetic_adapter()
    data['adapter_identity'] = adapter
    for pair in data['pairs']:
        for role, source, count, radius in (
                ('comparator', SOURCE_A, comparator_calls, .1),
                ('challenger', SOURCE_B, candidate_calls, .11)):
            pair[role] = synthetic_receipt(source, pair['seed'], adapter_identity=adapter,
                scheduled_calls=count, raw_sum=radius)
    return data


def test_target_allows_explicit_equal_or_distinct_base_and_comparator_and_detaches():
    value = target(('candidate_elapsed_seconds', 'decrease'), ('restart_total_starts', 'increase'))
    original = copy.deepcopy(value)
    checked = targets.validate_target(value, sources={SOURCE_A, SOURCE_B})
    checked['diagnostics'][0]['direction'] = 'increase'
    assert value == original
    assert targets.validate_target(target(base=SOURCE_B), sources=[SOURCE_A, SOURCE_B])['base_source_sha256'] == SOURCE_B
    assert targets.validate_target(target(), sources=(SOURCE_A,)) == target()
    assert targets.validate_target(target(), sources=frozenset({SOURCE_A})) == target()


@pytest.mark.parametrize('mutate', [
    lambda v: v.update(schema='sciel-comparison-target/0'),
    lambda v: v.update(schema=True),
    lambda v: v.pop('diagnostics'),
    lambda v: v.update(epsilon=1e-9),
    lambda v: v.update(quality_gate='exact_only'),
    lambda v: v.update(quality_gate=True),
    lambda v: v.update(base_source_sha256='A' * 64),
    lambda v: v.update(comparator_source_sha256='a' * 63),
    lambda v: v.update(base_source_sha256=None),
    lambda v: v.update(change=' \n\t'),
    lambda v: v.update(change='x' * 1025),
    lambda v: v.update(change=True),
    lambda v: v.update(change=float('nan')),
    lambda v: v.update(diagnostics={}),
    lambda v: v.update(diagnostics=None),
    lambda v: v.update(diagnostics=[{'observable': 'unknown_cost', 'direction': 'decrease'}]),
    lambda v: v.update(diagnostics=[{'observable': True, 'direction': 'decrease'}]),
    lambda v: v.update(diagnostics=[{'observable': 'candidate_elapsed_seconds', 'direction': 'near_zero'}]),
    lambda v: v.update(diagnostics=[{'observable': 'candidate_elapsed_seconds', 'direction': True}]),
    lambda v: v.update(diagnostics=[{'observable': 'candidate_elapsed_seconds', 'direction': 'decrease', 'epsilon': 0}]),
    lambda v: v.update(diagnostics=[{'observable': 'candidate_elapsed_seconds'}]),
    lambda v: v.update(diagnostics=[{'observable': 'candidate_elapsed_seconds', 'direction': 'decrease'}] * 5),
])
def test_malformed_or_weakened_target_is_rejected(mutate):
    value = target()
    mutate(value)
    with pytest.raises(ValueError, match='comparison target:'):
        targets.validate_target(value)


@pytest.mark.parametrize('second', ['decrease', 'increase'])
def test_same_observable_cannot_be_repeated_or_assigned_conflicting_directions(second):
    with pytest.raises(ValueError, match='repeated or conflicting'):
        targets.validate_target(target(('candidate_elapsed_seconds', 'decrease'),
                                      ('candidate_elapsed_seconds', second)))


@pytest.mark.parametrize('sources', [set(), {SOURCE_B}, [SOURCE_B], {'a' * 63}, {True},
                                     [float('nan')], SOURCE_A, {SOURCE_A: True}])
def test_source_allowlist_is_not_inferred_or_bypassed_when_empty(sources):
    with pytest.raises(ValueError):
        targets.validate_target(target(), sources=sources)


def test_schema_and_instruction_are_detached_and_cannot_offer_an_exact_only_gate():
    wire = targets.json_schema()
    assert wire['additionalProperties'] is False
    assert set(wire['required']) == set(target()) == set(wire['properties'])
    assert wire['properties']['quality_gate']['enum'] == [targets.QUALITY_GATE]
    assert 'At most 4' in wire['properties']['diagnostics']['description']
    assert wire['properties']['diagnostics']['items']['properties']['observable']['enum'] == [
        'candidate_elapsed_seconds', 'restart_total_starts']
    wire['properties']['quality_gate']['enum'].append('weakened')
    assert targets.json_schema()['properties']['quality_gate']['enum'] == [targets.QUALITY_GATE]
    text = targets.instruction()
    for phrase in ('all three', 'raw AND exact', 'No epsilon', 'Unknown is not a tie',
                   'diagnostics=[]', 'causal proof', 'base_source_sha256', 'comparator_source_sha256'):
        assert phrase in text
    pending = [wire]
    while pending:
        node = pending.pop()
        if type(node) is dict:
            assert not {'pattern', 'minLength', 'maxLength', 'maxItems', 'uniqueItems'} & set(node)
            pending.extend(node.values())
        elif type(node) is list:
            pending.extend(node)


def test_quality_only_confirmation_does_not_claim_mechanism_execution_or_valid_source_edit():
    value, data = target(base='c' * 64), inputs()
    group = build_comparison(**data)
    original = copy.deepcopy((value, group))
    result = targets.evaluate_target(value, group, candidate_source_sha256=SOURCE_B)
    assert result['quality']['verdict'] == result['combined_verdict'] == 'CONFIRMED'
    assert result['quality']['all_pairs_no_loss'] is True
    assert result['diagnostics'] == [] and result['diagnostics_requested'] is False
    assert result['source_contrast'] == dict(base_source_sha256='c' * 64,
        comparator_source_sha256=SOURCE_A, candidate_source_sha256=SOURCE_B)
    assert result['target_sha256'] == canonical_sha(value)
    assert result['comparison_sha256'] == canonical_sha(group)
    for flag in ('causal_proof', 'execution_verified', 'preregistration_verified', 'source_edit_validity_verified'):
        assert result[flag] is False
    result['quality']['pairs'][0]['certified_delta']['numerator'] = 0
    assert (value, group) == original


@pytest.mark.parametrize('mode', ['tie', 'raw_gain_only', 'exact_gain_only'])
def test_complete_no_loss_requires_a_strict_gain_in_each_metric(mode):
    data = inputs(old=(.1, .1, .1), new=(.1, .1, .1))
    for pair in data['pairs']:
        if mode == 'raw_gain_only':
            pair['challenger'] = receipt(SOURCE_B, pair['seed'], .10000001, x=.1)
        elif mode == 'exact_gain_only':
            pair['comparator'] = receipt(SOURCE_A, pair['seed'], .10000001, x=.1)
            pair['challenger'] = receipt(SOURCE_B, pair['seed'], .10000001)
    result = evaluate(data=data)
    assert result['quality']['all_pairs_no_loss'] is True
    assert result['quality']['verdict'] == result['combined_verdict'] == 'REFUTED'
    assert result['quality']['reason'] == 'no_strict_gain_in_each_metric'


def test_strict_gains_may_be_on_different_shared_seeds():
    data = inputs(old=(.1, .1, .1), new=(.1, .1, .1))
    data['pairs'][0]['challenger'] = receipt(SOURCE_B, 11, .10000001, x=.1)
    data['pairs'][1]['comparator'] = receipt(SOURCE_A, 12, .10000001, x=.1)
    data['pairs'][1]['challenger'] = receipt(SOURCE_B, 12, .10000001)
    result = evaluate(data=data)
    a, b, c = result['quality']['pairs']
    assert a['raw_delta'] > 0 and Fraction(**a['certified_delta']) == 0
    assert b['raw_delta'] == 0 and Fraction(**b['certified_delta']) > 0
    assert c['raw_delta'] == Fraction(**c['certified_delta']) == 0
    assert result['combined_verdict'] == 'CONFIRMED'


@pytest.mark.parametrize('mode', ['mixed', 'sub_epsilon_loss', 'raw_exact_disagreement'])
def test_a_known_loss_refutes_even_if_the_mean_or_exact_metric_improves(mode):
    data = inputs(new=(.11, .099, .11))
    if mode == 'sub_epsilon_loss':
        data = inputs(new=(.11, math.nextafter(.1, 0.), .11))
    elif mode == 'raw_exact_disagreement':
        for pair in data['pairs']:
            pair['comparator'] = receipt(SOURCE_A, pair['seed'], .10000005, x=.1)
            pair['challenger'] = receipt(SOURCE_B, pair['seed'], .10000002)
    result = evaluate(data=data)
    assert result['quality']['verdict'] == 'REFUTED'
    assert result['quality']['reason'] == 'known_paired_loss'


@pytest.mark.parametrize('missing', [None, UNKNOWN_RECEIPT, 'candidate_failure', 'infrastructure'])
@pytest.mark.parametrize('known_loss', [False, True])
def test_unknown_quality_is_inconclusive_unless_another_known_pair_loses(missing, known_loss):
    data = inputs(new=(.09 if known_loss else .11, .11, .11))
    data['pairs'][1]['challenger'] = (receipt(SOURCE_B, 12, status=missing)
        if type(missing) is str else copy.deepcopy(missing))
    result = evaluate(data=data)
    assert result['quality']['verdict'] == ('REFUTED' if known_loss else 'INCONCLUSIVE')
    assert result['quality']['pairs'][1]['raw_delta'] is None
    assert result['quality']['pairs'][1]['certified_delta'] is None
    assert result['quality']['pairs'][1]['no_loss'] is None
    assert result['quality']['all_pairs_valid'] is False


@pytest.mark.parametrize('damage', ['candidate', 'comparator', 'derived', 'bool_seed', 'nan_time', 'bool_time'])
def test_group_or_source_binding_cannot_be_substituted(damage):
    value, group, candidate = target(), build_comparison(**inputs()), SOURCE_B
    if damage == 'candidate': candidate = 'c' * 64
    elif damage == 'comparator': value['comparator_source_sha256'] = 'c' * 64
    elif damage == 'derived': group['derived']['pairs'][0]['raw_delta'] = 9
    elif damage == 'bool_seed': group['pairs'][0]['seed'] = True
    elif damage == 'nan_time': group['pairs'][0]['challenger']['candidate_elapsed_seconds'] = float('nan')
    else: group['pairs'][0]['challenger']['candidate_elapsed_seconds'] = True
    with pytest.raises(ValueError):
        targets.evaluate_target(value, group, candidate_source_sha256=candidate)


@pytest.mark.parametrize('direction,old,new', [
    ('decrease', (1., 1., 1.), (.5, .5, .5)),
    ('increase', (.5, .5, .5), (1., 1., 1.))])
def test_elapsed_diagnostic_requires_every_pair_to_move_strictly(direction, old, new):
    result = evaluate(target(('candidate_elapsed_seconds', direction)), timing_data(old, new))
    diagnostic, = result['diagnostics']
    assert diagnostic['verdict'] == result['quality']['verdict'] == result['combined_verdict'] == 'CONFIRMED'
    assert all(row['verdict'] == 'CONFIRMED' for row in diagnostic['pairs'])
    assert result['causal_proof'] is False


@pytest.mark.parametrize('new,expected', [((.5, 1., .5), 'REFUTED'),
    ((.5, 2., .5), 'REFUTED'), ((.5, None, .5), 'INCONCLUSIVE'),
    ((.5, None, 1.), 'REFUTED')])
def test_timing_ties_opposite_directions_and_unknowns_do_not_confirm(new, expected):
    result = evaluate(target(('candidate_elapsed_seconds', 'decrease')), timing_data(new=new))
    assert result['quality']['verdict'] == 'CONFIRMED'
    assert result['diagnostics'][0]['verdict'] == result['combined_verdict'] == expected


def test_elapsed_diagnostic_does_not_substitute_total_wall_or_certification_time():
    data = timing_data(new=(None, None, None))
    for pair in data['pairs']:
        pair['challenger']['elapsed_seconds'] = .01
        pair['challenger']['certification_elapsed_seconds'] = .001
    result = evaluate(target(('candidate_elapsed_seconds', 'decrease')), data)
    assert result['diagnostics'][0]['verdict'] == 'INCONCLUSIVE'
    assert all(row['candidate']['unavailable_reason'] == 'candidate_elapsed_seconds_missing'
               for row in result['diagnostics'][0]['pairs'])


def test_validated_restart_total_uses_full_run_count_beyond_retained_call_prefix():
    data = traced_data(comparator_calls=17, candidate_calls=1)
    result = evaluate(target(('restart_total_starts', 'decrease')), data)
    diagnostic, = result['diagnostics']
    assert diagnostic['verdict'] == result['combined_verdict'] == 'CONFIRMED'
    for row in diagnostic['pairs']:
        assert row['comparator']['value'] == 65 and row['candidate']['value'] == 1
        assert row['delta'] == -64
        assert row['candidate']['observation_sha256'] is not None
        assert row['candidate']['trace_sha256'] is not None
    trace = data['pairs'][0]['comparator']['restart_trace']['trace']
    assert len(trace['calls']) == 16 < 65


@pytest.mark.parametrize('damage', ['absent', 'unsupported', 'source', 'seed', 'runtime',
                                     'bool_starts', 'false_digest', 'quality_join'])
def test_unbound_or_unavailable_trace_is_unknown_not_a_zero_start_count(damage):
    data = traced_data()
    receipt_value = data['pairs'][1]['challenger']
    trace = receipt_value['restart_trace']['trace']
    if damage == 'absent':
        del receipt_value['restart_trace']
    elif damage == 'false_digest':
        trace['trace_sha256'] = 'c' * 64
    else:
        if damage == 'unsupported': trace['schema'] = 'unsupported-trace/99'
        elif damage == 'source': trace['identity']['source_sha256'] = 'c' * 64
        elif damage == 'seed': trace['identity']['seed'] = 13
        elif damage == 'runtime': trace['identity']['runtime_sha256'] = 'c' * 64
        elif damage == 'bool_starts': trace['total_starts'] = True
        else:
            # This is a valid bound trace of different final geometry. It must
            # not establish telemetry for the receipt's saved quality result.
            trace = synthetic_receipt(SOURCE_B, 12, adapter_identity=data['adapter_identity'],
                                      raw_sum=.2)['restart_trace']['trace']
        receipt_value['restart_trace']['trace'] = seal_trace(trace)
    result = evaluate(target(('restart_total_starts', 'decrease')), data)
    diagnostic, = result['diagnostics']
    assert result['quality']['verdict'] == 'CONFIRMED'
    assert diagnostic['verdict'] == result['combined_verdict'] == 'INCONCLUSIVE'
    assert diagnostic['pairs'][1]['candidate']['value'] is None
    assert diagnostic['pairs'][1]['candidate']['unavailable_reason'] is not None


def test_diagnostic_success_cannot_rescue_quality_loss_and_one_failure_refutes_combination():
    data = traced_data()
    for pair in data['pairs']:
        pair['comparator']['candidate_elapsed_seconds'] = 1.
        pair['challenger']['candidate_elapsed_seconds'] = .5
    value = target(('candidate_elapsed_seconds', 'decrease'), ('restart_total_starts', 'increase'))
    result = evaluate(value, data)
    assert result['quality']['verdict'] == 'CONFIRMED'
    assert [d['verdict'] for d in result['diagnostics']] == ['CONFIRMED', 'REFUTED']
    assert result['combined_verdict'] == 'REFUTED'
    failed_quality = timing_data()
    failed_quality['pairs'][1]['challenger'] = receipt(SOURCE_B, 12, .09)
    failed_quality['pairs'][1]['challenger']['candidate_elapsed_seconds'] = .5
    result = evaluate(target(('candidate_elapsed_seconds', 'decrease')), failed_quality)
    assert result['quality']['verdict'] == 'REFUTED'
    assert result['diagnostics'][0]['verdict'] == 'CONFIRMED'
    assert result['combined_verdict'] == 'REFUTED'
