"""Saved task evidence, never a candidate or numerical optimizer execution."""
import copy

import pytest

from sciel.knowledge_base import detached_optimizer_facts as facts
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.paired_comparisons import build_comparison, UNKNOWN_RECEIPT
from sciel.hypothesis import restart_optimizer_trace as codec
from detached_optimizer_fixtures import matched_group



@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Saved evidence reduction attempted candidate, transport or process execution.')
    for name in ('subprocess.Popen', 'sciel.eval.repeat_draw.draw',
                 'sciel.eval.repeat_draw.adapter_identity', 'scipy.optimize.minimize',
                 'sciel.campaign.budget.BudgetedLLM.generate'):
        monkeypatch.setattr(name, forbidden)


@pytest.fixture(scope='module')
def authored_group():
    return matched_group()


def rebuild(group):
    keys = ('namespace','batch_id','branch','origin_experiment_id','comparator_source_sha256',
            'challenger_source_sha256','adapter_identity','limits','pairs','provenance','task_metric','source_pair')
    return build_comparison(**{k:group[k] for k in keys if k in group})


def test_recorded_motion_and_same_input_control_gain_survive_without_forecast_credit(authored_group):
    value = facts.build_facts(authored_group)
    active = [c for d in value['draws'] if d['role']=='challenger' for c in d['calls'] if c['complete_accepted_return']]
    assert len(active) == 14
    assert sum(c['optimizer_result_center_hausdorff_from_best'] >= 1e-4 for c in active) == 9
    assert sum(c['incumbent_gain'] > 0 for c in active) == 1
    matched = next(r for r in value['matched_inputs'] if r['seed']==3084817015 and r['restart_index']==8)
    assert matched['inputs_byte_identical'] is True
    assert matched['comparator_gain'] == 0.002881161755933004
    assert matched['challenger_gain'] == 0.002881170166179725
    assert matched['local_gain_delta'] == 8.410246721268777e-9
    assert value['forecasts_declared'] is False
    assert all(c['optimizer_success'] is True and c['result_tolerance_valid'] is True
               and c['result_strict_valid'] is False for c in active)


def test_unmatched_input_contrast_is_not_reported_as_local_effect(authored_group):
    value = facts.build_facts(authored_group)
    later = next(r for r in value['matched_inputs'] if r['seed']==3084817015 and r['restart_index']==16)
    assert later['inputs_byte_identical'] is False
    assert later['local_gain_delta'] is None
    assert later['comparator_gain'] is None and later['challenger_gain'] is None


def test_unknown_intent_and_unissued_draw_are_distinct_and_have_no_motion(authored_group):
    group=copy.deepcopy(authored_group)
    group['pairs'][2]['challenger']=copy.deepcopy(UNKNOWN_RECEIPT)
    value=facts.build_facts(rebuild(group))
    left,right=value['draws'][-2:]
    assert left['receipt_status']=='not_dispatched' and right['receipt_status']=='unknown'
    assert left['receipt_sha256'] is None and right['receipt_sha256']==canonical_sha(UNKNOWN_RECEIPT)
    assert left['calls']==right['calls']==[]


def test_missing_optimizer_array_stays_unknown_despite_complete_success(authored_group):
    group=copy.deepcopy(authored_group)
    trace=group['pairs'][1]['challenger']['restart_trace']['trace']
    call=next(c for c in trace['calls'] if c['restart_index']==8)
    call['optimizer_result']={'ref':None,'reason':'array_unavailable'}
    used={r for c in trace['calls'] for r in codec._legacy._rt_record_refs(c)}
    trace['vectors']={k:v for k,v in trace['vectors'].items() if k in used}
    codec._seal(trace)
    value=facts.build_facts(rebuild(group))
    draw=next(d for d in value['draws'] if d['seed']==3084817015 and d['role']=='challenger')
    row=next(c for c in draw['calls'] if c['restart_index']==8)
    assert row['complete_accepted_return'] is True and row['optimizer_success'] is True
    assert row['optimizer_result_center_hausdorff_from_best'] is None
    assert row['distance_reason']=='result:array_unavailable'
    assert row['result_tolerance_valid'] is None
    assert row['incumbent_gain']>0  # Missing geometry is not rewritten as no local gain.


def test_raw_trace_source_transplant_yields_no_diagnostics_even_after_rehash(authored_group):
    group=copy.deepcopy(authored_group)
    trace=group['pairs'][1]['challenger']['restart_trace']['trace']
    trace['identity']['source_sha256']=group['comparator_source_sha256']
    codec._seal(trace)
    value=facts.build_facts(rebuild(group))
    draw=next(d for d in value['draws'] if d['seed']==3084817015 and d['role']=='challenger')
    assert draw['reason']=='diagnostic_invalid_or_unsupported'
    assert draw['calls']==[] and draw['trace_sha256'] is None
    assert draw['receipt_status']=='valid'  # Independently validated final packing remains evidence.


def test_hausdorff_is_unlabeled_and_nonfinite_arithmetic_is_unknown():
    centers=[coordinate for i in range(26) for coordinate in (i/26, i/52)]
    left=tuple(centers+[.001]*26)
    right=tuple(centers[2:]+centers[:2]+[.4]*26)
    assert facts._hausdorff(left,right)==0
    huge=tuple([1e308]*52+[0.]*26)
    assert facts._hausdorff(huge,tuple([-1e308]*52+[0.]*26)) is None


def test_compact_table_keeps_all_accepted_rows_and_unfavorable_matches(authored_group):
    full=facts.build_facts(authored_group);compact=facts.compact_facts(full)
    assert compact['facts_sha256']==full['facts_sha256']
    assert len(compact['matched_inputs'])==len(full['matched_inputs'])
    assert sum(len(d['accepted_return_calls']) for d in compact['draws'])==14
    for packed,raw in zip(compact['matched_inputs'],full['matched_inputs'],strict=True):
        assert dict(zip(compact['matched_columns'],packed))=={k:raw[k] for k in compact['matched_columns']}
