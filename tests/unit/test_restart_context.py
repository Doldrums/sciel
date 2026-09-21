"""Pure saved-array fixtures; no candidate, helper, optimizer or provider runs."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from sciel.hypothesis.restart_trace import RestartTrace
from sciel.knowledge_base import restart_context as context
from sciel.knowledge_base import paired_comparisons
from test_paired_comparisons import inputs


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def traced_group(*, traces='first', mutate=None):
    data = inputs()
    adapter = {'schema': 'sciel-repeat-adapter/1', 'runtime': {'fixture': 'saved numbers; no runtime'}}
    adapter['identity_sha256'] = digest(adapter)
    data['adapter_identity'] = adapter
    for index, pair in enumerate(data['pairs']):
        for role in ('comparator', 'challenger'):
            receipt = pair[role]
            receipt['adapter_identity'] = deepcopy(adapter)
            if traces == 'none' or (traces == 'first' and (index, role) != (0, 'comparator')):
                continue
            identity = dict(source_sha256=receipt['source_sha256'], seed=pair['seed'],
                runtime_sha256=digest(adapter['runtime']), scaffold_sha256='d'*64, invocation_id='e'*32)
            collector = RestartTrace(identity)
            best = np.array([v for row in receipt['capture']['centers'] for v in row] + receipt['capture']['radii'])
            saved = dict(best=best.copy(), initial=best.copy(), noise=np.zeros(78), ordinary=best.copy())
            score = receipt['raw_sum']
            collector.begin_start(0, incumbent_sum=score, improvements=0)
            collector.begin_helper(saved)
            collector.helper_end(raw_kind='none', raw_type='NoneType', post_inputs=saved,
                effective=saved['ordinary'], origin='saved_ordinary_fallback', incumbent_sum=score, improvements=0)
            collector.optimizer_enter(incumbent_sum=score, improvements=0)
            collector.optimizer_exit(incumbent_sum=score, improvements=0, result=best.copy())
            collector.end_start('completed', incumbent_sum=score, improvements=0)
            trace = collector.finish('restart_limit', incumbent_sum=score, improvements=0)
            envelope = dict(schema='sciel-repeat-restart-trace/1', status='captured', reason=None, trace=trace,
                **{k: identity[k] for k in ('source_sha256', 'seed', 'runtime_sha256', 'invocation_id')},
                adapter_identity_sha256=adapter['identity_sha256'])
            receipt['restart_trace'] = dict(status='captured', reason=None, trace=trace, sidecar_sha256=digest(envelope))
    if mutate:
        mutate(data)
    return paired_comparisons.build_comparison(**data)


def test_legacy_no_trace_is_absent_and_does_not_call_codec(monkeypatch):
    group = traced_group(traces='none')
    monkeypatch.setattr(context.motion_observations, 'build_observation', lambda *a, **k: pytest.fail('no diagnostic request'))
    assert context.build_context([group]) is None
    assert context.build_context([]) is None


def test_whole_views_exact_external_bindings_detached_without_quality_changes():
    group = traced_group()
    before = deepcopy(group)
    wire = context.build_context([group], max_chars=64000)
    result = context.expand_context(wire)
    assert result['selected_count'] == result['presented_count'] == 6
    assert result['omitted_count'] == 0
    assert result['groups'][0]['group_sha256'] == digest(group)
    assert result['cards'][0]['observation_view']['observation']['status'] == 'available'
    assert [c['observation_view']['observation']['reason'] for c in result['cards'][1:]] == ['trace_absent'] * 5
    for row in result['cards']:
        receipt = next(p for p in group['pairs'] if p['seed'] == row['seed'])[row['role']]
        assert row['source_sha256'] == receipt['source_sha256']
        assert row['receipt_sha256'] == digest(receipt)
    assert 'data_base64' not in json.dumps(result)
    assert context.validate_context(wire) == wire
    assert context._pack(result) == wire
    detached = context.expand_context(wire)
    detached['cards'][0]['observation_view']['observation']['reason'] = 'tamper copy'
    assert result['cards'][0]['observation_view']['observation']['reason'] is None
    assert group == before


def test_missing_unknown_and_bad_diagnostic_are_distinct_from_numeric_quality():
    def alter(data):
        data['pairs'][0]['challenger'] = None
        data['pairs'][1]['comparator'] = deepcopy(paired_comparisons.UNKNOWN_RECEIPT)
        receipt = data['pairs'][1]['challenger']
        receipt['restart_trace']['trace']['identity']['source_sha256'] = 'f' * 64
        trace = receipt['restart_trace']['trace']
        trace['trace_sha256'] = digest({k: v for k, v in trace.items() if k != 'trace_sha256'})
    group = traced_group(traces='all', mutate=alter)
    before = deepcopy(group)
    result = context.expand_context(context.build_context([group], max_chars=64000))
    unavailable = {(row['seed'], row['role']): row for row in result['cards'] if row['observation_view'] is None}
    assert unavailable[(11, 'challenger')]['unavailable_reason'] == 'receipt_not_dispatched'
    assert unavailable[(12, 'comparator')]['unavailable_reason'] == 'dispatch_unknown'
    assert unavailable[(12, 'challenger')]['unavailable_reason'] == 'invalid_diagnostic'
    assert unavailable[(12, 'challenger')]['receipt_status'] == 'valid'
    assert group['pairs'][1]['challenger']['capture']['valid'] is True
    assert group == before and paired_comparisons.validate_comparison(group) == before


def test_item_limit_keeps_every_receipt_omission_without_extra_codec_calls(monkeypatch):
    group = traced_group()
    calls = []
    original = context.motion_observations.build_observation
    def observed(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(context.motion_observations, 'build_observation', observed)
    result = context.expand_context(context.build_context([group], max_items=1, max_chars=64000))
    assert len(calls) == result['presented_count'] == 1
    assert result['omitted_count'] == 5
    assert all(row['reason'] == 'item_limit' for row in result['omissions'])
    assert {(r['role'], r['seed']) for r in result['cards'] + result['omissions']} == {
        (role, seed) for role in ('comparator', 'challenger') for seed in (11, 12, 13)}


def test_zero_and_tiny_budget_do_not_displace_existing_facts():
    group = traced_group()
    assert context.build_context([group], max_chars=0) is None
    assert context.build_context([group], max_chars=1) is None
    only_metadata = context.build_context([group], max_items=0, max_chars=64000)
    assert only_metadata['cards'] == [] and only_metadata['omitted_count'] == 6


def test_exact_context_boundary_never_partially_clips_a_view():
    group = traced_group()
    original = context.build_context([group], max_chars=64000)
    exact = context.build_context([group], max_chars=original['rendered_chars'])
    # Account for the decimal width of the stored limit itself.
    for _ in range(3):
        if exact is not None and exact['max_chars'] != exact['rendered_chars']:
            exact = context.build_context([group], max_chars=exact['rendered_chars'])
    assert exact is not None and exact['presented_count'] == 6
    assert exact['rendered_chars'] == exact['max_chars'] == len(context._wire(exact))
    smaller = context.build_context([group], max_chars=exact['rendered_chars'] - 1)
    assert smaller is None or (smaller['presented_count'] < 6 and smaller['rendered_chars'] <= smaller['max_chars'])
    if smaller:
        expanded = context.expand_context(smaller)
        assert any(row['reason'] == 'memory_char_limit' for row in expanded['omissions'])
        assert all(row['observation_view'] is None or row['observation_view']['included'] for row in expanded['cards'])


def test_oversize_whole_view_is_explicitly_omitted(monkeypatch):
    group = traced_group()
    original = context.observations.observation_view
    monkeypatch.setattr(context.observations, 'observation_view', lambda value, **kwargs: original(value, max_chars=700))
    result = context.expand_context(context.build_context([group], max_chars=64000))
    assert result['presented_count'] == 0 and result['omitted_count'] == 6
    assert all(row['reason'] == 'card_char_limit' for row in result['omissions'])


@pytest.mark.parametrize('mutation', ['source', 'receipt', 'coverage', 'count', 'unavailable', 'ref', 'bool_ref'])
def test_rehashed_context_cannot_break_declared_joins(mutation):
    value = context.build_context([traced_group()], max_chars=64000)
    if mutation == 'source':
        value['groups'][0]['comparator_source_sha256'] = 'f' * 64
    elif mutation == 'receipt':
        value['receipt_refs'][0][3] = 'f' * 64
    elif mutation == 'coverage':
        value['cards'].pop()
        value['presented_count'] -= 1
        value['selected_count'] -= 1
    elif mutation == 'count':
        value['presented_count'] += 1
    elif mutation == 'ref':
        value['cards'][0]['receipt_ref'] = 99
    elif mutation == 'bool_ref':
        value['cards'][0]['receipt_ref'] = False
    else:
        value['cards'][0].update(observation_refs=None, view_sha256=None, unavailable_reason='receipt_not_dispatched')
    with pytest.raises(ValueError):
        context.validate_context(context._seal(value))


def test_duplicate_group_identity_is_rejected():
    group = traced_group()
    with pytest.raises(ValueError, match='duplicate group'):
        context.build_context([group, deepcopy(group)])


@pytest.mark.parametrize('kwargs', [{'max_chars': True}, {'max_items': False}, {'max_chars': -1}, {'max_items': 385}])
def test_strict_budget_arguments(kwargs):
    with pytest.raises(ValueError):
        context.build_context([], **kwargs)
