"""Receipt/proposal authentication of warnings, not model-authored validity.

All providers and executor leaves are inert fakes; no submitted code executes.
"""
from copy import deepcopy
import json

import pytest

from sciel.engine.research_cells_context import compact_context
from sciel.methods.models import content_hash
from sciel.research_tools.comparison_assessment import from_record, validate
from test_research_experimental_protocol import comparison, protocol
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result, collect
from test_sequential_research_episode import build, decision


def run_check(tmp_path, *, conditions=None, observations=None, enabled=True,
              design=None, budget=4096, background=False):
    design = comparison() if design is None else design
    rows = [dict(arm_id=arm, seed=seed, cpu_seconds=.1, best_result={'value': seed},
                 saved_state={'seed': seed, 'state': 's' * 3000},
                 conditions={'inputs': 'same'} if conditions is None else conditions)
            for arm in ('a', 'b') for seed in (11, 17)]
    observed = {'comparison_results': rows} if observations is None else observations
    tools, backend, events, costs = build(tmp_path, [result(observed)],
        experimental_protocol=enabled, shared_observations=enabled,
        max_context_chars=budget, yield_after_success=True, background=background)
    action = decision(**({'experimental_protocol':design} if enabled else {}))
    if background:
        action.update(execution='background', cpu_seconds=8, wall_seconds=20)
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current, LLM([action]), io=IO())
    tools.commit(episode)
    if background:
        collect(tools, episode['submitted_jobs'][0]['job_id'])
    card = next(iter(tools.state()['lifecycle']['cards'].values()))
    rid = card['source']['receipt_id']
    record = tools._read_artifact(rid, 'invocation')
    return tools, backend, events, costs, card, record


@pytest.mark.parametrize('conditions,shape', [(None, 'complete'), ({}, 'invalid')])
def test_warning_is_host_owned_and_fits_small_card_with_actual_rows(tmp_path, conditions, shape):
    tools, backend, _, costs, card, record = run_check(tmp_path, conditions=conditions)
    warning = card['comparison_assessment']
    assert card['schema'] == 'sciel-research-check/5'
    assert card['observations']['status'] == 'success'  # Execution is separate.
    assert warning['receipt_id'] == card['source']['receipt_id']
    assert warning['proposal_id'] == record['experiment_proposal_id']
    assert warning['status'] == 'assessed'
    assert warning['assessment']['report_shape'] == shape
    assert warning['assessment']['adherence'] == 'tool_reported_unverified'
    assert warning['assessment']['problems'] == ([] if shape == 'complete' else ['reported conditions differ'])
    assert len(tools.format_context([card])) <= 4096
    projection = json.loads(card['observations']['output_excerpt'])
    assert projection['comparison_projection']['shown_rows'] >= 2
    assert validate(warning, card['source']['receipt_id'])
    saved = json.loads(json.dumps(tools.state()))
    tools.load_state(saved)
    assert tools.state() == saved and costs == [.03] and len(backend.calls) == 1


def test_missing_rows_and_authored_fake_assessment_cannot_become_valid(tmp_path):
    tools, _, _, _, card, _ = run_check(tmp_path, observations={
        'comparison_assessment': {'report_shape':'complete', 'adherence':'verified'}, 'best_result':7})
    a = card['comparison_assessment']['assessment']
    assert a['report_shape'] == 'invalid'
    assert a['problems'] == ['comparison_results missing or not an array']
    assert a['adherence'] == 'tool_reported_unverified'
    assert '"adherence":"verified"' in card['observations']['output_excerpt']  # Honest authored data, separate host field.


@pytest.mark.parametrize('enabled,status', [(False, 'unavailable'), (True, 'not_comparison')])
def test_absent_declaration_is_distinct_from_declared_diagnostic(tmp_path, enabled, status):
    tools, _, _, _, card, _ = run_check(tmp_path, enabled=enabled, design=protocol(), observations={'gain':0})
    a = card['comparison_assessment']
    assert a['status'] == status and a['assessment'] is None
    assert (a['proposal_id'] is not None) == enabled
    tools.load_state(json.loads(json.dumps(tools.state())))


@pytest.mark.parametrize('field', ['report', 'receipt', 'proposal', 'missing', 'downgrade'])
def test_cold_restore_rejects_forged_or_removed_host_binding(tmp_path, field):
    tools, _, _, _, card, _ = run_check(tmp_path, conditions={})
    original = tools.state()
    changed = deepcopy(original)
    target = changed['lifecycle']['cards'][card['cell_id']]
    warning = target['comparison_assessment']
    if field == 'report':
        warning['assessment'].update(report_shape='complete', problems=[])
    elif field == 'receipt':
        warning['receipt_id'] = 'f' * 64
    elif field == 'proposal':
        warning['proposal_id'] = 'f' * 64
    else:
        del target['comparison_assessment']
        if field == 'downgrade':
            target['schema'] = 'sciel-research-check/3'
    with pytest.raises(ValueError, match='card|assessment'):
        tools.load_state(changed)
    assert tools.state() == original


def test_delivery_rejects_self_consistent_forged_assessment(tmp_path):
    tools, _, _, _, card, _ = run_check(tmp_path, conditions={})
    card['comparison_assessment']['assessment'].update(report_shape='complete', problems=[])
    assert validate(card['comparison_assessment'], card['source']['receipt_id'])
    with pytest.raises(ValueError, match='immutable evidence'):
        tools.lifecycle.delivery_context([card])


@pytest.mark.parametrize('source', [['bad'], 'bad', 4])
def test_malformed_source_is_explicitly_omitted_without_attribute_error(tmp_path, source):
    _, _, _, _, card, _ = run_check(tmp_path)
    card['source'] = source
    view = compact_context([card])
    assert view['cards'] == [] and view['omission_reasons'] == {'invalid_card':1}


def test_background_collection_carries_same_bound_warning_cold(tmp_path):
    tools, backend, _, costs, card, record = run_check(tmp_path, conditions={}, background=True)
    assert card['comparison_assessment']['assessment']['problems'] == ['reported conditions differ']
    assert card['comparison_assessment']['proposal_id'] == record['experiment_proposal_id']
    tools.load_state(json.loads(json.dumps(tools.state())))
    assert len(backend.calls) == 1 and costs == [.03]


def test_pure_reader_refuses_another_attempt_and_retains_absent_legacy_declaration(tmp_path):
    tools, _, _, _, card, record = run_check(tmp_path, conditions={})
    proposal = tools._read_artifact(record['experiment_proposal_id'], 'proposal')
    assert from_record(record, card['source']['receipt_id'], lambda _: proposal) == card['comparison_assessment']
    wrong = deepcopy(proposal)
    wrong['attempt'] += 1
    with pytest.raises(ValueError, match='exact invocation attempt'):
        from_record(record, card['source']['receipt_id'], lambda _: wrong)
    legacy = {k:v for k,v in record.items() if k != 'experiment_proposal_id'}
    def unread(_):
        raise AssertionError('Legacy missing declaration must not read arbitrary proposals')
    missing = from_record(legacy, card['source']['receipt_id'], unread)
    assert missing['status'] == 'unavailable' and missing['assessment'] is None
    # A saved, scoped invocation proposal without opt-in protocol stays unknown.
    undeclared = deepcopy(proposal)
    action = json.loads(undeclared['response']['text'])
    action.pop('experimental_protocol')
    undeclared['response']['text'] = json.dumps(action)
    missing = from_record(record, card['source']['receipt_id'], lambda _: undeclared)
    assert missing['status'] == 'unavailable' and missing['proposal_id'] == record['experiment_proposal_id']


def test_warm_cache_cannot_replace_exact_tool_binding(tmp_path):
    tools, _, _, _, card, _ = run_check(tmp_path, conditions={})
    tools.lifecycle.delivery_context([card])  # Authenticate/cache the real receipt.
    card['source']['tool_id'] = 'f' * 64
    with pytest.raises(ValueError, match='immutable receipt'):
        tools.lifecycle.delivery_context([card])
