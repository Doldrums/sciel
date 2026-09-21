"""Later decisions preserve genuine prior observations without a new invocation."""
from copy import deepcopy
import json

import pytest

from sciel.engine.research_cells_context import compact_context, merge_context
from sciel.research_tools.decision_projection import authenticate_projection
from sciel.research_tools.retained_observations import MAX_PRIOR_CARDS
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import controller, result
from test_sequential_research_episode import decision, finish


def later_close(tmp_path, output=None, **config):
    tools, backend, _, costs = controller(tmp_path, [output or result({'gain':0, 'seed':17})],
        max_scientific_steps=3, yield_after_success=True, shared_observations=True,
        share_between_islands=False, **config)
    first_snap = snapshot()
    first = tools.execute(prepare(tools, first_snap), first_snap, LLM([decision()]), io=IO())
    tools.commit(first)
    original = deepcopy(next(iter(tools.lifecycle._state['cards'].values())))
    second_snap = snapshot(iteration=1)
    last = tools.execute(prepare(tools, second_snap), second_snap, LLM([finish(
        conclusion='inconclusive', reason='Measured zero improvement; this does not refute the method. ' + 'detail ' * 120)]), io=IO())
    tools.commit(last)
    return tools, backend, costs, first, original, last


def test_one_visible_cell_preserves_both_and_restores_own_latest_decision(tmp_path):
    tools, backend, costs, first, original, last = later_close(tmp_path, max_visible_tools=1)
    state = deepcopy(tools.state())
    terminal = deepcopy(next(iter(tools.lifecycle._state['cards'].values())))
    for cold in (False, True):
        if cold:
            tools.load_state(deepcopy(state))
        view = tools.context_view(snapshot(1, 2))
        assert view['omitted_count'] == 0 and len(view['cards']) == 2
        measured, closed = view['cards']
        assert measured == original
        assert closed == terminal  # No unnecessary 900-character reason shortening.
        assert closed['source']['receipt_id'] is None
        own = tools.lifecycle.delivery_context([original], recipient_cell_id=last['plan']['cell_id'])
        assert own['cards'] == view['cards']  # An old same-cell measure cannot hide latest decision.
        merged = merge_context([original, terminal], view)
        assert len(merged['cards']) == 2
        assert [c['source']['receipt_id'] for c in merged['cards']].count(first['receipt_ids'][0]) == 1
        assert tools.state() == state
    assert len(backend.calls) == 1 and costs == [.03, .03]


def test_unsupported_new_cell_does_not_consume_visible_cell_limit(tmp_path):
    tools, backend, _, _ = controller(tmp_path, [result({'gain':0})], max_scientific_steps=3,
        yield_after_success=True, shared_observations=True, share_between_islands=False, max_visible_tools=1)
    first = snapshot()
    measured = tools.execute(prepare(tools, first), first, LLM([decision()]), io=IO())
    tools.commit(measured)
    other = snapshot(1, 1)
    closed = tools.execute(prepare(tools, other), other, LLM([finish(
        conclusion='inconclusive', reason='No experiment on this question.')]), io=IO())
    tools.commit(closed)
    view = tools.context_view(snapshot(2, 2))
    assert len(view['cards']) == 1
    assert view['cards'][0]['source']['receipt_id'] == measured['receipt_ids'][0]
    assert view['omission_reasons'] == {'unmeasured_interpretation':1}
    assert len(backend.calls) == 1


@pytest.mark.parametrize('budget', [4096, 6000, 8000, 10000])
def test_tight_escaped_budget_reserves_decision_and_discloses_loss(tmp_path, budget):
    from test_paired_result_projection import witness, rows
    output = result(witness()['output']['observations'])
    output['observations']['large_vector'] = ['雪 \"line\"\n' * 5000]
    tools, backend, _, first, original, last = later_close(tmp_path, output, max_context_chars=budget)
    view = tools.context_view(snapshot(1, 2))
    assert len(json.dumps(view, ensure_ascii=True, sort_keys=True, separators=(',', ':'))) <= budget
    assert len(tools.format_context(view['cards'])) <= budget
    terminal = next(c for c in view['cards'] if c['source']['receipt_id'] is None)
    proposal = tools._read_artifact(terminal['decision']['proposal_id'], 'proposal')
    authenticate_projection(terminal['decision'], proposal, terminal['decision']['proposal_id'])
    measured = [c for c in view['cards'] if c['source']['receipt_id']]
    if budget >= 8000:
        assert len(measured) == 1
    if measured:
        assert measured[0]['source'] == original['source']
        wrapper = json.loads(measured[0]['observations']['output_excerpt'])
        assert wrapper.get('omitted_fields', 0) > 0 or wrapper.get('omitted') is True
        if budget >= 8000:
            original_rows = output['observations']['comparison_results']
            shown = rows(wrapper)
            assert len(shown) >= 3  # At least one complete matched seed, never zero rows.
            for row in shown:
                assert row['values'] == {k:original_rows[row['index']][k] for k in
                    ('arm_id', 'seed', 'cpu_seconds', 'best_result', 'conditions')}
            assert wrapper['comparison_projection']['shown_rows'] + wrapper['comparison_projection']['omitted_rows'] == len(original_rows)
        else:
            assert wrapper.get('observations_state') != 'complete'
        assert measured[0]['observations']['resources'] == original['observations']['resources']
    else:
        assert view['omission_reasons']['character_limit'] >= 1
    assert len(backend.calls) == 1


@pytest.mark.parametrize('own_plan', [False, True])
def test_prior_reference_cap_and_duplicate_delivery_are_explicit(tmp_path, own_plan):
    count = MAX_PRIOR_CARDS + 2
    tools, backend, _, _ = controller(tmp_path, [result({'sequence':n}) for n in range(count)],
        max_scientific_steps=count+1, yield_after_success=True, shared_observations=True,
        max_context_chars=10000)
    ids = []
    for n in range(count):
        snap = snapshot(iteration=n)
        episode = tools.execute(prepare(tools, snap), snap, LLM([decision()]), io=IO())
        tools.commit(episode)
        ids.extend(episode['receipt_ids'])
    snap = snapshot(iteration=count)
    episode = tools.execute(prepare(tools, snap), snap, LLM([finish()]), io=IO())
    tools.commit(episode)
    current = snapshot(1, count+1)
    if own_plan:
        current.tool_episode_plan = {'cell_id':episode['plan']['cell_id']}
    view = tools.context_view(current)
    assert view['omission_reasons']['prior_evidence_limit'] == count-MAX_PRIOR_CARDS
    receipts = [c['source']['receipt_id'] for c in view['cards'] if c['source']['receipt_id']]
    assert len(receipts) == len(set(receipts)) <= MAX_PRIOR_CARDS
    assert len(receipts) + view['omitted_count'] == count  # Each prior receipt counted exactly once.
    assert set(receipts) <= set(ids[-MAX_PRIOR_CARDS:])
    assert any(c['source']['receipt_id'] is None for c in view['cards'])
    assert len(backend.calls) == count


def test_prior_failed_invocation_is_delivered_with_original_failure_and_resource(tmp_path):
    tools, backend, _, _ = controller(tmp_path, [result(status='tool_error', error='Exact saved error')],
        max_scientific_steps=3, shared_observations=True)
    current = snapshot()
    first = tools.execute(prepare(tools, current), current, LLM([decision(), finish(conclusion='inconclusive')]), io=IO())
    tools.commit(first)
    later = snapshot(iteration=1)
    close = tools.execute(prepare(tools, later), later, LLM([finish(conclusion='inconclusive')]), io=IO())
    tools.commit(close)
    tools.load_state(deepcopy(tools.state()))
    cards = tools.context(snapshot(1, 2))
    measured = next(c for c in cards if c['source']['receipt_id'])
    assert measured['source']['receipt_id'] == first['receipt_ids'][0]
    assert measured['observations']['status'] == 'tool_error'
    assert 'Exact saved error' in measured['observations']['output_excerpt']
    assert measured['observations']['resources'] == backend.receipts[0]['resources']
    assert len(backend.calls) == 1


@pytest.mark.parametrize('mutation', ['resources','target','question','receipt','artifact'])
def test_reconstructed_observation_never_accepts_forged_terminal_provenance(tmp_path, mutation):
    tools, _, _, first, original, last = later_close(tmp_path)
    terminal = deepcopy(next(iter(tools.lifecycle._state['cards'].values())))
    if mutation == 'resources': terminal['observations']['resources'] = {'cpu_seconds':123}
    elif mutation == 'target': terminal['source']['target_source_sha256'] = 'f'*64
    elif mutation == 'question': terminal['question'] = 'A forged observation'
    elif mutation == 'receipt': terminal['decision']['receipt_ids'] = ['f'*64]
    else:
        tools.context_view(snapshot(1,2))  # Warm caches do not excuse later mutation.
        path = tools.path / (first['receipt_ids'][0]+'.json')
        path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises((ValueError, FileNotFoundError)):
        tools.lifecycle.delivery_context([terminal])


def test_final_omission_metadata_and_unicode_obey_actual_envelope_limit():
    from test_research_cell_context import card
    reasons = {key:123 for key in ('invalid_card','character_limit','card_limit',
        'shared_memory_limit','research_context_limit','unmeasured_interpretation',
        'prior_evidence_limit','cell_limit')}
    first = card(text='measurement 雪 "quote"\n' * 60)
    second = card('b', text='other measurement ' * 40)
    value = {'schema':'sciel-research-cells-context/1', 'cards':[first,second]+[{}]*7,
             'omitted_count':sum(reasons.values()), 'omission_reasons':reasons}
    for budget in range(4096, 8001, 37):
        shown = compact_context(value, max_chars=budget)
        encoded = json.dumps(shown, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
        assert len(encoded) <= budget
        assert shown['omission_reasons']['invalid_card'] == 130
        assert shown['omitted_count'] == sum(shown['omission_reasons'].values())
        assert compact_context(shown,max_chars=budget) == shown
