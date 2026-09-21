"""Successful findings survive saved cards and planner budgets.

The controller uses deterministic model and executor fakes. Authored Python is
registered as data and never imported or executed; there are no provider calls.
"""
from copy import deepcopy
import json

import pytest

from sciel.methods.models import content_hash
from sciel.research_tools.controller import RESULTS_VIEW_MARKER, _research_results_prompt
from sciel.research_tools.result_projection import PROJECTION_SCHEMA
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_contract import result
from test_research_tool_result_flow import action, controller


def delivered_view(prompt):
    return json.JSONDecoder().raw_decode(prompt.split(RESULTS_VIEW_MARKER, 1)[1])[0]


def value_at(value, pointer):
    for key in pointer.split('/')[1:]:
        value = value[key.replace('~1', '/').replace('~0', '~')]
    return value


def findings(kind):
    if kind == 'diagnostics':
        return result({'measured_effect': .125, 'n_cases': 7},
                      {'log': 'd' * 34000}), {
                          '/output/observations/measured_effect': .125,
                          '/output/observations/n_cases': 7}
    if kind == 'nested':
        return result({'a_samples': [float(n) for n in range(4000)],
                       'z_summary': {'measured_effect': .125, 'n_cases': 7}},
                      {'log': 'unrelated diagnostic ' * 700}), {
                          '/output/observations/z_summary/measured_effect': .125,
                          '/output/observations/z_summary/n_cases': 7}
    return result({'样本/结论~': {'label': '雪🙂 "quoted"\n',
                             'measured_effect': -.125, 'n_cases': 0}},
                  {'log': '日志🙂 ' * 2800}), {
                      '/output/observations/样本~1结论~0/label': '雪🙂 "quoted"\n',
                      '/output/observations/样本~1结论~0/measured_effect': -.125,
                      '/output/observations/样本~1结论~0/n_cases': 0}


@pytest.mark.parametrize('planner_budget', [8000, 64000])
@pytest.mark.parametrize('kind', ['diagnostics', 'nested', 'unicode'])
def test_saved_successful_findings_survive_card_restore_and_next_planner(
        tmp_path, kind, planner_budget):
    output, expected = findings(kind)
    # These are permitted output sizes, not executor-limit failure fixtures.
    assert len(json.dumps(output, ensure_ascii=False).encode()) < 65536
    before = deepcopy(output)
    config = dict(max_context_chars=4096, max_prompt_chars=planner_budget)
    tools, backend, _, costs = controller(tmp_path, [output], **config)
    source = snapshot()
    initial_llm = LLM([action()])
    episode = tools.execute(prepare(tools, source), source, initial_llm, io=IO())
    assert len(backend.calls) == len(initial_llm.prompts) == 1
    assert costs == [.03] and episode['known_llm_cost'] == .03
    assert len(episode['context']) <= 4096
    card, = episode['cards']
    assert ('Result projections retain exact values at original JSON paths; '
            'sparse tree branches are partial. Omitted fields are unavailable, '
            'not negative observations.') in card['limitations']
    projected = json.loads(card['observations']['output_excerpt'])
    assert projected['schema'] == PROJECTION_SCHEMA
    fields = {field['path']: field['value'] for field in projected['fields']}
    for pointer, expected_value in expected.items():
        assert fields[pointer] == expected_value
    assert projected['observations_state'] == ('partial' if kind == 'nested' else 'complete')
    assert projected['omitted_fields'] > 0
    if kind == 'nested':
        assert '/output/observations/a_samples' not in fields
        assert projected['omitted_observation_fields'] == 1
    else:
        assert projected['omitted_observation_fields'] == 0
    assert '/output/diagnostics/log' not in fields
    for pointer, value in fields.items():
        if pointer.startswith('/output/'):
            assert value_at({'output': output}, pointer) == value

    receipt_id, = episode['receipt_ids']
    artifact_path = tools.path / (receipt_id + '.json')
    artifact_bytes = artifact_path.read_bytes()
    artifact = json.loads(artifact_bytes)
    assert content_hash(artifact) == receipt_id
    assert artifact['payload']['receipt']['output'] == output == before
    assert card['source']['receipt_id'] == receipt_id
    tools.commit(episode)
    state = json.loads(json.dumps(tools.state()))
    next_source = snapshot(iteration=1)
    committed_cards = tools.context(next_source)
    committed_card, = committed_cards
    # Commit advances the cell revision; it must retain the exact receipt and
    # projected observations under that new, immutable revision identity.
    for key in ('cell_id', 'source', 'observations'):
        assert committed_card[key] == card[key]

    # A fresh controller loads only committed state and immutable artifacts.
    restored, resumed_backend, _, resumed_costs = controller(tmp_path, [], **config)
    restored.load_state(state)
    assert restored.state() == state
    next_source.research_cells_context = restored.context(next_source)
    assert next_source.research_cells_context == committed_cards
    llm = LLM([{'action': 'skip', 'reason': 'Use the recorded check.'}])
    restored.execute(prepare(restored, next_source), next_source, llm, io=IO())
    assert resumed_backend.calls == [] and resumed_costs == [.03]
    assert len(llm.prompts) == 1 and len(llm.prompts[0]) <= planner_budget
    view = delivered_view(llm.prompts[0])
    assert len(view['cards']) + view['omitted_count'] == 1
    if planner_budget == 64000:
        row, = view['cards']
        assert row['observations'] == card['observations']
        assert row['output_excerpt_omitted_chars'] == 0
    for row in view['cards']:
        assert row['source'] == card['source']
        assert row['card_sha256'] == content_hash(committed_card)
        text = row['observations']['output_excerpt']
        if text:
            assert text == card['observations']['output_excerpt']
            assert json.loads(text) == projected
            assert row['output_excerpt_omitted_chars'] == 0
        else:
            assert row['output_excerpt_omitted_chars'] == len(card['observations']['output_excerpt'])
    if not view['cards']:
        assert view['omission_reasons']['planner_prompt_budget'] == 1
    assert artifact_path.read_bytes() == artifact_bytes and output == before


@pytest.mark.parametrize('projected', [False, True], ids=['full_result', 'field_projection'])
def test_planner_budget_never_prefix_slices_a_typed_result(tmp_path, projected):
    output = findings('nested')[0] if projected else result({'whole_value': 'x' * 1100})
    tools, _, _, _ = controller(tmp_path, [output], max_context_chars=4096)
    source = snapshot()
    episode = tools.execute(prepare(tools, source), source, LLM([action()]), io=IO())
    card, = episode['cards']
    original = card['observations']['output_excerpt']
    parsed = json.loads(original)
    assert (parsed.get('schema') == PROJECTION_SCHEMA) is projected
    # A range straddling identity-only and complete-card capacity exercises the
    # branch that retains receipt identity while omitting the entire JSON value.
    saw_whole_omission = False
    for budget in range(1000, 3601, 200):
        prompt = _research_results_prompt([card], budget)
        assert len(prompt) <= budget
        view = delivered_view(prompt)
        assert len(view['cards']) + view['omitted_count'] == 1
        for row in view['cards']:
            assert row['source'] == card['source']
            retained = row['observations']['output_excerpt']
            if retained:
                assert retained == original and json.loads(retained) == parsed
                assert row['output_excerpt_omitted_chars'] == 0
            else:
                saw_whole_omission = True
                assert row['output_excerpt_omitted_chars'] == len(original)
    assert saw_whole_omission


def test_unicode_question_cannot_evict_small_measured_fields_at_minimum_context(tmp_path):
    output, expected = findings('unicode')
    tools, _, _, _ = controller(tmp_path, [output], max_context_chars=4096)
    source = snapshot()
    proposal = action()
    proposal['reason'] = '输入🙂' * 180
    episode = tools.execute(prepare(tools, source), source, LLM([proposal]), io=IO())
    assert len(episode['context']) <= 4096
    card, = episode['cards']
    projected = json.loads(card['observations']['output_excerpt'])
    fields = {field['path']: field['value'] for field in projected['fields']}
    for pointer, value in expected.items():
        assert fields[pointer] == value
