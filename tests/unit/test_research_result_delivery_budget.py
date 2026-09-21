"""Nested numerical evidence fits actual memory before any lossy projection.

The inert JSON is the complete 5,290-character result of the V32 two-target
probe (receipt fa1af65a...). It contains no source to execute. Engine tests use
only fake provider, evaluator and executor leaves.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from research_target_fixtures import binding
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.methods.models import canonical_bytes, content_hash
from sciel.research_tools.controller import ResearchTools
from sciel.research_tools.result_contract import assess_receipt
from sciel.research_tools.result_projection import PROJECTION_SCHEMAS, projection_fields
from test_parallel_engine import config, evaluator, task_fixture
from test_research_terminal_delivery_engine import delivered_cards
from test_research_tool_controller import IMAGE, snapshot
from test_research_tool_result_contract import receipt
from test_research_tool_result_flow import SavedExecutor
from test_research_yield_engine_integration import YieldProvider, generation_prompts
from test_sequential_research_episode import decision
from test_tool_result_projection import pointer


FIXTURE = Path(__file__).parents[1] / 'fixtures/research_tool_nested_result_v32.json'


def saved_output():
    output = json.loads(FIXTURE.read_text())
    assert len(canonical_bytes(output)) == 5290
    return output


def card_for(output, budget):
    tools = ResearchTools.__new__(ResearchTools)
    tools.config = {'max_context_chars': budget}
    observation = dict(output=output, result=assess_receipt(receipt(output)),
        replay_policy='independent_execution', tool_id='a'*64, invocation_id='b'*64,
        receipt_id='c'*64, target_binding=binding(snapshot()), resources={'cpu_seconds': .75},
        question='Does additional computation improve either saved result under the same constraints?')
    plan = {'cell_id':'d'*64, 'cell_revision_id':'e'*64, 'stage':'research',
            'question':observation['question']}
    return tools, tools._card(plan, observation)


def assert_complete_measurements(card, output):
    wrapper = json.loads(card['observations']['output_excerpt'])
    assert wrapper['output'] == output
    for role in ('selected', 'population_best'):
        observed = wrapper['output']['observations']['roles'][role]
        expected = output['observations']['roles'][role]
        assert observed['lp_at_saved_centers'] == expected['lp_at_saved_centers']
        assert observed['polish'] == expected['polish']
    assert wrapper['output']['observations']['interpretation_scope'] == output['observations']['interpretation_scope']
    assert card['observations']['status'] == 'success'
    assert 'not canonical task score or benefit' in ' '.join(card['limitations'])


def test_complete_two_target_result_fits_real_card_budget_without_size_ranked_omissions():
    output = saved_output()
    before = deepcopy(output)
    tools, card = card_for(output, 10000)
    assert 6000 < len(tools.format_context([card])) <= 10000
    assert_complete_measurements(card, output)
    assert output == before


@pytest.mark.parametrize('budget', [4096, 8000, 10000])
def test_genuinely_oversized_result_keeps_whole_values_and_explicit_omissions(budget):
    output = saved_output()
    output['observations']['large_records'] = ['snow 雪 "quoted"\n' * 2000]
    before = deepcopy(output)
    tools, card = card_for(output, budget)
    assert len(tools.format_context([card])) <= budget
    projected = json.loads(card['observations']['output_excerpt'])
    assert projected['schema'] in PROJECTION_SCHEMAS
    assert projected['omitted_fields'] > 0 and projected['omitted_observation_fields'] > 0
    assert projected['observations_state'] == 'partial'
    for field in projection_fields(projected):
        if field['path'].startswith('/output/'):
            assert field['value'] == pointer({'output': output}, field['path'])
    for array in projected.get('observation_rows', {}).get('arrays', []):
        original = pointer({'output': output}, array['path'])
        assert array['original_sha256'] == content_hash(original)
        for row in array['rows']:
            assert row['value'] == original[row['index']]
            assert row['original_sha256'] == content_hash(row['value'])
    assert card['source']['receipt_id'] == 'c'*64
    assert output == before


@pytest.mark.parametrize('research_enabled', [False, True])
def test_complete_result_reaches_actual_G_and_retained_G_after_checkpoint(
        tmp_path, monkeypatch, research_enabled):
    output = saved_output()
    frozen = deepcopy(output)
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    evaluations, _ = evaluator(monkeypatch)
    backend = SavedExecutor([output])
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor', lambda image: backend)
    provider = YieldProvider([decision()])
    cfg = config(task, tmp_path/'capacity', workers=1, num_islands=1, pop_cap=1,
        max_iterations=2,
        research={'enabled':research_enabled, 'reflection_interval':0, 'max_aux_calls':None},
        research_tools={'enabled':True, 'image_id':IMAGE, 'interval':100,
            'proactive_every_attempts':100, 'max_scientific_steps':3,
            'yield_after_success':True, 'max_requests':2, 'max_context_chars':10000,
            'share_between_islands':False})
    directory = tmp_path / 'run'

    def build():
        return build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)

    first = build()
    first._stop_at = 1
    stopped = first.run()
    assert stopped['reason'] == 'paused' and stopped['iteration_no'] == 1
    first_card = delivered_cards(generation_prompts(provider)[0], research_enabled)[0]
    assert_complete_measurements(first_card, frozen)
    evidence = first.research_tools._read_artifact(first_card['source']['receipt_id'], 'invocation')
    assert evidence['receipt']['output'] == frozen
    assert evidence['receipt']['output_sha256'] == content_hash(frozen)
    before_counts = (len(provider.planner_prompts), len(backend.calls), len(evaluations))

    resumed = build()
    checkpoint.load(resumed, str(directory/'checkpoint.json'))
    assert (len(provider.planner_prompts), len(backend.calls), len(evaluations)) == before_counts
    completed = resumed.run()
    assert completed['iteration_no'] == completed['B_evals'] == 2
    assert len(provider.planner_prompts) == len(backend.calls) == 1
    assert len(evaluations) == 3  # Seed and two ordinary candidates.
    cards = [delivered_cards(prompt, research_enabled)[0] for prompt in generation_prompts(provider)]
    assert len(cards) == 2
    for card in cards:
        assert_complete_measurements(card, frozen)
        assert card['source']['receipt_id'] == first_card['source']['receipt_id']
        assert not card.get('decision')  # Yield is not an interpreted closure.
    assert cards[0]['observations'] == cards[1]['observations']
    assert backend.receipts[0]['output'] == output == frozen
    assert completed['cost_total'] == pytest.approx(.11 if research_enabled else .09)


def test_later_terminal_preserves_complete_5290_result_when_bundle_fits(tmp_path):
    from test_research_tool_controller import IO, LLM, prepare
    from test_research_tool_result_flow import controller
    from test_sequential_research_episode import finish
    output = saved_output()
    tools, backend, _, _ = controller(tmp_path, [output], max_scientific_steps=3,
        yield_after_success=True, shared_observations=True, max_context_chars=10000,
        max_visible_tools=1)
    first = snapshot()
    episode = tools.execute(prepare(tools, first), first, LLM([decision()]), io=IO())
    tools.commit(episode)
    original = deepcopy(next(iter(tools.lifecycle._state['cards'].values())))
    later = snapshot(iteration=1)
    closed = tools.execute(prepare(tools, later), later, LLM([finish(
        conclusion='inconclusive', reason='Keep the measured selected/population comparison separate from this closure.')]), io=IO())
    tools.commit(closed)
    for cold in (False,True):
        if cold:
            tools.load_state(deepcopy(tools.state()))
        view = tools.context_view(snapshot(1, 2))
        assert len(view['cards']) == 2 and view['omitted_count'] == 0
        measured, terminal = view['cards']
        assert_complete_measurements(measured, output)
        # Mandatory host assessment can require compact host boilerplate,
        # but the complete scientific payload and all bindings remain exact.
        for key in ('observations', 'source', 'question', 'revision_id', 'comparison_assessment'):
            assert measured[key] == original[key]
        assert terminal['source']['receipt_id'] is None
        assert terminal['decision']['interpretation'].startswith('Keep the measured')
        assert len(json.dumps(view, ensure_ascii=True, separators=(',', ':'))) <= 10000
        assert len(tools.format_context(view['cards'])) <= 10000
    assert len(backend.calls) == 1
