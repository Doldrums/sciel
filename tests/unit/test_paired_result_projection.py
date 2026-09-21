"""Inert exact V43 measurements; no candidate/tool source is executed."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from sciel.methods.models import content_hash
from sciel.research_tools.result_projection import (
    COMPARISON_PROJECTION_SCHEMA, MIN_CONTEXT_CHARS, PROJECTION_SCHEMA,
    success_context, verify_projection, SPARSE_PROJECTION_SCHEMA, projection_fields,
)
from sciel.research_tools.shared_evidence import prompt as shared_prompt
from sciel.research_tools.controller import _research_results_prompt, RESULTS_VIEW_MARKER
from sciel.engine.research_cells_context import compact_context
from test_research_cell_context import replay_card


FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/research_comparison_v43.json'


def witness():
    saved = json.loads(FIXTURE.read_text())
    obs = saved['observations']
    obs['candidate_source'] = 'x' * saved['provenance']['excluded_candidate_source_chars']
    return {'output': {'observations': obs}, 'error': None}


def project(item, limit):
    before = deepcopy(item)
    text = success_context(item, limit)
    assert text.isascii() and len(text) <= limit and item == before
    value = json.loads(text)
    verify_projection(value, item)
    return value


def rows(value):
    return [row for group in value['comparison_projection']['groups'] for row in group['rows']]


def test_actual_six_rows_survive_bulky_state_and_source_without_invented_measurements():
    item = witness()
    value = project(item, 8000)
    assert value['schema'] in (COMPARISON_PROJECTION_SCHEMA, SPARSE_PROJECTION_SCHEMA)
    assert len(rows(value)) == 6
    assert [g['seed'] for g in value['comparison_projection']['groups']] == [0, 1]
    original = item['output']['observations']['comparison_results']
    for row in rows(value):
        index = row['index']
        assert row['path'] == f'/output/observations/comparison_results/{index}'
        assert row['original_sha256'] == content_hash(original[index])
        assert row['values'] == {k: original[index][k] for k in
            ('arm_id', 'seed', 'cpu_seconds', 'best_result', 'conditions')}
        assert row['omitted_values']['saved_state']['sha256'] == content_hash(original[index]['saved_state'])
    assert value['comparison_projection']['omitted_rows'] == 0
    assert value['observations_state'] == 'partial'  # Full geometries/source were not delivered.
    assert '/output/observations/comparison_results' in value['omitted_paths']
    assert 'not verified adherence or benefit' in value['comparison_projection']['scope']


def test_budget_never_selects_only_first_arm_from_arm_major_input():
    item = witness()
    # This budget fits one full three-arm seed, not all six rows.
    value = project(item, 3900)
    assert value['schema'] in (COMPARISON_PROJECTION_SCHEMA, SPARSE_PROJECTION_SCHEMA)
    assert value['comparison_projection']['shown_rows'] == 3
    assert value['comparison_projection']['omitted_rows'] == 3
    assert [row['index'] for row in rows(value)] == [0, 2, 4]
    assert {row['values']['arm_id'] for row in rows(value)} == {'lp_fixed_centers', 'slsqp_polish', 'shake_polish'}


@pytest.mark.parametrize('limit', [MIN_CONTEXT_CHARS, 100, 500, 1200, 2500])
def test_tiny_budget_explicit_legacy_omission_never_fabricates_complete_comparison(limit):
    value = project(witness(), limit)
    assert value['schema'] in (PROJECTION_SCHEMA, SPARSE_PROJECTION_SCHEMA)
    assert 'comparison_projection' not in value
    assert value.get('observations_state') != 'complete'


@pytest.mark.parametrize('damage', ['value', 'bool', 'index', 'path', 'row_hash', 'array_hash',
    'count', 'duplicate', 'partial_seed', 'state_hash', 'original_hash', 'full_claim'])
def test_tampered_rows_values_and_coverage_fail_closed(damage):
    item = witness(); value = project(item, 8000); comp = value['comparison_projection']; row = comp['groups'][0]['rows'][0]
    if damage == 'value': row['values']['best_result']['sum_radii'] += 1
    elif damage == 'bool': row['values']['cpu_seconds'] = False
    elif damage == 'index': row['index'] = 1
    elif damage == 'path': row['path'] = row['path'].replace('/0', '/1')
    elif damage == 'row_hash': row['original_sha256'] = 'a' * 64
    elif damage == 'array_hash': comp['original_sha256'] = 'b' * 64
    elif damage == 'count': comp['shown_rows'] = 5
    elif damage == 'duplicate': comp['groups'].append(deepcopy(comp['groups'][0]))
    elif damage == 'partial_seed': comp['groups'][0]['rows'].pop()
    elif damage == 'state_hash': row['omitted_values']['saved_state']['sha256'] = 'c' * 64
    elif damage == 'original_hash': value['original_sha256'] = 'd' * 64
    else: value['observations_state'] = 'complete'
    with pytest.raises(ValueError): verify_projection(value, item)


def test_oversized_core_or_missing_member_omits_whole_seed_and_keeps_other_seed():
    for change in ('oversized', 'missing'):
        item = witness(); row = item['output']['observations']['comparison_results'][2]
        if change == 'oversized': row['best_result']['trace'] = 'x' * 30000
        else: del row['best_result']
        value = project(item, 8000)
        assert [group['seed'] for group in value['comparison_projection']['groups']] == [1]
        assert value['comparison_projection']['shown_rows'] == 3
        assert value['comparison_projection']['omitted_rows'] == 3


def test_invalid_raw_values_and_duplicate_arms_remain_exact_not_certified():
    item = witness(); original = item['output']['observations']['comparison_results']
    original[0]['best_result'] = {'a/b~雪': [None, False, 0, '雪']}
    original[0]['cpu_seconds'] = -1
    original[0]['conditions'] = None
    original.append(deepcopy(original[0]))
    value = project(item, 10000)
    assert len(rows(value)) == 7
    assert rows(value)[0]['values']['best_result'] == original[0]['best_result']
    assert rows(value)[0]['values']['cpu_seconds'] == -1
    assert rows(value)[0]['values']['conditions'] is None
    assert 'not verified adherence' in value['comparison_projection']['scope']


def test_shared_prompt_and_structured_own_excerpt_keep_integrity_at_smaller_budget():
    item = witness(); entry = dict(receipt_id='a'*64, tool_id='b'*64, cell_id='c'*64,
        island_id=0, candidate_id='p1', target_source_sha256='d'*64, task={'task':'fixture'},
        question='Compare methods', status='success', resources={'cpu_seconds':33.419089},
        publication='unpublished', decision=None, decision_scope_matches_check=False, measurement=item)
    text = shared_prompt(dict(total_checks=1,source_islands=[0],entries=[entry]), 11000)
    view = json.JSONDecoder().raw_decode(text.split('\n',1)[1])[0]
    measured = view['entries'][0]['measurement']
    verify_projection(measured,item)
    assert len(rows(measured)) == 6
    card = replay_card()
    card['observations']['output_excerpt'] = success_context(item,8000)
    for budget in (1800,3000,11000):
        rendered = _research_results_prompt(compact_context([card]),budget)
        data = json.JSONDecoder().raw_decode(rendered.split(RESULTS_VIEW_MARKER,1)[1])[0]
        assert len(rendered) <= budget
        for shown in data['cards']:
            excerpt = shown['observations']['output_excerpt']
            if excerpt:
                verify_projection(json.loads(excerpt),item)
            else:
                assert shown['output_excerpt_omitted_chars'] == len(card['observations']['output_excerpt'])


def test_legacy_atomic_arrays_and_whole_output_still_verify():
    item = {'output': {'observations': {'ordinary_array':[{'x':'x'*1000}]*10,'gain':0}},'error':None}
    value=project(item,1000)
    assert value['schema']==PROJECTION_SCHEMA
    assert 'comparison_projection' not in value
    verify_projection({'omitted':True},item)
    project(item,100000)


@pytest.mark.parametrize('budget', [8000, 64000])
def test_actual_next_planner_saved_state_and_shared_measurements(tmp_path, budget):
    from test_research_tool_controller import IO, LLM, prepare, snapshot
    from test_research_tool_result_contract import result
    from test_research_tool_result_flow import action, controller
    output = result(witness()['output']['observations'])
    tools, backend, _, _ = controller(tmp_path,[output],max_prompt_chars=budget,
        max_context_chars=10000, shared_observations=True)
    source = snapshot()
    episode = tools.execute(prepare(tools,source),source,LLM([action()]),io=IO())
    assert len(backend.calls)==1
    tools.commit(episode)
    saved = json.loads(json.dumps(tools.state()))
    restored, backend, _, _ = controller(tmp_path,[],max_prompt_chars=budget,
        max_context_chars=10000,shared_observations=True)
    restored.load_state(saved)
    next_source=snapshot(iteration=1)
    next_source.research_cells_context=restored.context(next_source)
    llm=LLM([{'action':'skip','reason':'Use saved evidence.'}])
    restored.execute(prepare(restored,next_source),next_source,llm,io=IO())
    assert not backend.calls
    prompt=llm.prompts[0]
    assert len(prompt)<=budget
    shared=json.JSONDecoder().raw_decode(prompt.split('SHARED_RESEARCH_OBSERVATIONS_JSON:\n',1)[1])[0]
    own=json.JSONDecoder().raw_decode(prompt.split(RESULTS_VIEW_MARKER,1)[1])[0]
    if budget==64000:
        paired=shared['entries'][0]['measurement']
        assert len(rows(paired))==6
        verify_projection(paired,{'output':{'observations':output['observations']},'error':None})
    else:
        # No fake complete comparison at minimum budget; omitted groups/entry
        # remain explicit when a full matched seed cannot fit.
        for entry in shared['entries']:
            measured=entry['measurement']
            if measured.get('schema')==COMPARISON_PROJECTION_SCHEMA:
                assert len(rows(measured)) in (3,6)
            else:
                assert measured.get('observations_state')!='complete'
        assert len(shared['entries'])+shared['omitted_count']==1
    for card in own['cards']:
        text=card['observations']['output_excerpt']
        if text:
            assert isinstance(json.loads(text),dict)  # No prefix-cut /2 JSON.
        else:
            assert card['output_excerpt_omitted_chars']>0


@pytest.mark.parametrize('output', ['non-object output', ['not', 'an', 'object'], 9, True, None])
def test_nonobject_output_preserves_legacy_projection_behavior(output):
    item={'output':output,'diagnostic':'x'*10000}
    text=success_context(item,1000)
    value=json.loads(text)
    assert value['schema']==PROJECTION_SCHEMA
    assert value['observations_state']=='not_present'
    verify_projection(value,item)


def test_small_whole_comparison_array_keeps_legacy_complete_values_and_state():
    values=[dict(arm_id='a',seed=0,cpu_seconds=.1,best_result=1,conditions={},saved_state=[1,2]),
            dict(arm_id='b',seed=0,cpu_seconds=.2,best_result=0,conditions={},saved_state=[3,4])]
    item={'output':{'observations':{'comparison_results':values,'candidate_source':'x'*20000,'gain':1}},'error':None}
    value=project(item,2500)
    assert value['schema']==PROJECTION_SCHEMA
    assert next(row['value'] for row in value['fields'] if row['path']=='/output/observations/comparison_results')==values
    assert next(row['value'] for row in value['fields'] if row['path']=='/output/observations/gain')==1
