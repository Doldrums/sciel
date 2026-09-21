"""Accepted finish reasoning reaches G and survives immutable checkpoint checks.

All model responses and execution receipts are fixtures; no generated code runs.
"""
from copy import deepcopy
import json

import pytest

from sciel.engine.research_cells_context import compact_context
from sciel.research_tools.decision_projection import authenticate_projection
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish


@pytest.mark.parametrize('conclusion', ['supported', 'refuted', 'inconclusive'])
def test_terminal_interpretation_is_separate_from_successful_measurement(tmp_path, conclusion):
    observed = {'claimed_value': 7.5, 'samples': [[1.0, 1.0, .5]]}
    tools, backend, events, _ = build(tmp_path, [result(observed)])
    current = snapshot()
    final = finish(conclusion=conclusion, question='Does the reported value meet the task conditions?',
        reason='The reported value is invalid under the declared conditions; execution success is not validity.',
        next_decision='Repair the feasibility check, then use the ordinary evaluator.')
    episode = tools.execute(prepare(tools, current), current, LLM([decision(), final]), io=IO())
    card = episode['cards'][0]
    assert card['schema'] == 'sciel-research-check/5'
    assert card['comparison_assessment']['status'] == 'unavailable'
    assert card['question'] == decision()['question']  # Original receipt binding, not finish.question.
    assert card['observations']['status'] == 'success'
    assert json.loads(card['observations']['output_excerpt'])['output']['observations'] == observed
    assert card['decision']['question'] == final['question']
    assert card['decision']['interpretation'] == final['reason']
    assert card['decision']['conclusion'] == conclusion
    assert final['reason'] in episode['context'] and final['next_decision'] in episode['context']
    assert 'Interpretation remains open' not in episode['context']
    assert len(backend.calls) == 1 and episode['known_llm_cost'] == pytest.approx(.06)
    projection = compact_context(episode['cards'])
    assert projection['omitted_count'] == 0
    tools.commit(episode)
    saved = tools.state()
    tools.load_state(deepcopy(saved))
    retained = tools.context(snapshot(iteration=1))[0]
    assert retained['decision'] == card['decision']
    assert retained['observations'] == card['observations']
    assert retained['revision_id'] != card['revision_id']
    proposal_id = card['decision']['proposal_id']
    assert {'kind':'research_decision','id':proposal_id} in tools.lifecycle.cells._read(
        tools.lifecycle.cells.revisions, retained['revision_id'])['evidence']


@pytest.mark.parametrize('with_failure', [False, True])
def test_finish_without_success_does_not_invent_a_tool_measurement(tmp_path, with_failure):
    tools, backend, _, _ = build(tmp_path, [result(status='tool_error', error='Missing input')])
    current = snapshot()
    actions = ([decision()] if with_failure else []) + [finish(conclusion='inconclusive',
        reason='No valid observation supports this question.', next_decision='Obtain the missing input.')]
    episode = tools.execute(prepare(tools, current), current, LLM(actions), io=IO())
    card = episode['cards'][0]
    assert card['decision']['conclusion'] == 'inconclusive'
    assert 'No valid observation' in episode['context']
    if with_failure:
        assert card['observations']['status'] == 'tool_error'
        assert card['source']['receipt_id'] == episode['receipt_ids'][0]
    else:
        assert card['observations'] == {'status':'not_invoked_this_episode','output_excerpt':'','resources':{}}
        assert all(card['source'][key] is None for key in ('tool_id','receipt_id','invocation_id'))
        assert card['decision']['receipt_ids'] == [] and backend.calls == []
    tools.commit(episode)
    tools.load_state(deepcopy(tools.state()))


def test_later_finish_can_reference_prior_receipt_without_relabeling_it_as_new(tmp_path):
    tools, backend, _, _ = build(tmp_path, [result({'gain':0})])
    current = snapshot()
    first = tools.execute(prepare(tools,current),current,LLM([decision(),finish()]),io=IO())
    tools.commit(first)
    later = snapshot(iteration=1)
    second = tools.execute(prepare(tools,later),later,LLM([finish(
        reason='The prior measurement showed no gain; no new measurement was made.')]),io=IO())
    card = second['cards'][0]
    assert card['source']['receipt_id'] is None
    assert card['observations']['status'] == 'not_invoked_this_episode'
    assert card['decision']['receipt_ids'] == first['receipt_ids']
    assert card['decision']['action_id'] != first['cards'][0]['decision']['action_id']
    assert len(backend.calls) == 1
    tools.commit(second)
    tools.load_state(deepcopy(tools.state()))


@pytest.mark.parametrize('budget', [4096, 8000])
def test_large_decision_and_output_keep_enum_provenance_and_explicit_omissions(tmp_path, budget):
    tools, _, _, _ = build(tmp_path, [result({'payload':'x'*50000, 'claimed_value':7.5})],
                           max_context_chars=budget, max_prompt_chars=8000)
    current = snapshot()
    final = finish(conclusion='inconclusive', question='QUESTION_START '+('中'*3000)+' QUESTION_END',
        reason='INVALID_RESULT '+('detail '*1000)+' INVALID_CONSTRAINTS',
        next_decision='REPAIR_START '+('next '*1000)+' REPAIR_END')
    episode = tools.execute(prepare(tools,current),current,LLM([decision(),final]),io=IO())
    assert len(episode['context']) <= budget and len(episode['cards']) == 1
    card = episode['cards'][0]
    terminal = card['decision']
    assert terminal['conclusion'] == 'inconclusive'
    assert terminal['receipt_ids'] == episode['receipt_ids']
    assert terminal['interpretation'].startswith('INVALID_RESULT')
    assert terminal['interpretation'].endswith('INVALID_CONSTRAINTS')
    assert terminal['omitted_chars']['interpretation'] > 0
    assert terminal['original_chars']['interpretation'] == len(final['reason'])
    assert 'complete text is in the saved proposal' in terminal['interpretation']
    proposal = tools._read_artifact(terminal['proposal_id'],'proposal')
    assert authenticate_projection(terminal,proposal,terminal['proposal_id']) == terminal
    tools.commit(episode)
    tools.load_state(deepcopy(tools.state()))


@pytest.mark.parametrize('field', ['interpretation','conclusion','proposal_id','action_id','receipt_ids','omitted_chars'])
def test_tampered_terminal_decision_cannot_survive_resume(tmp_path, field):
    tools, _, _, _ = build(tmp_path, [result({'gain':0})])
    current = snapshot()
    episode = tools.execute(prepare(tools,current),current,LLM([decision(),finish(
        reason='This measurement does not support an improvement.')]),io=IO())
    tools.commit(episode)
    saved = deepcopy(tools.state())
    card = next(iter(saved['lifecycle']['cards'].values()))
    changed = card['decision']
    if field == 'receipt_ids': changed[field] = ['f'*64]
    elif field == 'omitted_chars': changed[field]['interpretation'] += 1
    elif field in ('action_id','proposal_id'): changed[field] = 'f'*64
    elif field == 'conclusion': changed[field] = 'supported'
    else: changed[field] = 'A fabricated successful finding.'
    with pytest.raises((ValueError, FileNotFoundError)):
        tools.load_state(saved)


def test_rejected_finish_and_skip_cannot_leak_a_terminal_conclusion(tmp_path):
    tools, _, _, _ = build(tmp_path, [])
    current = snapshot()
    bad = finish(conclusion='supported', reason='UNSUPPORTED_FINISH')
    skip = decision(action='skip', reason='Missing input, no experiment performed.')
    episode = tools.execute(prepare(tools,current),current,LLM([bad,skip]),io=IO())
    assert not episode['cards']
    assert 'terminal_decision_proposal_id' not in episode
    assert 'UNSUPPORTED_FINISH' not in episode['context']


def test_no_finish_retains_original_card_semantics(tmp_path):
    tools, _, _, _ = build(tmp_path, [result({'interpretation':'Tool-owned interpretation','gain':0})]*3)
    current = snapshot()
    episode = tools.execute(prepare(tools,current),current,LLM([decision()]*4),io=IO())
    assert episode['cards'][0]['schema'] == 'sciel-research-check/5'
    assert episode['cards'][0]['comparison_assessment']['status'] == 'unavailable'
    assert episode['cards'][0]['interpretation'] == 'Tool-owned interpretation'
    assert 'decision' not in episode['cards'][0]


def test_malformed_finish_reason_is_repaired_before_terminal_projection(tmp_path):
    tools, _, _, _ = build(tmp_path, [])
    current = snapshot()
    episode = tools.execute(prepare(tools,current),current,LLM([
        finish(conclusion='inconclusive',reason={'invalid':'not model text'}),
        finish(conclusion='inconclusive',reason='No experiment was performed.')]),io=IO())
    assert episode['research_state']['steps'][0]['status'] == 'proposal_invalid'
    assert episode['cards'][0]['decision']['interpretation'] == 'No experiment was performed.'
    tools.commit(episode)
    tools.load_state(deepcopy(tools.state()))


def test_failed_final_trial_keeps_its_failure_beside_terminal_interpretation(tmp_path):
    tools, _, _, _ = build(tmp_path, [result({'gain':0}),result(status='tool_error',error='Invalid input')])
    current = snapshot()
    episode = tools.execute(prepare(tools,current),current,LLM([decision(),decision(),
        finish(conclusion='inconclusive',reason='The second trial failed; the first showed no gain.')]),io=IO())
    card = episode['cards'][0]
    assert card['observations']['status'] == 'tool_error'
    assert card['source']['receipt_id'] == episode['receipt_ids'][-1]
    assert card['decision']['receipt_ids'] == episode['receipt_ids']
    assert 'Invalid input' in card['observations']['output_excerpt']
    tools.commit(episode)
    tools.load_state(deepcopy(tools.state()))


def test_old_finish_is_neither_attached_to_new_trial_nor_accepted_by_resume(tmp_path):
    tools, _, _, _ = build(tmp_path, [result({'gain':0})]*4)
    current = snapshot()
    first = tools.execute(prepare(tools,current),current,LLM([decision(),finish(reason='OLD_CLOSURE')]),io=IO())
    tools.commit(first)
    later = snapshot(iteration=1)
    second = tools.execute(prepare(tools,later),later,LLM([decision()]*4),io=IO())
    assert second['cards'][0]['schema'] == 'sciel-research-check/5'
    assert 'OLD_CLOSURE' not in second['context']
    tools.commit(second)
    saved = deepcopy(tools.state())
    card = next(iter(saved['lifecycle']['cards'].values()))
    card['schema'] = 'sciel-research-check/5'
    card['decision'] = deepcopy(first['cards'][0]['decision'])
    with pytest.raises(ValueError,match='another observation episode'):
        tools.load_state(saved)


@pytest.mark.parametrize('budget', [4096, 8000])
@pytest.mark.parametrize('characters', ['😀', '中'])
def test_unicode_invocation_and_terminal_questions_remain_bound_and_fit(tmp_path,budget,characters):
    tools, _, _, _ = build(tmp_path,[result({'value':7.5,'details':characters*5000})],
                           max_context_chars=budget,max_prompt_chars=8000)
    current = snapshot()
    question = characters*600
    final = finish(conclusion='inconclusive',question=question,reason=characters*2000,
                   next_decision=characters*600)
    episode = tools.execute(prepare(tools,current),current,
                            LLM([decision(question=question),final]),io=IO())
    card = episode['cards'][0]
    receipt = tools._read_artifact(episode['receipt_ids'][0],'invocation')
    assert card['question'] == receipt['question']
    assert len(json.dumps(card['question'],ensure_ascii=True)) <= min(620,budget//12)
    assert card['source']['target_source_sha256'] == receipt['target_binding']['target_source_sha256']
    assert card['decision']['conclusion'] == 'inconclusive'
    assert card['decision']['receipt_ids'] == episode['receipt_ids']
    assert len(episode['context']) <= budget
    assert compact_context(episode['cards'])['omitted_count'] == 0
    tools.commit(episode)
    tools.load_state(deepcopy(tools.state()))
