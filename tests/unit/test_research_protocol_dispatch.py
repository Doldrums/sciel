"""Protocol decisions govern actual controller dispatch and its full cost ledger."""
import json

import pytest

from test_research_experimental_protocol import protocol, comparison
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish, current_receipt


def designed(action, **changes):
    value = protocol()
    value.update(changes)
    return dict(action, experimental_protocol=value)


def test_same_opportunity_repeat_is_refused_before_tool_even_when_renamed(tmp_path):
    tools, backend, events, costs = build(tmp_path, [result({'gain': 0})],
        experimental_protocol=True, shared_observations=True)
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current, LLM([
        designed(decision()),
        designed(decision(name='Renamed but identical question', code=decision()['code'] + '# rename\n')),
        designed(finish(conclusion='refuted'), kind='closure', budget_action='none')]), io=IO())
    assert len(backend.calls) == 1 and costs == [.03] * 3
    assert episode['research_state']['known_llm_cost'] == pytest.approx(.09)
    assert episode['research_state']['status'] == 'closed'
    rejected = episode['research_state']['steps'][1]
    assert rejected['status'] == 'proposal_invalid' and 'new_uncertainty' in rejected['error']
    assert len([e for e in events if e['phase'] == 'registered']) == 1
    tools.commit(episode)
    saved = tools.state()
    tools.load_state(json.loads(json.dumps(saved)))
    assert tools.state() == saved


def test_next_stage_can_cite_uncommitted_exact_receipt_and_preserve_state(tmp_path):
    saved_state = {'seed': 11, 'incumbent': [0, 2], 'rng_state': [17, 81]}
    tools, backend, events, costs = build(tmp_path,
        [result({'gain': 0, 'continuation_state': saved_state}), result({'gain': 1})],
        experimental_protocol=True, shared_observations=True)
    current = snapshot()

    def next_stage(prompt):
        receipt = current_receipt(prompt)
        return designed(decision(code=decision()['code'] + '# alternative start\n', input_receipt_id=receipt),
            kind='follow_up', prior_receipt_ids=[receipt], new_uncertainty='Does another starting state improve within the same cap?',
            budget_action='reallocate', budget_reason='Initial segment did not improve; test a different start.')

    episode = tools.execute(prepare(tools, current), current, LLM([
        designed(decision()), next_stage,
        designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none')]), io=IO())
    assert len(backend.calls) == 2 and backend.calls[1]['inputs']['continuation_state'] == saved_state
    assert costs == [.03] * 3
    audits = [event for event in events if event['phase'] == 'experiment_protocol']
    assert audits[1]['repeated_receipt_ids'] == [episode['receipt_ids'][0]]
    tools.commit(episode)
    tools.load_state(json.loads(json.dumps(tools.state())))


def test_equal_budget_comparison_report_is_audited_without_turning_it_into_score(tmp_path):
    design = comparison()
    rows = [dict(arm_id=arm['id'], seed=seed, cpu_seconds=1.5,
                 best_result={'score': 999}, saved_state={'seed': seed}, conditions={'inputs': 'same'})
            for arm in design['arms'] for seed in design['matched_seeds']]
    tools, backend, events, costs = build(tmp_path, [result({'comparison_results': rows})],
        experimental_protocol=True, shared_observations=True)
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current, LLM([
        decision(experimental_protocol=design),
        designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none')]), io=IO())
    assessments = [event for event in events if event['phase'] == 'comparison_report_assessment']
    assert len(assessments) == 1
    assert assessments[0]['report_shape'] == 'complete'
    assert assessments[0]['adherence'] == 'tool_reported_unverified'
    assert 'candidate_submission' not in episode
    assert episode['research_state']['resources']['cpu_seconds'] == .01
    assert costs == [.03] * 2 and len(backend.calls) == 1


def test_failed_check_allows_repair_of_the_same_question_without_invented_uncertainty(tmp_path):
    tools, backend, _, costs = build(tmp_path,
        [result(status='tool_error', error='fixture failed to read input'), result({'gain': 0})],
        experimental_protocol=True, shared_observations=True)
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current, LLM([
        designed(decision()), designed(decision(code=decision()['code'] + '# input repair\n')),
        designed(finish(conclusion='refuted'), kind='closure', budget_action='none')]), io=IO())
    assert len(backend.calls) == 2 and costs == [.03] * 3
    assert [step['status'] for step in episode['research_state']['steps']] == ['tool_error', 'success', 'closed']
    assert episode['research_state']['scientific_steps'] == 1
