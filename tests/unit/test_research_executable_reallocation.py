"""Selected follow-up work must execute, using saved fake I/O and real queues."""
from copy import deepcopy

import pytest

from sciel.research_tools.experiment_protocol import validate_dispatch
from test_research_continuation_queue import queued, lease
from test_research_experimental_protocol import protocol
from test_research_tool_controller import IO, LLM
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish, current_receipt
from test_research_tool_controller import snapshot


def design(**changes):
    return {**protocol(), **changes}


def close(**changes):
    return finish(experimental_protocol=design(kind='closure', budget_action='none'), **changes)


@pytest.mark.parametrize('action', ['finish', 'skip', 'poll', 'cancel', 'submit_candidate'])
@pytest.mark.parametrize('allocation', ['reallocate', 'escalate'])
def test_declared_new_work_cannot_disappear_through_nonexecuting_action(action, allocation):
    proposed = decision(action=action, experimental_protocol=design(
        kind='follow_up', budget_action=allocation, new_uncertainty='Different starts?',
        budget_reason='The tested local refinement did not help.'))
    with pytest.raises(ValueError, match='selects executable research'):
        validate_dispatch(proposed, {}, 'f' * 64, default_limits={'cpu_seconds': 20})


@pytest.mark.parametrize('field', ['new_uncertainty', 'budget_reason'])
def test_allocation_requires_a_concrete_reason_even_for_a_new_question(field):
    spec = design(budget_action='reallocate', new_uncertainty='Another starting state?',
                  budget_reason='Local refinement gave no gain.')
    spec[field] = ''
    with pytest.raises(ValueError, match='requires new_uncertainty and budget_reason'):
        validate_dispatch(decision(experimental_protocol=spec), {}, 'f' * 64,
                          default_limits={'cpu_seconds': 20})


def test_finish_reallocation_repairs_to_experiment_survives_resume_and_submits_exact_candidate(tmp_path):
    source = 'def run_search():\n    return 17\n'
    saved = {'best': [1.23456789012345, 0], 'seed': 17, 'conditions': {'budget': 2}}
    tools, backend, events, costs = build(tmp_path, [
        result({'gain': 0, 'continuation_state': saved}),
        result({'gain': 1, 'candidate_source': source})],
        independent_research=True, shared_observations=True, experimental_protocol=True,
        yield_after_success=True, max_requests=2)
    cell = queued(tools, snapshot())
    _, current, plan = lease(tools)
    first = tools.execute(plan, current, LLM([decision(experimental_protocol=design())]), io=IO())
    tools.commit(first)
    original_receipt = first['receipt_ids'][0]

    _, current, plan = lease(tools)
    # This is the live failure pattern: a selected alternative hidden inside finish.
    alternate = design(kind='follow_up', question_key='alternative_start',
        budget_action='reallocate', prior_receipt_ids=[original_receipt],
        new_uncertainty='Does a different start improve on zero-gain local refinement?',
        budget_reason='Compare another start with the retained incumbent.')
    invalid = finish(experimental_protocol=alternate,
        next_decision='Run the alternative starting-state experiment next.')
    def repaired(prompt):
        assert 'selects executable research' in prompt
        assert original_receipt in prompt
        return decision(question='Does another start help?', experimental_protocol=alternate,
            input_receipt_id=original_receipt, arguments_json='{"start":17}',
            code=decision()['code'] + '# alternative experiment\n')
    second = tools.execute(plan, current, LLM([invalid, repaired]), io=IO())
    tools.commit(second)
    assert len(backend.calls) == 2
    assert backend.calls[1]['inputs']['continuation_state'] == saved
    assert backend.calls[1]['inputs']['research_stage']['prior_receipt_id'] == original_receipt
    assert second['research_state']['status'] == 'open'
    assert second['research_state']['question'] == 'Does another start help?'
    assert tools.state()['continuations']['items'][cell]['status'] == 'ready'
    assert len([s for s in second['research_state']['steps'] if s['status'] == 'proposal_invalid']) == 1
    checkpoint = deepcopy(tools.state())
    tools.load_state(checkpoint)
    assert tools.state() == checkpoint

    evaluated = []
    def evaluate(code, provenance):
        evaluated.append((code, provenance))
        return {'status': 'evaluated', 'score': .2}
    _, current, plan = lease(tools)
    third = tools.execute(plan, current, LLM([
        lambda prompt: decision(action='submit_candidate', candidate_receipt_id=current_receipt(prompt),
            experimental_protocol=design(kind='follow_up', budget_action='none')),
        close(conclusion='inconclusive', reason='Canonical candidate measured; transfer benefit remains unknown.')
    ]), io=IO(), candidate_evaluator=evaluate)
    tools.commit(third)
    assert len(evaluated) == 1 and evaluated[0][0] == source
    assert third['candidate_submission']['source'] == source
    assert costs == [.03] * 5  # includes the rejected allocation and the repair
    assert third['research_state']['resources']['cpu_seconds'] == pytest.approx(.02)
    assert len(backend.calls) == 2


def test_repeated_finish_reallocation_exhausts_repairs_without_tools_or_false_closure(tmp_path):
    tools, backend, events, costs = build(tmp_path, [], independent_research=True,
        shared_observations=True, experimental_protocol=True, max_requests=2)
    cell = queued(tools, snapshot())
    _, current, plan = lease(tools)
    invalid = finish(conclusion='inconclusive', experimental_protocol=design(
        kind='follow_up', budget_action='reallocate', new_uncertainty='Another start?',
        budget_reason='A different branch needs checking.'))
    episode = tools.execute(plan, current, LLM([invalid, invalid]), io=IO())
    tools.commit(episode)
    assert backend.calls == [] and costs == [.03, .03]
    assert episode['research_state']['status'] == 'repair_exhausted'
    assert episode['research_state']['conclusion'] is None
    assert all(s['status'] == 'proposal_invalid' for s in episode['research_state']['steps'])
    assert tools.state()['continuations']['items'][cell]['status'] == 'closed'


def test_legitimate_unmeasured_closure_still_needs_no_tool(tmp_path):
    tools, backend, _, costs = build(tmp_path, [], independent_research=True,
        shared_observations=True, experimental_protocol=True)
    cell = queued(tools, snapshot())
    _, current, plan = lease(tools)
    episode = tools.execute(plan, current, LLM([close(conclusion='inconclusive',
        reason='Required external inputs are unavailable; no supported check is selected.')]), io=IO())
    tools.commit(episode)
    assert backend.calls == [] and costs == [.03]
    assert episode['research_state']['status'] == 'closed'
    assert tools.state()['continuations']['items'][cell]['status'] == 'closed'
