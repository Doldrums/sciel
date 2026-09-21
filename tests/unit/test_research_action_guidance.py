"""Delivered planner permissions, using fake provider/executor leaves only.

These tests prove the instructions reach actual requests and preserve valid
closure. They do not predict whether a model will choose a useful experiment.
"""
from copy import deepcopy

import pytest

from sciel.engine.research_cells_context import compact_context
from test_research_experimental_protocol import protocol
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish


def assert_permissions(prompt, budget, independent):
    assert len(prompt) <= budget
    assert ('choose an unresolved task/candidate decision using supplied evidence' in prompt
            or 'resolve a task/candidate decision with a control' in prompt)
    for text in (
        'CPU search, comparisons and candidate transforms are allowed with existing source/results',
        'compute for new evidence',
        'Prior model-only closures are interpretations, not tool policy',
        'no worthwhile feasible check',
    ):
        assert text in prompt
    assert 'No tools for known metadata' not in prompt
    assert 'do not invoke a tool only to reformat known metadata' not in prompt
    assert 'Already answered?' not in prompt
    for text in ('kind=closure', 'budget_action=none', 'reallocate', 'escalate'):
        assert text in prompt
    if independent:
        assert 'open work resumes independently' in prompt
        assert 'Select another experiment with create/use or eligible continue' in prompt
    else:
        assert 'issue create/use (or eligible continue) now' in prompt
    assert 'finish closes without executing next_decision advice' in prompt


@pytest.mark.parametrize('budget', [8000, 64000])
@pytest.mark.parametrize('independent', [False, True])
def test_initial_repair_and_result_requests_keep_positive_permissions_and_replay(tmp_path, budget, independent):
    tools, backend, _, costs = build(tmp_path, [result({'gain': 0, 'tested_iterations': 17})],
        background=True, experimental_protocol=True, shared_observations=True,
        independent_research=independent, max_prompt_chars=budget)
    current = snapshot()
    current.kb_lessons = ['Unrelated study material 雪🙂 ' * 4000]
    plan = prepare(tools, current)
    if independent:
        def refresh(saved, state, **work):
            saved.research_cells_context = compact_context(tools.context(saved))
            return saved
        _, current = tools.reserve_continuations(refresh)[0]
        plan = current.tool_episode_plan
    actions = ['not JSON', decision(experimental_protocol=protocol()),
        finish(reason='The measured pilot has zero gain; no further feasible test is selected.',
            next_decision='Close this bounded question.',
            experimental_protocol={**protocol(), 'kind': 'closure', 'budget_action': 'none'})]
    llm, io = LLM(deepcopy(actions)), IO()
    episode = tools.execute(plan, current, llm, io=io)
    assert len(llm.prompts) == 3 and len(backend.calls) == 1
    for prompt in llm.prompts:
        assert_permissions(prompt, budget, independent)
    assert 'PREVIOUS ATTEMPT FEEDBACK:' in llm.prompts[1]
    assert episode['research_state']['status'] == 'closed'
    assert episode['research_state']['scientific_steps'] == 1
    assert costs == [.03] * 3
    replay = tools.execute(plan, current, LLM(deepcopy(actions)), io=io)
    assert replay['receipt_ids'] == episode['receipt_ids']
    assert costs == [.03] * 3 and len(backend.calls) == 1


@pytest.mark.parametrize('budget', [8000, 64000])
def test_answered_static_question_can_still_close_without_execution(tmp_path, budget):
    tools, backend, _, costs = build(tmp_path, [], background=True,
        experimental_protocol=True, shared_observations=True, max_prompt_chars=budget)
    current = snapshot()
    def close(prompt):
        assert_permissions(prompt, budget, False)
        return finish(question='Does the supplied source define the documented entrypoint?',
            conclusion='inconclusive', reason='The source defines run_search; no experiment is needed for this static inspection, and no measured method claim is made.',
            next_decision='Close; no additional experiment selected.',
            experimental_protocol={**protocol(), 'kind': 'closure', 'budget_action': 'none'})
    episode = tools.execute(prepare(tools, current), current, LLM([close]), io=IO())
    assert episode['research_state']['status'] == 'closed'
    assert not backend.calls and costs == [.03]
