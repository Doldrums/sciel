"""Regression for ordinary evaluation IDs mistaken for research receipt citations."""
import json
import re
from pathlib import Path

import pytest

from sciel.eval.registry import load_task
from sciel.eval.task_prompt import compose_task_prompt, split_task_prompt
from sciel.research_tools.experiment_protocol import (
    parse_protocol, response_property, validate_dispatch)
from test_research_experimental_protocol import protocol
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish, current_receipt

ROOT = Path(__file__).resolve().parents[2]
ORDINARY_ID = 'evaluation:fixture-run:wave:1:portfolio:0:1'


@pytest.mark.parametrize('budget', [8000, 64000])
@pytest.mark.parametrize('task_name', ['circle_packing', 'heilbronn_triangle', 'function_minimization_oracle'])
def test_actual_first_and_followup_planner_distinguish_evaluation_from_receipt(tmp_path, budget, task_name):
    tools, backend, _, costs = build(tmp_path, [result({'gain': 0})],
        experimental_protocol=True, shared_observations=True,
        background=True, max_prompt_chars=budget)
    current = snapshot()
    task = load_task(str(ROOT / 'tasks/oe' / task_name))
    current.task_prompt = compose_task_prompt(task)
    current.research_search_evidence = {
        'schema': 'sciel-search-evidence/3',
        'recent_evaluations': [{'evaluation_id': ORDINARY_ID, 'score': 1.0}],
    }
    public, _ = split_task_prompt(current.task_prompt)
    prompts = []

    def verify(prompt):
        prompts.append(prompt)
        assert len(prompt) <= budget and public in prompt
        assert 'prior_receipt_ids:' in prompt and 'lowercase SHA256' in prompt
        assert 'canonical_evaluations[].artifact_id' in prompt
        assert '[] if none' in prompt
        assert 'Never use/hash' in prompt and 'evaluation_id (evaluation:...)' in prompt

    def first(prompt):
        verify(prompt)
        assert not current.research_cells_context
        if budget == 64000:
            assert ORDINARY_ID in prompt
        else:
            assert ORDINARY_ID in prompt or 'recent_evaluations_omitted' in prompt
        return decision(experimental_protocol=protocol())

    def close(prompt):
        verify(prompt)
        actual = current_receipt(prompt)
        design = protocol()
        design.update(kind='closure', budget_action='none', prior_receipt_ids=[actual])
        return finish(conclusion='inconclusive', experimental_protocol=design)

    episode = tools.execute(prepare(tools, current), current, LLM([first, close]), io=IO())
    assert len(prompts) == 2 and len(backend.calls) == 1 and costs == [.03, .03]
    assert not [step for step in episode['research_state']['steps']
                if step['status'] == 'proposal_invalid']
    tools.commit(episode)
    tools.load_state(json.loads(json.dumps(tools.state())))


def test_schema_and_repair_explain_actual_raw_id_failure_without_rewriting_evidence():
    schema = response_property()['properties']['prior_receipt_ids']
    assert re.fullmatch(schema['items']['pattern'], 'a' * 64)
    for ident in (ORDINARY_ID, 'p25', 'A' * 64, 'a' * 63):
        assert not re.fullmatch(schema['items']['pattern'], ident)
    assert '[] if none' in schema['description']
    value = {**protocol(), 'prior_receipt_ids': [ORDINARY_ID]}
    with pytest.raises(ValueError) as error:
        parse_protocol(value)
    assert 'canonical_evaluations[].artifact_id' in str(error.value)
    assert '[] if none' in str(error.value)
    assert value['prior_receipt_ids'] == [ORDINARY_ID]


def test_hash_format_alone_does_not_grant_citation_rights():
    action = decision(experimental_protocol={**protocol(), 'prior_receipt_ids': ['a' * 64]})
    with pytest.raises(ValueError, match='available in this research plan'):
        validate_dispatch(action, {}, 'f' * 64, default_limits={'cpu_seconds': 20})
    plan = {'shared_observations': {'canonical_evaluation_index': [{'artifact_id': 'a' * 64}]}}
    assert validate_dispatch(action, plan, 'f' * 64, default_limits={'cpu_seconds': 20})['declared']['prior_receipt_ids'] == ['a' * 64]
