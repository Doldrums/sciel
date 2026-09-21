"""Reporting guidance reaches the next planner without altering saved measurements."""
from copy import deepcopy
import json

import pytest

from sciel.research_tools.experiment_protocol import response_property
from test_research_experimental_protocol import comparison, protocol
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish


@pytest.mark.parametrize('budget', [8000, 64000])
def test_next_planner_gets_exact_reporting_guidance_and_honest_invalid_report(tmp_path, budget):
    design = comparison()
    design['conditions_json'] = '{"resource_cap_enforced":true}'
    rows = [dict(arm_id=arm['id'], seed=seed, cpu_seconds=1.5,
                 best_result={'measured_value': 7}, saved_state={'seed': seed},
                 conditions={'enforces_resource_cap': True})
            for arm in design['arms'] for seed in design['matched_seeds']]
    # A real positive overshoot and a renamed metadata key must remain visible,
    # not be normalized into a compliant claim by this guidance-only change.
    rows[0]['cpu_seconds'] = design['cpu_seconds_per_arm'] + .00025
    observations = {'comparison_results': rows}
    frozen = deepcopy(observations)
    tools, backend, events, costs = build(tmp_path, [result(observations)],
        experimental_protocol=True, shared_observations=True, max_prompt_chars=budget)
    current = snapshot()
    next_prompt = []

    def close(prompt):
        next_prompt.append(prompt)
        assert len(prompt) <= budget
        assert ('Row conditions must equal parsed conditions_json' in prompt
                or 'conditions=conditions_json' in prompt)
        assert 'actual changes' in prompt and 'diagnostics' in prompt and 'outside' in prompt
        assert 'overshoot' in prompt and 'no tolerance' in prompt
        assert ('Never round down/clamp CPU' in prompt or 'no tolerance/round/clamp' in prompt)
        assert 'reported conditions differ' in prompt
        assert 'reported CPU missing, invalid or above declared cap' in prompt
        closure = protocol()
        closure.update(kind='closure', budget_action='none')
        return finish(conclusion='inconclusive', experimental_protocol=closure)

    episode = tools.execute(prepare(tools, current), current,
        LLM([decision(experimental_protocol=design), close]), io=IO())
    assert len(next_prompt) == 1 and len(backend.calls) == 1 and costs == [.03, .03]
    assert observations == frozen
    step = episode['research_state']['steps'][0]
    assert step['status'] == 'success'
    assessment = step['comparison_report_assessment']
    assert assessment['report_shape'] == 'invalid'
    assert assessment['adherence'] == 'tool_reported_unverified'
    assert set(assessment['problems']) == {
        'reported conditions differ', 'reported CPU missing, invalid or above declared cap'}
    receipt_id = episode['receipt_ids'][0]
    assert receipt_id in next_prompt[0]
    saved = tools._read_artifact(receipt_id, 'invocation')
    assert saved['receipt']['output']['observations'] == frozen
    assert 'candidate_submission' not in episode
    tools.commit(episode)
    state = deepcopy(tools.state())
    tools.load_state(json.loads(json.dumps(state)))
    assert tools.state() == state and costs == [.03, .03]


def test_conditions_schema_describes_exact_keys_without_blind_echo():
    description = response_property()['properties']['conditions_json']['description']
    assert 'compared by JSON equality' in description
    assert 'booleans differ from numbers' in description
    assert 'finite 1 equals 1.0' in description
    assert 'report changes truthfully' in description
    assert 'extra diagnostics outside conditions' in description
