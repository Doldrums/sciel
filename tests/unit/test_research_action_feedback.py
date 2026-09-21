"""Real malformed-action shapes identify their field and resource arithmetic."""
import json
from types import SimpleNamespace

import pytest

from sciel.research_tools.config import validate_config
from sciel.research_tools.controller import _proposal_failure, parse_action
from sciel.research_tools.experiment_protocol import validate_dispatch
from test_research_experimental_protocol import comparison
from test_research_tool_controller import IO, LLM, create, prepare, setup, snapshot
from test_sequential_research_episode import decision


@pytest.mark.parametrize('arguments', ['', '{', '{"x":NaN}', '{"x":1,"x":2}'])
def test_nested_argument_failure_is_not_outer_provider_truncation(arguments):
    text = json.dumps(decision(arguments_json=arguments))
    with pytest.raises(ValueError, match='arguments_json') as caught:
        parse_action(text, validate_config())
    assert not isinstance(caught.value, json.JSONDecodeError)
    response = SimpleNamespace(text=text, finish_reason='length')
    feedback = _proposal_failure(caught.value, response, 'a' * 64)
    assert 'arguments_json' in feedback['reason'] and 'use "{}"' in feedback['reason']
    assert 'partial proposal' not in feedback['reason']


def test_outer_truncation_keeps_provider_diagnostic():
    with pytest.raises(json.JSONDecodeError) as caught:
        parse_action('{"action":', validate_config())
    feedback = _proposal_failure(caught.value, SimpleNamespace(text='{"action":', finish_reason='length'), 'a' * 64)
    assert 'partial proposal' in feedback['reason']


def test_nested_field_feedback_reaches_paid_repair_before_executor(tmp_path):
    tools, executor, _, charges = setup(tmp_path, max_requests=2)
    malformed = {**create(), 'arguments_json': ''}

    def repair(prompt):
        assert 'arguments_json must encode a valid finite JSON object' in prompt
        assert not executor.calls
        return create()

    current = snapshot()
    episode = tools.execute(prepare(tools, current), current, LLM([malformed, repair]), io=IO())
    assert len(executor.calls) == 1 and len(charges) == 2
    assert sum(charges) == pytest.approx(.06)


def test_budget_feedback_identifies_oracle_allocation_without_weakening_gate():
    design = comparison()
    design['cpu_seconds_per_arm'] = 8
    action = decision(experimental_protocol=design)
    with pytest.raises(ValueError) as caught:
        validate_dispatch(action, {}, 'f' * 64, default_limits={'cpu_seconds': 20})
    message = str(caught.value)
    assert 'requested 20 CPU seconds; required 32' in message
    assert '2 arms × 2 seeds × 8 CPU seconds per arm/seed' in message
    assert validate_dispatch(action, {}, 'f' * 64, default_limits={'cpu_seconds': 32})['declared_cpu_seconds'] == 32
