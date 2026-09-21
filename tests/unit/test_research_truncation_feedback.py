"""Provider finish metadata informs bounded repair; no provider/source executes."""
import json

import pytest

from sciel.engine.llm.client import LLMResponse
from test_research_tool_controller import IO, prepare, snapshot
from test_research_tool_result_flow import action, controller, result
from test_sequential_research_episode import decision


class RecordedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, model, temperature=None, **kwargs):
        self.prompts.append(prompt)
        text, finish_reason, cost = self.responses.pop(0)
        return LLMResponse(text if isinstance(text, str) else json.dumps(text),
            model, cost, 12, 8, finish_reason=finish_reason)


def setup(tmp_path, sequential, outputs, **config):
    return controller(tmp_path, outputs, max_scientific_steps=3 if sequential else 1,
        yield_after_success=sequential, **config)


def proposal_artifacts(tools):
    return [row['payload'] for path in tools.path.glob('*.json')
            if (row := json.loads(path.read_text()))['kind'] == 'proposal']


@pytest.mark.parametrize('sequential', [False, True])
@pytest.mark.parametrize('finish_reason', ['length', 'max_tokens'])
def test_output_limit_feedback_preserves_full_response_and_exact_cost(tmp_path, sequential, finish_reason):
    tools, backend, _, costs = setup(tmp_path, sequential, [result({'gain':0})])
    partial = '{"action":"create","code":"def run(payload):\\n'
    fixed = decision() if sequential else action()
    llm = RecordedLLM([(partial, finish_reason, .047), (fixed, 'stop', .031)])
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    prompt = llm.prompts[1]
    for text in ('finish_reason='+finish_reason, 'partial proposal was not executed',
                 'shorter complete JSON object', 'Do not append', 'Unterminated string'):
        assert text in prompt
    assert len(backend.calls) == 1 and len(llm.prompts) == 2
    assert costs == [.047, .031]
    assert episode['known_llm_cost'] == pytest.approx(.078)
    original = next(row for row in proposal_artifacts(tools) if row['response']['text'] == partial)
    assert original['response']['finish_reason'] == finish_reason
    assert original['response']['cost'] == .047
    assert all(row['status'] == 'ok' for row in backend.receipts)
    if sequential:
        assert [s['status'] for s in episode['research_state']['steps']] == ['proposal_invalid','success']


@pytest.mark.parametrize('sequential', [False, True])
def test_repeated_truncation_keeps_existing_retry_bound_and_charge(tmp_path, sequential):
    tools, backend, _, costs = setup(tmp_path, sequential, [], max_requests=2)
    llm = RecordedLLM([('{"action":', 'length', .07)]*2)
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    assert len(llm.prompts) == len(proposal_artifacts(tools)) == 2
    assert backend.calls == [] and costs == [.07, .07]
    assert episode['known_llm_cost'] == pytest.approx(.14)
    if sequential:
        assert episode['research_state']['status'] == 'repair_exhausted'
        assert episode['research_state']['scientific_steps'] == 0


@pytest.mark.parametrize('sequential', [False, True])
def test_valid_json_is_accepted_even_with_length_finish_reason(tmp_path, sequential):
    tools, backend, _, costs = setup(tmp_path, sequential, [result({'gain':0})])
    llm = RecordedLLM([(decision() if sequential else action(), 'length', .03)])
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    assert len(backend.calls) == len(llm.prompts) == 1 and costs == [.03]
    assert episode['invocations'][0]['result']['status'] == 'success'


@pytest.mark.parametrize('sequential', [False, True])
@pytest.mark.parametrize('finish_reason', ['stop', None, 'error'])
def test_empty_provider_error_is_distinct_from_truncation(tmp_path, sequential, finish_reason):
    tools, backend, _, costs = setup(tmp_path, sequential, [result({'gain':0})])
    llm = RecordedLLM([('', finish_reason, 0),
        (decision() if sequential else action(), 'stop', .03)])
    current = snapshot()
    tools.execute(prepare(tools, current), current, llm, io=IO())
    prompt = llm.prompts[1]
    assert 'Expecting value' in prompt and 'shorter complete JSON' not in prompt
    assert ('Provider finish_reason=error; empty response' in prompt) is (finish_reason == 'error')
    assert len(backend.calls) == 1 and costs == [0, .03]
