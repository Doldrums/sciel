"""Exact planner preflight on both current execution paths, without a provider."""
from copy import deepcopy
import hashlib

import pytest

from test_research_tool_controller import IO, LLM, prepare, setup, snapshot


class CountedLLM(LLM):
    local_inference = True
    max_tokens = 1536
    prompt_tokenizer_identity = {'fixture': 'utf8-bytes/1'}

    def __init__(self):
        super().__init__([{'action': 'skip'}])
        self.measured = []
        self.options = []

    def count_prompt_tokens(self, prompt):
        self.measured.append(prompt)
        return len(prompt.encode('utf-8'))

    def generate(self, prompt, model, temperature=None, **options):
        self.options.append(deepcopy(options))
        return super().generate(prompt, model, temperature, **options)


@pytest.mark.parametrize('steps', [1, 2])
def test_complete_oversize_planner_prompt_is_retained_without_dispatch(tmp_path, steps):
    tools, backend, events, costs = setup(tmp_path, max_scientific_steps=steps,
        max_requests=1, max_tokens=3072, context_tokens=4096)
    source = snapshot()
    llm, io = CountedLLM(), IO()
    result = tools.execute(prepare(tools, source), source, llm, io=io)
    assert len(llm.measured) == 1
    assert llm.prompts == llm.options == backend.calls == costs == []
    assert not any(e['phase'] == 'llm_start' for e in events)
    skipped = [e for e in events if e['phase'] == 'llm_skipped']
    assert len(skipped) == 1
    artifact = tools._read_artifact(skipped[0]['receipt_id'], 'proposal_budget')
    assert artifact['prompt'] == llm.measured[0]
    assert artifact['generation_dispatch'] == 'not_dispatched'
    assert artifact['generation_returned'] is False
    assert artifact['current_call_cost_unknown'] is False
    facts = artifact['token_budget']
    assert facts['prompt_sha256'] == hashlib.sha256(artifact['prompt'].encode()).hexdigest()
    assert facts['reserved_output_tokens'] == 3072
    assert facts['total_reserved_tokens'] > 4096 and facts['fits'] is False
    assert result['tool_ids'] == result['receipt_ids'] == []
    assert 'context_token_limit' in result['context']


@pytest.mark.parametrize('steps', [1, 2])
def test_fitting_prompt_dispatches_exact_measured_bytes_and_explicit_output_cap(tmp_path, steps):
    tools, backend, events, costs = setup(tmp_path, max_scientific_steps=steps,
        max_requests=1, max_tokens=3072, context_tokens=65536)
    source, llm = snapshot(), CountedLLM()
    tools.execute(prepare(tools, source), source, llm, io=IO())
    assert llm.measured == llm.prompts and len(llm.prompts) == 1
    assert llm.options[0]['max_tokens'] == 3072
    assert llm.options[0]['response_format']['json_schema']['strict'] is True
    assert costs == [.03] and backend.calls == []
    assert [e['token_budget']['fits'] for e in events if e['phase'] == 'prompt_budget'] == [True]
    assert not any(e['phase'] == 'llm_skipped' for e in events)
