"""Missing provider usage stays unknown; all transport is a local fake."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.engine.llm.client import EnsembleLLM, known_cost


def generate(usage):
    llm = EnsembleLLM.__new__(EnsembleLLM)
    llm.temperature, llm.max_tokens, llm.max_retries = .7, 100, 1
    llm._client = Mock()
    llm._client.chat.completions.create.return_value = SimpleNamespace(
        usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content="returned text"))])
    result = llm.generate("prompt", "gpt-4.1-mini")
    assert llm._client.chat.completions.create.call_count == 1
    assert result.text == "returned text"
    return result


@pytest.mark.parametrize("usage", [None, SimpleNamespace(),
    SimpleNamespace(prompt_tokens=12), SimpleNamespace(completion_tokens=14),
    SimpleNamespace(prompt_tokens=12, completion_tokens=None),
    SimpleNamespace(prompt_tokens=True, completion_tokens=14),
    SimpleNamespace(prompt_tokens=-1, completion_tokens=14),
    SimpleNamespace(prompt_tokens=12, completion_tokens="14"),
    SimpleNamespace(prompt_tokens=12, completion_tokens=float("nan"))])
def test_incomplete_or_invalid_usage_is_unknown_and_cannot_be_accumulated(usage):
    response = generate(usage)
    assert response.cost is None
    with pytest.raises(ValueError, match="unknown or invalid"):
        known_cost(response)


def test_absent_token_counts_are_not_fabricated_as_zero():
    response = generate(None)
    assert response.prompt_tokens is None and response.completion_tokens is None


@pytest.mark.parametrize("billed", [0, .123])
def test_reported_cost_remains_authoritative_without_token_counts(billed):
    response = generate(SimpleNamespace(cost=billed))
    assert known_cost(response) == billed
    assert response.prompt_tokens is None and response.completion_tokens is None


@pytest.mark.parametrize("billed", [None, -1, True, float("nan"), float("inf"), "unknown"])
def test_complete_counts_allow_finite_estimate_when_provider_cost_is_unusable(billed):
    response = generate(SimpleNamespace(cost=billed, prompt_tokens=1000, completion_tokens=2000))
    assert known_cost(response) == pytest.approx(.0036)
    assert response.prompt_tokens == 1000 and response.completion_tokens == 2000


def test_measured_zero_usage_is_a_known_zero_cost():
    response = generate(SimpleNamespace(prompt_tokens=0, completion_tokens=0))
    assert known_cost(response) == 0


@pytest.mark.parametrize("cost", [None, True, -.1, float("nan"), float("inf"), 10**1000])
def test_known_cost_rejects_nonfinite_or_unknown_counters(cost):
    with pytest.raises(ValueError, match="unknown or invalid"):
        known_cost(SimpleNamespace(cost=cost))
