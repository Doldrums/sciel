"""Unknown returned charges stop legacy generation and postmortems locally."""
from unittest.mock import Mock

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.components.tail import extract_lesson
from sciel.engine.llm.client import LLMResponse
from sciel.engine.types import Snapshot
from sciel.engine.worker import run_worker


def response(cost, text=""):
    return LLMResponse(text, "fake", cost, 1, 1)


@pytest.mark.parametrize("cost", [None, float("nan"), float("inf"), -.1, True])
def test_worker_keeps_prior_known_cost_and_stops_before_evaluation_or_retry(cost):
    llm = Mock()
    llm.generate.side_effect = [response(.125), response(cost, "```python\nx = 1\n```")]
    evaluate = Mock()
    snapshot = Snapshot(0, "full_rewrite", "fake", 1, "parent", "x = 0\n", 0)
    with pytest.raises(CampaignStopped) as caught:
        run_worker("full_rewrite", snapshot, llm, evaluate, dup_cap=4)
    assert caught.value.known_cost == .125
    assert caught.value.current_call_cost_unknown is True
    assert caught.value.stage == "generation"
    assert llm.generate.call_count == 2
    evaluate.assert_not_called()


def test_worker_accepts_a_known_zero_charge():
    llm, evaluate = Mock(), Mock()
    llm.generate.return_value = response(0)
    snapshot = Snapshot(0, "full_rewrite", "fake", 1, "parent", "x = 0\n", 0)
    result = run_worker("full_rewrite", snapshot, llm, evaluate, dup_cap=1)
    assert result.reason == "empty_response" and result.cost == 0
    evaluate.assert_not_called()


@pytest.mark.parametrize("cost", [None, float("nan"), -.1])
def test_postmortem_unknown_cost_never_returns_a_lesson_or_zero(cost):
    llm = Mock()
    llm.generate.return_value = response(cost, "LESSON: example\nMECHANISM: example")
    with pytest.raises(CampaignStopped) as caught:
        extract_lesson(llm, "fake", "parent", "child", "mechanism")
    assert caught.value.known_cost == 0
    assert caught.value.current_call_cost_unknown is True
    assert caught.value.stage == "postmortem"
    assert llm.generate.call_count == 1


def test_postmortem_preserves_paid_unparseable_response():
    llm = Mock()
    llm.generate.return_value = response(.125, "no lesson")
    assert extract_lesson(llm, "fake", "parent", "child", "mechanism") == (None, .125)


def test_postmortem_does_not_swallow_explicit_campaign_stop():
    llm = Mock()
    llm.generate.side_effect = CampaignStopped("budget exhausted")
    with pytest.raises(CampaignStopped, match="budget exhausted"):
        extract_lesson(llm, "fake", "parent", "child", "mechanism")
