"""Safe auxiliary failure receipts; SDK requests are local mocks only."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.config import apply_defaults
from sciel.engine.llm.client import EnsembleLLM, LLMError, llm_error_metadata
from sciel.engine.research import ResearchController


SECRET = 'private-provider-body-Bearer-secret-token'


class RateLimitError(Exception):
    status_code = 429

    def __str__(self):
        raise AssertionError('exception text must not be read')

    @property
    def body(self):
        raise AssertionError('provider body must not be read')

    @property
    def response(self):
        raise AssertionError('response and headers must not be read')


def model(*, error=None, response=None):
    llm = EnsembleLLM.__new__(EnsembleLLM)
    llm.models, llm.temperature, llm.max_tokens, llm.max_retries = ['fixture'], .7, 100, 1
    llm._client = Mock()
    request = llm._client.chat.completions.create
    request.side_effect, request.return_value = error, response
    return llm, request


def controller(llm, *, fail_stop=False):
    events, charges = [], []
    def emit(**row):
        if fail_stop and row['phase'] == 'aux_stopped':
            raise OSError(SECRET)
        events.append(copy.deepcopy(row))
    cfg = apply_defaults({'research': {'enabled': True, 'reflection_interval': 0}})['research']
    return (ResearchController(cfg, llm, SimpleNamespace(research=emit), charges.append),
            events, charges)


@pytest.mark.parametrize(('error', 'reason', 'cause_type', 'status'), [
    (RateLimitError(SECRET), 'http_error', 'RateLimitError', 429),
    (TimeoutError(SECRET), 'timeout', 'TimeoutError', None),
    (ConnectionError(SECRET), 'connection_error', 'ConnectionError', None),
    (RuntimeError(SECRET), 'request_failed', 'RuntimeError', None),
])
def test_auxiliary_client_error_is_sanitized_unknown_cost_and_never_retried(error, reason, cause_type, status):
    llm, request = model(error=error)
    research, events, charges = controller(llm)
    with pytest.raises(CampaignStopped, match='auxiliary request stopped: LLMError') as stopped:
        research._aux('conclusion', 'private prompt', 'r1')
    receipt = events[-1]
    assert receipt['phase'] == 'aux_stopped'
    assert receipt['call_index'] == 1 and receipt['stage'] == 'request'
    assert receipt['generation_returned'] is False and receipt['generation_dispatch'] == 'unknown'
    assert receipt['cost'] is None and receipt['current_call_cost_unknown'] is True
    expected = dict(stage='request', reason=reason, cause_type=cause_type)
    if status is not None:
        expected['http_status'] = status
    assert receipt['llm_error'] == expected
    assert SECRET not in json.dumps(events) and SECRET not in str(stopped.value)
    assert 'private prompt' not in json.dumps(events)
    assert charges == [] and research.aux_cost == 0
    with pytest.raises(CampaignStopped):
        research._aux('reflection', 'another prompt', 'r2')
    assert request.call_count == 1 and research.aux_calls == 1


@pytest.mark.parametrize(('response', 'reason'), [
    (SimpleNamespace(choices=[]), 'empty_choices'),
    (SimpleNamespace(choices=[SimpleNamespace(message=None)]), 'malformed_response'),
])
def test_malformed_provider_response_has_response_stage_without_claiming_charge(response, reason):
    llm, request = model(response=response)
    research, events, charges = controller(llm)
    with pytest.raises(CampaignStopped):
        research._aux('conclusion', 'fixture', 'r1')
    receipt = events[-1]
    assert receipt['llm_error']['stage'] == 'response'
    assert receipt['llm_error']['reason'] == reason
    # The provider returned, but generate() did not return a cost-bearing response.
    assert receipt['generation_returned'] is False
    assert receipt['current_call_cost_unknown'] is True and receipt['cost'] is None
    assert request.call_count == 1 and charges == []


@pytest.mark.parametrize('status', [True, '429', -1, 600, None])
def test_non_http_status_values_are_never_serialized(status):
    error = RuntimeError(SECRET)
    error.status_code = status
    assert llm_error_metadata(LLMError(cause=error)) == dict(
        stage='request', reason='request_failed', cause_type='RuntimeError')


def test_unknown_exception_class_and_untrusted_metadata_cannot_add_secrets():
    error = type(SECRET, (Exception,), {})(SECRET)
    assert llm_error_metadata(LLMError(cause=error))['cause_type'] == 'OtherError'
    wrapped = LLMError(SECRET)
    wrapped._metadata = {'stage': SECRET, 'reason': [SECRET], 'cause_type': SECRET,
                         'http_status': SECRET, 'body': SECRET, 'request_id': SECRET}
    assert llm_error_metadata(wrapped) == dict(
        stage='request', reason='request_failed', cause_type='OtherError')


def test_request_not_dispatched_is_explicitly_zero_cost_but_still_stops():
    llm = SimpleNamespace(models=['fixture'], max_retries=1,
                          generate=Mock(side_effect=RequestNotDispatched(SECRET)))
    research, events, charges = controller(llm)
    with pytest.raises(RequestNotDispatched):
        research._aux('conclusion', 'fixture', 'r1')
    receipt = events[-1]
    assert receipt['cost'] == 0 and receipt['current_call_cost_unknown'] is False
    assert receipt['generation_dispatch'] == 'not_dispatched'
    assert SECRET not in json.dumps(events) and charges == []
    with pytest.raises(CampaignStopped):
        research._aux('conclusion', 'fixture', 'r1')
    assert llm.generate.call_count == 1


def test_failed_stop_receipt_still_fails_closed_without_leaking_or_retrying():
    llm, request = model(error=TimeoutError(SECRET))
    research, _, charges = controller(llm, fail_stop=True)
    with pytest.raises(CampaignStopped, match='stop persistence failed: OSError') as error:
        research._aux('conclusion', 'fixture', 'r1')
    assert SECRET not in str(error.value) and charges == []
    with pytest.raises(CampaignStopped):
        research._aux('conclusion', 'fixture', 'r1')
    assert request.call_count == 1
