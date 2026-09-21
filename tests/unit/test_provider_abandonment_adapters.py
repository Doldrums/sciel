"""Authenticated unknown requests terminate only their work; local stubs only."""
from copy import deepcopy

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine.llm.client import LLMError
from sciel.engine.provider_abandonment import AbandonedProviderRequest
from sciel.engine.types import NoChild
from sciel.engine.worker import run_worker

from test_conclusion_format_repair import setup as conclusion_setup, response as conclusion_text
from test_research_aux_failures import setup as aux_setup, response, worker_setup


RECEIPT = {
    'recovery_id': 'recovery-fixture', 'job_call': 'C:r000001a1:0',
    'request_sha256': 'a' * 64, 'receipt_sha256': 'b' * 64,
    'unknown_cost': True, 'retry': False,
}
DIAGNOSTIC = {
    'stage': 'request', 'reason': 'http_error',
    'cause_type': 'RateLimitError', 'http_status': 429,
}


def abandoned(job_call='C:r000001a1:0'):
    return AbandonedProviderRequest({**RECEIPT, 'job_call': job_call}, DIAGNOSTIC)


@pytest.mark.parametrize('kind', ['conclusion', 'reflection'])
def test_abandoned_aux_preserves_unknown_and_consumes_once_without_stopping_siblings(kind):
    controller, model, rows, charges = aux_setup([])
    request = controller.prepare_aux(kind, 'failed physical prompt', 'r1')
    sibling = controller.prepare_aux(kind, 'known sibling prompt', 'r2')
    error = abandoned()
    assert controller.consume_aux(request, error=error) is None
    assert controller.stopped_reason is None and charges == []
    assert controller.aux_cost == 0 and controller.aux_calls == 2
    terminal = rows[-2:]
    assert [row['phase'] for row in terminal] == ['aux_abandoned', 'aux_skipped']
    assert terminal[-1]['reason'] == 'provider_request_abandoned'
    for row in terminal:
        assert row['cost'] is None and row['current_call_cost_unknown'] is True
        assert row['generation_dispatch'] == 'unknown'
        assert row['generation_returned'] is False
        assert row['provider_abandonment'] == RECEIPT
        assert row['llm_error'] == DIAGNOSTIC
        assert row['call_index'] == request['call_index'] and row['reference'] == 'r1'
    error.receipt['unknown_cost'] = False
    assert terminal[-1]['provider_abandonment']['unknown_cost'] is True
    with pytest.raises(ValueError, match='already consumed'):
        controller.consume_aux(request, error=abandoned())
    assert controller.consume_aux(sibling, response=response(text='known sibling')) == 'known sibling'
    assert charges == [.03] and controller.aux_cost == .03 and not controller._aux_requests
    assert model.calls == []
    next_request = controller.prepare_aux(kind, 'new logical request', 'r3')
    assert next_request['call_index'] == 3


@pytest.mark.parametrize('kind', ['conclusion', 'reflection'])
def test_plain_aux_provider_error_remains_fail_closed(kind):
    controller, _, rows, charges = aux_setup([])
    request = controller.prepare_aux(kind, 'ordinary error', 'r1')
    with pytest.raises(CampaignStopped):
        controller.consume_aux(request, error=LLMError(cause=ConnectionError()))
    assert controller.stopped_reason and charges == []
    assert rows[-1]['phase'] == 'aux_stopped'
    assert rows[-1]['current_call_cost_unknown'] is True
    assert not any(row['phase'] == 'aux_abandoned' for row in rows)


@pytest.mark.parametrize('during_repair', [False, True])
def test_abandoned_conclusion_cannot_create_lesson_or_repeat_request_after_restore(during_repair):
    items = ([conclusion_text(too_long=True)] if during_repair else []) + [abandoned()]
    controller, model, rows, charges = conclusion_setup(items)
    original = deepcopy(controller.knowledge.experiments)
    controller.complete_step()
    assert controller.conclusions['r000001a1']['status'] == ('invalid' if during_repair else 'skipped')
    assert not controller.knowledge.lessons and controller.knowledge.experiments == original
    assert controller.queue == [] and controller.stopped_reason is None
    assert len(model.calls) == controller.aux_calls == 1 + during_repair
    assert charges == ([.03] if during_repair else [])
    assert controller.aux_cost == (.03 if during_repair else 0)
    assert sum(row['phase'] == 'aux_abandoned' for row in rows) == 1
    assert not any(row['phase'] == 'aux_stopped' for row in rows)
    restored, fresh, _, restored_charges = conclusion_setup([])
    restored.queue = []
    restored.load_state(controller.state())
    restored.complete_step()
    assert restored.state() == controller.state()
    assert not fresh.calls and not restored_charges


def test_abandoned_reflection_preserves_existing_lessons_and_cadence_without_fake_success():
    controller, model, rows, charges = conclusion_setup(
        [conclusion_text(lessons=True), abandoned('R:1:0')], reflection_interval=1)
    controller.complete_step()
    assert controller.conclusions['r000001a1']['status'] == 'parsed'
    assert len(controller.knowledge.lessons) == 1
    before = deepcopy(controller.knowledge.lessons)
    reflection = next(row for row in rows if row['phase'] == 'reflection')
    assert reflection['status'] == 'skipped' and reflection['applied'] is None
    assert reflection['changed'] is False
    assert controller.reflected_through == 1 and controller.reflections == 0
    assert controller.stopped_reason is None and len(model.calls) == 2 and charges == [.03]
    controller.finish_conclusions()
    assert len(model.calls) == 2 and charges == [.03] and controller.knowledge.lessons == before
    restored, fresh, _, new_charges = conclusion_setup([], reflection_interval=1)
    restored.queue = []
    restored.load_state(controller.state())
    restored.finish_conclusions()
    assert restored.state() == controller.state() and not fresh.calls and not new_charges


@pytest.mark.parametrize('research_on', [False, True])
@pytest.mark.parametrize('prior_response', [False, True])
def test_abandoned_generation_returns_only_prior_known_subtotal_without_eval_or_hypothesis(
        research_on, prior_response):
    research, model, rows, charges, snapshot = worker_setup(research_on=research_on)
    first = model.responses[0]
    model.responses = ([first] if prior_response else []) + [abandoned('G:1:1')]
    result = run_worker('diff', snapshot, model, lambda _: pytest.fail('must not evaluate'),
        dup_cap=4, research=research if research_on else None)
    assert isinstance(result, NoChild) and result.reason == 'llm_failed'
    assert len(model.calls) == 1 + prior_response
    assert result.cost == (.01 if prior_response else 0)
    assert charges == []  # Coordinator commit owns this known subtotal exactly once.
    assert result.meta['current_call_cost_unknown'] is True
    assert result.meta['generation_dispatch'] == 'unknown'
    assert result.meta['generation_returned'] is False
    assert result.meta['provider_abandonment'] == {**RECEIPT, 'job_call': 'G:1:1'}
    assert result.meta['llm_error'] == DIAGNOSTIC
    attempt = result.meta['validation_trace']['attempts'][-1]
    assert attempt['cost'] is None and attempt['current_call_cost_unknown'] is True
    assert attempt['disposition'] == 'llm_failed' and attempt['decisions'] == []
    if not research_on:
        return
    assert research.attempts == research.outcomes == 1 + prior_response
    assert not research.pending and not research.queue and not research.knowledge.lessons
    assert research.stopped_reason is None
    record = research.knowledge.experiments[-1]
    assert record['status'] == 'generation_stopped' and record['generation_dispatch'] == 'unknown'
    assert record['hypothesis'] is record['score'] is record['actual_delta'] is None
    assert record['generation_cost'] is record['evaluation'] is None
    assert any(row['phase'] == 'generation_abandoned' for row in rows)
    research.complete_step()
    fresh, fresh_model, _, new_charges = aux_setup([])
    fresh.load_state(research.state())
    fresh.complete_step()
    assert fresh.state() == research.state() and not fresh_model.calls and not new_charges


@pytest.mark.parametrize('research_on', [False, True])
def test_plain_generation_error_does_not_receive_abandonment_permission(research_on):
    research, model, rows, charges, snapshot = worker_setup(research_on=research_on)
    model.responses[1] = LLMError(cause=ConnectionError())
    with pytest.raises(CampaignStopped):
        run_worker('diff', snapshot, model, lambda _: pytest.fail('must not evaluate'),
            dup_cap=4, research=research if research_on else None)
    assert len(model.calls) == 2 and charges == ([.01] if research_on else [])
    assert not any(row['phase'] == 'generation_abandoned' for row in rows)


def test_abandoned_aux_journal_failure_is_not_swallowed_as_a_successful_skip():
    controller, _, _, charges = aux_setup([], fail_journal=('aux_abandoned',))
    request = controller.prepare_aux('conclusion', 'physical request', 'r1')
    with pytest.raises(OSError):
        controller.consume_aux(request, error=abandoned())
    assert charges == [] and not controller._aux_requests
    with pytest.raises(ValueError, match='already consumed'):
        controller.consume_aux(request, error=abandoned())
