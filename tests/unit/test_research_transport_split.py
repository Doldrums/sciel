"""Coordinator/transport boundaries; local receipts only, no provider execution."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import threading
from types import SimpleNamespace as NS

import pytest

from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.engine.research import digest

from test_conclusion_format_repair import setup, response as conclusion_text
from test_research_aux_failures import setup as aux_setup, response
from test_research_feedback_retention import full_window
from sciel.engine.prompt.build import build


def two_jobs(**options):
    controller, model, events, charges = setup([], **options)
    record = deepcopy(controller.knowledge.experiments[0])
    record['id'] = 'r000002a1'
    controller.knowledge.record(record)
    controller.queue = deepcopy(controller.knowledge.experiments)
    controller.attempts = controller.outcomes = controller.valid_hypotheses = 2
    return controller, model, events, charges


def text_for(job, *, lesson=True):
    value = json.loads(conclusion_text(lessons=lesson))
    if lesson:
        value['lessons'][0]['rule'] = 'Measured regression for ' + job['record']['id']
        value['lessons'][0]['evidence_ids'] = [job['record']['id']]
    return json.dumps(value)


def receipt(text, *, cost=.03, finish='stop'):
    return NS(text=text, cost=cost, model='fixture', prompt_tokens=10,
              completion_tokens=10, finish_reason=finish, refusal=None)


def test_preparation_freezes_source_evidence_and_options_without_dispatch_or_charge():
    c, model, events, charges = two_jobs()
    jobs = c.prepare_conclusions()
    assert c.queue == [] and len(jobs) == 2 and model.calls == []
    assert charges == [] and c.aux_calls == 0 and not c.conclusions
    assert [j['record']['id'] for j in jobs] == ['r000001a1', 'r000002a1']
    original = jobs[0]['evidence']
    jobs[0]['evidence']['hypothesis']['text'] = 'changed detached copy'
    c.knowledge._experiments[0]['hypothesis']['text'] = 'changed live knowledge'
    assert jobs[0]['evidence'] == original
    with pytest.raises(TypeError):
        jobs[0]['prompt'] = 'changed'
    with pytest.raises(FrozenInstanceError):
        jobs[0].payload = '{}'
    request = c.prepare_aux('conclusion', jobs[0]['prompt'], jobs[0]['record']['id'],
        conclusion_evidence=jobs[0]['evidence'])
    original_options = request['options']
    request['options']['response_format']['json_schema']['strict'] = False
    assert request['options'] == original_options
    assert request['prompt_sha256'] == digest(jobs[0]['prompt'])
    assert charges == [] and model.calls == [] and c.aux_calls == 1
    assert sum(r['phase'] == 'conclusion_reference_catalog' for r in events) == 2


def test_parallel_transports_do_not_mutate_state_and_reverse_receipts_keep_call_binding():
    c, _, events, charges = two_jobs(reflection_interval=1)
    jobs = c.prepare_conclusions()
    requests = [c.prepare_aux('conclusion', j['prompt'], j['record']['id'],
        conclusion_evidence=j['evidence']) for j in jobs]
    before = deepcopy(events)
    coordinator_thread = threading.get_ident()

    def transport(index):
        assert threading.get_ident() != coordinator_thread
        return receipt(text_for(jobs[index]), cost=.01 * (index + 1))

    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(pool.map(transport, range(2)))
    assert events == before and charges == [] and not c.conclusions
    texts = {}
    for index in (1, 0):
        texts[index] = c.consume_aux(requests[index], response=replies[index])
    results = [r for r in events if r['phase'] == 'aux_result']
    assert [(r['call_index'], r['reference'], r['cost']) for r in results] == [
        (2, 'r000002a1', .02), (1, 'r000001a1', .01)]
    assert charges == [.02, .01] and c.aux_cost == pytest.approx(.03)
    assert not c.conclusions and not c.knowledge.lessons

    for index, job in enumerate(jobs):
        result, correction = c.decode_conclusion(job, texts[index])
        assert result is not None and correction is None
        c.apply_conclusion(job, result, texts[index])
    assert [r['experiment_id'] for r in events if r['phase'] == 'conclusion'] == [
        'r000001a1', 'r000002a1']
    assert [l['evidence_ids'] for l in c.knowledge.lessons] == [
        ['r000001a1'], ['r000002a1']]
    assert [l['id'] for l in c.knowledge.lessons] == ['l000001', 'l000002']
    # Defer R without consuming its cadence; the wave's committed C is visible.
    c.finish_conclusions(reflect=False)
    assert c.reflected_through == 0 and not any(r['phase'] == 'reflection' for r in events)
    c.llm.generate = lambda *args, **kwargs: receipt(json.dumps(dict(lessons=[], retire_ids=[])))
    c.finish_conclusions()
    phases = [r['phase'] for r in events]
    assert phases.index('reflection_context') > max(i for i, p in enumerate(phases) if p == 'conclusion')
    assert c.reflected_through == 2 and phases.count('reflection') == 1
    c.finish_conclusions()
    assert [r['phase'] for r in events].count('reflection') == 1


@pytest.mark.parametrize('failure', [ConnectionError('offline fixture'), RequestNotDispatched('not sent')])
def test_stopped_sibling_does_not_erase_known_receipt_and_neither_can_be_consumed_twice(failure):
    c, _, rows, charges = aux_setup([])
    first = c.prepare_aux('conclusion', 'one', 'r1')
    second = c.prepare_aux('conclusion', 'two', 'r2')
    with pytest.raises(CampaignStopped):
        c.consume_aux(second, error=failure)
    stop = next(r for r in rows if r['phase'] == 'aux_stopped')
    assert stop['call_index'] == 2 and stop['reference'] == 'r2'
    assert stop['current_call_cost_unknown'] is (not isinstance(failure, RequestNotDispatched))
    assert c.consume_aux(first, response=response(text='known sibling')) == 'known sibling'
    assert charges == [.03] and c.aux_cost == .03
    assert c.stopped_reason and c._aux_requests == {}
    for request in (first, second):
        with pytest.raises(ValueError, match='already consumed'):
            c.consume_aux(request, response=response())
    with pytest.raises(CampaignStopped):
        c.prepare_aux('conclusion', 'new', 'r3')
    assert charges == [.03] and c.aux_calls == 2


def test_changed_request_rejected_before_charge_and_original_remains_consumable():
    c, _, _, charges = aux_setup([])
    request = c.prepare_aux('reflection', 'original', 4)
    altered = dict(request)
    altered['reference'] = 99
    with pytest.raises(ValueError, match='changed'):
        c.consume_aux(altered, response=response())
    assert charges == []
    assert c.consume_aux(request, response=response()) == '{}'
    assert charges == [.03]


def test_outstanding_requests_block_reflection_and_reserved_calls_obey_cap():
    c, _, rows, charges = aux_setup([])
    c.config['max_aux_calls'] = 1
    request = c.prepare_aux('conclusion', 'first', 'r1')
    assert c.prepare_aux('conclusion', 'second', 'r2') is None
    with pytest.raises(ValueError, match='before queued research'):
        c.finish_conclusions()
    assert rows[-1]['phase'] == 'aux_skipped' and charges == []
    c.consume_aux(request, response=response())
    c.finish_conclusions()
    assert c.aux_calls == 1 and charges == [.03]


def test_correction_uses_frozen_original_prompt_and_skipped_repair_preserves_invalid_text():
    c, _, rows, charges = setup([], max_aux_calls=1)
    job, = c.prepare_conclusions()
    request = c.prepare_aux('conclusion', job['prompt'], job['record']['id'],
        conclusion_evidence=job['evidence'])
    original = conclusion_text(too_long=True)
    text = c.consume_aux(request, response=receipt(original))
    result, repair_prompt = c.decode_conclusion(job, text, attempt=0)
    assert result is None and job['prompt'] in repair_prompt
    assert c.prepare_aux('conclusion', repair_prompt, job['record']['id'],
        repair_evidence=job['evidence']) is None
    c.apply_conclusion(job, result, text)
    c.finish_conclusions()
    assert c.conclusions[job['record']['id']]['status'] == 'invalid'
    assert charges == [.03] and len([r for r in rows if r['phase'] == 'conclusion_repair']) == 1
    with pytest.raises(ValueError, match='correction attempt'):
        c.decode_conclusion(job, original, attempt=2)


def test_truncated_response_is_charged_and_skipped_without_correction_or_partial_admission():
    c, _, rows, charges = setup([], incomplete_auxiliary='skip')
    job, = c.prepare_conclusions()
    request = c.prepare_aux('conclusion', job['prompt'], job['record']['id'],
        conclusion_evidence=job['evidence'])
    text = c.consume_aux(request, response=receipt(conclusion_text(lessons=True), finish='length'))
    assert text is None
    result, correction = c.decode_conclusion(job, text)
    assert result is correction is None
    c.apply_conclusion(job, result, text)
    c.finish_conclusions()
    assert c.conclusions[job['record']['id']]['status'] == 'skipped'
    assert not c.knowledge.lessons and charges == [.03]
    assert not any(r['phase'] == 'conclusion_repair' for r in rows)


def test_interleaved_frozen_generation_jobs_keep_their_own_feedback_after_eviction():
    c, _, _, events, first, source_id, _ = full_window(2)
    second = deepcopy(first)
    second.iteration_no += 1
    second.research_context['iteration'] += 1
    one, _ = c.begin(first, 0, 'full_rewrite', build('full_rewrite', first))
    two, _ = c.begin(second, 0, 'full_rewrite', build('full_rewrite', second))
    c.resolve(one, status='no_child')
    assert source_id not in c.conclusions
    retry_one, prompt_one = c.begin(first, 1, 'full_rewrite', build('full_rewrite', first))
    c.resolve(retry_one, status='no_child')
    c.resolve(two, status='no_child')
    retry_two, prompt_two = c.begin(second, 1, 'full_rewrite', build('full_rewrite', second))
    assert source_id in prompt_one and source_id in prompt_two
    attempts = [r for r in events if r['phase'] == 'attempt'
                and r['experiment_id'] in (one, two, retry_one, retry_two)]
    assert len(attempts) == 4
    assert all(r['conclusion_feedback'] == first.research_context['conclusion_feedback'] for r in attempts)
    # Same bytes in a new, unvalidated snapshot are not an authorization token.
    with pytest.raises(ValueError, match='conclusion feedback'):
        fresh = deepcopy(first)
        c.begin(fresh, 1, 'full_rewrite', build('full_rewrite', fresh))
    c.resolve(retry_two, status='no_child')
    c.finish_conclusions(reflect=False)
    assert c._feedback_delivery is None


@pytest.mark.parametrize('field', ['parent_id', 'parent_code', 'iteration_no'])
def test_validated_generation_context_cannot_move_to_another_parent_or_iteration(field):
    c, _, _, _, snapshot, _, _ = full_window(2)
    c.begin(snapshot, 0, 'full_rewrite', build('full_rewrite', snapshot))
    old = getattr(snapshot, field)
    setattr(snapshot, field, old + 1 if type(old) is int else old + ' changed')
    with pytest.raises(ValueError, match='conclusion feedback'):
        c.begin(snapshot, 1, 'full_rewrite', build('full_rewrite', snapshot))
