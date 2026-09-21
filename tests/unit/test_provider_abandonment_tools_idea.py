"""Authenticated abandoned requests preserve facts, without provider or tool calls."""
from copy import deepcopy
import json

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine.idea import TwoStageIdea
from sciel.engine.llm.client import LLMError, LLMResponse
from sciel.engine.loop import Engine
from sciel.engine.provider_abandonment import AbandonedProviderRequest
from sciel.engine.types import NoChild
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.research_tools.controller import PROMPT_MARKER
from test_research_tool_controller import Executor, IO, LLM, create, prepare, setup, snapshot
from test_two_stage_idea import CARD, response, setup as idea_setup
from test_idea_card_repair import LONG_CARD, repaired_setup
from test_novelty_worker_retries import callback, registry_with


RECEIPT = {'recovery_id':'a'*64, 'job_call':'G:1:0', 'request_sha256':'b'*64,
           'receipt_sha256':'c'*64, 'unknown_cost':True, 'retry':False}
DIAGNOSTIC = {'stage':'request', 'reason':'http_error', 'cause_type':'RateLimitError',
              'http_status':429}


def abandoned(_prompt=None):
    raise AbandonedProviderRequest(RECEIPT, DIAGNOSTIC)


def test_first_tool_proposal_abandonment_is_not_free_response_or_retry(tmp_path):
    tools, executor, events, costs = setup(tmp_path)
    source = snapshot()
    plan = prepare(tools, source)
    llm = LLM([abandoned, create()])
    episode = tools.execute(plan, source, llm, io=IO())
    assert len(llm.prompts) == 1 and llm.prompts[0].startswith(PROMPT_MARKER)
    assert executor.calls == costs == []
    assert episode['known_llm_cost'] == 0  # Known subtotal, not the failed request's price.
    assert episode['provider_abandonment'] == RECEIPT
    assert episode['registered_ids'] == episode['invocations'] == episode['cards'] == []
    assert 'outcome and cost remain unknown' in episode['context']
    event, = [row for row in events if row['phase'] == 'llm_abandoned']
    assert event['cost'] is None and event['current_call_cost_unknown'] is True
    assert event['generation_returned'] is False and event['generation_dispatch'] == 'unknown'
    artifact = tools._read_artifact(event['receipt_id'], 'proposal_abandonment')
    assert artifact['provider_abandonment'] == RECEIPT
    assert artifact['prompt'] == llm.prompts[0] and artifact['llm_error'] == DIAGNOSTIC
    assert artifact['cost'] is None
    finished, = [row for row in events if row['phase'] == 'episode_finished']
    assert finished['status'] == 'provider_abandoned' and finished['attempts'] == 1
    assert finished['provider_abandonment'] == RECEIPT
    tools.commit(episode)
    tools.commit(episode)
    state = tools.state()
    assert state['committed_episodes'] == [plan['action_id']]
    assert not state['lifecycle']['cells']['cells'][plan['cell_id']]['active']
    assert state['registered'] == state['published'] == []
    tools.load_state(deepcopy(state))
    assert tools.state() == state


def test_abandoned_repair_preserves_prior_tool_execution_cost_and_observation(tmp_path):
    tools, executor, events, costs = setup(tmp_path, executor=Executor(('failed',)))
    source, io = snapshot(), IO()
    plan = prepare(tools, source)
    episode = tools.execute(plan, source, LLM([create(), abandoned, create()]), io=io)
    assert costs == [.03] and episode['known_llm_cost'] == .03
    assert len(executor.calls) == len(episode['registered_ids']) == len(episode['invocations']) == 1
    assert len(episode['cards']) == 1 and 'fixture failure' in episode['context']
    assert episode['invocations'][0]['status'] == 'failed'
    assert episode['provider_abandonment'] == RECEIPT
    # Receipt replay must neither execute the old method nor book its known cost again.
    replayed = tools.execute(plan, source, LLM([create(), abandoned]), io=io)
    assert len(executor.calls) == 1 and costs == [.03]
    assert replayed['receipt_ids'] == episode['receipt_ids']
    tools.commit(episode)
    state = tools.state()
    assert state['registered'] == episode['registered_ids'] and state['published'] == []
    assert state['lifecycle']['methods'] == {}
    assert not state['lifecycle']['cells']['cells'][plan['cell_id']]['active']
    tools.load_state(deepcopy(state))
    assert tools.state() == state


def test_unclassified_tool_planner_error_is_still_fatal(tmp_path):
    tools, executor, events, costs = setup(tmp_path)
    def failure(_prompt):
        raise LLMError('ordinary transport failure is not authorization')
    source = snapshot()
    with pytest.raises(LLMError):
        tools.execute(prepare(tools, source), source, LLM([failure]), io=IO())
    assert executor.calls == costs == []
    assert not any(row['phase'] in ('llm_abandoned', 'episode_finished') for row in events)


def test_real_engine_execute_step_continues_to_ordinary_generation(tmp_path):
    tools, executor, events, costs = setup(tmp_path)
    source = snapshot()
    source.tool_episode_plan = prepare(tools, source)
    calls, evaluated = [], []
    class Model:
        def generate(self, prompt, model, temperature=None, **options):
            calls.append(prompt)
            if prompt.startswith(PROMPT_MARKER):
                abandoned()
            return LLMResponse('```python\ndef run_search():\n    return 2\n```', model, .07, 12, 8)
    def evaluate(code):
        evaluated.append(code)
        return CascadeOutcome(EvalResult(True, .3, {}), .3, 'full', False)
    engine = Engine.__new__(Engine)
    engine.research_tools, engine._parallel_io, engine.llm = tools, IO(), Model()
    engine.cfg = {'research_tools': {'max_context_chars':10000}, 'dup_cap':1}
    engine.research = engine.idea = engine.intervention = engine.circle_generation = None
    engine._candidate_novelty_callback = lambda _island: None
    result = engine._execute_step({'snap':source, 'op':'full_rewrite'}, evaluate_fn=evaluate)
    assert result.ok and len(calls) == 2 and len(evaluated) == 1
    assert not calls[1].startswith(PROMPT_MARKER)
    assert result.cost == .07 and costs == [] and executor.calls == []
    assert result.meta['research_tools']['provider_abandonment'] == RECEIPT
    tools.commit(result.meta['research_tools'], child_program_id='p_new')
    assert not tools.state()['lifecycle']['cells']['cells'][source.tool_episode_plan['cell_id']]['active']


@pytest.mark.parametrize('stage', ['card', 'card_repair', 'code', 'code_retry'])
def test_idea_abandonment_returns_nochild_without_evaluation_or_lost_known_cost(stage):
    error = AbandonedProviderRequest(RECEIPT, DIAGNOSTIC)
    rows = [error]
    if stage == 'card_repair':
        rows = [response(LONG_CARD, .03), error]
    elif stage == 'code':
        rows = [response(json.dumps(CARD), .03), error]
    elif stage == 'code_retry':
        # Missing hypothesis rejects the first paid code without evaluating it.
        rows = [response(json.dumps(CARD), .03), response('no program', .07), error]
    factory = repaired_setup if stage == 'card_repair' else idea_setup
    research, idea, snap, model, log, charged, evaluated, run = factory(rows)
    result = run(**({'novelty_check':callback(registry_with(snap.parent_code))}
                    if stage == 'code_retry' else {}))
    assert isinstance(result, NoChild) and result.reason == 'provider_request_abandoned'
    expected = .10 if stage == 'code_retry' else .03 if stage in ('card_repair', 'code') else 0
    assert result.cost == pytest.approx(expected) and idea.known_cost == pytest.approx(expected)
    assert result.meta['provider_abandonment'] == result.meta['idea']['provider_abandonment'] == RECEIPT
    assert result.meta['idea']['current_call_cost_unknown'] is True
    assert charged == evaluated == [] and research.queue == [] and research.pending == {}
    assert research.stopped_reason is idea.stopped_reason is None
    assert idea.pending is None and idea.finished == idea.attempts == 1
    assert len(model.calls) == len(rows) == idea.calls_started
    assert idea.calls_returned == len(rows) - 1
    abandoned_event, = [e for e in log.events if e.get('action') == 'provider_abandoned']
    assert abandoned_event['cost'] is None and abandoned_event['current_call_cost_unknown'] is True
    assert abandoned_event['generation_dispatch'] == 'unknown'
    assert abandoned_event['provider_abandonment'] == RECEIPT
    if stage in ('code', 'code_retry'):
        record = research.knowledge.experiments[-1]
        assert record['status'] == 'generation_stopped'
        assert record['score'] is record['actual_delta'] is record['evaluation'] is None
        assert record['generation_cost'] is None and record['generation_dispatch'] == 'unknown'
    else:
        assert research.knowledge.experiments == []
    other = TwoStageIdea(research, card_repair_attempts=idea.card_repair_attempts)
    other.load_state(json.loads(json.dumps(idea.state())))
    assert other.state() == idea.state()


def test_ordinary_idea_transport_failure_remains_stopped():
    research, idea, _, model, _, _, evaluated, run = idea_setup([LLMError('ordinary')])
    with pytest.raises(CampaignStopped):
        run()
    assert len(model.calls) == 1 and evaluated == []
    assert research.stopped_reason and idea.stopped_reason
