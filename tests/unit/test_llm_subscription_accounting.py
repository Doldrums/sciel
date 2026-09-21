"""Completed subscription responses are unpriced, never free or unknown calls.

Provider, evaluator and tool leaves are inert local fixtures. Real Engine,
checkpoint, response receipts and cooperative replay exercise accounting.
"""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from sciel.engine.llm.client import (LLMResponse, known_cost, accounted_cost,
    response_accounting, response_cost_provenance, SUBSCRIPTION_ENDPOINT_SHA256)
from sciel.engine.parallel_calls import _encode, _decode

MARKER = {'schema': 'sciel-llm-cost-provenance/1', 'source': 'subscription_unpriced',
          'provider': 'codex_responses', 'response_status': 'completed',
          'endpoint_sha256': SUBSCRIPTION_ENDPOINT_SHA256}


def unpriced(response=None):
    response = response or LLMResponse('completed answer', 'fixture', .1, 20, 30)
    return replace(response, cost=None, finish_reason='stop', cost_provenance=deepcopy(MARKER))


def test_unpriced_raw_receipt_roundtrip_and_known_usd_are_distinct():
    response = unpriced()
    with pytest.raises(ValueError):
        known_cost(response)
    assert response_accounting(response) == {'known_usd': 0., 'unpriced_calls': 1}
    assert accounted_cost(response) == 0.
    raw = _encode('llm', response)
    assert raw['cost'] is None and raw['cost_provenance'] == MARKER
    restored = _decode('llm', raw)
    assert restored == response
    assert response_accounting(restored)['unpriced_calls'] == 1
    assert response_accounting(replace(response, cost=.25, cost_provenance=None)) == {
        'known_usd': .25, 'unpriced_calls': 0}
    assert response_accounting(replace(response, cost=0., cost_provenance=None)) == {
        'known_usd': 0., 'unpriced_calls': 0}


@pytest.mark.parametrize('changes', [
    {'cost_provenance': None},
    {'cost_provenance': {'schema': MARKER['schema'], 'source': 'unavailable'}},
    {'cost_provenance': {**MARKER, 'endpoint_sha256': 'invalid'}},
    {'cost_provenance': {**MARKER, 'endpoint_sha256': 'a' * 64}},
    {'cost_provenance': {**MARKER, 'provider': 'other'}},
    {'cost_provenance': {**MARKER, 'response_status': 'incomplete'}},
    {'cost_provenance': {**MARKER, 'authorization': 'must not survive'}},
    {'finish_reason': None}, {'finish_reason': 'length'}, {'finish_reason': 'error'},
    {'response_error': {'reason': 'malformed_response'}},
    {'cost': True}, {'cost': -1}, {'cost': float('nan')}, {'cost': float('inf')},
])
def test_unavailable_or_malformed_never_becomes_accepted_unpriced(changes):
    response = replace(unpriced(), **changes)
    with pytest.raises(ValueError):
        response_accounting(response)
    assert response_cost_provenance(response) is None or changes['cost_provenance']['source'] == 'unavailable'


def converted(provider, *, priced_role=None):
    original = provider.generate
    def generate(*args, **kwargs):
        response = original(*args, **kwargs)
        if priced_role and provider.roles[-1] == priced_role:
            return response
        return unpriced(response)
    provider.generate = generate
    return provider


@pytest.mark.parametrize('idea', [False, True])
def test_real_engine_generation_conclusion_reflection_idea_and_mixed_usd(tmp_path, monkeypatch, idea):
    from test_parallel_engine import ResearchProvider, run_engine, events
    provider = converted(ResearchProvider(), priced_role='C')
    overrides = {}
    if idea:
        monkeypatch.setattr('sciel.engine.loop.OpBandit.pick', lambda *a, **k: ('full_rewrite', 1., {}))
        overrides.update(op_fixed=None, idea_island=True,
                         idea_two_stage={'enabled': True, 'card_repair_attempts': 1})
    engine, provider, calls, _, result, directory = run_engine(tmp_path, monkeypatch, provider=provider,
        research={'enabled': True, 'reflection_interval': 4, 'max_aux_calls': None}, **overrides)
    assert result['iteration_no'] == 4
    expected = sum(role != 'C' for role in provider.roles)
    assert engine.llm_unpriced_calls == result['llm_unpriced_calls'] == expected
    assert result['cost_total'] == pytest.approx(.04)
    assert len(engine.research.conclusions) == 4
    saved = json.loads((directory / 'checkpoint.json').read_text())
    assert saved['llm_unpriced_calls'] == expected
    physical = [json.loads(path.read_text())['payload'] for path in directory.glob('.parallel/wave-*/calls/*/receipt.json')]
    responses = [row['value'] for row in physical if 'text' in row.get('value', {})]
    assert sum(row['cost'] is None for row in responses) == expected
    assert all(row['cost_provenance'] == MARKER for row in responses if row['cost'] is None)
    traces = [json.loads(line) for line in (directory / 'artifacts/llm_calls.jsonl').read_text().splitlines()]
    unpriced_traces = [row for row in traces if row.get('event') == 'response' and row.get('cost') is None]
    assert len(unpriced_traces) == expected
    assert all(row['current_call_cost_unknown'] is False for row in unpriced_traces)


def test_real_engine_stopped_after_receipt_replays_without_call_or_double_count(tmp_path, monkeypatch):
    from sciel.engine import checkpoint
    from sciel.engine.factory import build_engine
    from sciel.engine.llm.mock import MockLLM
    from sciel.engine.parallel_store import WaveTransaction
    from sciel.campaign.budget import CampaignStopped
    from test_parallel_engine import config, evaluator, Provider, task_fixture
    tasks = tmp_path / 'tasks'; task = task_fixture(tasks)
    cfg = config(task, tmp_path / 'capacity', workers=1, max_iterations=1)
    provider = converted(Provider('run_search'))
    calls, _ = evaluator(monkeypatch)
    directory = tmp_path / 'run'
    first = build_engine(cfg, str(tasks), str(directory), llm=provider)
    commit = WaveTransaction.commit
    def stop_before_commit(transaction):
        if transaction.wave_id == '1':
            raise RuntimeError('offline stop after durable response')
        return commit(transaction)
    with monkeypatch.context() as stopped:
        stopped.setattr(WaveTransaction, 'commit', stop_before_commit)
        with pytest.raises(CampaignStopped, match='offline stop'):
            first.run()
    assert len(provider.calls) == 1 and len(calls) == 2
    class ReplayOnly(MockLLM):
        def generate(self, *args, **kwargs):
            pytest.fail('replay may not dispatch the provider')
    def no_evaluation(*args, **kwargs):
        pytest.fail('replay may not execute a candidate')
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', no_evaluation)
    resumed = build_engine(cfg, str(tasks), str(directory), llm=ReplayOnly())
    checkpoint.load(resumed, str(directory / 'checkpoint.json'))
    assert resumed.llm_unpriced_calls == 0
    result = resumed.run()
    assert result['iteration_no'] == 1 and result['cost_total'] == 0.
    assert result['llm_unpriced_calls'] == resumed.llm_unpriced_calls == 1
    checkpointed = json.loads((directory / 'checkpoint.json').read_text())
    assert checkpointed['llm_unpriced_calls'] == 1
    again = build_engine(cfg, str(tasks), str(directory), llm=ReplayOnly())
    checkpoint.load(again, str(directory / 'checkpoint.json'))
    assert again.llm_unpriced_calls == 1
    again.close()


def test_research_planner_charge_and_lifecycle_continue_and_resume(tmp_path, monkeypatch):
    from test_independent_research_engine import setup, designed
    from test_sequential_research_episode import decision, finish
    from test_research_tool_result_flow import result
    build, provider, backend, evaluations, directory = setup(tmp_path, monkeypatch,
        [designed(decision()), designed(finish(), kind='closure', budget_action='none')],
        [result({'marker': 'actual fake measured observation'})])
    converted(provider)
    engine = build(); summary = engine.run()
    assert summary['iteration_no'] == 2 and summary['cost_total'] == 0.
    assert summary['llm_unpriced_calls'] == 4
    assert len(backend.calls) == 1
    charges = engine.research_tools.state()['charges']
    assert len(charges) == 2 and all(row['unpriced_calls'] == 1 for row in charges.values())
    restored = build(resume=True)
    assert restored.llm_unpriced_calls == 4
    assert restored.research_tools.state()['charges'] == charges
    restored.close()


def test_serial_direct_response_and_postmortem_counts(tmp_path, monkeypatch):
    from test_parallel_engine import Provider, run_engine
    from sciel.components.tail import extract_lesson
    provider = converted(Provider('run_search'))
    engine, _, _, _, summary, directory = run_engine(tmp_path, monkeypatch,
        provider=provider, workers=1, max_iterations=1, parallel_search={'enabled': False})
    assert summary['llm_unpriced_calls'] == 1 and summary['cost_total'] == 0.
    # Postmortem uses the same accounting rule; no source execution.
    lesson, subtotal = extract_lesson(SimpleNamespace(generate=lambda *a, **k: unpriced()),
                                      'fixture', 'parent', 'child', 'mechanism')
    assert lesson is None and subtotal == 0.


@pytest.mark.parametrize('tamper', [None, 'missing', 'other_endpoint', 'incomplete', 'numeric_bool'])
def test_external_reducer_accepts_only_typed_completed_subscription_history(tmp_path, tamper):
    # Author the source history with inert model/measurement fixtures; the
    # reducer never executes candidates, evaluators, or provider calls.
    from test_external_research_v2_reducer import actual as actual_fixture
    from sciel.engine import external_research_v2 as history
    capsule, final = actual_fixture.__wrapped__()
    rows = list(map(json.loads, capsule['journals'][1]['journal'].splitlines()))
    terminal = next(row for row in rows if row.get('phase') == 'conclusion')
    result = next(row for row in rows if row.get('phase') == 'aux_result')
    result.update(cost=None, cost_provenance=deepcopy(MARKER), known_usd=0., unpriced_calls=1)
    if tamper == 'missing':
        result.pop('cost_provenance')
    elif tamper == 'other_endpoint':
        result['cost_provenance']['endpoint_sha256'] = 'a' * 64
    elif tamper == 'incomplete':
        result['cost_provenance']['response_status'] = 'incomplete'
    elif tamper == 'numeric_bool':
        result['cost'] = True
    if tamper is None:
        decoded = history._auxiliary(rows, terminal, 'conclusion', terminal['experiment_id'])
        assert decoded is not None
        assert result['cost'] is None  # No rewrite to zero during reconstruction.
    else:
        with pytest.raises(ValueError, match='complete usable auxiliary response'):
            history._auxiliary(rows, terminal, 'conclusion', terminal['experiment_id'])


def test_validation_trace_never_labels_unpriced_generation_as_free():
    from sciel.engine.validation import validate_candidate, trace_attempt
    parent = 'def run_search():\n    return 1\n'
    child = 'def run_search():\n    return 2\n'
    response = unpriced(LLMResponse(child, 'fixture', .1, 5, 6))
    validation = validate_candidate(child, parent_code=parent,
        required_entrypoints=['run_search'], code_hashes=set())
    trace = trace_attempt(validation, extracted_code=child, response=response,
        attempt=1, operator='full_rewrite', temperature=.7, cost=0.)
    assert trace['cost'] is None
    assert trace['known_usd'] == 0. and trace['unpriced_calls'] == 1
    assert trace['cost_provenance'] == MARKER
