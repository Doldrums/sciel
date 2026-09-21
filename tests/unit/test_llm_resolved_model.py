"""Provider identities are observations, not model-selection aliases.

SDK responses, WAL leaves and canonical evaluator returns are inert fixtures.
No provider request or generated program is executed.
"""
from contextlib import nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from sciel.engine.llm.client import (LLMResponse, response_model_identity,
    response_resolved_model, validate_resolved_model)
from sciel.engine.llm.trace import TracedLLM
from sciel.engine.parallel_calls import CallLedger, _decode, _encode
from sciel.methods.proposals import response_record
from test_llm_cost_provenance import provider


REQUESTED = 'provider/model-latest'
RESOLVED = 'provider/model-2026-09-21'


@pytest.mark.parametrize('choices', [True, False])
def test_sdk_identity_is_retained_even_for_paid_malformed_content(choices):
    llm = provider({'cost': .02, 'prompt_tokens': 4, 'completion_tokens': 3},
                   model=RESOLVED, choices=choices)
    reply = llm.generate('inert request', REQUESTED)
    assert reply.model == REQUESTED and reply.resolved_model == RESOLVED
    assert reply.cost == .02 and bool(reply.response_error) is not choices
    assert llm._client.chat.completions.create.call_args.kwargs['model'] == REQUESTED
    assert response_record(reply)['resolved_model'] == RESOLVED


@pytest.mark.parametrize('value', [None, '', ' ', 'x\ny', 'x y', 'x\x00y', '\ud800',
                                 'x' * 257, 'é' * 129, False, 1, {}, [], Mock()])
def test_invalid_optional_identity_cannot_hide_paid_response(value):
    llm = provider({'cost': .02}, model=value)
    reply = llm.generate('inert request', REQUESTED)
    assert reply.model == REQUESTED and reply.cost == .02
    assert reply.resolved_model is None and validate_resolved_model(value) is None
    encoded = _encode('llm', reply)
    assert 'resolved_model' not in encoded
    replay = _decode('llm', {**encoded, 'resolved_model': value})
    assert replay.resolved_model is None and replay.cost == .02


def test_throwing_optional_metadata_is_unknown_not_a_lost_charge():
    class Body:
        usage = NS(cost=.2, prompt_tokens=1, completion_tokens=1)
        choices = [NS(message=NS(content='saved', refusal=None), finish_reason='stop')]
        @property
        def model(self):
            raise RuntimeError('private arbitrary SDK object')
    llm = provider()
    llm._client.chat.completions.create.return_value = Body()
    reply = llm.generate('p', REQUESTED)
    assert reply.text == 'saved' and reply.cost == .2 and reply.resolved_model is None


def test_legacy_receipt_does_not_infer_resolved_version():
    old = dict(text='saved', model=REQUESTED, cost=.02, prompt_tokens=1,
               completion_tokens=1, finish_reason='stop', refusal=None)
    reply = _decode('llm', old)
    assert response_resolved_model(reply) is None and response_model_identity(reply) is None
    assert _encode('llm', reply) == old


def test_real_sdk_wal_cold_replay_and_trace_keep_both_identities(tmp_path):
    llm = provider({'cost': .02, 'prompt_tokens': 4, 'completion_tokens': 3}, model=RESOLVED)
    context = dict(role='G', run_id='run', segment_id='segment', reference='r1', experiment_id='r1')
    traced = TracedLLM(llm, tmp_path, lambda _: context)
    capacity = NS(acquire=lambda *a, **k: nullcontext())
    ledger = CallLedger(tmp_path / 'wave', capacity, 'local')
    request = {'prompt': 'fixed request', 'model': REQUESTED}
    directory, digest, cached = ledger.prepare('G:1:0', 'llm', request)
    payload = ledger.perform(directory, digest, 'llm', lambda: traced.generate(request['prompt'], REQUESTED))
    before = (directory / 'receipt.json').read_bytes()
    cold = CallLedger(tmp_path / 'wave', NS(acquire=Mock(side_effect=AssertionError('no dispatch'))), 'local')
    _, _, cached = cold.prepare('G:1:0', 'llm', request)
    replay = cold.result('llm', cached)
    assert replay == ledger.result('llm', payload)
    assert replay.model == REQUESTED and replay.resolved_model == RESOLVED
    assert (directory / 'receipt.json').read_bytes() == before
    assert llm._client.chat.completions.create.call_count == 1
    trace = [json.loads(x) for x in (tmp_path / 'artifacts/llm_calls.jsonl').read_text().splitlines()]
    assert 'resolved_model' not in trace[0]
    assert trace[1]['resolved_model'] == payload['value']['resolved_model'] == RESOLVED


def test_budgeted_sdk_keeps_requested_price_and_provider_identity(tmp_path):
    from test_campaign_budget import llm as budgeted
    llm, sdk, ledger = budgeted(tmp_path)
    sdk.chat.completions.create.return_value.model = RESOLVED
    sdk.chat.completions.create.return_value.usage.cost = .0001
    reply = llm.generate('p', 'm')
    assert reply.model == 'm' and reply.resolved_model == RESOLVED
    assert reply.cost == .0001 and sdk.chat.completions.create.call_count == 1
    assert next(iter(json.loads(ledger.path.read_text())['calls'].values()))['model'] == 'm'


def test_model_identity_survives_real_research_context_and_cold_evidence():
    from test_method_exposure_research import prepared, execute
    from sciel.knowledge_base import KnowledgeBase
    research, llm, rows, snap = prepared()
    original = llm.generate
    def generate(*args, **kwargs):
        response = original(*args, **kwargs)
        response.resolved_model = RESOLVED
        return response
    llm.generate = generate
    result = execute(research, llm, snap)
    identity = dict(schema='sciel-llm-model-identity/1', requested_model=snap.model, resolved_model=RESOLVED)
    record = research.knowledge.experiments[0]
    assert record['generation_model_identity'] == identity
    assert result.meta['validation_trace']['attempts'][0]['model_identity'] == identity
    assert next(r for r in rows if r['phase'] == 'hypothesis')['resolved_model'] == RESOLVED
    research.complete_step()
    clone = KnowledgeBase(max_lessons=10, max_experiments=20, max_fragments=5)
    clone.load_state(research.knowledge.state())
    assert clone.state() == research.knowledge.state()
    # This is the actual evidence family consumed by reflection/model review,
    # not a standalone log of the response string.
    context = clone.reflection_context(10)
    assert RESOLVED in json.dumps(context) and snap.model in json.dumps(context)
    changed = deepcopy(research.knowledge.state())
    changed['experiments'][0]['generation_model_identity']['resolved_model'] = 'invented-version'
    with pytest.raises(ValueError, match='identity mismatch'):
        clone.load_state(changed)


def test_parallel_engine_replay_uses_original_model_identity_in_conclusion_and_trace(tmp_path, monkeypatch):
    from sciel.engine import parallel_recovery
    from test_parallel_provider_recovery import (FailingProvider, stopped_wave,
        stop_after_publication, physical_files)
    class ResolvedProvider(FailingProvider):
        def generate(self, *args, **kwargs):
            response = super().generate(*args, **kwargs)
            response.resolved_model = RESOLVED
            return response
    first, build, provider_, _, directory = stopped_wave(tmp_path, monkeypatch, provider=ResolvedProvider())
    path = directory / 'checkpoint.json'
    permission = parallel_recovery.inspect_parallel_stop(path)
    parallel_recovery.authorize_parallel_recovery(first, path, permission)
    first.close()
    before = physical_files(directory)
    def forbidden(*args, **kwargs):
        pytest.fail('Replay must not call provider or evaluator')
    monkeypatch.setattr(provider_, 'generate', forbidden)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', forbidden)
    stop_after_publication(monkeypatch, directory)
    resumed = build(resume=True)
    result = resumed.run()
    assert result['cost_total'] == pytest.approx(.15)
    assert physical_files(directory) == before
    assert all(r['generation_model_identity']['resolved_model'] == RESOLVED
               for r in resumed.research.knowledge.experiments)
    traces = [json.loads(x) for x in (directory / 'artifacts/llm_calls.jsonl').read_text().splitlines()]
    conclusions = [r for r in traces if r['event'] == 'request' and r['role'] == 'C']
    assert len(conclusions) == 4
    for row in conclusions:
        prompt = (directory / 'artifacts/llm_calls' / (row['call_id'] + '.prompt.txt')).read_text()
        assert 'generation_model_identity' in prompt and RESOLVED in prompt
    returned = [row for row in traces if row['event'] == 'response']
    assert len(returned) == 7 and all(row['resolved_model'] == RESOLVED for row in returned)
    assert all(row['model'] == provider_.models[0] for row in returned)
