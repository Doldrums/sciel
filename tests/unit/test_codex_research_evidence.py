"""Inert saved-response replay and typed cost evidence through real research.

No model transport, candidate source or evaluator is executed. Sol response
bytes are immutable JSON inputs; non-circle receipts are authored offline.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from sciel.engine.parallel_calls import _decode, digest as wire_digest
from sciel.engine.research import ResearchController, digest
from sciel.engine.types import Program, Snapshot
from sciel.engine.worker import run_worker
from sciel.eval.output_identity import adapt_output
from sciel.eval.registry import load_task
from sciel.hypothesis.measurement import metric_context
from sciel.knowledge_base import KnowledgeBase
from sciel.engine.research_cells_context import compact_context
from test_llm_subscription_accounting import MARKER, unpriced
from test_method_exposure_research import prepared, measured, execute
from test_research_cell_context import card

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'tests/fixtures/codex_sol_first_canonical.json'


def saved():
    raw = json.loads(FIXTURE.read_text())
    for key in ('llm', 'evaluation'):
        receipt = raw[key]['receipt']
        assert wire_digest(receipt['payload']) == receipt['sha256']
        assert receipt['payload']['status'] == 'returned'
    intent = raw['evaluation']['intent']
    assert wire_digest({'kind': 'evaluation', 'request': intent['request']}) == intent['request_sha256']
    assert intent['request_sha256'] == raw['evaluation']['receipt']['payload']['request_sha256']
    return raw


@pytest.mark.parametrize('cards', [None, [], [card()]])
def test_exact_saved_sol_response_and_geometry_complete_research_and_restore(cards):
    raw = saved()
    response = _decode('llm', raw['llm']['receipt']['payload']['value'])
    outcome = _decode('evaluation', raw['evaluation']['receipt']['payload']['value'])
    source = raw['evaluation']['intent']['request']['source']
    task = load_task(str(ROOT / 'tasks/oe/circle_packing'))
    original = deepcopy(raw)
    calls = []
    class Model:
        models = [response.model]
        def generate(self, *args, **kwargs):
            calls.append('saved response')
            return deepcopy(response)
    events = []
    research = ResearchController(dict(model=None, max_lessons=10, max_experiments=20,
        max_fragments=5, retrieve_k=3, max_aux_calls=0, reflection_interval=0), Model(),
        NS(research=lambda **kw: events.append(kw)), lambda _: pytest.fail('auxiliary call'),
        metric=metric_context(task))
    h = raw['hypothesis_record']
    parent = Program(h['parent_id'], Path(task.initial_program_path).read_text(),
        h['parent_score'], h['parent_cases'], 0, 'seed')
    assert digest(parent.code) == h['parent_source_sha256']
    context = research.context(NS(_all={parent.id: parent}), parent, 0, 0)
    if cards is not None:
        context['research_cells_context'] = compact_context(cards)
    snapshot = Snapshot(0, 'full_rewrite', response.model, 0, parent.id, parent.code,
        parent.score, required_entrypoints=['run_packing'], research_context=context)
    def returned(code):
        assert code == source
        calls.append('saved evaluation')
        return deepcopy(outcome)
    result = run_worker('full_rewrite', snapshot, research.llm, returned,
                        research=research, dup_cap=1)
    assert calls == ['saved response', 'saved evaluation']
    assert result.child_code == source and result.ok and result.score == .9487666033206833
    record = research.knowledge.experiments[0]
    assert record['status'] == 'evaluated'
    assert record['evaluation']['cases']['sum_radii'] == 2.4999999997500004
    assert record['evaluation']['native_geometry'] == json.loads(outcome.result.artifacts['native_geometry'])
    assert record['generation_cost'] == 0. and record['generation_cost_accounting'] == {
        'known_usd': 0., 'unpriced_calls': 1}
    assert record['generation_cost_provenance'] == MARKER
    assert ('method_exposures' in record) == (cards is not None)
    research.complete_step()
    clone = KnowledgeBase(max_lessons=10, max_experiments=20, max_fragments=5)
    clone.load_state(research.knowledge.state())
    assert clone.state() == research.knowledge.state()
    binding = dict(task_hash=task.task_hash, source_sha256=digest(source), island_id=0,
        receipt_sha256=raw['evaluation']['receipt']['sha256'],
        request_sha256=raw['evaluation']['intent']['request_sha256'])
    identity = adapt_output(task, digest(source), outcome, binding)
    assert identity is not None  # Previously silently labelled "unsupported".
    assert identity['output_sha256'] == record['evaluation']['native_geometry']['numeric_sha256']
    assert raw == original and response.cost is None


@pytest.mark.parametrize('status', ['evaluated', 'eval_failed', 'hypothesis_invalid'])
@pytest.mark.parametrize('cards', [None, [], [card()]])
def test_unpriced_generation_outcomes_keep_typed_cost_with_or_without_methods(status, cards):
    research, llm, rows, snap = prepared(cards=cards)
    reply = unpriced(llm.generate('inert fixture', snap.model))
    if status == 'hypothesis_invalid': reply.text = 'No forecast.'
    llm.generate = lambda *args, **kwargs: deepcopy(reply)
    def evaluator(code):
        assert status != 'hypothesis_invalid'
        value = measured(code)
        if status == 'eval_failed':
            value.result.ok = False
            value.result.score = 0.
        return value
    run_worker('full_rewrite', snap, llm, evaluator, research=research, dup_cap=1)
    record = research.knowledge.experiments[0]
    assert record['status'] == status
    assert record['generation_cost_accounting'] == {'known_usd': 0., 'unpriced_calls': 1}
    assert record['generation_cost_provenance'] == MARKER
    assert record['generation_dispatch'] == 'returned'
    clone = KnowledgeBase(max_lessons=10, max_experiments=20, max_fragments=5)
    clone.load_state(research.knowledge.state())
    assert clone.state() == research.knowledge.state()


@pytest.mark.parametrize('priced', [False, True])
def test_optional_cost_fields_survive_evicted_lesson_fragment_and_cold_state(priced):
    research, llm, rows, snap = prepared(cards=[card()])
    research.knowledge.max_experiments = 1
    research.config['max_experiments'] = 1
    reply = llm.generate('inert fixture', snap.model)
    if not priced: reply = unpriced(reply)
    llm.generate = lambda *args, **kwargs: deepcopy(reply)
    execute(research, llm, snap)
    record = research.knowledge.experiments[0]
    assert ('generation_cost_accounting' in record) != priced
    research.knowledge.add_lessons([{'rule': 'Tentative observed association.', 'polarity': 'positive',
        'scope': 'branch', 'evidence_ids': [record['id']]}], allowed_evidence_ids=[record['id']])
    research.complete_step()
    execute(research, llm, snap)
    research.complete_step()
    state = research.state()
    assert all(r['id'] != record['id'] for r in state['knowledge']['experiments'])
    evidence = state['knowledge']['lessons'][0]['evidence_snapshots'][0]
    assert ('generation_cost_provenance' in evidence) != priced
    assert state['knowledge']['fragments']
    clone, _, _, _ = prepared(cards=[card()])
    clone.knowledge.max_experiments = 1
    clone.config['max_experiments'] = 1
    clone.load_state(state)
    assert clone.state() == state


@pytest.mark.parametrize('damage', ['missing_accounting', 'missing_provenance', 'extra', 'bool_usd',
    'bool_count', 'count', 'subtotal', 'cost', 'endpoint', 'provider', 'status', 'dispatch'])
def test_malformed_or_inconsistent_accounting_never_enters_knowledge(damage):
    research, llm, _, snap = prepared(cards=[card()])
    reply = unpriced(llm.generate('fixture', snap.model))
    llm.generate = lambda *args, **kwargs: deepcopy(reply)
    execute(research, llm, snap)
    record = research.knowledge.experiments[0]
    if damage == 'missing_accounting': del record['generation_cost_accounting']
    elif damage == 'missing_provenance': del record['generation_cost_provenance']
    elif damage == 'extra': record['generation_cost_accounting']['inferred_usd'] = 0
    elif damage == 'bool_usd': record['generation_cost_accounting']['known_usd'] = False
    elif damage == 'bool_count': record['generation_cost_accounting']['unpriced_calls'] = True
    elif damage == 'count': record['generation_cost_accounting']['unpriced_calls'] = 2
    elif damage == 'subtotal': record['generation_cost_accounting']['known_usd'] = .1
    elif damage == 'cost': record['generation_cost'] = .1
    elif damage == 'endpoint': record['generation_cost_provenance']['endpoint_sha256'] = 'a'*64
    elif damage == 'provider': record['generation_cost_provenance']['provider'] = 'unverified'
    elif damage == 'status': record['generation_cost_provenance']['response_status'] = 'incomplete'
    else: record['generation_dispatch'] = 'unknown'
    kb = KnowledgeBase(max_lessons=10, max_experiments=20, max_fragments=5)
    before = kb.state()
    with pytest.raises(ValueError): kb.record(record)
    assert kb.state() == before


@pytest.mark.parametrize('task_name', ['heilbronn', 'oracle'])
def test_authored_non_circle_records_resolve_and_restore_with_returned_geometry_or_cases(task_name):
    from sciel.hypothesis.measurement import evaluation_observation
    from research_receipt_fixture import non_circle_receipt
    from sciel.engine.parallel_calls import _encode
    research, task, raw = non_circle_receipt(task_name)
    before = deepcopy(raw)
    intent, receipt, record = raw['evaluation_intent'], raw['evaluation_receipt'], raw['hypothesis_record']
    assert wire_digest(receipt['payload']) == receipt['sha256']
    assert wire_digest({'kind': 'evaluation', 'request': intent['request']}) == intent['request_sha256']
    assert receipt['payload']['request_sha256'] == intent['request_sha256']
    assert record['generation_cost_provenance'] == MARKER and 'method_exposures' in record
    assert digest(raw['parent_code']) == record['parent_source_sha256']
    assert digest(intent['request']['source']) == record['source_sha256']
    returned = _decode('evaluation', receipt['payload']['value'])
    assert returned.result.ok
    assert _encode('evaluation', returned) == receipt['payload']['value']
    if task_name == 'heilbronn':
        binding = dict(task_hash=task.task_hash, source_sha256=record['source_sha256'], island_id=0,
            receipt_sha256=receipt['sha256'], request_sha256=intent['request_sha256'])
        assert adapt_output(task, record['source_sha256'], returned, binding) is not None
        assert json.loads(returned.result.artifacts['full_output'])['output']['points'] == [[0., 0.]] * 11
    else:
        assert returned.result.cases['objective_queries'] == 1.
    resolved = research.resolve(record['id'], status='evaluated', score=returned.result.score,
        source=intent['request']['source'], evaluation=evaluation_observation(returned))
    assert resolved['status'] == 'evaluated' and resolved['evaluation']['cases'] == returned.result.cases
    assert resolved['generation_cost_accounting'] == {'known_usd': 0., 'unpriced_calls': 1}
    assert not research.pending and raw == before
    assert resolved['metric'] == metric_context(task)
    clone = KnowledgeBase(**research.knowledge.state()['limits'])
    clone.load_state(research.knowledge.state())
    assert clone.state() == research.knowledge.state()


def test_real_engine_method_exposure_unpriced_response_wal_replay_is_exactly_once(tmp_path, monkeypatch):
    from sciel.engine import checkpoint
    from sciel.engine.factory import build_engine
    from sciel.engine.llm.mock import MockLLM
    from sciel.engine.parallel_store import WaveTransaction
    from sciel.campaign.budget import CampaignStopped
    from test_parallel_engine import config, evaluator, ResearchProvider, task_fixture
    from test_llm_subscription_accounting import converted
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    cfg = config(task, tmp_path / 'capacity', workers=1, max_iterations=1,
        research={'enabled': True, 'max_aux_calls': 0, 'reflection_interval': 0})
    original_context = ResearchController.context
    def context(self, *args, **kwargs):
        value = original_context(self, *args, **kwargs)
        value['research_cells_context'] = compact_context([card()])
        return value
    monkeypatch.setattr(ResearchController, 'context', context)
    provider = converted(ResearchProvider(concurrent=1))
    evaluations, _ = evaluator(monkeypatch)
    directory = tmp_path / 'run'
    first = build_engine(cfg, str(tasks), str(directory), llm=provider)
    commit = WaveTransaction.commit
    def stop_after_receipt(transaction):
        if transaction.wave_id == '1':
            raise RuntimeError('offline stop after durable unpriced research result')
        return commit(transaction)
    with monkeypatch.context() as stopped:
        stopped.setattr(WaveTransaction, 'commit', stop_after_receipt)
        with pytest.raises(CampaignStopped, match='offline stop'):
            first.run()
    assert len(provider.calls) == 1 and len(evaluations) == 2
    class ReplayOnly(MockLLM):
        def generate(self, *args, **kwargs):
            pytest.fail('returned provider response must replay')
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate',
        lambda *args, **kwargs: pytest.fail('returned evaluator response must replay'))
    resumed = build_engine(cfg, str(tasks), str(directory), llm=ReplayOnly())
    checkpoint.load(resumed, str(directory / 'checkpoint.json'))
    assert resumed.llm_unpriced_calls == 0
    result = resumed.run()
    assert result['llm_unpriced_calls'] == resumed.llm_unpriced_calls == 1
    assert result['cost_total'] == 0. and result['iteration_no'] == 1
    record, = resumed.research.knowledge.experiments
    assert record['generation_cost_provenance'] == MARKER
    assert record['generation_cost_accounting'] == {'known_usd': 0., 'unpriced_calls': 1}
    assert len(record['method_exposures']['cards']) == 1 and record['status'] == 'evaluated'
    again = build_engine(cfg, str(tasks), str(directory), llm=ReplayOnly())
    checkpoint.load(again, str(directory / 'checkpoint.json'))
    assert again.llm_unpriced_calls == 1
    assert again.research.knowledge.state() == resumed.research.knowledge.state()
    again.close()
