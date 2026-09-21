"""Proven no-dispatch is not a model exposure; no API or candidate execution."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.campaign.budget import BudgetedLLM, CampaignStopped, CostLedger, RequestNotDispatched
from sciel.knowledge_base import KnowledgeBase
from sciel.hypothesis.measurement import metric_context

from test_research_knowledge import experiment, lesson


def provider(tmp_path, limit=1):
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        id='fixture', usage=SimpleNamespace(cost=.0001, prompt_tokens=5, completion_tokens=8),
        choices=[SimpleNamespace(message=SimpleNamespace(content='inert response'))])
    ledger = CostLedger(tmp_path / 'ledger.json', limit)
    cfg = {'models': ['m'], 'weights': [1.], 'max_tokens': 100,
           'api_base': 'https://openrouter.ai/api/v1'}
    prices = {'models': [{'id': 'm', 'pricing': {'prompt': '.000001', 'completion': '.000002'}}]}
    wrapped = BudgetedLLM(cfg, prices, ledger, 'fixture', 'inert-key', client)
    return wrapped, client, ledger


def stopped(number, injected, **changes):
    record = experiment(number, status='generation_stopped', hypothesis=None,
                        injected=injected, generation_dispatch='not_dispatched')
    record.update(changes)
    return record


def seeded():
    kb = KnowledgeBase()
    kb.record(experiment(1))
    lid = lesson(kb, 'r000001a1')
    return kb, lid


def test_insufficient_reserve_is_typed_nondispatch_without_receipt_or_provider_call(tmp_path):
    llm, client, ledger = provider(tmp_path, limit=.00001)
    before = ledger.snapshot()
    with pytest.raises(RequestNotDispatched):
        llm.generate('prompt', 'm')
    client.chat.completions.create.assert_not_called()
    assert ledger.snapshot() == before


def test_previous_unknown_receipt_is_preserved_but_next_blocked_call_is_nondispatched(tmp_path):
    llm, client, ledger = provider(tmp_path)
    client.chat.completions.create.side_effect = TimeoutError('fixture lost response')
    with pytest.raises(CampaignStopped) as caught:
        llm.generate('prompt', 'm')
    assert not isinstance(caught.value, RequestNotDispatched)
    before = ledger.snapshot()
    prior_receipts = json.loads((tmp_path / 'ledger.json').read_text())['calls']
    assert before['calls'] == 1 and float(before['reserved_or_unknown_usd']) > 0
    with pytest.raises(RequestNotDispatched):
        llm.generate('next', 'm')
    assert client.chat.completions.create.call_count == 1
    assert ledger.snapshot() == before
    assert json.loads((tmp_path / 'ledger.json').read_text())['calls'] == prior_receipts


@pytest.mark.parametrize('mode', ['missing_cost', 'overrun'])
def test_dispatched_charge_failure_is_never_relabelled_as_free_nondispatch(tmp_path, mode):
    llm, client, ledger = provider(tmp_path)
    client.chat.completions.create.return_value.usage.cost = None if mode == 'missing_cost' else .5
    with pytest.raises(CampaignStopped) as caught:
        llm.generate('prompt', 'm')
    assert not isinstance(caught.value, RequestNotDispatched)
    assert client.chat.completions.create.call_count == 1
    assert ledger.snapshot()['calls'] == 1


def test_nondispatched_attempt_keeps_intended_card_membership_but_no_exposure():
    kb, lid = seeded()
    previous = kb.lessons
    kb.record(stopped(2, [lid]))
    assert kb.experiments[-1]['injected_ids'] == [lid]
    assert kb.experiments[-1]['generation_dispatch'] == 'not_dispatched'
    assert kb.experiments[-1]['actual_delta'] is None
    assert kb.lessons == previous
    assert len(kb.state()['seen_experiments']) == 2
    with pytest.raises(ValueError, match='duplicate'):
        kb.record(stopped(2, [lid]))


@pytest.mark.parametrize('dispatch', [None, 'unknown', 'returned'])
def test_legacy_unknown_and_returned_failures_remain_unmeasured_exposure_intent(dispatch):
    kb, lid = seeded()
    record = stopped(2, [lid])
    if dispatch is None:
        del record['generation_dispatch']
    else:
        record['generation_dispatch'] = dispatch
    kb.record(record)
    result = kb.lessons[0]
    assert result['exposure_ids'] == ['r000002a1']
    assert result['support_ids'] == result['refute_ids'] == []
    assert result['confidence'] == .5


@pytest.mark.parametrize('change', [
    {'status': 'evaluated', 'score': .6, 'actual_delta': .6 - .5, 'source_sha256': 'a'*64},
    {'hypothesis': experiment(2)['hypothesis']}, {'source_sha256': 'a'*64},
    {'proposed_sha256': 'a'*64}, {'generation_cost': 0.}, {'change_excerpt': 'code edit'},
    {'child_id': 'p2'}, {'source_repaired': True},
    {'generation_dispatch': True}, {'generation_dispatch': None},
    {'generation_dispatch': 'sent'}, {'generation_dispatch': []},
])
def test_nondispatch_cannot_contradict_returned_response_or_malformed_marker(change):
    kb, lid = seeded()
    before = kb.state()
    with pytest.raises(ValueError):
        kb.record(stopped(2, [lid], **change))
    assert kb.state() == before


@pytest.mark.parametrize('with_metric', [False, True])
@pytest.mark.parametrize('with_dispatch', [False, True])
def test_all_supported_old_new_snapshot_shapes_roundtrip_without_adding_fields(with_metric, with_dispatch):
    kb = KnowledgeBase()
    r = experiment(1)
    if with_metric:
        r.update(metric=metric_context(), parent_cases={}, evaluation=None)
    if with_dispatch:
        r['generation_dispatch'] = 'returned'
    kb.record(r)
    lesson(kb, r['id'])
    original = json.dumps(kb.state(), sort_keys=True)
    loaded = KnowledgeBase()
    loaded.load_state(json.loads(original))
    assert json.dumps(loaded.state(), sort_keys=True) == original
    assert ('generation_dispatch' in loaded.experiments[0]) is with_dispatch


@pytest.mark.parametrize('inherited', [False, True])
def test_saved_nondispatch_cannot_be_forged_into_direct_or_inherited_exposure(inherited):
    kb, lid = seeded()
    kb.record(stopped(2, [lid]))
    state = kb.state()
    snap = copy.deepcopy(state['experiments'][-1])
    snap.pop('code_fragment')
    snap['code_fragment_sha256'] = None
    target = state['lessons'][0]
    if inherited:
        target['inherited_exposures'] = [dict(from_lesson_id=lid,
            from_rule=target['rule'], experiment=snap)]
    else:
        target['exposure_snapshots'] = [snap]
        target['exposure_ids'] = [snap['id']]
    before = kb.state()
    with pytest.raises(ValueError, match='exposure'):
        kb.load_state(state)
    assert kb.state() == before


@pytest.mark.parametrize('dispatched', [False, True])
def test_real_budget_worker_controller_chain_distinguishes_never_sent_from_unknown(tmp_path, dispatched):
    from test_research_injection import controller, populated
    from sciel.engine.types import Snapshot
    from sciel.engine.worker import run_worker

    llm, client, ledger = provider(tmp_path, limit=1 if dispatched else .00001)
    if dispatched:
        client.chat.completions.create.side_effect = TimeoutError('fixture unknown request')
    research, db, parent, _ = controller(populated())
    ctx = research.context(db, parent, 0, 3)
    assert ctx['injected_ids']
    snapshot = Snapshot(0, 'diff', 'm', 2, parent.id, parent.code, parent.score,
        research_context=ctx, kb_lessons=ctx['lesson_texts'])
    before = research.knowledge.lessons
    evaluate = Mock(side_effect=AssertionError('no evaluator should run'))
    with pytest.raises(CampaignStopped):
        run_worker('diff', snapshot, llm, evaluate, research=research)
    evaluate.assert_not_called()
    assert client.chat.completions.create.call_count == int(dispatched)
    record = research.knowledge.experiments[-1]
    assert record['status'] == 'generation_stopped' and record['actual_delta'] is None
    assert record['injected_ids'] == ctx['injected_ids']
    assert record['generation_dispatch'] == ('unknown' if dispatched else 'not_dispatched')
    if dispatched:
        shown = {l['id']: l for l in research.knowledge.lessons}
        assert all(shown[ident]['exposure_ids'] == [record['id']] for ident in ctx['injected_ids'])
        assert float(ledger.snapshot()['reserved_or_unknown_usd']) > 0
    else:
        assert research.knowledge.lessons == before and ledger.snapshot()['calls'] == 0


@pytest.mark.parametrize('stage', ['card', 'code'])
def test_two_stage_nondispatch_preserves_prior_stage_cost_and_stage_specific_trace(stage):
    from test_two_stage_idea import CARD, response, setup
    from test_research_injection import populated
    from sciel.engine.types import Program

    rows = [RequestNotDispatched('fixture pre-dispatch gate')]
    if stage == 'code':
        rows.insert(0, response(json.dumps(CARD), .03))
    research, idea, snapshot, _, log, charged, evaluated, run = setup(rows)
    research.knowledge = populated()
    research.attempts = research.outcomes = 2
    parent = Program('p1', snapshot.parent_code, snapshot.parent_score, {}, 0, 'seed')
    snapshot.research_context = research.context(SimpleNamespace(_all={'p1': parent}), parent, 0, 3)
    snapshot.kb_lessons = snapshot.research_context['lesson_texts']
    previous_lessons = research.knowledge.lessons
    with pytest.raises(CampaignStopped):
        run()
    assert not evaluated
    assert charged == ([.03] if stage == 'code' else [0.])
    event = next(e for e in reversed(log.events) if e.get('action') == 'stopped')
    assert event['stage'] == stage and event['generation_dispatch'] == 'not_dispatched'
    assert event['current_call_cost_unknown'] is False
    assert research.knowledge.lessons == previous_lessons
    if stage == 'code':
        assert research.knowledge.experiments[-1]['generation_dispatch'] == 'not_dispatched'
        assert idea.calls_returned == 1
        card_request = next(e for e in log.events if e.get('stage') == 'card' and e.get('action') == 'request_start')
        assert card_request['injected_ids'] == snapshot.research_context['injected_ids']
        assert 'RESEARCH_LESSON_JSON' in card_request['prompt']
    else:
        assert len(research.knowledge.experiments) == 2 and idea.calls_returned == 0


@pytest.mark.parametrize('failure', ['nontext', 'validation_raised'])
def test_two_stage_returned_code_receipt_survives_preproposal_failure(failure):
    from test_two_stage_idea import CARD, code_response, response, setup
    from sciel.engine.types import NoChild

    second = None if failure == 'nontext' else code_response()
    research, idea, _, _, _, charged, evaluated, run = setup([
        response(json.dumps(CARD), .03), response(second, .07)])
    if failure == 'nontext':
        result = run()
        assert isinstance(result, NoChild) and result.cost == pytest.approx(.1)
        assert charged == []  # The returned NoChild carries the charge to Engine.
    else:
        def invalid_callback(_source):
            raise RuntimeError('inert validation fixture')
        with pytest.raises(CampaignStopped):
            run(novelty_check=invalid_callback)
        assert charged == [pytest.approx(.1)]
    assert evaluated == [] and idea.calls_returned == 2
    record = research.knowledge.experiments[-1]
    assert record['generation_dispatch'] == 'returned'
    assert record['generation_cost'] == .07
    assert record['actual_delta'] is record['score'] is record['hypothesis'] is None
    assert record['status'] == ('no_child' if failure == 'nontext' else 'generation_stopped')
