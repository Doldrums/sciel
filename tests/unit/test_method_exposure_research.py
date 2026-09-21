"""Actual G prompt/transport/outcome bindings; inert providers/evaluators only."""
from copy import deepcopy
import json

from greenlet import greenlet, getcurrent
import pytest

from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.engine.llm.client import LLMError
from sciel.engine.provider_abandonment import AbandonedProviderRequest
from sciel.engine.research import digest
from sciel.engine.research_cells_context import compact_context
from sciel.engine.types import NoChild
from sciel.engine.worker import run_worker
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis.rendering import decode_evidence, serialize_context
from sciel.knowledge_base import KnowledgeBase
from sciel.methods.models import content_hash
from test_research_cell_context import card
from test_research_failure_context import setup, CANDIDATE


def measured(code):
    assert code == CANDIDATE
    return CascadeOutcome(EvalResult(ok=True, score=.6, cases={'sample': .6}), .6, 'full')


def prepared(*, cards=None):
    research, llm, rows, snapshot = setup()
    snap = snapshot()
    if cards is not None:
        snap.research_context['research_cells_context'] = compact_context(cards)
    return research, llm, rows, snap


def actual_context(prompt):
    raw = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1])[0]
    return decode_evidence(raw)


def execute(research, llm, snap, **options):
    return run_worker('full_rewrite', snap, llm, measured, dup_cap=1,
                      research=research, **options)


def test_old_card_without_new_episode_is_bound_to_actual_request_and_canonical_outcome():
    research, llm, rows, snap = prepared(cards=[card()])
    assert snap.tool_episode_plan is None
    result = execute(research, llm, snap)
    record = research.knowledge.experiments[0]
    exposure = record['method_exposures']
    actual = actual_context(result.prompt)['research_cells_context']
    assert exposure['prompt_sha256'] == digest(result.prompt) == digest(llm.prompts[0])
    assert exposure['cards'][0]['card_sha256'] == content_hash(actual['cards'][0])
    assert exposure['cards'][0]['source'] == actual['cards'][0]['source']
    assert record['generation_response_sha256'] == digest(result.response)
    assert record['generation_dispatch'] == 'returned'
    assert result.meta['research_outcome_ids'] == [record['id']] == [result.meta['research_id']]
    emitted = next(r['evidence'] for r in rows if r['phase'] == 'outcome')
    assert result.meta['method_followup_records'] == [emitted]
    assert result.meta['method_followup_records'][0] is not emitted
    assert record['source_sha256'] == digest(result.child_code)
    assert record['score'] == .6 and record['actual_delta'] == .6 - .5
    assert record['evaluation']['score'] == .6
    assert 'method_reward' not in record and research.knowledge.lessons == []
    assert next(r for r in rows if r['phase'] == 'attempt')['evidence']['method_exposures'] == exposure
    research.complete_step()
    clone, _, _, _ = setup()
    clone.load_state(research.state())
    assert clone.knowledge.state() == research.knowledge.state()


def test_legacy_attempt_has_no_new_optional_evidence_fields():
    research, llm, _, snap = prepared()
    execute(research, llm, snap)
    record = research.knowledge.experiments[0]
    assert 'method_exposures' not in record and 'generation_response_sha256' not in record
    clone = KnowledgeBase(max_lessons=10, max_experiments=20, max_fragments=5)
    clone.load_state(research.knowledge.state())
    assert clone.state() == research.knowledge.state()


def test_final_context_budget_omits_card_before_exposure_capture(monkeypatch):
    research, llm, _, snap = prepared(cards=[card(text='LARGE_CARD ' + 'x' * 7000)])
    monkeypatch.setattr('sciel.engine.research.MAX_CONTEXT_CHARS', 2000)
    result = execute(research, llm, snap)
    actual = actual_context(result.prompt)['research_cells_context']
    exposure = research.knowledge.experiments[0]['method_exposures']
    assert actual['cards'] == exposure['cards'] == []
    assert exposure['omitted_count'] == actual['omitted_count'] == 1
    assert exposure['omission_reasons'] == {'research_context_limit': 1}
    assert 'LARGE_CARD' not in result.prompt


def test_removed_whole_context_never_credits_stale_local_card(monkeypatch):
    baseline, _, _, baseline_snap = prepared()
    _, prompt = baseline.begin(baseline_snap, 0, 'full_rewrite', 'base prompt')
    limit = len(serialize_context(actual_context(prompt))) + 5
    research, _, _, snap = prepared(cards=[card()])
    monkeypatch.setattr('sciel.engine.research.MAX_CONTEXT_CHARS', limit)
    ident, prompt = research.begin(snap, 0, 'full_rewrite', 'base prompt')
    assert 'research_cells_context' not in actual_context(prompt)
    assert 'method_exposures' not in research.pending[ident]


@pytest.mark.parametrize('failure,dispatch', [
    (RequestNotDispatched('capacity refused'), 'not_dispatched'),
    (LLMError('unknown transport outcome'), 'unknown'),
    (RuntimeError('unknown custom provider failure'), 'unknown'),
])
def test_nonreturned_generation_never_gets_response_hash_or_measured_quality(failure, dispatch):
    research, llm, _, snap = prepared(cards=[card()])
    charged = []
    research.charge = charged.append
    def fail(*args):
        raise failure
    llm.generate = fail
    with pytest.raises(CampaignStopped):
        execute(research, llm, snap)
    record = research.knowledge.experiments[0]
    assert record['generation_dispatch'] == dispatch
    assert 'generation_response_sha256' not in record
    assert record['generation_cost'] is record['score'] is record['actual_delta'] is None
    assert record['hypothesis'] is record['evaluation'] is None
    assert sum(charged) == 0


def test_abandoned_generation_is_one_unknown_outcome_without_retry():
    research, llm, _, snap = prepared(cards=[card()])
    calls = []
    def fail(*args):
        calls.append(args)
        raise AbandonedProviderRequest({'unknown_cost': True, 'retry': False}, {})
    llm.generate = fail
    result = execute(research, llm, snap)
    record = research.knowledge.experiments[0]
    assert len(calls) == 1 and result.cost == 0
    assert result.meta['research_outcome_ids'] == [record['id']]
    assert record['generation_dispatch'] == 'unknown' and record['generation_cost'] is None
    assert 'generation_response_sha256' not in record


@pytest.mark.parametrize('mode', ['malformed', 'unknown_cost', 'nontext'])
def test_returned_unusable_response_still_binds_actual_transport(mode):
    research, llm, _, snap = prepared(cards=[card()])
    reply = llm.generate('fixture', snap.model)
    llm.prompts.clear()
    if mode == 'malformed':
        reply.response_error = {'reason': 'invalid_response'}
    elif mode == 'unknown_cost':
        reply.cost = None
    else:
        reply.text = None
    llm.generate = lambda *args: reply
    charged = []
    research.charge = charged.append
    if mode == 'malformed':
        result = execute(research, llm, snap)
        assert result.cost == .01 and charged == []
    else:
        with pytest.raises(CampaignStopped):
            execute(research, llm, snap)
        assert sum(charged) == (0 if mode == 'unknown_cost' else .01)
    record = research.knowledge.experiments[0]
    assert record['generation_dispatch'] == 'returned'
    assert record['generation_cost'] == (None if mode == 'unknown_cost' else .01)
    if mode == 'nontext':
        assert 'generation_response_sha256' not in record
    else:
        assert record['generation_response_sha256'] == digest(reply.text)
    assert record['evaluation'] is record['score'] is record['actual_delta'] is None


def test_capture_failure_preserves_known_paid_cost(monkeypatch):
    research, llm, _, snap = prepared(cards=[card()])
    charged = []
    research.charge = charged.append
    def fail(*args):
        raise ValueError('inconsistent response binding')
    monkeypatch.setattr(research, 'record_generation_response', fail)
    with pytest.raises(CampaignStopped):
        execute(research, llm, snap)
    assert charged == [.01]
    record = research.knowledge.experiments[0]
    assert record['generation_cost'] == .01 and record['generation_dispatch'] == 'returned'


def test_all_rejected_attempts_are_returned_without_history_scan():
    research, llm, _, snap = prepared(cards=[card()])
    good = llm.generate('fixture', snap.model)
    bad = deepcopy(good)
    bad.text = 'No forecast or code.'
    replies = iter([bad, good])
    llm.generate = lambda *args: next(replies)
    result = run_worker('full_rewrite', snap, llm, measured, dup_cap=2, research=research)
    records = research.knowledge.experiments
    assert [r['status'] for r in records] == ['hypothesis_invalid', 'evaluated']
    assert result.meta['research_outcome_ids'] == [r['id'] for r in records]
    assert result.cost == .02
    assert [r['generation_response_sha256'] for r in records] == [digest(bad.text), digest(good.text)]


def test_real_greenlet_interleaving_does_not_mix_outcome_ids():
    research, llm, _, first = prepared(cards=[card('a')])
    second = deepcopy(first)
    second.iteration_no += 1
    second.research_context = deepcopy(first.research_context)
    second.research_context['iteration'] += 1
    reply = llm.generate('fixture', first.model)
    def suspend(*args):
        getcurrent().parent.switch('at_provider')
        return deepcopy(reply)
    llm.generate = suspend
    one = greenlet(lambda: execute(research, llm, first))
    two = greenlet(lambda: execute(research, llm, second))
    assert one.switch() == two.switch() == 'at_provider'
    result_two = two.switch()
    result_one = one.switch()
    assert one.dead and two.dead
    assert result_one.meta['research_outcome_ids'] == ['r000001a1']
    assert result_two.meta['research_outcome_ids'] == ['r000002a1']
    assert [r['id'] for r in research.knowledge.experiments] == ['r000002a1', 'r000001a1']


def test_detached_records_survive_one_record_retention_with_interleaved_retries():
    research, llm, _, first = prepared(cards=[card('a')])
    research.knowledge.max_experiments = 1
    second = deepcopy(first)
    reply = llm.generate('fixture', first.model)
    bad = deepcopy(reply)
    bad.text = 'No hypothesis.'
    counts = {}
    def suspend(*args):
        current = getcurrent()
        counts[current] = counts.get(current, 0) + 1
        current.parent.switch('at_provider')
        return deepcopy(bad if counts[current] == 1 else reply)
    llm.generate = suspend
    def worker(snap):
        return run_worker('full_rewrite', snap, llm, measured, dup_cap=2, research=research)
    one, two = greenlet(lambda: worker(first)), greenlet(lambda: worker(second))
    assert one.switch() == two.switch() == 'at_provider'
    assert one.switch() == two.switch() == 'at_provider'
    result_two, result_one = two.switch(), one.switch()
    assert len(research.knowledge.experiments) == 1
    assert result_one.meta['research_outcome_ids'] == ['r000001a1', 'r000003a2']
    assert result_two.meta['research_outcome_ids'] == ['r000002a1', 'r000004a2']
    for result in (result_one, result_two):
        records = result.meta['method_followup_records']
        assert [r['id'] for r in records] == result.meta['research_outcome_ids']
        assert [r['status'] for r in records] == ['hypothesis_invalid', 'evaluated']
        assert [r['generation_response_sha256'] for r in records] == [digest(bad.text), digest(reply.text)]
    # Copies remain independent of the live, bounded memory and of each other.
    result_two.meta['method_followup_records'][0]['failure'] = 'modified copy'
    assert result_one.meta['method_followup_records'][0]['failure'] != 'modified copy'


def test_structured_response_fallback_binds_raw_return_before_source_validation():
    from test_research_circle_body import fixture, response, body
    f = fixture()
    # End the fixture's unexposed preparation; prepare a genuine exposed body
    # request through begin, without running candidate source.
    f.research.resolve(f.ident, status='no_child', failure='Unexposed fixture setup')
    f.snapshot.research_context['research_cells_context'] = compact_context([card()])
    ident, prompt = f.research.begin(f.snapshot, 0, 'circle_generate', 'Write one body.',
        response_instruction=body.response_instruction(), context_scope=body.OPERATION)
    reply = response(code='invalid syntax!')
    decoded = f.research.record_response(ident, reply, response_format=body.OPERATION)
    assert decoded['proposal'] is None
    record = f.research.resolve(ident, status='no_child', failure='Invalid body')
    assert record['generation_response_sha256'] == digest(reply.text)
    assert record['method_exposures']['prompt_sha256'] == digest(prompt)
    assert record['source_sha256'] is None and record['actual_delta'] is None


def test_proposal_fallback_binds_return_without_ordinary_worker_hook():
    from types import SimpleNamespace
    research, llm, _, snap = prepared(cards=[card()])
    ident, prompt = research.begin(snap, 0, 'full_rewrite', 'direct generation')
    reply = llm.generate(prompt, snap.model)
    research.proposal(ident, reply, CANDIDATE, SimpleNamespace(code=CANDIDATE))
    record = research.resolve(ident, status='no_child', failure='Not evaluated')
    assert record['generation_response_sha256'] == digest(reply.text)
    assert record['generation_dispatch'] == 'returned'
    assert record['actual_delta'] is None


def test_common_wrapper_captures_specialized_early_return():
    research, llm, _, snap = prepared(cards=[card()])
    class Idea:
        def __init__(self):
            self.research = research
        def run(self, snapshot, *args, **kwargs):
            ident, _ = research.begin(snapshot, 0, 'idea', 'specialized prompt')
            record = research.resolve(ident, status='no_child', failure='Fixture abstention')
            return NoChild(island=0, reason='fixture', meta={'research_id': record['id']})
    result = run_worker('idea', snap, llm, measured, research=research, idea=Idea())
    assert result.meta['research_outcome_ids'] == [result.meta['research_id']]


@pytest.mark.parametrize('damage', ['exposure_digest', 'response_digest', 'unknown_dispatch'])
def test_invalid_optional_evidence_is_rejected_before_kb_mutation(damage):
    research, llm, _, snap = prepared(cards=[card()])
    execute(research, llm, snap)
    record = research.knowledge.experiments[0]
    if damage == 'exposure_digest':
        record['method_exposures']['prompt_sha256'] = 'f' * 64
    elif damage == 'response_digest':
        record['generation_response_sha256'] = 'not-a-hash'
    else:
        record['generation_dispatch'] = 'unknown'
    kb = KnowledgeBase(max_lessons=10, max_experiments=20, max_fragments=5)
    before = kb.state()
    with pytest.raises(ValueError):
        kb.record(record)
    assert kb.state() == before


def test_bound_exposure_survives_lesson_snapshot_and_restore():
    research, llm, _, snap = prepared(cards=[card()])
    execute(research, llm, snap)
    record = research.knowledge.experiments[0]
    research.knowledge.add_lessons([{'rule': 'A tentative association.', 'polarity': 'positive',
        'scope': 'branch', 'evidence_ids': [record['id']]}], allowed_evidence_ids=[record['id']])
    state = research.knowledge.state()
    clone = KnowledgeBase(max_lessons=10, max_experiments=20, max_fragments=5)
    clone.load_state(state)
    assert clone.state() == state
    assert clone.lessons[0]['evidence_snapshots'][0]['method_exposures'] == record['method_exposures']


def test_bad_exposure_checkpoint_load_is_atomic():
    research, llm, _, snap = prepared(cards=[card()])
    execute(research, llm, snap)
    state = research.knowledge.state()
    clone = KnowledgeBase(max_lessons=10, max_experiments=20, max_fragments=5)
    clone.load_state(state)
    before = clone.state()
    state['experiments'][0]['method_exposures']['cards'][0]['source']['receipt_id'] = 'f' * 64
    with pytest.raises(ValueError):
        clone.load_state(state)
    assert clone.state() == before
