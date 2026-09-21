"""Controlled-operation dispatch with fake LLM/outcomes; no candidate execution."""
import copy
import hashlib
import json
from types import SimpleNamespace
import pytest
from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.config import apply_defaults
from sciel.engine.db import Database, normalized_hash
from sciel.engine.intervention import CircleIntervention
from sciel.engine.llm.client import LLMResponse
from sciel.engine.prompt.sources import source_digest
from sciel.engine.research import ResearchController
from sciel.engine.types import NoChild, Result, Snapshot
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_interventions as edits
from intervention_worker_fixtures import (
    plan, response, Log, Model,
    setup, run, journal, move_parent,
)


def test_one_paid_forecast_builds_exact_edit_before_eval_and_preserves_real_measurement():
    s = setup([response()])
    result = run(s)
    assert isinstance(result, Result) and result.ok
    assert result.child_code == edits.source_for('incumbent') == s.evaluated[0]
    assert len(s.model.calls) == 1 and result.cost == .03 and s.charged == []
    assert result.raw_first_score == .88 and result.timing == .012 and result.reached == 'full'
    assert result.artifacts == {'test_only': {'candidate_executed': False}}
    proof = result.meta['intervention']['proof']
    assert proof == edits.apply(s.snapshot.parent_code, 'incumbent')['proof']
    record = s.research.knowledge.experiments[-1]
    assert record['actual_delta'] == .9 - .8 and record['parent_cases'] == s.parent.cases
    assert record['evaluation']['cases'] == result.cases
    assert record['hypothesis']['text'] == f'{edits.OPERATION}:incumbent'
    assert not record['source_repaired']
    assert record['source_sha256'] == record['proposed_sha256'] == proof['child_source_sha256']
    assert 'guess = incumbent._params' in record['change_excerpt']
    assert s.worker.seen_plans == [proof['plan_key']]
    assert s.worker.known_cost == .03 and s.worker.evaluations == 1
    prompt = s.model.calls[0][0]
    assert 'COMPLETE Python program' not in prompt
    assert 'Return exactly HYPOTHESIS_JSON:' in prompt
    allowed = json.JSONDecoder().raw_decode(prompt.split('ALLOWED_INTERVENTIONS_JSON:\n', 1)[1])[0]
    assert [p['policy'] for p in allowed] == ['incumbent', 'alternating']


@pytest.mark.parametrize('text', [
    '', '{}', 'prose\n' + plan(), plan() + '\nprose',
    plan() + '\n```python\nraise RuntimeError("must never execute")\n```',
    plan('initial'), plan('not-a-policy'), plan(hypothesis='Use the incumbent'),
    plan(hypothesis=f' {edits.OPERATION}:incumbent '),
    plan(extra='unknown'), plan(predicted_low=True), plan(confidence=float('nan')),
    plan().replace('"confidence": 0.5', '"confidence": 0.5,"confidence": 0.6'),
])
def test_bad_or_noop_plan_is_paid_but_never_code_or_quality(text):
    s = setup([response(text)])
    result = run(s)
    assert isinstance(result, NoChild) and result.reason == 'intervention_plan_invalid'
    assert result.cost == .03 and len(s.model.calls) == 1 and not s.evaluated
    assert not s.worker.seen_plans and s.charged == []
    record = s.research.knowledge.experiments[-1]
    assert record['status'] == 'hypothesis_invalid'
    assert record['actual_delta'] is record['score'] is None
    assert record['generation_cost'] == .03 and record['generation_dispatch'] == 'returned'
    assert not s.research.knowledge.lessons


def test_seen_plan_persists_after_resume_even_without_child_in_active_population():
    s = setup([response(), response(plan(reasoning='Different words, same operation')),
               response(plan('alternating'))])
    first = run(s)
    state = json.loads(json.dumps(s.worker.state()))
    replacement = CircleIntervention(s.research)
    replacement.load_state(state)
    assert replacement.state() == state
    s.worker = replacement
    second = run(s)
    assert second.reason == 'intervention_plan_repeated' and second.cost == .03
    assert len(s.evaluated) == 1 and s.worker.seen_plans == [first.meta['intervention']['plan_key']]
    allowed = json.JSONDecoder().raw_decode(s.model.calls[1][0].split('ALLOWED_INTERVENTIONS_JSON:\n')[1])[0]
    assert [p['policy'] for p in allowed] == ['alternating']
    run(s)
    exhausted = run(s)
    assert exhausted.reason == 'intervention_exhausted' and exhausted.cost == 0
    assert len(s.model.calls) == 3 and len(s.evaluated) == 2
    assert s.research.attempts == s.research.outcomes == 3


def test_active_sources_make_exhaustion_free_without_fabricated_research_outcome():
    s = setup([])
    s.snapshot.code_hashes.update(normalized_hash(edits.source_for(p)) for p in edits.POLICIES)
    result = run(s)
    assert result.reason == 'intervention_exhausted' and result.cost == 0
    assert s.research.attempts == s.research.outcomes == 0
    assert not s.model.calls and not s.evaluated and s.worker.seen_plans == []
    other = CircleIntervention(s.research)
    other.load_state(json.loads(json.dumps(s.worker.state())))
    assert other.state() == s.worker.state()


@pytest.mark.parametrize('key,value', [('parent_code', 'different'), ('parent_source_sha256', '0' * 64),
                                     ('parent_id', 'other'), ('parent_score', .7)])
def test_mismatched_measured_parent_rejected_before_request(key, value):
    s = setup([])
    s.snapshot.research_context[key] = value
    with pytest.raises(ValueError, match='exact measured parent'):
        run(s)
    assert not s.model.calls and not s.evaluated and s.research.attempts == 0


def test_unknown_source_never_prompts_model_or_evaluates():
    s = setup([])
    s.snapshot.parent_code += '# not the pinned family\n'
    s.snapshot.research_context.update(parent_code=s.snapshot.parent_code,
        parent_source_sha256=source_digest(s.snapshot.parent_code))
    result = run(s)
    assert result.reason == 'intervention_parent_unknown' and result.cost == 0
    assert not s.model.calls and not s.evaluated and s.research.attempts == 0


@pytest.mark.parametrize('error,dispatch,unknown', [
    (RequestNotDispatched('budget'), 'not_dispatched', False),
    (CampaignStopped('unknown reservation'), 'unknown', True),
    (RuntimeError('do not copy secret provider text'), 'unknown', True)])
def test_dispatch_stop_distinguishes_unknown_from_proven_unpaid_and_cannot_resume(error, dispatch, unknown):
    s = setup([error])
    with pytest.raises(CampaignStopped):
        run(s)
    assert not s.evaluated and sum(s.charged) == 0
    record = s.research.knowledge.experiments[-1]
    assert record['generation_dispatch'] == dispatch and record['actual_delta'] is None
    stopped = s.log.events[-1]
    assert stopped['action'] == 'stopped' and stopped['current_call_cost_unknown'] is unknown
    assert 'secret provider text' not in json.dumps(s.log.events)
    with pytest.raises(CampaignStopped):
        run(s)
    with pytest.raises(ValueError, match='unfinished'):
        CircleIntervention(s.research).load_state(s.worker.state())
    assert len(s.model.calls) == 1


@pytest.mark.parametrize('changes,expected_cost,cost_unknown', [
    ({'text': None}, .03, False), ({'prompt_tokens': True}, .03, False),
    ({'model': None}, .03, False), ({'cost': float('nan')}, 0., True),
    ({'cost': True}, 0., True), ({'cost': 10 ** 1000}, 0., True),
    ({'cost': 0., 'text': None}, 0., False)])
def test_malformed_return_stops_and_retains_known_charge_once(changes, expected_cost, cost_unknown):
    row = response()
    for key, value in changes.items():
        setattr(row, key, value)
    s = setup([row])
    with pytest.raises(CampaignStopped):
        run(s)
    assert not s.evaluated and s.charged == [expected_cost]
    assert s.worker.known_cost == expected_cost
    record = s.research.knowledge.experiments[-1]
    assert record['generation_dispatch'] == 'returned' and record['actual_delta'] is None
    assert record['generation_cost'] == (None if cost_unknown else expected_cost)
    assert s.log.events[-1]['current_call_cost_unknown'] is cost_unknown
    assert any(e.get('action') == 'request_result' for e in s.log.events)


def test_novelty_rejection_records_built_plan_but_no_evaluation_or_fake_quality():
    s = setup([response()])
    calls = []
    def novelty(code):
        calls.append(code)
        return {'accept': False, 'similarity': .999}
    result = run(s, novelty_check=novelty)
    assert result.reason == 'gate_rejected' and result.cost == .03
    assert calls == [edits.source_for('incumbent')] and not s.evaluated
    assert len(s.worker.seen_plans) == 1
    record = s.research.knowledge.experiments[-1]
    assert record['status'] == 'no_child' and record['actual_delta'] is None


def test_evaluator_exception_preserves_source_failure_and_charge_without_retry():
    s = setup([response()])
    def broken(code):
        assert code == edits.source_for('incumbent')
        raise RuntimeError('fixture failure')
    with pytest.raises(CampaignStopped):
        run(s, broken)
    record = s.research.knowledge.experiments[-1]
    assert record['status'] == 'eval_failed' and record['actual_delta'] is record['score'] is None
    assert record['source_sha256'] == source_digest(edits.source_for('incumbent'))
    assert s.charged == [.03] and s.worker.evaluations == 1


@pytest.mark.parametrize('ok', [True, False])
def test_invalid_geometry_zero_is_distinct_from_returned_evaluator_failure(ok):
    s = setup([response()])
    result = run(s, lambda _: CascadeOutcome(EvalResult(ok, 0., {'validity': 0.} if ok else {},
        artifacts={'error': 'failed'} if not ok else {}), 0., 'full', False))
    assert isinstance(result, Result) and result.ok is ok and result.cost == .03
    record = s.research.knowledge.experiments[-1]
    assert record['status'] == ('evaluated' if ok else 'eval_failed')
    assert record['actual_delta'] == (-.8 if ok else None)
    assert record['evaluation']['ok'] is ok and not s.charged


@pytest.mark.parametrize('change', [
    {'identity': {}}, {'pending': {}}, {'known_cost': True}, {'known_cost': float('inf')},
    {'calls_started': True}, {'calls_returned': 0}, {'evaluations': 2},
    {'seen_plans': []}, {'seen_plans': [], 'seen_plan_count': 0},
    {'seen_plans': ['invented']}, {'stopped_reason': ''}, {'unexpected': True},
])
def test_corrupt_component_state_is_rejected_atomically(change):
    s = setup([response()])
    run(s)
    state = copy.deepcopy(s.worker.state())
    state.update(change)
    other = CircleIntervention(s.research)
    before = other.state()
    with pytest.raises(ValueError):
        other.load_state(state)
    assert other.state() == before


def test_fragment_catalog_and_whole_lesson_reach_request_without_extra_calls():
    s = setup([response()])
    code = 'saved_fragment = 1\n'
    s.research.knowledge.record({'id': 'r900000a1', 'branch': s.parent.id, 'parent_id': s.parent.id,
        'parent_score': .5, 'score': .6, 'actual_delta': .6 - .5, 'status': 'evaluated',
        'source_sha256': source_digest(code), 'proposed_sha256': source_digest(code),
        'hypothesis': None, 'injected_ids': [], 'code_fragment': code})
    ids = s.research.knowledge.add_lessons([{'rule': 'Restart refinement improved in this observation.',
        'polarity': 'positive', 'scope': 'branch', 'evidence_ids': ['r900000a1']}],
        allowed_evidence_ids=['r900000a1'])
    ctx = s.research.context(s.db, s.parent, 0, 1)
    s.snapshot.research_context, s.snapshot.kb_lessons = ctx, ctx['lesson_texts']
    run(s)
    prompt = s.model.calls[0][0]
    assert len(s.model.calls) == 1 and all(text in prompt for text in ctx['lesson_texts'])
    assert ctx['injected_ids'] == ids
    catalog = json.loads(prompt.rsplit('SOURCE_CATALOG_JSON:\n', 1)[1])['catalog']
    assert catalog[source_digest(code)] == code
    assert catalog[source_digest(s.snapshot.parent_code)] == s.snapshot.parent_code
    assert s.research.knowledge.experiments[-1]['injected_ids'] == ids


def test_consumed_plans_are_bound_to_actual_ordered_journal_and_only_anchored_prefix(tmp_path):
    s = setup([response(), response(plan('alternating'))])
    run(s)
    run(s)
    path = tmp_path / 'events.jsonl'
    anchor = journal(path, s.log.events)
    s.worker.assert_journal(path, anchor)
    # An unrelated later event is outside this already authenticated prefix.
    # The enclosing research loader separately forbids paid post-checkpoint work.
    with path.open('ab') as stream:
        stream.write(b'{"event":"run_end"}\n')
    s.worker.assert_journal(path, anchor)
    changed = copy.deepcopy(s.worker.state())
    changed['seen_plans'].reverse()
    restored = CircleIntervention(s.research)
    restored.load_state(changed)  # Individually valid keys/counters alone are insufficient.
    with pytest.raises(ValueError, match='journal proofs'):
        restored.assert_journal(path, anchor)


@pytest.mark.parametrize('change', ['valid_key_substitution', 'proof_span', 'duplicate_plan',
                                  'causal_claim', 'prefix_bytes', 'duplicate_json_key'])
def test_journal_guards_valid_looking_state_substitution_and_malformed_proofs(tmp_path, change):
    s = setup([response()])
    run(s)
    events = copy.deepcopy(s.log.events)
    bound = next(e for e in events if e.get('action') == 'source_binding')
    state = copy.deepcopy(s.worker.state())
    if change == 'valid_key_substitution':
        state['seen_plans'] = [edits.apply(edits.source_for('initial'), 'alternating')['proof']['plan_key']]
    elif change == 'proof_span':
        bound['proof']['after']['byte_span'][1] += 1
    elif change == 'duplicate_plan':
        events.append(copy.deepcopy(bound))
    elif change == 'causal_claim':
        bound['semantic_or_causal_claim'] = True
    path = tmp_path / 'events.jsonl'
    anchor = journal(path, events)
    if change == 'prefix_bytes':
        path.write_bytes(path.read_bytes().replace(b'returned', b'retirned', 1))
    if change == 'duplicate_json_key':
        data = path.read_bytes().replace(b'"event": "research"',
                                        b'"event": "research", "event": "research"', 1)
        path.write_bytes(data)
        anchor = {'byte_offset': len(data), 'prefix_sha256': hashlib.sha256(data).hexdigest()}
    restored = CircleIntervention(s.research)
    restored.load_state(state)
    with pytest.raises(ValueError):
        restored.assert_journal(path, anchor)


def test_journal_failure_after_paid_response_never_loses_cost_or_replays():
    s = setup([response()])
    original = s.log.research
    def broken(**event):
        if event.get('action') == 'request_result':
            raise OSError('fixture disk failure')
        original(**event)
    s.log.research = broken
    with pytest.raises(CampaignStopped):
        run(s)
    assert not s.evaluated and s.charged == [.03]
    assert s.research.knowledge.experiments[-1]['generation_cost'] == .03
    with pytest.raises(CampaignStopped):
        run(s)
    assert len(s.model.calls) == 1


def test_repaired_builder_source_is_refused_even_if_the_validator_accepts_it(monkeypatch):
    s = setup([response()])
    from sciel.engine import intervention
    actual = intervention.validate_candidate
    def changed(code, **kwargs):
        result = actual(code, **kwargs)
        result.code += '\n# unexpected repair\n'
        return result
    monkeypatch.setattr(intervention, 'validate_candidate', changed)
    with pytest.raises(CampaignStopped):
        run(s)
    assert not s.evaluated and s.charged == [.03]
    assert s.research.knowledge.experiments[-1]['actual_delta'] is None


def test_removed_seed_is_still_consumed_when_improved_child_becomes_parent_after_resume():
    s = setup([response(), response(plan('initial')), response(plan('alternating'))])
    run(s)
    saved = json.loads(json.dumps(s.worker.state()))
    s.worker = CircleIntervention(s.research)
    s.worker.load_state(saved)
    move_parent(s, 'incumbent')
    repeated = run(s)
    assert repeated.reason == 'intervention_plan_repeated'
    allowed = json.JSONDecoder().raw_decode(s.model.calls[-1][0].split('ALLOWED_INTERVENTIONS_JSON:\n')[1])[0]
    assert [row['policy'] for row in allowed] == ['alternating']
    assert len(s.evaluated) == 1
    run(s)
    move_parent(s, 'alternating')
    exhausted = run(s)
    assert exhausted.reason == 'intervention_exhausted' and exhausted.cost == 0
    assert len(s.model.calls) == 3 and len(s.evaluated) == 2


def test_failed_nonadmitted_source_cannot_be_rediscovered_from_a_different_parent():
    s = setup([response(), response(plan('alternating'))])
    failure = lambda _: CascadeOutcome(EvalResult(False, 0., {}, artifacts={'error': 'fixture'}),
                                      0., 'full', False)
    first = run(s, failure)
    assert not first.ok
    run(s)
    # Incumbent was never admitted; initial can also leave the active population.
    # Their prior source identities remain derivable from the retained proofs.
    move_parent(s, 'alternating')
    before = len(s.model.calls)
    exhausted = run(s)
    assert exhausted.reason == 'intervention_exhausted' and exhausted.cost == 0
    assert len(s.model.calls) == before


def test_prompt_labels_the_measured_policy_for_initial_then_alternating_parent():
    s = setup([response(plan('alternating')), response(plan('incumbent'))])
    # A misleading model-written card must never define the authoritative policy.
    s.snapshot.parent_card = 'Every restart uses the incumbent.'
    first = run(s)
    move_parent(s, 'alternating')
    s.snapshot.parent_card = 'Every restart uses the initial grid.'
    second = run(s)
    assert [result.response for result in (first, second)] == [plan('alternating'), plan('incumbent')]
    assert first.cost == second.cost == .03 and s.worker.known_cost == .06
    assert s.charged == [] and len(s.model.calls) == len(s.evaluated) == 2
    for index, (parent, target) in enumerate([('initial', 'alternating'), ('alternating', 'incumbent')]):
        prompt = s.model.calls[index][0]
        current = json.JSONDecoder().raw_decode(prompt.split('CURRENT_PARENT_POLICY_JSON:\n', 1)[1])[0]
        allowed = json.JSONDecoder().raw_decode(prompt.split('ALLOWED_INTERVENTIONS_JSON:\n', 1)[1])[0]
        proof = edits.apply(edits.source_for(parent), target)['proof']
        assert current['policy'] == parent
        assert current['source_sha256'] == source_digest(edits.source_for(parent))
        assert current['assignment'] == proof['before']
        assert current['description'] == edits.policy_descriptor(parent)['change']
        assert current['uncertain_effect'] == edits.policy_descriptor(parent)['uncertain_effect']
        assert 'unverified interpretations, not measured benefits' in prompt
        assert 'do not assume the parent always uses the original-grid policy' in prompt
        chosen = next(row for row in allowed if row['policy'] == target)
        assert (chosen['from_policy'], chosen['to_policy']) == (parent, target)
        assert chosen['before_assignment'] == current['assignment']
        assert chosen['after_assignment'] == proof['after']
        start, end = current['assignment']['byte_span']
        assert edits.source_for(parent).encode()[start:end].decode() == current['assignment']['text']
    # The second forecast/effect uses the newly measured alternating parent,
    # rather than reusing the original baseline's score or operation name.
    records = s.research.knowledge.experiments
    assert records[0]['actual_delta'] == .9 - .8
    assert records[1]['parent_score'] == .95 and records[1]['actual_delta'] == .9 - .95
    second_allowed = json.JSONDecoder().raw_decode(s.model.calls[1][0].split('ALLOWED_INTERVENTIONS_JSON:\n', 1)[1])[0]
    assert [row['policy'] for row in second_allowed] == ['incumbent']


@pytest.mark.parametrize('field', ['schema', 'identity'])
def test_parent_policy_prompt_contract_rejects_v1_checkpoint_atomically(field):
    s = setup([response()])
    run(s)
    saved = json.loads(json.dumps(s.worker.state()))
    assert saved['schema'] == saved['identity']['schema'] == 'sciel-circle-intervention-worker/2'
    if field == 'schema':
        saved['schema'] = 'sciel-circle-intervention-worker/1'
    else:
        saved['identity']['schema'] = 'sciel-circle-intervention-worker/1'
    other = CircleIntervention(s.research)
    before = other.state()
    with pytest.raises(ValueError, match='checkpoint'):
        other.load_state(saved)
    assert other.state() == before
