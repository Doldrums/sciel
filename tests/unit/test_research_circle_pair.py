"""Real Research/KB integration with pure body construction, never model/eval I/O."""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults
from sciel.engine.db import Database
from sciel.engine.research import ResearchController, digest
from sciel.engine.types import Snapshot
from sciel.hypothesis import circle_proposal_pair as pair
from sciel.hypothesis import circle_restart_body as body
from sciel.hypothesis.circle_proposal_group import build_group, gate, group_card
from sciel.hypothesis.rendering import decode_evidence, presentation_identity
from sciel.knowledge_base.catalogs import canonical_json


FIRST = {'hypothesis': 'Change center noise.', 'predicted_low': -.1, 'predicted_high': .2,
         'reasoning': 'First independent prospective forecast.', 'confidence': .4}
SECOND = {'hypothesis': 'Change radius noise.', 'predicted_low': .01, 'predicted_high': .02,
          'reasoning': 'Second independent prospective forecast.', 'confidence': .8}
BODY0 = 'x = best_params.copy()\nx[:52] += uniform_noise[:52] / 2\nreturn x\n'
BODY1 = 'x = best_params.copy()\nx[52:] *= 0.99\nreturn x\n'


def response(first=BODY0, second=BODY1, **overrides):
    value = {'candidates': [{'hypothesis': FIRST, 'body': first}, {'hypothesis': SECOND, 'body': second}]}
    fields = dict(text=json.dumps(value, ensure_ascii=False), cost=.003, model='fake',
                  prompt_tokens=31, completion_tokens=29, finish_reason='stop', refusal=None)
    fields.update(overrides)
    return NS(**fields)


def fixture(*, inject=True):
    events, charges = [], []
    cfg = apply_defaults({'research': {'enabled': True, 'max_aux_calls': 0,
        'reflection_interval': 0, 'inject_lessons': inject}})['research']
    model = NS(models=['fake'], max_retries=1)
    log = NS(research=lambda **event: events.append(copy.deepcopy(event)))
    research = ResearchController(cfg, model, log, charges.append)
    db = Database(1)
    source = body.source_for()
    parent = db.add(source, .8, {'sum_radii': 2.108}, 0, 'seed')
    snapshot = Snapshot(0, 'circle_generate', 'fake', 0, parent.id, source, .8)
    snapshot.research_context = research.context(db, parent, 0, 1)
    ident, prompt = research.begin(snapshot, 0, 'circle_generate', 'Write two new bodies.',
        response_instruction=pair.response_instruction(), context_scope=body.OPERATION)
    return NS(research=research, events=events, charges=charges, db=db, parent=parent,
              snapshot=snapshot, ident=ident, prompt=prompt)


def group_for(f, r, *, selected=0, pending=False, provider_error=None):
    decoded = pair.decode_response(r.text)
    gates = []
    for item in decoded['entries']:
        if item['error'] is not None or provider_error is not None:
            gates.append(gate('not_applicable'))
        elif pending:
            gates.append(gate('pending'))
        elif selected is None or item['index'] < selected:
            gates.append(gate('rejected', reason='known_source', diagnostic='Already consumed.'))
        else:
            gates.append(gate('passed', evidence_sha256='a' * 64))
    built = (body.apply(f.snapshot.parent_code, decoded['entries'][selected]['proposal'].body)
             if selected is not None and not pending and provider_error is None else None)
    group = build_group(request_id=f.ident, raw_response=r.text,
        parent_source_sha256=digest(f.snapshot.parent_code), gates=gates,
        provider_error=provider_error, selected_proof=built['proof'] if built else None)
    return group, built


def recorded(f, r, *, selected=0):
    f.research.record_response(f.ident, r, response_format=pair.CONTRACT)
    group, built = group_for(f, r, selected=selected)
    f.research.bind_pair_group(f.ident, r, group)
    return group, built


def commit(f, r, built):
    return f.research.proposal(f.ident, r, built['code'], NS(code=built['code']),
        response_format=pair.CONTRACT, controlled_change=built['proof'])


def test_pair_raw_return_and_known_cost_are_durable_before_any_decoding(monkeypatch):
    f, r = fixture(), response()
    original = pair.decode_response
    def inspect(raw):
        assert f.events[-1]['phase'] == 'structured_response'
        assert f.events[-1]['response'] == raw == r.text
        record = f.research.pending[f.ident]
        assert record['generation_cost'] == r.cost and record['generation_dispatch'] == 'returned'
        assert record['hypothesis'] is record['source_sha256'] is record['proposed_sha256'] is None
        assert f.research.valid_hypotheses == 0
        return original(raw)
    monkeypatch.setattr(pair, 'decode_response', inspect)
    decoded = f.research.record_response(f.ident, r, response_format=pair.CONTRACT)
    assert len(decoded['entries']) == 2 and decoded['provider_error'] is None
    assert f.research.valid_hypotheses == 0 and f.research.pending[f.ident]['hypothesis'] is None
    assert f.events[-1]['phase'] == 'structured_pair'
    assert [e['hypothesis']['text'] for e in f.events[-1]['entries']] == [FIRST['hypothesis'], SECOND['hypothesis']]


def test_second_selected_forecast_alone_binds_one_outcome_and_exact_source():
    f, r = fixture(), response(first='return np.random.choice(26)')
    group, built = recorded(f, r, selected=1)
    assert f.research.valid_hypotheses == 1
    record = f.research.pending[f.ident]
    assert record['hypothesis'] == group['entries'][1]['hypothesis']
    assert record['hypothesis']['text'] == SECOND['hypothesis']
    assert record['source_sha256'] is record['proposed_sha256'] is None
    assert commit(f, r, built)
    assert f.events[-1]['phase'] == 'hypothesis' and f.events[-1]['response'] == r.text
    outcome = f.research.resolve(f.ident, status='evaluated', score=.9, source=built['code'])
    assert outcome['forecast_verdict'] == 'REFUTED' and outcome['actual_delta'] == .9 - .8
    assert outcome['proposal_group'] == group
    assert outcome['controlled_change'] == built['proof']
    assert outcome['source_sha256'] == digest(built['code'])
    assert outcome['generation_cost'] == r.cost
    assert outcome['proposal_group']['entries'][0]['hypothesis']['text'] == FIRST['hypothesis']
    assert all(not ({'score', 'actual_delta', 'forecast_verdict', 'child_id'} & set(e)) for e in group['entries'])
    f.research.complete_step()
    assert f.research.attempts == f.research.outcomes == f.research.valid_hypotheses == 1
    assert len(f.research.knowledge.experiments) == 1
    assert set(f.research.knowledge._evidence()) == {f.ident}
    assert not f.research._structured_responses and not f.research.pending


def test_both_eligible_bind_first_without_counting_alternative_as_experiment():
    f, r = fixture(), response()
    group, built = recorded(f, r)
    assert group['selected_index'] == 0 and group['entries'][1]['disposition'] == 'eligible_unselected'
    assert f.research.pending[f.ident]['hypothesis']['text'] == FIRST['hypothesis']
    assert f.research.valid_hypotheses == 1
    commit(f, r, built)
    f.research.resolve(f.ident, status='eval_failed', source=built['code'], failure='candidate failed')
    f.research.complete_step()
    assert f.research.knowledge.experiments[0]['score'] is None
    assert f.research.knowledge.experiments[0]['proposal_group'] == group


@pytest.mark.parametrize('mode', ['rejected', 'invalid_entries', 'bad_envelope', 'refused', 'length'])
def test_terminal_no_selection_retains_alternatives_but_no_ordinary_hypothesis(mode):
    f = fixture()
    r = (response(first='return np.random.choice(26)', second='return np.linalg.norm(best_params)')
         if mode == 'invalid_entries' else response(text='{bad') if mode == 'bad_envelope' else
         response(refusal='declined') if mode == 'refused' else
         response(finish_reason='length') if mode == 'length' else response())
    parsed = f.research.record_response(f.ident, r, response_format=pair.CONTRACT)
    group, _ = group_for(f, r, selected=None, provider_error=parsed['provider_error'])
    f.research.bind_pair_group(f.ident, r, group)
    record = f.research.resolve(f.ident, status='no_child', failure='no eligible alternative')
    assert record['proposal_group'] == group and group['selected_index'] is None
    assert all(record[k] is None for k in ('hypothesis', 'source_sha256', 'proposed_sha256', 'score', 'actual_delta', 'evaluation'))
    assert record['forecast_verdict'] == 'INCONCLUSIVE' and record['generation_cost'] == r.cost
    assert f.research.valid_hypotheses == 0 and f.research.outcomes == f.research.attempts == 1
    assert len(f.research.knowledge.experiments) == 1
    assert f.research.knowledge.lessons == []


def test_pending_assessment_cannot_bind_or_be_resolved_as_normal_nochild():
    f, r = fixture(), response()
    f.research.record_response(f.ident, r, response_format=pair.CONTRACT)
    group, _ = group_for(f, r, pending=True)
    before = canonical_json(f.research.state())
    with pytest.raises(ValueError, match='pending admission'):
        f.research.bind_pair_group(f.ident, r, group)
    with pytest.raises(ValueError, match='terminal assessed'):
        f.research.resolve(f.ident, status='no_child')
    assert canonical_json(f.research.state()) == before


def test_abort_before_group_keeps_known_raw_charge_and_does_not_fabricate_group(monkeypatch):
    f, r = fixture(), response()
    def broken(_):
        raise RuntimeError('internal pair decoder failure')
    monkeypatch.setattr(pair, 'decode_response', broken)
    with pytest.raises(RuntimeError):
        f.research.record_response(f.ident, r, response_format=pair.CONTRACT)
    assert f.events[-1]['response'] == r.text
    f.research.abort(f.ident, 'decode stopped', spent=r.cost)
    record = f.research.knowledge.experiments[0]
    assert record['status'] == 'generation_stopped' and record['generation_cost'] == r.cost
    assert record['generation_dispatch'] == 'returned' and record['hypothesis'] is None
    assert 'proposal_group' not in record and f.charges == [r.cost]


def test_abort_after_selection_before_proposal_preserves_unmeasured_selected_forecast():
    f, r = fixture(), response()
    group, _ = recorded(f, r)
    f.research.abort(f.ident, 'source persistence stopped', spent=r.cost)
    record = f.research.knowledge.experiments[0]
    assert record['proposal_group'] == group
    assert record['hypothesis'] == group['entries'][0]['hypothesis']
    assert all(record[k] is None for k in ('source_sha256', 'proposed_sha256', 'score', 'actual_delta'))
    assert record['forecast_verdict'] == 'INCONCLUSIVE' and 'controlled_change' not in record


def test_no_double_record_or_binding_and_group_is_detached_from_caller():
    f, r = fixture(), response()
    group, _ = recorded(f, r)
    before = canonical_json(f.research.state())
    with pytest.raises(ValueError, match='already recorded'):
        f.research.record_response(f.ident, r, response_format=pair.CONTRACT)
    with pytest.raises(ValueError, match='already committed'):
        f.research.bind_pair_group(f.ident, r, group)
    group['entries'][0]['hypothesis']['reasoning'] = 'mutated caller object'
    assert canonical_json(f.research.state()) == before and f.research.valid_hypotheses == 1


@pytest.mark.parametrize('key,value', [('request_id', 'r999999a1'), ('parent_source_sha256', '0' * 64),
    ('response_sha256', '0' * 64), ('selected_index', True), ('selected_source_sha256', '0' * 64),
    ('selected_proof_sha256', '0' * 64)])
def test_group_binding_rejects_identity_index_and_source_tamper_atomically(key, value):
    f, r = fixture(), response()
    f.research.record_response(f.ident, r, response_format=pair.CONTRACT)
    group, _ = group_for(f, r)
    group[key] = value
    before, count = canonical_json(f.research.state()), len(f.events)
    with pytest.raises(ValueError):
        f.research.bind_pair_group(f.ident, r, group)
    assert canonical_json(f.research.state()) == before and len(f.events) == count


@pytest.mark.parametrize('field,value,error', [
    ('text', ' ', 'durable raw response'),
    ('cost', .004, 'durable raw response'),
    # Malformed accounting is rejected before comparing the durable charge.
    ('cost', True, 'LLM response cost is unknown or invalid'),
    ('finish_reason', 'length', 'durable raw response'),
    ('refusal', 'later refusal', 'durable raw response')])
def test_binding_cannot_replace_original_raw_response_or_charge(field, value, error):
    f, r = fixture(), response()
    f.research.record_response(f.ident, r, response_format=pair.CONTRACT)
    group, _ = group_for(f, r)
    changed = copy.copy(r)
    setattr(changed, field, r.text + value if field == 'text' else value)
    before = canonical_json(f.research.state())
    receipt = copy.deepcopy(f.research._structured_responses)
    events, charges = copy.deepcopy(f.events), list(f.charges)
    with pytest.raises(ValueError, match=error):
        f.research.bind_pair_group(f.ident, changed, group)
    assert canonical_json(f.research.state()) == before
    assert f.research._structured_responses == receipt
    assert f.events == events and f.charges == charges
    assert f.research.pending[f.ident]['hypothesis'] is None and f.research.valid_hypotheses == 0


def test_selected_proposal_cannot_substitute_other_body_proof_or_single_wire():
    f, r = fixture(), response()
    group, built = recorded(f, r, selected=1)
    other = body.apply(f.snapshot.parent_code, BODY0)
    before = canonical_json(f.research.state())
    with pytest.raises(ValueError, match='selected pair source or proof'):
        commit(f, r, other)
    single = copy.copy(r)
    single.text = json.dumps(json.loads(r.text)['candidates'][1])
    with pytest.raises(ValueError, match='durable raw response'):
        commit(f, single, built)
    with pytest.raises(ValueError, match='explicit transport'):
        f.research.proposal(f.ident, r, built['code'], NS(code=built['code']))
    assert canonical_json(f.research.state()) == before


@pytest.mark.parametrize('mutation', ['group', 'hypothesis', 'cost', 'proof'])
def test_resolve_rejects_changes_to_bound_selection_and_forecast(mutation):
    f, r = fixture(), response()
    _, built = recorded(f, r)
    commit(f, r, built)
    record = f.research.pending[f.ident]
    if mutation == 'group':
        record['proposal_group']['selected_index'] = 1
    elif mutation == 'hypothesis':
        record['hypothesis']['reasoning'] = 'unsubmitted forecast'
    elif mutation == 'cost':
        record['generation_cost'] = .4
    else:
        record['controlled_change']['target_body'] = BODY1
    with pytest.raises(ValueError):
        f.research.resolve(f.ident, status='evaluated', score=.9, source=built['code'])
    assert f.research.outcomes == 0 and f.research.pending[f.ident]['score'] is None


@pytest.mark.parametrize('inject', [True, False])
def test_branch_context_and_begin_preserve_whole_group_and_explicit_omissions(inject):
    f, r = fixture(inject=inject), response()
    group, built = recorded(f, r)
    commit(f, r, built)
    f.research.resolve(f.ident, status='evaluated', score=.9, source=built['code'])
    f.research.complete_step()
    ctx = f.research.context(f.db, f.parent, 0, 2)
    block = ctx['proposal_groups']
    assert block['retained_count'] == 1
    assert block['cards'] == ([group_card(group)] if inject else [])
    other = f.db.add(f.parent.code, .8, f.parent.cases, 0, 'seed')
    assert 'proposal_groups' not in f.research.context(f.db, other, 0, 2)
    snap = copy.copy(f.snapshot)
    snap.research_context = ctx
    _, prompt = f.research.begin(snap, 0, 'circle_generate', 'Write alternatives.',
        response_instruction=pair.response_instruction(), context_scope=body.OPERATION)
    serialized = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1])[0]
    assert decode_evidence(serialized)['proposal_groups'] == block
    assert 'Unselected and rejected alternatives are unmeasured in that original request' in prompt
    assert 'historical admission, not current permission to resubmit' in prompt
    if inject:
        shown = block['cards'][0]
        assert shown['resubmission_eligibility'] == 'not_assessed'
        assert shown['entries'][0]['eligible_at_original_request'] is True


def test_completed_checkpoint_roundtrip_and_tampered_alternative_rejected():
    f, r = fixture(), response()
    group, built = recorded(f, r)
    commit(f, r, built)
    f.research.resolve(f.ident, status='evaluated', score=.9, source=built['code'])
    f.research.complete_step()
    state = f.research.state()
    # Use a fresh controller; pending work itself is intentionally not replayable.
    restored = ResearchController(f.research.config, NS(models=['fake'], max_retries=1),
                                 NS(research=lambda **_: None), lambda _: None)
    restored.load_state(json.loads(json.dumps(state)))
    assert restored.state() == state
    damaged = copy.deepcopy(state)
    damaged['knowledge']['experiments'][0]['proposal_group']['entries'][1]['body'] = 'return initial_params'
    before = canonical_json(restored.state())
    with pytest.raises(ValueError):
        restored.load_state(damaged)
    assert canonical_json(restored.state()) == before


@pytest.mark.parametrize('old_version', [None, 'sciel-circle-proposal-card/1'])
def test_proposal_card_identity_rejects_old_presentation_atomically(old_version):
    f, r = fixture(), response()
    _, built = recorded(f, r)
    commit(f, r, built)
    f.research.resolve(f.ident, status='evaluated', score=.9, source=built['code'])
    f.research.complete_step()
    saved = copy.deepcopy(f.research.state())
    before = canonical_json(f.research.state())
    assert saved['presentation']['proposal_group_card'] == 'sciel-circle-proposal-card/2'
    if old_version is None:
        del saved['presentation']['proposal_group_card']
    else:
        saved['presentation']['proposal_group_card'] = old_version
    with pytest.raises(ValueError, match='presentation'):
        f.research.load_state(saved)
    assert canonical_json(f.research.state()) == before
    identity = presentation_identity()
    identity['proposal_group_card'] = 'caller mutation'
    assert presentation_identity()['proposal_group_card'] == 'sciel-circle-proposal-card/2'


@pytest.mark.parametrize('old_version', [None, 'sciel-circle-proposal-card/1'])
def test_forced_research_only_checkpoint_cannot_restore_old_proposal_presentation(tmp_path, old_version):
    import hashlib
    from sciel.engine import checkpoint
    f, r = fixture(), response()
    _, built = recorded(f, r)
    commit(f, r, built)
    f.research.resolve(f.ident, status='evaluated', score=.9, source=built['code'])
    f.research.complete_step()
    journal = tmp_path/'events.jsonl'
    journal.write_bytes(b'{"event":"run_start"}\n')
    engine = NS(task=None, cfg={'research': f.research.config}, research=f.research,
                log=NS(path=str(journal)))
    identity, saved = checkpoint._identity(engine), f.research.state()
    for presentation in (identity['research']['presentation'], saved['presentation']):
        if old_version is None:
            del presentation['proposal_group_card']
        else:
            presentation['proposal_group_card'] = old_version
    state = {'identity': identity, 'research': saved, 'provenance': {
        'schema': checkpoint.PROVENANCE_SCHEMA, 'counters': {'db_counter': None},
        'journal_anchor': {'byte_offset': journal.stat().st_size,
                           'prefix_sha256': hashlib.sha256(journal.read_bytes()).hexdigest()}}}
    file = tmp_path/'checkpoint.json'
    file.write_text(json.dumps(state))
    before = canonical_json(f.research.state())
    with pytest.raises(checkpoint.IdentityMismatch, match='presentation'):
        checkpoint.load(engine, str(file), allow_identity_mismatch=True)
    assert canonical_json(f.research.state()) == before


def test_pair_parser_oversize_raw_return_is_saved_exactly_before_rejection():
    f, r = fixture(), response(text='x' * 270000)
    decoded = f.research.record_response(f.ident, r, response_format=pair.CONTRACT)
    assert decoded['error'] == 'invalid_json_envelope'
    saved = next(e for e in f.events if e['phase'] == 'structured_response')
    assert saved['response'] == r.text and saved['response_truncated'] is False
    assert saved['response_sha256'] == digest(r.text) and saved['generation_cost'] == r.cost


def test_two_unselected_forecasts_create_one_unmeasured_advice_exposure():
    f, r = fixture(), response()
    _, built = recorded(f, r)
    commit(f, r, built)
    f.research.resolve(f.ident, status='evaluated', score=.9, source=built['code'])
    f.research.complete_step()
    lesson_id, = f.research.knowledge.add_lessons([{
        'rule': 'This fixture source had a positive measured delta.',
        'polarity': 'positive', 'scope': 'branch', 'evidence_ids': [f.ident]}],
        allowed_evidence_ids=[f.ident])
    ctx = f.research.context(f.db, f.parent, 0, 2)
    assert ctx['injected_ids'] == [lesson_id]
    snap = copy.copy(f.snapshot)
    snap.research_context, snap.kb_lessons = ctx, ctx['lesson_texts']
    ident, _ = f.research.begin(snap, 0, 'circle_generate', '\n'.join(ctx['lesson_texts']),
        response_instruction=pair.response_instruction(), context_scope=body.OPERATION)
    second = NS(**{**vars(f), 'ident': ident, 'snapshot': snap})
    group, _ = recorded(second, response(), selected=None)
    f.research.resolve(ident, status='no_child', failure='both local gates rejected')
    f.research.complete_step()
    lesson = f.research.knowledge.lessons[0]
    assert lesson['exposure_ids'] == [ident]
    assert len(lesson['exposure_snapshots']) == 1
    exposure = lesson['exposure_snapshots'][0]
    assert exposure['proposal_group'] == group and exposure['score'] is None
    assert lesson['support_ids'] == lesson['refute_ids'] == []
    assert f.research.attempts == f.research.outcomes == 2
    assert f.research.valid_hypotheses == 1 and f.research.aux_calls == 0
    assert f.charges == []  # Normal generation charge is owned by the worker.
