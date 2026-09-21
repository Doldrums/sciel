"""Bounded card format repair, accounting and recovery; no paid or candidate calls."""
import json
from pathlib import Path

import pytest

from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.engine import checkpoint
from sciel.engine.idea import TwoStageIdea, validate_card
from sciel.engine.llm.client import LLMError, LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.engine.recovery import ProviderRecoveryError, inspect_provider_stop, prepare_provider_recovery
from sciel.engine.types import NoChild, Result
from sciel.engine.worker import run_worker
from test_two_stage_idea import setup, response, code_response, CARD
from test_novelty_worker_retries import callback, registry_with


LONG_CARD = json.dumps({**CARD, 'mechanism': 'x' * 1029})
QUOTED_CARD = '{"family":"grid", "strategy":"Use "bad" quotes", "mechanism":"spacing"}'


def repaired_setup(responses):
    research, _, snap, model, log, charged, evaluated, _ = setup(responses)
    idea = TwoStageIdea(research, card_repair_attempts=1)
    def evaluate(code):
        from test_engine_recovery_review_fixes import outcome
        evaluated.append(code)
        return outcome(.7)
    def run(**kwargs):
        return run_worker('idea', snap, model, evaluate, research=research, idea=idea, **kwargs)
    return research, idea, snap, model, log, charged, evaluated, run


@pytest.mark.parametrize('length,accepted', [(1024, True), (1025, False), (1029, False)])
def test_mechanism_limit_stays_exact_and_reports_actual_length(length, accepted):
    raw = json.dumps({**CARD, 'mechanism': 'x' * length})
    card, errors = validate_card(raw)
    assert (card is not None) is accepted
    assert errors == ([] if accepted else [dict(kind='field_too_long', field='mechanism', length=length, maximum=1024)])


@pytest.mark.parametrize('malformed,kind', [(LONG_CARD, 'field_too_long'), (QUOTED_CARD, 'invalid_json')])
def test_format_retry_retains_exact_response_and_proceeds_only_after_corrected_card(malformed, kind):
    research, idea, snap, model, log, charged, evaluated, run = repaired_setup([
        response(malformed), response(json.dumps(CARD)), response(code_response())])
    result = run()
    assert isinstance(result, Result) and len(model.calls) == 3 and len(evaluated) == 1
    assert result.cost == pytest.approx(.09) and charged == []
    assert 'mechanism <=640 characters' in model.calls[0]
    previous = model.calls[1].split('PREVIOUS_CARD_RESPONSE_JSON (exact previous text; null if no text was returned):\n')[1]
    assert json.JSONDecoder().raw_decode(previous)[0] == malformed
    rejected, = [r for r in log.events if r.get('action') == 'card_rejected']
    assert rejected['retry_scheduled'] is True and rejected['diagnostics'][0]['kind'] == kind
    assert idea.card_invalid == 0 and idea.card_retries == 1 and research.attempts == 1
    assert result.meta['idea']['card'] == CARD
    other = TwoStageIdea(research, card_repair_attempts=1)
    other.load_state(json.loads(json.dumps(idea.state())))
    assert other.state() == idea.state()
    with pytest.raises(ValueError):
        TwoStageIdea(research).load_state(idea.state())


def test_exhausted_card_repair_is_explicit_and_all_paid_costs_are_returned():
    research, idea, snap, model, log, charged, evaluated, run = repaired_setup([
        response(LONG_CARD), response(LONG_CARD), response(code_response())])
    result = run()
    assert isinstance(result, NoChild) and result.reason == 'idea_card_repair_exhausted'
    assert len(model.calls) == 2 and evaluated == [] and result.cost == pytest.approx(.06)
    assert research.attempts == 0 and research.queue == [] and charged == []
    assert idea.card_invalid == idea.card_retries == 1
    assert result.meta['card_validation_errors'][0]['length'] == 1029
    assert [r['retry_scheduled'] for r in log.events if r.get('action') == 'card_rejected'] == [True, False]


def test_card_repair_and_novelty_code_retries_have_separate_bounded_counters():
    duplicate = code_response('def run_search():\n    return 1.0\n')
    research, idea, snap, model, log, charged, evaluated, run = repaired_setup([
        response(LONG_CARD), response(json.dumps(CARD)), response(duplicate),
        response(duplicate), response(code_response())])
    result = run(novelty_check=callback(registry_with(snap.parent_code)))
    assert isinstance(result, Result) and len(model.calls) == 5 and len(evaluated) == 1
    assert idea.card_retries == 1 and idea.code_retries == 2
    assert result.cost == pytest.approx(.15) and charged == []
    assert len(research.queue) == 1
    other = TwoStageIdea(research, card_repair_attempts=1)
    other.load_state(json.loads(json.dumps(idea.state())))
    assert other.state() == idea.state()


@pytest.mark.parametrize('failure', [RequestNotDispatched('budget exhausted'), LLMError(cause=ConnectionError())])
def test_card_retry_stops_preserve_known_charge_and_never_reach_code(failure):
    research, idea, snap, model, log, charged, evaluated, run = repaired_setup([response(LONG_CARD), failure])
    with pytest.raises(CampaignStopped):
        run()
    assert charged == [pytest.approx(.03)] and evaluated == [] and research.attempts == 0
    stopped, = [r for r in log.events if r.get('action') == 'stopped']
    assert stopped['known_cost'] == pytest.approx(.03)
    assert stopped['current_call_cost_unknown'] is (not isinstance(failure, RequestNotDispatched))
    assert idea.calls_started == 2 and idea.calls_returned == 1 and idea.card_retries == 1


def make_stop(folder, monkeypatch, fail_call):
    from test_engine_recovery_review_fixes import construct, settings, outcome
    seed = (Path(__file__).resolve().parents[2] / 'tasks/test/toy_quadratic/initial_program.py').read_text()
    class Provider(MockLLM):
        calls = 0
        def generate(self, prompt, model, *args, **kwargs):
            self.calls += 1
            if self.calls == fail_call:
                raise LLMError(cause=ConnectionError())
            raw = LONG_CARD if self.calls == 1 else json.dumps(CARD) if self.calls == 2 else code_response(seed)
            return LLMResponse(raw, model, .03, 10, 10)
    provider = Provider()
    cfg = settings(op_fixed='idea', idea_two_stage={'enabled': True, 'card_repair_attempts': 1},
        candidate_novelty={'enabled': True, 'max_resamples': 2},
        research={'enabled': True, 'max_aux_calls': None})
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', lambda *args, **kwargs: outcome())
    engine = construct(folder, cfg, provider)
    with pytest.raises(CampaignStopped, match='two-stage idea stopped'):
        engine.run()
    return folder / 'checkpoint.json', provider


@pytest.mark.parametrize('fail_call', [2, 3, 5])
def test_card_repair_unknown_request_recovers_without_replay_or_lost_cost(tmp_path, monkeypatch, fail_call):
    from test_engine_recovery_review_fixes import construct
    path, provider = make_stop(tmp_path, monkeypatch, fail_call)
    receipt = inspect_provider_stop(path, allow_legacy=False)
    known = .03 * (fail_call - 1)
    assert receipt['request']['call_index'] == fail_call
    assert receipt['request']['stage'] == ('card' if fail_call == 2 else 'code')
    assert receipt['known_attempt_cost'] == pytest.approx(known)
    saved = json.loads(path.read_bytes())
    assert saved['cost_total'] == pytest.approx(known)
    assert saved['idea_two_stage']['card_retries'] == 1
    engine = construct(tmp_path, saved['config'], provider)
    checkpoint.load(engine, str(path))
    prepare_provider_recovery(engine, path, allow_legacy=False)
    engine.request_stop()
    assert engine.run()['reason'] == 'paused'
    assert provider.calls == fail_call and engine.cost_total == pytest.approx(known)
    assert engine.idea.card_retries == 1


@pytest.mark.parametrize('mutation', [
    lambda state, rows: next(r for r in rows if r.get('action') == 'card_rejected')['diagnostics'][0].update(maximum=4096),
    lambda state, rows: next(r for r in rows if r.get('action') == 'card_rejected').update(retry_scheduled=False),
    lambda state, rows: state['idea_two_stage'].update(card_retries=0),
    lambda state, rows: next(r for r in rows if r.get('action') == 'request_result').update(cost=None),
])
def test_recovery_rejects_unproven_card_repair_prefix(tmp_path, monkeypatch, mutation):
    from test_provider_recovery import rewrite_bound_fixture
    path, _ = make_stop(tmp_path, monkeypatch, 3)
    rewrite_bound_fixture(path, mutation)
    with pytest.raises(ProviderRecoveryError):
        inspect_provider_stop(path, allow_legacy=False)
