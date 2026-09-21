"""Offline provider-stop recovery using actual journal/checkpoint writers."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from sciel.engine import checkpoint
from sciel.engine.recovery import (ProviderRecoveryError, apply_provider_recovery,
    inspect_provider_stop, prepare_provider_recovery, validate_recoveries)
from sciel.run_log import RunLog
from provider_recovery_fixtures import (
    State, Research, make_engine, stopped,
    restored, idea_stop, rewrite_bound_fixture,
)


def test_explicit_recovery_preserves_full_state_and_persists_unknown_charge(tmp_path, monkeypatch):
    path = stopped(tmp_path, monkeypatch)
    original = path.read_bytes()
    before = json.loads(original)
    engine = restored(path, monkeypatch)
    receipt = prepare_provider_recovery(engine, path)
    assert engine.research.stopped_reason is not None  # preparing never clears anything
    engine.log.run_start(restored_checkpoint=engine._restored_checkpoint)
    assert apply_provider_recovery(engine) is True
    assert apply_provider_recovery(engine) is False
    after = json.loads(path.read_bytes())
    for key in checkpoint._SCALARS + checkpoint._COMPONENTS:
        assert after[key] == before[key]
    research = deepcopy(before['research'])
    research['stopped_reason'] = None
    assert after['research'] == research
    assert after['provider_recoveries'] == [receipt]
    assert receipt['unknown_cost'] is True and receipt['retry'] is False
    assert receipt['known_cost_total'] == 1.25
    assert next((tmp_path / 'recovery').iterdir()).read_bytes() == original
    engine.close()
    again = restored(path, monkeypatch)
    assert again.research.stopped_reason is None
    assert again.provider_recoveries == [receipt]
    with pytest.raises(ProviderRecoveryError):
        prepare_provider_recovery(again, path)
    again.close()


def test_normal_restore_never_clears_provider_stop(tmp_path, monkeypatch):
    path = stopped(tmp_path, monkeypatch)
    engine = restored(path, monkeypatch)
    assert apply_provider_recovery(engine) is False
    assert engine.research.stopped_reason == 'research auxiliary request stopped: LLMError'
    engine.close()


def test_legacy_generation_is_manual_only(tmp_path, monkeypatch):
    path = stopped(tmp_path, monkeypatch, generation=True, metadata=None)
    receipt = inspect_provider_stop(path)
    assert receipt['auto_recoverable'] is False
    assert receipt['provider_failure'] == {'category': 'legacy_generation_unknown'}
    with pytest.raises(ProviderRecoveryError, match='automatic recovery'):
        inspect_provider_stop(path, allow_legacy=False)


@pytest.mark.parametrize('generation', [False, True])
@pytest.mark.parametrize('reason,status,accepted', [
    ('timeout', None, True), ('http_error', 429, True), ('http_error', 503, True),
    ('http_error', 401, False), ('http_error', 400, False), ('request_failed', None, False),
    ('malformed_response', None, False)])
def test_only_transient_provider_categories_are_eligible(tmp_path, monkeypatch, generation, reason, status, accepted):
    metadata = {'stage': 'request', 'reason': reason}
    if status is not None:
        metadata['http_status'] = status
    path = stopped(tmp_path, monkeypatch, generation=generation, metadata=metadata)
    if accepted:
        assert inspect_provider_stop(path)['auto_recoverable'] is True
    else:
        with pytest.raises(ProviderRecoveryError, match='transient'):
            inspect_provider_stop(path)


@pytest.mark.parametrize('change', [
    lambda e: e.cfg.update(budget=100),
    lambda e: e.cfg.update(max_iterations=100),
    lambda e: e.cfg['research'].update(max_aux_calls=32),
    lambda e: e.cfg.update(repeat_population={'enabled': True}),
    lambda e: e.research.value.update(pending={'r2': {}}),
    lambda e: e.research.value.update(queue=[{'r': 1}]),
    lambda e: e.research.value.update(outcomes=0),
    lambda e: setattr(e, '_accounting_stopped_reason', 'response cost unknown'),
    lambda e: e.research.value.update(stopped_reason='research auxiliary response stopped: ValueError'),
    lambda e: setattr(e, 'idea', State({'attempts': 1, 'finished': 0, 'pending': {'id': 1}})),
])
def test_finite_dirty_specialized_and_accounting_stops_rejected(tmp_path, monkeypatch, change):
    path = stopped(tmp_path, monkeypatch, configure=change)
    with pytest.raises(ProviderRecoveryError):
        inspect_provider_stop(path)


def test_unknown_paid_tail_and_partial_journal_are_rejected(tmp_path, monkeypatch):
    path = stopped(tmp_path, monkeypatch)
    journal = path.with_name('sciel_events.jsonl')
    original = journal.read_bytes()
    journal.write_bytes(original + b'{"event":"research","phase":"attempt"}\n')
    with pytest.raises(ProviderRecoveryError):
        inspect_provider_stop(path)
    journal.write_bytes(original[:-1])
    with pytest.raises(ProviderRecoveryError, match='prefix'):
        inspect_provider_stop(path)


def test_no_ledger_or_identity_override_recovery(tmp_path, monkeypatch):
    path = stopped(tmp_path, monkeypatch)
    engine = restored(path, monkeypatch)
    engine.llm.ledger = object()
    with pytest.raises(ProviderRecoveryError, match='budget-ledger'):
        prepare_provider_recovery(engine, path)
    del engine.llm.ledger
    engine._restored_checkpoint['identity_mismatches'] = ['changed task']
    with pytest.raises(ProviderRecoveryError, match='identity-matching'):
        prepare_provider_recovery(engine, path)
    engine.close()


def test_checkpoint_failure_keeps_stop_and_receipt_can_be_reused(tmp_path, monkeypatch):
    path = stopped(tmp_path, monkeypatch)
    original = path.read_bytes()
    engine = restored(path, monkeypatch)
    prepare_provider_recovery(engine, path)
    engine.log.run_start(restored_checkpoint=engine._restored_checkpoint)
    save = checkpoint.save
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(checkpoint, 'save', fail)
    with pytest.raises(OSError, match='disk full'):
        apply_provider_recovery(engine)
    assert path.read_bytes() == original and engine.research.stopped_reason is not None
    engine.close()
    monkeypatch.setattr(checkpoint, 'save', save)
    engine = restored(path, monkeypatch)
    prepare_provider_recovery(engine, path)
    engine.log.run_start(restored_checkpoint=engine._restored_checkpoint)
    apply_provider_recovery(engine)
    rows = [json.loads(line) for line in path.with_name('sciel_events.jsonl').read_bytes().splitlines()]
    assert sum(r.get('policy') == 'provider_recovery' for r in rows) == 1
    engine.close()


def test_failed_receipt_write_never_clears_stop(tmp_path, monkeypatch):
    path = stopped(tmp_path, monkeypatch)
    engine = restored(path, monkeypatch)
    prepare_provider_recovery(engine, path)
    engine.log.run_start(restored_checkpoint=engine._restored_checkpoint)
    monkeypatch.setattr(engine.log, 'emit', lambda *args, **kwargs: False)
    with pytest.raises(ProviderRecoveryError, match='persist'):
        apply_provider_recovery(engine)
    assert engine.research.stopped_reason is not None
    engine.close()


def test_recovery_metadata_cannot_claim_a_known_zero_charge(tmp_path, monkeypatch):
    path = stopped(tmp_path, monkeypatch)
    receipt = inspect_provider_stop(path)
    receipt['unknown_cost'] = False
    with pytest.raises(ProviderRecoveryError, match='accounting'):
        validate_recoveries([receipt])
    state = json.loads(path.read_bytes())
    state['provider_recoveries'] = [receipt]
    path.write_text(json.dumps(state))
    engine = make_engine(tmp_path, monkeypatch)
    with pytest.raises(checkpoint.IdentityMismatch, match='accounting'):
        checkpoint.load(engine, str(path))


def test_cli_rejects_identity_override_before_opening_directory(monkeypatch):
    from sciel.cli.main import main
    assert main(['resume', '/nonexistent', '--force-identity', '--abandon-provider-failure']) == 2


def test_real_engine_persists_recovery_before_the_next_mock_request(tmp_path):
    """Run the production hook and research controller, with no real evaluator/API."""
    import sys
    from sciel.campaign.budget import CampaignStopped
    from sciel.engine.factory import build_engine
    from sciel.engine.llm.client import LLMError
    from sciel.engine.llm.mock import MockLLM
    from sciel.eval.cascade import CascadeOutcome
    from sciel.eval.facade import EvalResult

    class NextRequestObserved(BaseException):
        pass

    class BrokenProvider(MockLLM):
        def generate(self, *args, **kwargs):
            raise LLMError(cause=ConnectionError())

    cfg = dict(task='test/toy_quadratic', budget=None, max_iterations=None,
        schedule_horizon=10, checkpoint_interval=10, num_islands=1, inflight_cap=1,
        dup_cap=1, random_seed=23, schedule_policy='round_robin', milestone_fracs=[],
        embed_grid=False, embed_niche=False, portfolio_seed=False, dynamic_islands=False,
        idea_island=False, crossover=False, op_fixed='full_rewrite',
        policy={'enabled': False, 'plateau_shift': False}, llm={'mock': True},
        research={'enabled': True, 'max_aux_calls': None})
    tasks = str(Path(__file__).resolve().parents[2] / 'tasks')
    first = build_engine(cfg, tasks, str(tmp_path), python=sys.executable, llm=BrokenProvider())
    first._evaluate = lambda code: CascadeOutcome(
        result=EvalResult(ok=True, score=.1, cases={}, timing=.01),
        raw_first_score=.1, reached='full', confirmed=False)
    with pytest.raises(CampaignStopped, match='generation outcome unknown'):
        first.run()
    path = tmp_path / 'checkpoint.json'
    original = json.loads(path.read_bytes())
    terminal_id = next(json.loads(line)['experiment_id']
        for line in reversed((tmp_path / 'sciel_events.jsonl').read_bytes().splitlines())
        if json.loads(line).get('phase') == 'outcome')

    class ObservingProvider(MockLLM):
        def generate(self, *args, **kwargs):
            saved = json.loads(path.read_bytes())
            assert saved['milestone'] == 'provider_recovery'
            assert saved['research']['stopped_reason'] is None
            assert saved['provider_recoveries'][0]['request']['reference'] == terminal_id
            assert saved['provider_recoveries'][0]['unknown_cost'] is True
            for key in checkpoint._SCALARS + checkpoint._COMPONENTS:
                assert saved[key] == original[key]
            expected_research = deepcopy(original['research'])
            expected_research['stopped_reason'] = None
            assert saved['research'] == expected_research
            rows = [json.loads(line) for line in (tmp_path / 'sciel_events.jsonl').read_bytes().splitlines()]
            attempts = [r for r in rows if r.get('phase') == 'attempt']
            assert len(attempts) == 2
            assert attempts[-1]['experiment_id'] != terminal_id
            assert any(r.get('event') == 'checkpoint' and r.get('reason') == 'provider_recovery' for r in rows)
            raise NextRequestObserved()

    second = build_engine(original['config'], tasks, str(tmp_path), python=sys.executable,
                          llm=ObservingProvider())
    second._evaluate = lambda *args: pytest.fail('resume must not re-evaluate the seed')
    checkpoint.load(second, str(path))
    prepare_provider_recovery(second, path)
    with pytest.raises(NextRequestObserved):
        second.run()


@pytest.mark.parametrize('stage', ['card', 'code'])
def test_idea_recovery_skips_sealed_attempt_and_never_double_charges_card(tmp_path, stage):
    import sys
    from sciel.engine.factory import build_engine
    from sciel.engine.llm.client import LLMResponse
    from sciel.engine.llm.mock import MockLLM
    path = idea_stop(tmp_path, stage)
    before = json.loads(path.read_bytes())
    old_spend = 0.0 if stage == 'card' else .03
    assert before['cost_total'] == before['idea_two_stage']['known_cost'] == old_spend
    receipt = inspect_provider_stop(path, allow_legacy=False)
    assert receipt['auto_recoverable'] is True
    assert receipt['request'] == {'kind': 'idea', 'stage': stage, 'reference': 'idea000001',
        'experiment_id': None if stage == 'card' else 'r000001a1',
        'call_index': 1 if stage == 'card' else 2}
    assert receipt['known_attempt_cost'] == old_spend

    class NextCard(MockLLM):
        def generate(self, prompt, *args, **kwargs):
            saved = json.loads(path.read_bytes())
            assert saved['milestone'] == 'provider_recovery'
            for key in checkpoint._SCALARS + checkpoint._COMPONENTS:
                assert saved[key] == before[key]
            for key in ('research', 'idea_two_stage'):
                expected = deepcopy(before[key])
                expected['stopped_reason'] = None
                assert saved[key] == expected
            assert saved['provider_recoveries'] == [receipt]
            assert engine.idea.pending['id'] == 'idea000002'
            assert prompt.startswith('IDEA STAGE 1:')  # never replay the abandoned code stage
            engine.request_stop()
            return LLMResponse('invalid new card', 'mock-model', .02, 12, 10)

    engine = build_engine(before['config'], str(Path(__file__).resolve().parents[2] / 'tasks'),
                          str(tmp_path), python=sys.executable, llm=NextCard())
    engine._evaluate = lambda code: pytest.fail('no candidate or seed should be evaluated')
    checkpoint.load(engine, str(path))
    prepare_provider_recovery(engine, path, allow_legacy=False)
    assert engine.run()['reason'] == 'paused'
    after = json.loads(path.read_bytes())
    assert after['cost_total'] == pytest.approx(old_spend + .02)
    assert after['idea_two_stage']['known_cost'] == pytest.approx(old_spend + .02)
    assert after['idea_two_stage']['attempts'] == after['idea_two_stage']['finished'] == 2
    assert after['research']['attempts'] == before['research']['attempts']
    assert after['research']['stopped_reason'] is after['idea_two_stage']['stopped_reason'] is None


@pytest.mark.parametrize('mutation', [
    lambda s, rows: s['idea_two_stage'].update(pending={'id': 'idea000001'}),
    lambda s, rows: s['idea_two_stage'].update(finished=0),
    lambda s, rows: s['idea_two_stage'].update(stopped_reason=None),
    lambda s, rows: s['idea_two_stage'].update(calls_started=1),
    lambda s, rows: rows[-2].update(stage='card'),
    lambda s, rows: rows[-2].update(experiment_id='foreign'),
    lambda s, rows: rows[-2].update(generation_returned=True),
    lambda s, rows: rows[-2].update(generation_dispatch='not_dispatched'),
    lambda s, rows: rows[-2].update(known_cost=0.0),
    lambda s, rows: rows[-2].pop('llm_error'),
    lambda s, rows: rows[-1]['evidence'].update(generation_cost=.07),
    lambda s, rows: rows[-1]['evidence'].update(score=.2),
    lambda s, rows: next(r for r in rows if r.get('action') == 'card_decision').update(accepted=False),
    lambda s, rows: next(r for r in rows if r.get('action') == 'request_result').update(cost=None),
    lambda s, rows: next(r for r in rows if r.get('action') == 'request_start' and r.get('stage') == 'code').update(prompt_sha256='f'*64),
])
def test_idea_recovery_requires_clean_counters_exact_stage_and_unreturned_request(tmp_path, mutation):
    path = idea_stop(tmp_path, 'code')
    rewrite_bound_fixture(path, mutation)
    with pytest.raises(ProviderRecoveryError):
        inspect_provider_stop(path)


@pytest.mark.parametrize('stage', ['card', 'code'])
def test_idea_unknown_unclassified_or_permanent_errors_are_not_automatically_abandoned(tmp_path, stage):
    from sciel.engine.llm.client import LLMError
    path = idea_stop(tmp_path, stage, error=LLMError('unclassified'))
    with pytest.raises(ProviderRecoveryError, match='transient'):
        inspect_provider_stop(path)


def test_failed_idea_recovery_checkpoint_restores_both_stop_flags(tmp_path, monkeypatch):
    import sys
    from sciel.engine.factory import build_engine
    from sciel.engine.llm.mock import MockLLM
    path = idea_stop(tmp_path, 'code')
    original = json.loads(path.read_bytes())
    engine = build_engine(original['config'], str(Path(__file__).resolve().parents[2] / 'tasks'),
                          str(tmp_path), python=sys.executable, llm=MockLLM())
    checkpoint.load(engine, str(path))
    prepare_provider_recovery(engine, path)
    engine.log.run_start(restored_checkpoint=engine._restored_checkpoint)
    monkeypatch.setattr(checkpoint, 'save', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        apply_provider_recovery(engine)
    assert engine.research.stopped_reason == engine.idea.stopped_reason == 'two-stage idea stopped: LLMError'
    assert engine.cost_total == .03
    engine.close()


def test_second_unknown_idea_request_retains_first_abandonment_and_card_charge(tmp_path):
    import sys
    from sciel.campaign.budget import CampaignStopped
    from sciel.engine.factory import build_engine
    from sciel.engine.llm.client import LLMError
    from sciel.engine.llm.mock import MockLLM

    class BrokenCard(MockLLM):
        def generate(self, *args, **kwargs):
            raise LLMError(cause=TimeoutError())

    path = idea_stop(tmp_path, 'code')
    cfg = json.loads(path.read_bytes())['config']
    engine = build_engine(cfg, str(Path(__file__).resolve().parents[2] / 'tasks'),
                          str(tmp_path), python=sys.executable, llm=BrokenCard())
    checkpoint.load(engine, str(path))
    first = prepare_provider_recovery(engine, path, allow_legacy=False)
    with pytest.raises(CampaignStopped, match='two-stage idea stopped: LLMError'):
        engine.run()
    state = json.loads(path.read_bytes())
    second = inspect_provider_stop(path, allow_legacy=False)
    assert state['provider_recoveries'] == [first]
    assert state['cost_total'] == state['idea_two_stage']['known_cost'] == .03
    assert second['known_attempt_cost'] == 0.0 and second['known_cost_total'] == .03
    assert second['request']['reference'] == 'idea000002' and second['request']['stage'] == 'card'
    assert second['request']['call_index'] == 3
    assert second['recovery_id'] != first['recovery_id']
