"""Real four-job WAL recovery; local provider/evaluator fixtures only."""

from collections import Counter
import json
import math
import re
import threading

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint, parallel_recovery, parallel_store
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMError
from sciel.engine.loop import Engine
from sciel.engine.recovery import ProviderRecoveryError
from sciel.engine.run_lock import RunDirectoryBusy, RunDirectoryLock

from test_parallel_engine import ResearchProvider, config, evaluator, events, task_fixture


def transient(status=429):
    error = LLMError('synthetic provider error')
    error._metadata = dict(stage='request', reason='http_error',
        cause_type='RateLimitError' if status == 429 else 'AuthenticationError',
        http_status=status)
    return error


class FailingProvider(ResearchProvider):
    def __init__(self, role='C', status=429):
        super().__init__(concurrent=4)
        self.failed_role, self.status = role, status
        self.role_counts = Counter()
        self.phase_barrier = threading.Barrier(4)
        self.mixed_failure = False
        self.unknown_returned_cost = False
        self.duplicate_siblings = False

    def generate(self, prompt, model, temperature=None, **options):
        role = 'C' if 'Review this one experiment.' in prompt[:160] else 'G'
        with self.lock:
            self.role_counts[role] += 1
            ordinal = self.role_counts[role]
        if role == self.failed_role and ordinal <= 4:
            self.phase_barrier.wait(timeout=3)
        response = super().generate(prompt, model, temperature, **options)
        if role == self.failed_role and ordinal == 1:
            raise transient(self.status)
        if role == self.failed_role and ordinal == 2 and self.mixed_failure:
            raise ValueError('independent non-provider failure')
        if role == self.failed_role and ordinal == 2 and self.unknown_returned_cost:
            response.cost = None
        if role == 'G' and self.duplicate_siblings:
            response.text = re.sub(r'VALUE = \d+', 'VALUE = 0', response.text)
        return response


def stopped_wave(tmp_path, monkeypatch, provider=None, **overrides):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    cfg = config(task, tmp_path / 'capacity', max_iterations=None,
        research={'enabled': True, 'max_aux_calls': None, 'reflection_interval': 0})
    cfg.update(overrides)
    directory = tmp_path / 'run'
    provider = provider or FailingProvider()
    evaluations, _ = evaluator(monkeypatch)

    def build(resume=False):
        engine = build_engine(cfg, str(tasks), str(directory), llm=provider)
        if resume:
            checkpoint.load(engine, str(directory / 'checkpoint.json'))
        return engine

    first = build()
    # Real supervisor boundary: Engine.run finalizes runtime and closes the
    # writer before its exception reaches the caller. Authorization reacquires.
    try:
        with pytest.raises(CampaignStopped):
            first.run()
        assert first._writer_lock._file.closed
    except BaseException:
        first.close()
        raise
    return first, build, provider, evaluations, directory


def stop_after_publication(monkeypatch, directory):
    commit = parallel_store.WaveTransaction.commit

    def publish_then_stop(transaction):
        result = commit(transaction)
        if transaction.wave_id == '1':
            (directory / 'STOP').touch()
        return result

    monkeypatch.setattr(parallel_store.WaveTransaction, 'commit', publish_then_stop)


def physical_files(directory):
    return {str(p.relative_to(directory)): p.read_bytes()
            for name in ('intent.json', 'dispatched.json', 'receipt.json')
            for p in (directory / '.parallel').rglob(name)}


def test_four_job_conclusion_failure_replays_without_old_calls_or_double_cost(tmp_path, monkeypatch):
    first, build, provider, evaluations, directory = stopped_wave(tmp_path, monkeypatch)
    try:
        path = directory / 'checkpoint.json'
        baseline_bytes = path.read_bytes()
        baseline = json.loads(path.read_text())
        assert baseline['B_evals'] == baseline['iteration_no'] == 0
        assert baseline['cost_total'] == 0
        assert Counter(provider.roles) == {'G': 4, 'C': 4}
        assert len(evaluations) == 5
        saved = physical_files(directory)
        receipt = parallel_recovery.inspect_parallel_stop(path)
        assert receipt['retry'] is False and receipt['unknown_cost'] is True
        assert len(receipt['failed_calls']) == 1
        parallel_recovery.authorize_parallel_recovery(first, path, receipt)
        authorizations = {str(p): p.read_bytes() for p in
                          (directory / '.parallel').rglob('abandonment.json')}
        parallel_recovery.authorize_parallel_recovery(first, path, receipt)
        assert authorizations == {str(p): p.read_bytes() for p in
                                 (directory / '.parallel').rglob('abandonment.json')}
        assert path.read_bytes() == baseline_bytes
    finally:
        first.close()

    def forbidden(*args, **kwargs):
        pytest.fail('authenticated completed physical calls must replay, not redispatch')

    monkeypatch.setattr(provider, 'generate', forbidden)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', forbidden)
    stop_after_publication(monkeypatch, directory)
    resumed = build(resume=True)
    result = resumed.run()
    assert result['reason'] == 'paused'
    assert result['B_evals'] == result['iteration_no'] == 4
    assert result['cost_total'] == pytest.approx(.15)
    assert physical_files(directory) == saved
    trace_rows = [json.loads(line) for line in
                  (directory / 'artifacts/llm_calls.jsonl').read_text().splitlines()]
    trace_calls = [row for row in trace_rows if row['event'] in ('response', 'error')]
    unknown = [call for call in trace_calls if call['current_call_cost_unknown']]
    assert len(unknown) == 1 and unknown[0]['cost'] is None
    assert math.fsum(call['cost'] for call in trace_calls if call['cost'] is not None) == pytest.approx(.15)
    assert len(resumed.research.knowledge.lessons) == 3
    conclusions = list(resumed.research.conclusions.values())
    assert Counter(x['status'] for x in conclusions) == {'parsed': 3, 'skipped': 1}
    skipped = next(x for x in conclusions if x['status'] == 'skipped')
    assert skipped['llm'] is None and skipped['accepted_lesson_ids'] == []
    abandonment = [x for x in events(directory) if x.get('phase') == 'aux_abandoned']
    assert len(abandonment) == 1
    assert abandonment[0]['cost'] is None and abandonment[0]['current_call_cost_unknown']
    assert abandonment[0]['provider_abandonment']['recovery_id'] == receipt['recovery_id']
    assert [x['job_id'] for x in events(directory) if x.get('action') == 'job_committed'] == [
        'G:1', 'G:2', 'G:3', 'G:4']
    # The committed abandonment state must itself remain resumable and retain
    # exactly one accounting/admission event, without changing config identity.
    again = build(resume=True)
    again._stop_at = 4
    second_resume = again.run()
    assert second_resume['cost_total'] == pytest.approx(.15)
    assert len(again.research.knowledge.lessons) == 3
    assert len([x for x in events(directory) if x.get('phase') == 'aux_abandoned']) == 1
    assert physical_files(directory) == saved


def test_generation_429_is_one_nochild_and_never_retries_old_generation(tmp_path, monkeypatch):
    first, build, provider, evaluations, directory = stopped_wave(
        tmp_path, monkeypatch, FailingProvider(role='G'))
    try:
        assert Counter(provider.roles) == {'G': 4}
        assert len(evaluations) == 1  # Only canonical seed ran before abort.
        path = directory / 'checkpoint.json'
        receipt = parallel_recovery.inspect_parallel_stop(path)
        parallel_recovery.authorize_parallel_recovery(first, path, receipt)
        existing = physical_files(directory)
    finally:
        first.close()
    generate = provider.generate

    def only_new_conclusions(prompt, *args, **kwargs):
        assert 'Review this one experiment.' in prompt[:160], 'old G must never redispatch'
        return generate(prompt, *args, **kwargs)

    monkeypatch.setattr(provider, 'generate', only_new_conclusions)
    stop_after_publication(monkeypatch, directory)
    resumed = build(resume=True)
    result = resumed.run()
    assert result['reason'] == 'paused'
    assert result['iteration_no'] == 4 and result['B_evals'] == 3
    assert resumed.n_nochild == 1 and len(evaluations) == 4
    assert Counter(provider.roles) == {'G': 4, 'C': 3}
    assert result['cost_total'] == pytest.approx(.12)
    assert len(resumed.research.knowledge.lessons) == 3
    assert len([p for p in resumed.db._all.values() if p.origin != 'seed']) == 3
    assert all(physical_files(directory)[key] == value for key, value in existing.items())
    abandons = [row for row in events(directory) if row.get('phase') == 'generation_abandoned']
    assert len(abandons) == 1 and abandons[0]['cost'] is None
    failed_id = abandons[0]['experiment_id']
    assert failed_id not in resumed.research.conclusions
    assert not any(failed_id in lesson['evidence_ids'] for lesson in resumed.research.knowledge.lessons)


@pytest.mark.parametrize('mode', ['401', 'mixed_provider_failure', 'unknown_returned_cost'])
def test_real_failed_wave_refuses_ineligible_provider_recovery(tmp_path, monkeypatch, mode):
    provider = FailingProvider(status=401 if mode == '401' else 429)
    provider.mixed_failure = mode == 'mixed_provider_failure'
    provider.unknown_returned_cost = mode == 'unknown_returned_cost'
    first, _, _, _, directory = stopped_wave(tmp_path, monkeypatch, provider)
    try:
        before = physical_files(directory)
        with pytest.raises(ProviderRecoveryError):
            parallel_recovery.inspect_parallel_stop(directory / 'checkpoint.json')
        assert not list((directory / '.parallel').rglob('abandonment.json'))
        assert physical_files(directory) == before
    finally:
        first.close()


@pytest.mark.parametrize('damage', [
    'missing_terminal', 'receipt_hash', 'missing_dispatch', 'dispatch_binding',
    'frozen_jobs', 'baseline',
])
def test_recovery_inspection_rejects_incomplete_or_tampered_real_evidence(tmp_path, monkeypatch, damage):
    first, _, _, _, directory = stopped_wave(tmp_path, monkeypatch)
    try:
        path = directory / 'checkpoint.json'
        eligible = parallel_recovery.inspect_parallel_stop(path)
        wave = directory / '.parallel' / ('wave-' + eligible['wave_id'])
        failed = wave / 'calls' / eligible['failed_calls'][0]['directory'] / 'receipt.json'
        if damage == 'missing_terminal':
            failed.unlink()  # Durable dispatch remains; this is unresolved work.
        elif damage == 'receipt_hash':
            data = json.loads(failed.read_text())
            data['payload']['error']['http_status'] = 503
            failed.write_text(json.dumps(data))
        elif damage == 'missing_dispatch':
            failed.with_name('dispatched.json').unlink()
        elif damage == 'dispatch_binding':
            dispatch = failed.with_name('dispatched.json')
            data = json.loads(dispatch.read_text())
            data['request_sha256'] = '0' * 64
            dispatch.write_text(json.dumps(data))
        elif damage == 'frozen_jobs':
            with (wave / 'jobs.json').open('a') as stream:
                stream.write(' ')
        else:
            with path.open('a') as stream:
                stream.write(' ')
        with pytest.raises(ProviderRecoveryError):
            parallel_recovery.inspect_parallel_stop(path)
        with pytest.raises(ProviderRecoveryError):
            parallel_recovery.authorize_parallel_recovery(first, path, eligible)
        assert not list(wave.rglob('abandonment.json'))
    finally:
        first.close()


def test_provider_429_cannot_hide_a_sibling_commit_failure(tmp_path, monkeypatch):
    provider = FailingProvider(role='G')
    provider.duplicate_siblings = True
    commits = []

    def unrelated_commit_failure(engine, plan, result, **options):
        commits.append(type(result).__name__)
        raise ValueError('synthetic sibling commit defect')

    monkeypatch.setattr(Engine, '_commit_step', unrelated_commit_failure)
    first, _, _, _, directory = stopped_wave(tmp_path, monkeypatch, provider,
        candidate_novelty={'enabled': True, 'max_resamples': 0, 'neighbors': 3})
    try:
        assert commits == ['NoChild'], 'must actually reach sibling admission after the 429'
        stops = list((directory / '.parallel/wave-1/provider_stops').glob('*.json'))
        assert len(stops) == 1
        errors = json.loads(stops[0].read_text())['errors']
        assert any(error['kind'] == 'provider' for error in errors)
        assert any(error['kind'] == 'fatal' for error in errors)
        with pytest.raises(ProviderRecoveryError, match='non-provider failure'):
            parallel_recovery.inspect_parallel_stop(directory / 'checkpoint.json')
    finally:
        first.close()


def test_closed_engine_authorization_refuses_concurrent_writer(tmp_path, monkeypatch):
    first, _, _, _, directory = stopped_wave(tmp_path, monkeypatch)
    path = directory / 'checkpoint.json'
    receipt = parallel_recovery.inspect_parallel_stop(path)
    competing = RunDirectoryLock(directory)
    try:
        with pytest.raises(RunDirectoryBusy):
            parallel_recovery.authorize_parallel_recovery(first, path, receipt)
        assert not list((directory / '.parallel').rglob('abandonment.json'))
    finally:
        competing.close()
    parallel_recovery.authorize_parallel_recovery(first, path, receipt)
    assert len(list((directory / '.parallel').rglob('abandonment.json'))) == 1
    # Authorization must release its reacquired ownership on return as well.
    subsequent = RunDirectoryLock(directory)
    subsequent.close()


def test_authorization_survives_replay_crash_and_new_baseline_without_recharge(tmp_path, monkeypatch):
    first, build, provider, _, directory = stopped_wave(tmp_path, monkeypatch)
    path = directory / 'checkpoint.json'
    before_baseline = path.read_bytes()
    receipt = parallel_recovery.inspect_parallel_stop(path)
    parallel_recovery.authorize_parallel_recovery(first, path, receipt)
    original_files = physical_files(directory)
    marker_files = {str(p): p.read_bytes() for p in
                    (directory / '.parallel').rglob('abandonment.json')}

    def forbidden(*args, **kwargs):
        pytest.fail('authorized replay must not repeat any physical request')

    monkeypatch.setattr(provider, 'generate', forbidden)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', forbidden)
    commit = parallel_store.WaveTransaction.commit

    def crash_after_replay(transaction):
        if transaction.wave_id == '1':
            assert transaction.engine.cost_total == pytest.approx(.15)
            raise RuntimeError('synthetic crash after authorized replay before publication')
        return commit(transaction)

    with monkeypatch.context() as crashed:
        crashed.setattr(parallel_store.WaveTransaction, 'commit', crash_after_replay)
        attempt = build(resume=True)
        with pytest.raises(CampaignStopped, match='synthetic crash'):
            attempt.run()
    assert attempt._writer_lock._file.closed
    state = json.loads(path.read_text())
    assert state['cost_total'] == 0 and state['B_evals'] == state['iteration_no'] == 0
    assert path.read_bytes() != before_baseline, 'replay must exercise a regenerated provenance baseline'
    assert physical_files(directory) == original_files
    assert marker_files == {str(p): p.read_bytes() for p in
                            (directory / '.parallel').rglob('abandonment.json')}

    stop_after_publication(monkeypatch, directory)
    recovered = build(resume=True)
    result = recovered.run()
    assert result['reason'] == 'paused'
    assert result['B_evals'] == result['iteration_no'] == 4
    assert result['cost_total'] == pytest.approx(.15)
    assert len(recovered.research.knowledge.lessons) == 3
    assert len([x for x in events(directory) if x.get('phase') == 'aux_abandoned']) == 1
    assert physical_files(directory) == original_files
