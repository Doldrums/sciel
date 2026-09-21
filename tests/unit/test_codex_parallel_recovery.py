"""Actual parser errors through four-job WALs; all HTTP/evaluation is local fake."""
import json

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import parallel_recovery, parallel_store
from sciel.engine.llm.client import LLMError, COST_PROVENANCE_SCHEMA
from sciel.engine.llm.codex import CodexResponsesLLM
from sciel.engine.llm.codex_protocol import ENDPOINT_SHA256
from sciel.engine.recovery import ProviderRecoveryError, _provider_failure

from test_codex_responses import Opener
from test_parallel_provider_recovery import (
    FailingProvider, stopped_wave, physical_files, stop_after_publication, events,
)


class ProtocolProvider(FailingProvider):
    def __init__(self, role='C', mode='stream_eof'):
        super().__init__(role=role)
        self.mode = mode
        self.codex_config = dict(provider='codex_responses', models=['mock-model'],
                                weights=[1.], max_retries=1)
        self.opener = Opener([])
        self.adapter = CodexResponsesLLM(self.codex_config, 'fixture-only', opener=self.opener)

    @property
    def transport_identity(self):
        return self.adapter.transport_identity

    def request_metadata(self, *args, **kwargs):
        return self.adapter.request_metadata(*args, **kwargs)

    def generate(self, prompt, model, temperature=None, **options):
        try:
            response = super().generate(prompt, model, temperature, **options)
        except LLMError:
            # The existing fixture schedules exactly one failed role. Feed that
            # actual request through the real adapter/parser without networking.
            try:
                self.adapter.generate(prompt, model, temperature, **options)
            except LLMError as error:
                if self.mode == 'legacy':
                    error._metadata = dict(stage='response', reason='malformed_response', cause_type='ValueError')
                elif self.mode == 'wrong_body':
                    error._metadata['codex_protocol']['body_sha256'] = '0' * 64
                elif self.mode == 'wrong_model':
                    error._metadata['codex_protocol']['model'] = 'unrequested-model'
                elif self.mode != 'stream_eof':
                    error._metadata['codex_protocol']['code'] = self.mode
                raise
            pytest.fail('incomplete fake stream must raise')
        response.cost = None
        response.cost_provenance = dict(schema=COST_PROVENANCE_SCHEMA,
            source='subscription_unpriced', response_status='completed',
            provider='codex_responses', endpoint_sha256=ENDPOINT_SHA256)
        return response


def stopped(tmp_path, monkeypatch, *, role='C', mode='stream_eof', **options):
    provider = ProtocolProvider(role, mode)
    return stopped_wave(tmp_path, monkeypatch, provider,
                        llm=provider.codex_config, **options)


@pytest.mark.parametrize('role', ['G', 'C'])
def test_typed_parser_failure_abandons_only_missing_output_once(tmp_path, monkeypatch, role):
    first, build, provider, evaluations, directory = stopped(tmp_path, monkeypatch, role=role)
    path = directory / 'checkpoint.json'
    receipt = parallel_recovery.inspect_parallel_stop(path)
    assert receipt['unknown_cost'] is True and receipt['retry'] is False
    assert len(provider.opener.calls) == len(receipt['failed_calls']) == 1
    failed = receipt['failed_calls'][0]
    assert failed['error']['codex_protocol']['code'] == 'stream_eof'
    saved = physical_files(directory)
    parallel_recovery.authorize_parallel_recovery(first, path, receipt)
    parallel_recovery.authorize_parallel_recovery(first, path, receipt)
    first.close()
    generate = provider.generate

    def remaining_only(prompt, *args, **kwargs):
        assert role == 'G' and 'Review this one experiment.' in prompt[:160]
        return generate(prompt, *args, **kwargs)

    monkeypatch.setattr(provider, 'generate', remaining_only)
    if role == 'C':
        monkeypatch.setattr('sciel.engine.loop.cascade_evaluate',
                            lambda *a, **k: pytest.fail('completed evaluation replayed physically'))
    stop_after_publication(monkeypatch, directory)
    resumed = build(resume=True)
    result = resumed.run()
    assert result['reason'] == 'paused' and result['iteration_no'] == 4
    assert result['B_evals'] == (3 if role == 'G' else 4)
    assert result['cost_total'] == 0 and resumed.llm_unpriced_calls == (6 if role == 'G' else 7)
    assert len(provider.opener.calls) == 1
    assert all(physical_files(directory)[key] == value for key, value in saved.items())
    assert len(resumed.research.knowledge.lessons) == 3
    phase = 'generation_abandoned' if role == 'G' else 'aux_abandoned'
    abandoned = [row for row in events(directory) if row.get('phase') == phase]
    assert len(abandoned) == 1 and abandoned[0]['cost'] is None
    assert abandoned[0]['current_call_cost_unknown'] is True
    if role == 'C':
        skipped = [c for c in resumed.research.conclusions.values() if c['status'] == 'skipped']
        assert len(skipped) == 1 and skipped[0]['accepted_lesson_ids'] == []
    else:
        assert resumed.n_nochild == 1
        assert abandoned[0]['experiment_id'] not in resumed.research.conclusions
    # Second cold load must neither add accounting nor run a physical request.
    again = build(resume=True)
    again._stop_at = result['B_evals']
    again.run()
    assert again.llm_unpriced_calls == resumed.llm_unpriced_calls
    assert len([r for r in events(directory) if r.get('phase') == phase]) == 1


@pytest.mark.parametrize('mode', ['legacy', 'wrong_body', 'wrong_model',
    'response_identity', 'item_identity', 'text_conflict', 'delta_after_done'])
def test_ambiguous_or_unbound_errors_remain_stopped(tmp_path, monkeypatch, mode):
    first, _, _, _, directory = stopped(tmp_path, monkeypatch, mode=mode)
    try:
        with pytest.raises(ProviderRecoveryError):
            parallel_recovery.inspect_parallel_stop(directory / 'checkpoint.json')
        assert not list((directory / '.parallel').rglob('abandonment.json'))
    finally:
        first.close()


def test_typed_failure_does_not_broaden_serial_recovery_or_override_stop(tmp_path, monkeypatch):
    first, _, _, _, directory = stopped(tmp_path, monkeypatch)
    path = directory / 'checkpoint.json'
    receipt = parallel_recovery.inspect_parallel_stop(path)
    with pytest.raises(ProviderRecoveryError, match='request-stage'):
        _provider_failure(receipt['failed_calls'][0]['error'], legacy=False, allow_legacy=False)
    (directory / 'STOP').touch()
    with pytest.raises(ProviderRecoveryError, match='STOP'):
        parallel_recovery.authorize_parallel_recovery(first, path, receipt)
    assert not list((directory / '.parallel').rglob('abandonment.json'))
    first.close()


def test_authorized_parser_failure_survives_crash_before_publication(tmp_path, monkeypatch):
    first, build, provider, _, directory = stopped(tmp_path, monkeypatch)
    path = directory / 'checkpoint.json'
    receipt = parallel_recovery.inspect_parallel_stop(path)
    parallel_recovery.authorize_parallel_recovery(first, path, receipt)
    first.close()
    saved = physical_files(directory)
    monkeypatch.setattr(provider, 'generate', lambda *a, **k: pytest.fail('physical redispatch'))
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', lambda *a, **k: pytest.fail('reevaluation'))
    def crash(transaction):
        raise RuntimeError('synthetic publication crash')

    with monkeypatch.context() as local:
        local.setattr(parallel_store.WaveTransaction, 'commit', crash)
        engine = build(resume=True)
        with pytest.raises(CampaignStopped, match='publication crash'):
            engine.run()
    assert physical_files(directory) == saved
    stop_after_publication(monkeypatch, directory)
    resumed = build(resume=True)
    assert resumed.run()['reason'] == 'paused'
    assert resumed.llm_unpriced_calls == 7
    assert len(resumed.research.knowledge.lessons) == 3
    assert physical_files(directory) == saved


@pytest.mark.parametrize('damage', ['config_provider', 'request_metadata', 'missing_receipt'])
def test_request_bound_protocol_recovery_rejects_other_transport_or_changed_evidence(tmp_path, monkeypatch, damage):
    first, _, _, _, directory = stopped(tmp_path, monkeypatch)
    path = directory / 'checkpoint.json'
    receipt = parallel_recovery.inspect_parallel_stop(path)
    row = receipt['failed_calls'][0]
    wave = directory / '.parallel' / ('wave-' + receipt['wave_id'])
    call = wave / 'calls' / row['directory']
    if damage == 'config_provider':
        cfg = json.loads(path.read_text())['config']
        cfg['llm']['provider'] = 'openrouter'
        assert not parallel_recovery._recoverable_call(wave, row, cfg)
    elif damage == 'request_metadata':
        intent = json.loads((call / 'intent.json').read_text())
        intent['request']['provider_request']['schema'] = 'sciel-codex-responses/1'
        (call / 'intent.json').write_text(json.dumps(intent))
        with pytest.raises(ProviderRecoveryError):
            parallel_recovery.authorize_parallel_recovery(first, path, receipt)
    else:
        (call / 'receipt.json').unlink()
        with pytest.raises(ProviderRecoveryError, match='terminal receipt'):
            parallel_recovery.inspect_parallel_stop(path)
    assert not list((directory / '.parallel').rglob('abandonment.json'))
    first.close()
