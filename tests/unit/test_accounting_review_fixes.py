"""Physical request accounting and concurrent stop races, entirely offline."""
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.diagnostics import methods_observer as observer
from sciel.engine.llm.client import EnsembleLLM, LLMError
from sciel.engine.llm.trace import TracedLLM
from sciel.engine.types import NoChild
from sciel.engine.worker import run_worker
from test_llm_error_metadata import model
from test_llm_trace import rows
from test_parallel_calls import make_io
from test_research_retry_feedback import setup


@pytest.mark.parametrize('choices,reason', [
    ([], 'empty_choices'),
    ([SimpleNamespace(message=None)], 'malformed_response'),
    ([SimpleNamespace(message=SimpleNamespace(content={'secret': 'not text'}))], 'malformed_response'),
])
@pytest.mark.parametrize('research_on', [False, True])
def test_paid_malformed_response_is_charged_once_without_retry_evaluation_or_lessons(tmp_path, choices, reason, research_on):
    llm, request = model(response=SimpleNamespace(choices=choices,
        usage=SimpleNamespace(cost=.25, prompt_tokens=10, completion_tokens=20)))
    _, research, snap, events, evaluated, evaluate = setup([], research_on=research_on)
    traced = TracedLLM(llm, tmp_path, lambda _: {'role': 'G'})
    result = run_worker('diff', snap, traced, evaluate, dup_cap=4,
                        research=research if research_on else None)
    assert isinstance(result, NoChild) and result.reason == 'llm_failed' and result.cost == .25
    assert request.call_count == 1 and not evaluated
    assert result.meta['response_error']['reason'] == reason
    recorded = rows(tmp_path)[-1]
    assert recorded['event'] == 'response' and recorded['generation_returned'] is True
    assert recorded['cost'] == .25 and recorded['current_call_cost_unknown'] is False
    assert recorded['response_error'] == result.meta['response_error']
    assert 'secret' not in json.dumps(recorded)
    if research_on:
        assert research.pending == {} and research.queue == [] and research.knowledge.lessons == []
        record, = research.knowledge.experiments
        assert record['generation_cost'] == .25 and record['generation_dispatch'] == 'returned'
        assert record['status'] == 'generation_stopped' and record['hypothesis'] is None
        assert record['score'] is None


def test_paid_malformed_response_round_trips_durable_receipt_without_second_dispatch(tmp_path):
    llm, request = model(response=SimpleNamespace(choices=[], usage=SimpleNamespace(cost=.25)))
    def run():
        io = make_io(tmp_path)
        try:
            return io.run([('same', lambda: io.call('llm', {'prompt': 'same'},
                lambda: llm.generate('same', 'fixture')))])
        finally:
            io.close()
    first, errors = run()
    replay, replay_errors = run()
    assert not errors and not replay_errors and first == replay and request.call_count == 1
    assert replay['same'].cost == .25 and replay['same'].text == ''
    assert replay['same'].response_error['reason'] == 'empty_choices'


def test_known_malformed_final_response_keeps_prior_worker_charges():
    good_empty = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=''))],
                                usage=SimpleNamespace(cost=.125))
    bad = SimpleNamespace(choices=[], usage=SimpleNamespace(cost=.25))
    llm, request = model(error=[good_empty, bad])
    _, _, snapshot, _, evaluated, evaluate = setup([], research_on=False)
    result = run_worker('diff', snapshot, llm, evaluate, dup_cap=4)
    assert result.reason == 'llm_failed' and result.cost == .375
    assert request.call_count == 2 and not evaluated
    assert [r['cost'] for r in result.meta['validation_trace']['attempts']] == [.125, .25]


def test_authoritative_cost_survives_broken_optional_token_accessor():
    class Usage:
        cost = .25
        @property
        def prompt_tokens(self):
            raise RuntimeError('private SDK detail')
    llm, _ = model(response=SimpleNamespace(choices=[], usage=Usage()))
    result = llm.generate('same', 'fixture')
    assert result.cost == .25 and result.prompt_tokens is None and result.response_error


def test_legacy_retry_configuration_rejected_before_client_construction():
    with pytest.raises(ValueError, match='max_retries must be 1'):
        EnsembleLLM(['fixture'], [1], 'https://unused.invalid', 'unused', max_retries=3)


def test_timeout_followed_by_success_is_not_a_hidden_second_paid_dispatch():
    llm, request = model(error=[TimeoutError('private remote detail'),
        SimpleNamespace(choices=[], usage=SimpleNamespace(cost=.25))])
    llm.max_retries = 3
    with pytest.raises(LLMError) as exc:
        llm.generate('same', 'fixture')
    assert exc.value.metadata['reason'] == 'timeout' and request.call_count == 1


@pytest.mark.parametrize('reported,expected', [('accounting_stopped', 'unknown'), ('failed', 'failed')])
def test_stop_signal_does_not_mask_concurrent_failed_worker_receipt(tmp_path, monkeypatch, reported, expected):
    terminal = tmp_path / 'worker.json'
    class Child:
        pid, returncode = 123456789, None
        def poll(self):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            self.returncode = 2
            terminal.write_text(json.dumps(dict(schema='sciel-method-worker-terminal/1',
                pid=self.pid, exit_code=2, controller_cpu_seconds=.25)))
            (tmp_path/'run-report.json').write_text(json.dumps(dict(status=reported)))
            return self.returncode
    monkeypatch.setattr(observer.subprocess, 'Popen', Mock(return_value=Child()))
    monkeypatch.setattr(observer, 'process_identity', lambda _: {'kind': 'fixture', 'value': 'stable'})
    monkeypatch.setattr(observer, 'process_matches', lambda _: True)
    monkeypatch.setattr(observer, '_sample_cpu', lambda _: .1)
    monkeypatch.setattr(observer.os, 'kill', Mock())
    result = observer.observe(['offline-placeholder'], tmp_path, interval=.05, worker_terminal=terminal)
    assert result['status'] == expected and result['reason'] == reported
    assert result['stop_requested'] is True and result['stop_signal'] == signal.SIGTERM
    assert json.loads((tmp_path/'runtime.json').read_text())['status'] == expected


def test_supervisor_concurrent_operator_stop_preserves_unknown_outcome(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2]/'scripts/continue_live_run.py'
    spec = importlib.util.spec_from_file_location('accounting_supervisor_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = dict(budget=None, max_iterations=None, llm={'mock': True})
    engine = SimpleNamespace(iteration_no=1, research=SimpleNamespace(stopped_reason=None),
                             _stop_requested=True, close=Mock())
    monkeypatch.setattr(module.sys, 'argv', ['continue_live_run.py', str(tmp_path/'run'),
                                           '--fresh-config', str(tmp_path/'config.json')])
    monkeypatch.setattr(module, 'load_config', lambda _: config)
    monkeypatch.setattr(module, 'validate', lambda value: value)
    build, inspect, wait = Mock(return_value=engine), Mock(), Mock()
    monkeypatch.setattr(module, 'build_engine', build)
    monkeypatch.setattr(module, 'inspect_provider_stop', inspect)
    monkeypatch.setattr(module, 'backoff', wait)
    monkeypatch.setattr(module, '_run_engine', Mock(side_effect=CampaignStopped('outcome unknown')))
    with pytest.raises(CampaignStopped, match='outcome unknown'):
        module.main()
    engine.close.assert_called_once()
    build.assert_called_once()
    inspect.assert_not_called()
    wait.assert_not_called()
