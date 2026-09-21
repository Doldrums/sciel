"""Real Engine/WAL planner abandonment, with entirely local physical leaves.

The Codex adapter and SSE consumer, coordinator, recovery authorization and
research-tools adapter are real. Ordinary LLM/evaluation responses are fixtures;
candidate/tool code is never executed and no credentials or HTTP are used.
"""
from collections import Counter
import json
from pathlib import Path
import sys
import threading

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint, parallel_recovery
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import COST_PROVENANCE_SCHEMA
from sciel.engine.llm.codex import CodexResponsesLLM
from sciel.engine.llm.codex_protocol import ENDPOINT_SHA256
from sciel.engine.recovery import ProviderRecoveryError

from test_codex_responses import Opener
from test_parallel_engine import ResearchProvider, config, evaluator, events, number, task_fixture
from test_parallel_provider_recovery import physical_files, stop_after_publication
from test_research_tool_controller import IMAGE
from test_research_tool_result_flow import SavedExecutor


class PlannerProtocolProvider(ResearchProvider):
    def __init__(self, ordinary_done, *, mixed=None):
        super().__init__(concurrent=1)
        self.ordinary_done = ordinary_done
        self.planner_started = threading.Event()
        self.planner_prompts = []
        self.mixed = mixed
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
        if prompt.startswith('SCIEL_RESEARCH_TOOL_EPISODE'):
            self.planner_prompts.append(prompt)
            self.planner_started.set()
            assert self.ordinary_done.wait(5), 'ordinary sibling did not reach its physical leaf'
            # Empty fake SSE reaches the actual _consume EOF failure path.
            return self.adapter.generate(prompt, model, temperature, **options)
        if self.mixed == 'worker':
            self.ordinary_done.set()
            assert self.planner_started.wait(5)
            raise ValueError('independent ordinary worker failure')
        response = super().generate(prompt, model, temperature, **options)
        response.cost = None
        response.cost_provenance = dict(schema=COST_PROVENANCE_SCHEMA,
            source='subscription_unpriced', response_status='completed',
            provider='codex_responses', endpoint_sha256=ENDPOINT_SHA256)
        return response


def stopped_planner(tmp_path, monkeypatch, *, mixed=None):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    evaluated, _ = evaluator(monkeypatch)
    from sciel.engine import loop
    physical_evaluate = loop.cascade_evaluate
    ordinary_done = threading.Event()
    provider = PlannerProtocolProvider(ordinary_done, mixed=mixed)

    def evaluate(task, path, **options):
        is_candidate = number(Path(path).read_text()) != 0
        if is_candidate and mixed == 'evaluation':
            ordinary_done.set()
            assert provider.planner_started.wait(5)
            raise ValueError('independent evaluator infrastructure failure')
        result = physical_evaluate(task, path, **options)
        if is_candidate:
            ordinary_done.set()
        return result

    monkeypatch.setattr(loop, 'cascade_evaluate', evaluate)
    backend = SavedExecutor([])
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor', lambda image: backend)
    cfg = config(task, tmp_path / 'capacity', workers=1, max_iterations=None,
        llm=provider.codex_config,
        research={'enabled': True, 'max_aux_calls': None, 'reflection_interval': 0},
        research_tools={'enabled': True, 'image_id': IMAGE, 'interval': 1000,
            'proactive_every_attempts': 1000, 'max_scientific_steps': 3,
            'yield_after_success': True, 'independent_research': True,
            'research_jobs_per_wave': 1, 'shared_observations': True,
            'experimental_protocol': True, 'max_requests': 2,
            'share_between_islands': False})
    directory = tmp_path / 'run'

    def build(resume=False):
        engine = build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
        if resume:
            checkpoint.load(engine, str(directory / 'checkpoint.json'))
        return engine

    first = build()
    try:
        with pytest.raises(CampaignStopped):
            first.run()
        assert first._writer_lock._file.closed
    except BaseException:
        first.close()
        raise
    return first, build, provider, backend, evaluated, directory


def test_actual_codex_planner_error_is_abandoned_once_then_ordinary_evolution_commits(tmp_path, monkeypatch):
    first, build, provider, backend, evaluated, directory = stopped_planner(tmp_path, monkeypatch)
    path = directory / 'checkpoint.json'
    baseline = json.loads(path.read_text())
    assert baseline['iteration_no'] == baseline['B_evals'] == 0
    authorization = parallel_recovery.inspect_parallel_stop(path)
    assert authorization['unknown_cost'] is True and authorization['retry'] is False
    assert len(authorization['failed_calls']) == len(provider.opener.calls) == 1
    failed = authorization['failed_calls'][0]
    assert failed['key'].startswith('research:')
    assert failed['error']['codex_protocol']['code'] == 'stream_eof'
    assert failed['not_dispatched'] is False
    assert len(evaluated) == 2  # Canonical seed plus the ordinary sibling.
    saved = physical_files(directory)
    parallel_recovery.authorize_parallel_recovery(first, path, authorization)
    parallel_recovery.authorize_parallel_recovery(first, path, authorization)
    first.close()
    original = provider.generate

    def new_conclusion_only(prompt, *args, **kwargs):
        assert 'Review this one experiment.' in prompt[:160], 'paid planner/generation was retried'
        return original(prompt, *args, **kwargs)

    monkeypatch.setattr(provider, 'generate', new_conclusion_only)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate',
                        lambda *a, **k: pytest.fail('paid evaluation was repeated'))
    stop_after_publication(monkeypatch, directory)
    resumed = build(resume=True)
    result = resumed.run()
    assert result['reason'] == 'paused'
    assert result['iteration_no'] == result['B_evals'] == 1
    assert result['cost_total'] == 0 and resumed.llm_unpriced_calls == 2
    assert Counter(provider.roles) == {'G': 1, 'C': 1}
    assert len(provider.opener.calls) == len(provider.planner_prompts) == 1
    assert not backend.calls
    assert all(physical_files(directory)[key] == value for key, value in saved.items())
    abandoned = [row for row in events(directory) if row.get('phase') == 'llm_abandoned']
    assert len(abandoned) == 1
    assert abandoned[0]['cost'] is None and abandoned[0]['current_call_cost_unknown'] is True
    assert abandoned[0]['provider_abandonment']['request_sha256'] == failed['request_sha256']
    artifact = resumed.research_tools._read_artifact(abandoned[0]['receipt_id'], 'proposal_abandonment')
    assert artifact['llm_error'] == failed['error']
    assert artifact['prompt'] == provider.planner_prompts[0]
    state = resumed.research_tools.state()
    assert len(state['committed_episodes']) == 1 and not state['registered']
    episode = next(iter(state['lifecycle']['research_state'].values()))
    assert episode['status'] == 'provider_abandoned'
    assert episode['known_llm_cost'] == 0 and episode['unknown_llm_calls'] == 1
    assert episode['scientific_steps'] == 0 and episode['conclusion'] is None
    assert episode['steps'][0]['status'] == 'unknown'
    queued = next(iter(state['continuations']['items'].values()))
    assert queued['status'] == 'closed' and queued['reason'] == 'provider_abandoned'
    assert len(resumed.research.knowledge.lessons) == 1  # Ordinary measured candidate only.
    # Another cold load cannot repay, relog or revive the failed research call.
    again = build(resume=True)
    again._stop_at = result['B_evals']
    again.run()
    assert again.research_tools.state() == state
    assert again.llm_unpriced_calls == resumed.llm_unpriced_calls
    assert len([row for row in events(directory) if row.get('phase') == 'llm_abandoned']) == 1
    assert len(provider.opener.calls) == 1 and not backend.calls


@pytest.mark.parametrize('mixed', ['worker', 'evaluation'])
def test_typed_planner_error_cannot_hide_independent_worker_or_evaluator_failure(tmp_path, monkeypatch, mixed):
    first, _, provider, backend, _, directory = stopped_planner(tmp_path, monkeypatch, mixed=mixed)
    try:
        assert len(provider.opener.calls) == 1 and not backend.calls
        pointer = json.loads((directory / '.parallel/current.json').read_text())
        wave = directory / '.parallel' / ('wave-' + pointer['wave_id'])
        stop = json.loads((wave / 'provider-stop.json').read_text())
        sealed = json.loads((wave / 'provider_stops' / (stop['id'] + '.json')).read_text())
        assert {'provider', 'fatal'} <= {row['kind'] for row in sealed['errors']}
        errors = [row for row in sealed['calls'] if row['status'] == 'error']
        assert len(errors) == 2
        assert any(row['kind'] == ('evaluation' if mixed == 'evaluation' else 'llm')
                   and row['error'].get('cause_type') != '_CodexProtocolError' for row in errors)
        with pytest.raises(ProviderRecoveryError, match='unclassified or non-provider failure'):
            parallel_recovery.inspect_parallel_stop(directory / 'checkpoint.json')
        assert not list((directory / '.parallel').rglob('abandonment.json'))
    finally:
        first.close()
