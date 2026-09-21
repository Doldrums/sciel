"""Real Engine auxiliary STOP/replay; fixed fake provider/evaluator, no authored execution."""
from collections import Counter
import json
import threading

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.parallel import _pause_error
from sciel.engine.parallel_calls import ParallelPause
from sciel.engine.research import ResearchController
from test_parallel_engine import ResearchProvider, config, evaluator, events, task_fixture
from test_parallel_provider_recovery import physical_files, transient


def setup_aux_pause(tmp_path, monkeypatch, *, fatal=False, malformed_response=False):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    cfg = config(task, tmp_path / 'capacity', max_iterations=4,
        research={'enabled': True, 'reflection_interval': 0, 'max_aux_calls': None,
                  'structured_auxiliary': True, 'conclusion_source_refs': True,
                  'conclusion_repair_attempts': 1})
    cfg['parallel_search'].update(host_llm_slots=1, provider_slots=1)
    directory = tmp_path / 'run'
    provider = ResearchProvider(concurrent=1)
    evaluations, _ = evaluator(monkeypatch)
    original_generate = provider.generate
    paused_sibling = threading.Event()
    original_consume = ResearchController.consume_aux
    seen_pause = []

    def consume(controller, request, response=None, error=None):
        if isinstance(error, ParallelPause):
            seen_pause.append(request['call_index'])
            paused_sibling.set()
        return original_consume(controller, request, response=response, error=error)

    monkeypatch.setattr(ResearchController, 'consume_aux', consume)
    first_aux = [True]

    def generate(prompt, *args, **kwargs):
        is_aux = 'Review this one experiment.' in prompt[:160]
        response = original_generate(prompt, *args, **kwargs)
        if is_aux and first_aux[0]:
            first_aux[0] = False
            (directory / 'STOP').touch()
            # This paid C keeps the single provider lease while sibling C
            # requests are cancelled before dispatch. Its returned malformed
            # review then tries to prepare a repair after that pause was seen.
            assert paused_sibling.wait(3)
            if fatal:
                raise transient(401)
            if malformed_response:
                response.cost = None
            else:
                response.text = '{}'
        return response

    monkeypatch.setattr(provider, 'generate', generate)

    def build(resume=False):
        engine = build_engine(cfg, str(tasks), str(directory), llm=provider)
        if resume:
            checkpoint.load(engine, str(directory / 'checkpoint.json'))
        return engine

    return build, provider, evaluations, directory, seen_pause


def test_auxiliary_sibling_pause_keeps_paid_review_and_resumes_exactly_once(tmp_path, monkeypatch):
    build, provider, evaluations, directory, seen_pause = setup_aux_pause(tmp_path, monkeypatch)
    first = build()
    paused = first.run()
    assert seen_pause and paused['reason'] == 'paused'
    runtime = json.loads((directory / 'runtime.json').read_bytes())
    assert runtime['status'] == 'paused' and runtime['error_type'] is None
    baseline = json.loads((directory / 'checkpoint.json').read_bytes())
    assert baseline['research']['stopped_reason'] is None
    assert baseline['cost_total'] == baseline['iteration_no'] == 0
    assert Counter(provider.roles) == {'G': 4, 'C': 1}
    assert len(evaluations) == 5
    saved = physical_files(directory)
    paid_prompts = set(provider.calls)
    original_generate = provider.generate

    def no_repaid_prompt(prompt, *args, **kwargs):
        assert prompt not in paid_prompts, 'A returned paid G/C response must replay from its receipt'
        return original_generate(prompt, *args, **kwargs)

    monkeypatch.setattr(provider, 'generate', no_repaid_prompt)
    resumed = build(resume=True)
    complete = resumed.run()
    assert complete['iteration_no'] == complete['B_evals'] == 4
    assert complete['cost_total'] == pytest.approx(.17)  # 4 G + 4 initial C + 1 repair.
    assert len(evaluations) == 5
    assert Counter(provider.roles) == {'G': 4, 'C': 5}
    assert all(physical_files(directory)[path] == raw for path, raw in saved.items())
    assert resumed.research.stopped_reason is None
    saved_research = resumed.research.state()
    assert '_stopped_cause' not in saved_research
    resumed.research._stopped_cause = ParallelPause('stale transient cause fixture')
    resumed.research.load_state(saved_research)
    assert getattr(resumed.research, '_stopped_cause', None) is None
    assert len(resumed.research.conclusions) == 4
    auxiliary = [row for row in events(directory) if row.get('phase') == 'aux_result']
    assert len(auxiliary) == 5 and sum(row['cost'] for row in auxiliary) == pytest.approx(.05)
    assert not any(row.get('current_call_cost_unknown') for row in auxiliary)
    assert not list((directory / '.parallel').rglob('abandonment.json'))
    saved_after = physical_files(directory)
    again = build(resume=True)
    final = again.run()
    assert final['cost_total'] == pytest.approx(.17)
    assert physical_files(directory) == saved_after and Counter(provider.roles) == {'G': 4, 'C': 5}


@pytest.mark.parametrize('mode', ['paid_request_failure', 'returned_unknown_cost'])
def test_auxiliary_real_failure_after_sibling_pause_remains_fatal(tmp_path, monkeypatch, mode):
    build, provider, evaluations, directory, seen_pause = setup_aux_pause(tmp_path, monkeypatch,
        fatal=mode == 'paid_request_failure', malformed_response=mode == 'returned_unknown_cost')
    first = build()
    with pytest.raises(CampaignStopped) as failure:
        first.run()
    assert seen_pause and not _pause_error(failure.value)
    runtime = json.loads((directory / 'runtime.json').read_bytes())
    assert runtime['status'] == 'stopped' and runtime['error_type'] == 'CampaignStopped'
    assert Counter(provider.roles) == {'G': 4, 'C': 1}
    assert len(evaluations) == 5
    saved = physical_files(directory)
    resumed = build(resume=True)
    with pytest.raises(CampaignStopped) as retry:
        resumed.run()
    assert not _pause_error(retry.value)
    assert physical_files(directory) == saved
    assert Counter(provider.roles) == {'G': 4, 'C': 1}


def test_auxiliary_stop_persistence_failure_is_fatal_not_hidden_by_pause_context(tmp_path, monkeypatch):
    build, provider, evaluations, directory, seen_pause = setup_aux_pause(tmp_path, monkeypatch)
    original_emit = ResearchController.emit

    def fail_pause_event(controller, phase, **fields):
        if phase == 'aux_stopped' and fields.get('error_type') == 'ParallelPause':
            raise OSError('fixture stop journal write failed')
        return original_emit(controller, phase, **fields)

    monkeypatch.setattr(ResearchController, 'emit', fail_pause_event)
    first = build()
    with pytest.raises(CampaignStopped) as failure:
        first.run()
    assert seen_pause and not _pause_error(failure.value)
    assert 'persistence failed' in str(failure.value)
    runtime = json.loads((directory / 'runtime.json').read_bytes())
    assert runtime['status'] == 'stopped' and runtime['error_type'] == 'CampaignStopped'
    assert Counter(provider.roles) == {'G': 4, 'C': 1}
    assert len(evaluations) == 5
    paid = []
    for path in (directory / '.parallel').rglob('intent.json'):
        intent = json.loads(path.read_bytes())
        receipt_path = path.with_name('receipt.json')
        if intent['kind'] == 'llm' and receipt_path.exists():
            paid.append(json.loads(receipt_path.read_bytes())['payload'])
    assert len(paid) == 5 and all(row['status'] == 'returned' for row in paid)
    assert sum(row['value']['cost'] for row in paid) == pytest.approx(.13)
    assert not list((directory / '.parallel').rglob('abandonment.json'))
