"""Crash/replay, capacity and unknown-outcome semantics of tool episodes."""
from contextlib import nullcontext
import json
from pathlib import Path

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import parallel_calls
from sciel.engine.loop import Engine
from sciel.engine.parallel_calls import CallLedger, CooperativeIO, UnknownParallelCall

from test_research_tool_engine import FakeTools, setup_tools


@pytest.mark.parametrize('after_commit', [False, True])
def test_replay_does_not_repeat_tool_or_provider_and_charges_once(tmp_path, monkeypatch, after_commit):
    build, provider, evaluations, directory = setup_tools(tmp_path, monkeypatch)
    commit = Engine._commit_step
    crashed = False
    def crash(self, *args, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            if after_commit:
                commit(self, *args, **kwargs)
            raise RuntimeError('fixture tool wave crash')
        return commit(self, *args, **kwargs)
    monkeypatch.setattr(Engine, '_commit_step', crash)
    with pytest.raises(CampaignStopped, match='fixture tool wave crash'):
        build().run()
    baseline = json.loads((directory / 'checkpoint.json').read_text())
    assert baseline['cost_total'] == 0 and baseline['research_tools']['charged'] == []
    assert len(provider.calls) == 4 and len(evaluations) == 3
    assert FakeTools.invocations == ['1', '2']
    saved_artifacts = {path.name:path.read_bytes() for path in
                       (directory / 'research_tools' / 'artifacts').glob('*.json')}
    assert len(saved_artifacts) == 2
    expected_tools = [row['tool_id'] for row in sorted(
        (json.loads(raw)['payload'] for raw in saved_artifacts.values()), key=lambda row:row['iteration'])]
    engine = build(resume=True)
    result = engine.run()
    assert result['cost_total'] == pytest.approx(.12) and result['B_evals'] == 2
    assert len(provider.calls) == 4 and len(evaluations) == 3
    assert FakeTools.invocations == ['1', '2']
    assert engine.research_tools.charged == ['1', '2']
    assert engine.research_tools.visible == expected_tools
    assert {path.name:path.read_bytes() for path in
            (directory / 'research_tools' / 'artifacts').glob('*.json')} == saved_artifacts
    assert len(engine.research_tools.committed) == 2


def test_unknown_tool_dispatch_pauses_without_candidate_or_tool_replay(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup_tools(tmp_path, monkeypatch, workers=1, max_iterations=1)
    write = parallel_calls.durable_json
    def lose_receipt(path, value):
        if (Path(path).name == 'receipt.json'
                and value['payload'].get('value', {}).get('output') == {'probe': 999}):
            raise OSError('fixture lost execution receipt')
        return write(path, value)
    monkeypatch.setattr(parallel_calls, 'durable_json', lose_receipt)
    with pytest.raises(CampaignStopped):
        build().run()
    assert len(provider.calls) == len(evaluations) == len(FakeTools.invocations) == 1
    monkeypatch.setattr(parallel_calls, 'durable_json', write)
    with pytest.raises(CampaignStopped, match='no terminal receipt'):
        build(resume=True).run()
    assert len(provider.calls) == len(evaluations) == len(FakeTools.invocations) == 1


def test_tool_receipt_json_replay_uses_research_admission_and_separate_pending_count(tmp_path):
    slots, calls, states = [], [], []
    class Capacity:
        def acquire(self, resource, **kwargs):
            slots.append((resource, kwargs))
            return nullcontext()
    expected = {'status': 'failed', 'output': None, 'diagnostic': 'local tool error', 'cpu_seconds': 0.}
    def run():
        io = CooperativeIO(CallLedger(tmp_path, Capacity(), 'fixture'), 1, on_state=states.append)
        def invoke():
            calls.append('executed')
            return expected
        try:
            return io.run([('tool', lambda: io.call('tool_execution', {'id': 'a'}, invoke))])
        finally:
            io.close()
    assert run() == ({'tool': expected}, {})
    assert run() == ({'tool': expected}, {})
    assert calls == ['executed'] and len(slots) == 1 and slots[0][0] == 'research_evaluator'
    assert 'provider' not in slots[0][1]
    assert states[0] == {'pending_calls': 1, 'pending_llm_calls': 0,
                         'pending_evaluations': 0, 'pending_tool_calls': 1}
    assert states[-1]['pending_tool_calls'] == 0


def test_tool_exception_is_unknown_and_never_a_candidate_failure(tmp_path):
    class Capacity:
        def acquire(self, *args, **kwargs):
            return nullcontext()
    io = CooperativeIO(CallLedger(tmp_path, Capacity(), 'fixture'), 1)
    def broken():
        raise RuntimeError('no trustworthy receipt')
    try:
        results, errors = io.run([('tool', lambda: io.call('tool_execution', {}, broken))])
    finally:
        io.close()
    assert not results and isinstance(errors['tool'], UnknownParallelCall)
