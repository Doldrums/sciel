"""Legacy optional lessons still own physical requests and their charges."""
import json
import sys
from types import SimpleNamespace

import pytest

from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.components.tail import extract_lesson
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from test_parallel_engine import Provider, config, evaluator, task_fixture


def test_postmortem_timeout_stops_real_engine_without_hiding_generation_cost(tmp_path, monkeypatch):
    class TimeoutPostmortem(Provider):
        postmortems = 0
        def generate(self, prompt, model, temperature=None, **options):
            if prompt.startswith('An improvement event occurred:'):
                self.postmortems += 1
                raise TimeoutError('fixture timeout')
            return super().generate(prompt, model, temperature, **options)

    task = task_fixture(tmp_path / 'tasks')
    evaluator(monkeypatch)
    provider = TimeoutPostmortem('run_search')
    cfg = config(task, tmp_path / 'capacity', workers=1, num_islands=1,
                 max_iterations=2, parallel_search={'enabled': False}, c2_event_bets=True)
    engine = build_engine(cfg, str(tmp_path / 'tasks'), str(tmp_path / 'run'),
                          python=sys.executable, llm=provider)
    try:
        with pytest.raises(CampaignStopped) as stopped:
            engine.run()
        assert stopped.value.current_call_cost_unknown is True
        assert stopped.value.generation_returned is False
        assert stopped.value.stage == 'postmortem'
        assert provider.postmortems == len(provider.calls) == 1
        assert engine.cost_total == .03
        assert engine._accounting_stopped_reason
        rows = [json.loads(line) for line in (tmp_path / 'run/sciel_events.jsonl').read_text().splitlines()]
        unknown = [row for row in rows if row.get('action') == 'unknown_cost']
        assert len(unknown) == 1 and unknown[0]['cost'] is None
        assert unknown[0]['generation_dispatch'] == 'unknown'
        assert unknown[0]['llm_error']['cause_type'] == 'TimeoutError'
    finally:
        engine.close()


@pytest.mark.parametrize('text', ['not a lesson', None])
def test_returned_parse_rejection_retains_its_known_cost(text):
    llm = SimpleNamespace(generate=lambda *a, **k: LLMResponse(text, 'fixture', .02, 1, 1))
    assert extract_lesson(llm, 'fixture', 'parent', 'child', '') == (None, .02)


def test_not_dispatched_remains_distinct_from_unknown_outcome():
    expected = RequestNotDispatched('fixture capacity')
    def fail(*args, **kwargs):
        raise expected
    with pytest.raises(RequestNotDispatched) as stopped:
        extract_lesson(SimpleNamespace(generate=fail), 'fixture', 'p', 'c', '')
    assert stopped.value is expected
    assert not getattr(stopped.value, 'current_call_cost_unknown', False)
