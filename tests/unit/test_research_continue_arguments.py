"""Explicit continuation arguments replace, with paid I/O replay unchanged."""
from copy import deepcopy
import json

import pytest

from test_background_tool_controller import setup, launch, finish, snapshot, LLM, IO


@pytest.mark.parametrize('arguments', [{}, {'changed': 1}])
def test_continue_replaces_arguments_without_changing_saved_parent(tmp_path, arguments):
    tools, executor, _, costs = setup(tmp_path)
    parent = launch(tools)
    original = deepcopy(tools.jobs.spec(parent))
    assert original['request']['inputs']['arguments'] == {'probe': 3}
    finish(tools, parent)
    current = snapshot(iteration=2)
    current.research_cells_context = tools.context(current)
    plan = tools.prepare(current, signals={})
    before = json.loads(json.dumps(tools.state()))
    io = IO()
    action = dict(action='continue', job_id=parent, cpu_seconds=360, wall_seconds=480,
                  arguments_json=json.dumps(arguments))
    class SavedLLM:
        def __init__(self, model): self.model = model
        def generate(self, prompt, model, temperature=None, **options):
            return io.call('llm', dict(prompt=prompt, model=model,
                temperature=temperature, options=options),
                lambda: self.model.generate(prompt, model, temperature, **options))

    model = LLM([action])
    prior_costs = list(costs)
    first = tools.execute(plan, current, SavedLLM(model), io=io)
    child = first['submitted_jobs'][0]['job_id']
    spec = deepcopy(tools.jobs.spec(child))
    assert spec['request']['inputs']['arguments'] == arguments
    assert spec['request']['inputs']['continuation_state']['array'] == list(range(8000))
    assert spec['request']['inputs']['continuation_state']['rng'] == 'saved'
    assert spec['request']['inputs']['parent_source'] == original['request']['inputs']['parent_source']
    assert spec['request']['parent_job_id'] == parent
    assert tools.jobs.spec(parent) == original
    tools.load_state(before)
    costs[:] = prior_costs  # Wave rollback also restores its accounting sink.
    forbidden = LLM([])
    replay = tools.execute(plan, current, SavedLLM(forbidden), io=io)
    assert replay == first
    assert tools.jobs.spec(child) == spec
    assert len(executor.calls) == 1  # Continued job is queued, never executed here.
    tools.commit(replay)
    tools.load_state(json.loads(json.dumps(tools.state())))
    assert tools.jobs.spec(child) == spec
    assert len(model.prompts) == 1 and forbidden.prompts == []
    assert costs == [.03, .03]  # Original job and one continued-job proposal.
    assert first['known_llm_cost'] == replay['known_llm_cost'] == .03
