"""First eligible research episodes and persisted cooldowns, without real dispatch."""
from copy import deepcopy

import pytest

from sciel.research_tools.config import library_limits, validate_config
from sciel.research_tools.controller import ResearchTools
from sciel.research_tools.jobs import BackgroundJobs
from sciel.research_tools.library import ToolLibrary
from test_research_tool_controller import IMAGE, IO, LLM, snapshot
from test_research_tool_result_flow import SavedExecutor, action, collect, result


def controller(tmp_path, *, background=False, **changes):
    changes.setdefault('max_scientific_steps',1)
    config = validate_config({"enabled": True, "image_id": IMAGE, "interval": 4,
        "max_requests": 1, "jobs": {"enabled": background}, **changes})
    backend = SavedExecutor([result({"probe": "observed"})])
    run = tmp_path / "run"
    library = ToolLibrary(run, {"image_id": IMAGE}, library_limits(config), executor=backend)
    jobs = BackgroundJobs(run, library, config["jobs"], launch=lambda: None) if background else None
    return ResearchTools(config, run, task_identity={"task": "fixture/any", "task_hash": "fixture"},
        emit=lambda **event: None, charge=lambda value: None, library=library, jobs=jobs)


def signals(active):
    return {"schema": "sciel-stagnation-signals/1",
        "population_stall": {"threshold_reached": active, "last_record_B_evals": 0}}


def prepare(tools, *, active=True, island=0, iteration=0):
    current = snapshot(island=island, iteration=iteration)
    return current, tools.prepare(current, signals=signals(active))


def commit_skip(tools, current, plan):
    episode = tools.execute(plan, current, LLM([{"action": "skip"}]), io=IO())
    tools.commit(episode)


@pytest.mark.parametrize("healthy_visits", [1, 2, 3])
def test_first_trigger_after_healthy_visits_is_immediate(tmp_path, healthy_visits):
    tools = controller(tmp_path)
    for visit in range(healthy_visits):
        assert prepare(tools, active=False, iteration=visit)[1] is None
    assert tools.state()["last_episode_opportunity"] == {}
    assert tools.state()["lifecycle"]["cells"]["cells"] == {}

    _, plan = prepare(tools, iteration=healthy_visits)
    assert plan["opportunity"] == healthy_visits
    assert plan["last_episode_opportunity"] is None
    assert plan["cadence_reason"] == "first_episode"
    assert tools.state()["last_episode_opportunity"] == {"0": healthy_visits}


def test_cap_refusal_with_dormant_cell_does_not_delay_retry(tmp_path):
    tools = controller(tmp_path, max_active_cells=1)
    first, first_plan = prepare(tools)
    assert prepare(tools, island=1)[1] is None
    refused = tools.state()
    assert refused["last_episode_opportunity"] == {"0": 0}
    assert len(refused["lifecycle"]["cells"]["cells"]) == 2
    dormant_id = refused["lifecycle"]["affinity"]["1"][-1]
    assert tools.lifecycle.cells.get(dormant_id)["active"] is False

    commit_skip(tools, first, first_plan)
    _, retry = prepare(tools, island=1, iteration=1)
    assert retry["cell_id"] == dormant_id
    assert retry["opportunity"] == 1
    assert retry["cadence_reason"] == "first_episode"
    assert tools.state()["last_episode_opportunity"]["1"] == 1


def test_due_but_leased_cell_does_not_restart_cooldown(tmp_path):
    tools = controller(tmp_path)
    first, first_plan = prepare(tools)
    for visit in range(1, 5):
        assert prepare(tools, iteration=visit)[1] is None
    # Visit four was interval-due but the actual first episode still owns its cell.
    assert tools.state()["opportunities"] == {"0": 5}
    assert tools.state()["last_episode_opportunity"] == {"0": 0}

    commit_skip(tools, first, first_plan)
    _, retry = prepare(tools, iteration=5)
    assert retry["opportunity"] == 5
    assert retry["last_episode_opportunity"] == 0
    assert retry["cadence_reason"] == "interval_elapsed"


def test_skip_reserves_cooldown_and_restored_state_preserves_it(tmp_path):
    tools = controller(tmp_path)
    current, plan = prepare(tools)
    commit_skip(tools, current, plan)
    saved = tools.state()
    restored = controller(tmp_path)
    restored.load_state(saved)
    for visit in range(1, 4):
        assert prepare(restored, iteration=visit)[1] is None
    _, next_plan = prepare(restored, iteration=4)
    assert next_plan["last_episode_opportunity"] == 0
    assert next_plan["opportunity"] == 4
    assert next_plan["cadence_reason"] == "interval_elapsed"
    assert saved["opportunities"] == {"0": 1}  # State snapshots are detached.


def test_healthy_only_checkpoint_has_no_artificial_episode_anchor(tmp_path):
    tools = controller(tmp_path)
    for visit in range(3):
        assert prepare(tools, active=False, iteration=visit)[1] is None
    restored = controller(tmp_path)
    restored.load_state(tools.state())
    _, plan = prepare(restored, iteration=3)
    assert plan["opportunity"] == 3
    assert plan["cadence_reason"] == "first_episode"


def test_collected_background_result_bypasses_cooldown_once(tmp_path):
    tools = controller(tmp_path, background=True)
    current, plan = prepare(tools)
    submitted = tools.execute(plan, current, LLM([action(background=True)]), io=IO())
    tools.commit(submitted)
    job_id = submitted["submitted_jobs"][0]["job_id"]
    assert prepare(tools, active=False, iteration=1)[1] is None
    collect(tools, job_id)  # Mock executor returns saved JSON; source is never executed.

    current, review = prepare(tools, active=False, iteration=2)
    assert review["opportunity"] == 2
    assert review["last_episode_opportunity"] == 0
    assert review["cadence_reason"] == "fresh_result"
    assert review["visible_jobs"][0]["job_id"] == job_id
    commit_skip(tools, current, review)
    assert prepare(tools, active=False, iteration=3)[1] is None
    assert tools.state()["last_episode_opportunity"] == {"0": 2}


@pytest.mark.parametrize("anchors", [None, [], {"0": True}, {"0": -1}, {"0": 1},
    {"0": "0"}, {"1": 0}])
def test_invalid_anchor_restore_is_atomic(tmp_path, anchors):
    tools = controller(tmp_path)
    current, plan = prepare(tools)
    commit_skip(tools, current, plan)
    saved = tools.state()
    bad = deepcopy(saved)
    bad["last_episode_opportunity"] = anchors
    with pytest.raises(ValueError, match="cadence anchors"):
        tools.load_state(bad)
    assert tools.state() == saved


@pytest.mark.parametrize("field,value", [("schema", "sciel-island-research-tools/3"),
    ("opportunities", {0: 1}), ("opportunities", {"0": True})])
def test_legacy_contract_and_invalid_visit_cursors_fail_closed(tmp_path, field, value):
    tools = controller(tmp_path)
    current, plan = prepare(tools)
    commit_skip(tools, current, plan)
    saved = tools.state()
    bad = deepcopy(saved)
    bad[field] = value
    with pytest.raises(ValueError):
        tools.load_state(bad)
    assert tools.state() == saved
