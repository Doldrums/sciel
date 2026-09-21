"""Replay comparisons reach real model prompts without repeating executions.

The controller, immutable library, disk job collector and Engine generation path
are real. Only provider responses and execution/evaluation boundaries are local
deterministic fixtures; submitted tool or candidate source is never executed.
"""
import json
import sys

import pytest

from sciel.engine.factory import build_engine
from sciel.hypothesis.rendering import decode_evidence
from sciel.research_tools.config import validate_config
from test_research_tool_controller import IMAGE
from test_parallel_engine import ResearchProvider, config, evaluator, task_fixture
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_contract import result
from test_research_tool_result_flow import controller, execute_create, collect


def completed_replay(tmp_path, *, background=False, policy="exact_observations",
                     sample=2, noisy=False, **config_changes):
    common = {"unchanged_large_output": "evidence detail " * 2000} if noisy else {}
    tools, backend, events, costs = controller(tmp_path,
        [result({**common, "sample": 1}), result({**common, "sample": sample})],
        background=background, max_requests=1, **config_changes)
    author = snapshot()
    llm = LLM([])
    local = execute_create(tools, author, policy=policy, background=background)
    tools.commit(local)
    if background:
        collect(tools, local["submitted_jobs"][0]["job_id"])
    foreign = snapshot(1, 2)
    plan = prepare(tools, foreign)
    assert plan["stage"] == "independent_replay"
    io = IO()
    replay = tools.execute(plan, foreign, llm, io=io)
    tools.commit(replay)
    if background:
        collect(tools, replay["submitted_jobs"][0]["job_id"])
    assert len(backend.calls) == 2 and llm.prompts == [] and costs == [.03]
    assert backend.calls[0]["inputs"] == backend.calls[1]["inputs"]
    return tools, backend, events, costs, local, replay, foreign, io


def actual_generation_context(tmp_path, monkeypatch, tools, *, island=0):
    """Use the actual G transport after an already committed tool episode.

    No further tool episode is due in this fixture. The real context retrieval,
    Engine snapshot, merge, research prompt and provider boundary still run.
    """
    tasks = tmp_path / "candidate_tasks"
    task = task_fixture(tasks)
    provider = ResearchProvider(concurrent=1)
    evaluations, _ = evaluator(monkeypatch)
    engine = build_engine(config(task, tmp_path / "capacity", workers=1,
        num_islands=2, research={"enabled": True, "max_aux_calls": None}),
        str(tasks), str(tmp_path / "candidate_run"), python=sys.executable,
        llm=provider)
    try:
        engine.log.run_start()
        engine.seed(portfolio=False)
        engine.research_tools = tools
        engine.cfg["research_tools"] = tools.config
        with monkeypatch.context() as no_new_episode:
            no_new_episode.setattr(tools, "prepare", lambda *args, **kwargs: None)
            plan = engine._prepare_step(island, spawn=False)
            outcome = engine._execute_step(plan, llm=provider)
        assert outcome.ok and len(evaluations) == 2
        assert provider.roles == ["G"]
        prompt = provider.calls[0]
        context = decode_evidence(json.JSONDecoder().raw_decode(
            prompt.split("RESEARCH_CONTEXT_JSON:\n", 1)[1])[0])
        return context["research_cells_context"], prompt
    finally:
        engine.close()


def comparison_card(cards, tool_id):
    matches = [card for card in cards if card.get("replay_comparison")
               and card["source"]["tool_id"] == tool_id]
    assert matches, "the comparison must survive the actual G context budget"
    return matches[0]


def assert_mismatch(comparison, tool_id, *, require_change_detail=True):
    assert comparison["schema"] == "sciel-replay-feedback/1"
    assert comparison["tool_id"] == tool_id
    assert comparison["policy"] == "exact_observations"
    assert comparison["status"] == "not_reproduced"
    assert comparison["matched"] is False
    assert comparison["measurement_reproduced"] is False
    assert comparison["reason"] == "observations_differ"
    assert comparison["binding_mismatches"] == []
    assert comparison["reference_receipt_id"] != comparison["replay_receipt_id"]
    assert comparison["reference_observations_sha256"] != comparison["replay_observations_sha256"]
    changes = [row for row in comparison["changed_paths"] if "sample" in row["path"]]
    if require_change_detail or changes:
        assert changes
        assert json.loads(changes[0]["reference"]) == 1 and json.loads(changes[0]["replay"]) == 2
    else:
        assert comparison["changed_paths_omitted"] >= 1


@pytest.mark.parametrize("background", [False, True])
def test_failed_replay_reaches_actual_next_generation_and_original_author(tmp_path, monkeypatch, background):
    tools, backend, _, costs, local, replay, foreign, io = completed_replay(
        tmp_path, background=background)
    tool_id = local["registered_ids"][0]
    assert tools.state()["published"] == []
    assert tools.lifecycle.method(tool_id)["release_record"] is None
    # Host and declared execution success remain true: only reproducibility failed.
    assert all(receipt["status"] == "ok" for receipt in backend.receipts)
    assert all(receipt["output"]["status"] == "success" for receipt in backend.receipts)
    context, _ = actual_generation_context(tmp_path, monkeypatch, tools, island=1)
    card = comparison_card(context["cards"], tool_id)
    assert card["observations"]["status"] == "success"
    assert_mismatch(card["replay_comparison"], tool_id)

    # The comparison is inherited by the tool's author even though another cell
    # performed the independent execution. It is part of the frozen next plan.
    author = snapshot(0, 3)
    author_plan = prepare(tools, author)
    assert author_plan["stage"] == "research"
    assert_mismatch(author_plan["author_replay_feedback"][0], tool_id)
    llm = LLM([{"action": "skip"}])
    skipped = tools.execute(author_plan, author, llm, io=IO())
    tools.commit(skipped)
    feedback = json.JSONDecoder().raw_decode(
        llm.prompts[0].split("REPLAY_FEEDBACK_JSON:\n", 1)[1])[0]
    assert feedback["role"] == "original_author"
    assert_mismatch(feedback["comparisons"][0], tool_id)
    assert len(backend.calls) == 2 and costs == [.03, .03]

    # A fresh controller authenticates saved state; neither restore nor replaying
    # the already receipted assignment may run the executor or charge again.
    saved = tools.state()
    restored, unused_backend, _, new_costs = controller(tmp_path, [],
        background=background, max_requests=1)
    restored.load_state(saved)
    empty_llm = LLM([])
    again = restored.execute(replay["plan"], foreign, empty_llm, io=io)
    restored.commit(again)
    if background:
        restored.collect_background(IO())
    assert restored.state() == saved
    assert unused_backend.calls == [] and empty_llm.prompts == [] and new_costs == []


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("policy,sample,status,reproduced", [
    ("exact_observations", 1, "reproduced", True),
    ("independent_execution", 2, "independently_executed", False),
])
def test_replay_delivery_distinguishes_reproduction_from_independent_execution(
        tmp_path, monkeypatch, background, policy, sample, status, reproduced):
    tools, backend, _, costs, local, _, _, _ = completed_replay(tmp_path,
        background=background, policy=policy, sample=sample)
    tool_id = local["registered_ids"][0]
    context, _ = actual_generation_context(tmp_path, monkeypatch, tools, island=1)
    comparison = comparison_card(context["cards"], tool_id)["replay_comparison"]
    assert comparison["policy"] == policy
    assert comparison["status"] == status and comparison["matched"] is True
    assert comparison["measurement_reproduced"] is reproduced
    assert tools.state()["published"] == [tool_id]
    assert len(backend.calls) == 2 and costs == [.03]
    if policy == "independent_execution":
        assert comparison["reference_observations_sha256"] != comparison["replay_observations_sha256"]
        assert comparison["reason"] == "independent_execution_passed"


@pytest.mark.parametrize("background", [False, True])
def test_minimum_context_and_planner_budgets_preserve_failed_replay_criterion(
        tmp_path, monkeypatch, background):
    tools, backend, _, costs, local, _, _, _ = completed_replay(tmp_path,
        background=background, noisy=True, max_context_chars=4096, max_prompt_chars=8000)
    tool_id = local["registered_ids"][0]
    context, _ = actual_generation_context(tmp_path, monkeypatch, tools, island=1)
    comparison = comparison_card(context["cards"], tool_id)["replay_comparison"]
    assert_mismatch(comparison, tool_id, require_change_detail=False)
    author = snapshot(0, 3)
    author.task_prompt = "Task context " * 3000
    author.kb_lessons = ["Long historical lesson " * 3000]
    author.research_cells_context = {"unrelated": "Old context " * 3000}
    plan = prepare(tools, author)
    llm = LLM([{"action": "skip"}])
    tools.execute(plan, author, llm, io=IO())
    assert len(llm.prompts[0]) <= 8000
    feedback = json.JSONDecoder().raw_decode(
        llm.prompts[0].split("REPLAY_FEEDBACK_JSON:\n", 1)[1])[0]
    assert feedback["role"] == "original_author"
    assert_mismatch(feedback["comparisons"][0], tool_id, require_change_detail=False)
    assert len(backend.calls) == 2 and costs == [.03, .03]


def test_context_budget_that_cannot_hold_bound_replay_metadata_is_rejected():
    with pytest.raises(ValueError, match="max_context_chars"):
        validate_config({"enabled": True, "image_id": IMAGE, "max_context_chars": 2048})
