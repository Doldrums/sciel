"""The planner view documents the real frozen tool input, not another schema.

All measurement records are produced by ObservedChanges. Tool/controller paths
are real; deterministic local receipts and responses avoid executing any source.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from sciel.engine.observed_changes import ObservedChanges
from sciel.engine.research_targets import failed_provenance, freeze_targets, tool_inputs
from sciel.engine.types import Program, Snapshot
from sciel.methods.models import content_hash
from test_research_tool_controller import IO, LLM, prepare
from test_research_tool_result_flow import action, controller
from test_research_tool_result_contract import result


def measured(score):
    return dict(ok=True, score=score, cases={"quality": score}, artifacts={},
                reached="full", confirmed=False)


def target_fixture(kind="positive", *, huge=False):
    tail = ("# quoted \\\"source\\\" café🙂\n" * 3000) if huge else ""
    score = 0.0 if kind == "zero" else .2
    parent = Program("p_original", "def solve(): return 1\n" + tail,
        score if kind == "alias" else 23., {"quality": score}, 0, "seed")
    best = Program("p_best", "def solve(): return 2\n" + tail,
        999., {"quality": .8}, 1, "gen")
    history = ObservedChanges()
    if kind != "unavailable":
        history.observe(parent, measured(score))
    history.observe(best, measured(.8))
    if kind == "alias":
        alias = Program("p_alias", parent.code, parent.score, deepcopy(parent.cases),
            0, "migrant", parent_id=parent.id)
        history.bind_alias(alias, parent)
        parent = alias
    programs = {parent.id: parent, best.id: best}
    engine = SimpleNamespace(log=SimpleNamespace(run_id="input-view-fixture"),
        task=SimpleNamespace(task_hash="generic-task"), iteration_no=7, B_evals=9,
        db=SimpleNamespace(pop_best_id=None if kind == "absent_best" else best.id,
                           get=programs.__getitem__), observed_changes=history)
    source, failure = parent.code, None
    if kind == "failed":
        source = 'def solve(): raise RuntimeError("failed source")\n' + tail
        failure = failed_provenance(SimpleNamespace(child_code=source,
            meta={"evaluation_id": "failed-eval-9"}, reached="smoke", confirmed=False,
            artifacts={"error": "failed source"}), iteration=7, B_evals=9)
    current = Snapshot(island=0, op="error_repair" if kind == "failed" else "diff",
        model="fixture", iteration_no=7, parent_id=parent.id, parent_code=source,
        parent_score=0.0 if kind == "failed" else parent.score,
        task_prompt="A generic quality task.")
    current.research_targets = freeze_targets(engine, parent, source,
        failed=kind == "failed", failure=failure)
    return current


def pointer_value(payload, pointer):
    """A test-owned JSON Pointer resolver; never eval generated accessors."""
    assert pointer.startswith("/")
    value = payload
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def rendered_view(prompt):
    value, _ = json.JSONDecoder().raw_decode(
        prompt.split("FROZEN_INPUT_VIEW_JSON:\n", 1)[1])
    return value


def assert_view_matches_inputs(view, inputs):
    targets = inputs["research_targets"]
    assert view["schema"] == "sciel-research-targets-view/1"
    assert view["payload_schema"] == targets["schema"] == "sciel-research-targets/2"
    assert view["research_targets_sha256"] == targets["sha256"]
    assert targets["sha256"] == content_hash(
        {key: value for key, value in targets.items() if key != "sha256"})
    for field in ("run_id", "task_hash", "frozen_at", "ancestry_parent"):
        assert view[field] == targets[field]
    paths = view["access_paths"]
    expected_paths = {
        "source": "/source",
        "measurement_status": "/measurement/status",
        "measurement_record": "/measurement/record",
        "child_observation": "/measurement/record/child_observation",
        "score": "/measurement/record/child_observation/score",
        "saved_evaluation": "/saved_evaluation",
        "saved_artifacts": "/saved_evaluation/record/result/artifacts",
        "saved_timing": "/saved_evaluation/record/result/timing",
        "saved_conditions": "/saved_evaluation/record/conditions",
    }
    assert {key: paths["fields"][key] for key in expected_paths} == expected_paths
    references = {row["role"]: row for row in view["references"]}
    assert set(references) == {"selected", "population_best"}
    assert set(view["absence_rules"]) >= {
        "population_best", "measurement", "failed_candidate", "score"}
    for role, row in references.items():
        base = paths[role]
        assert base == "/research_targets/" + role
        actual = pointer_value(inputs, base)
        assert row["available"] is (actual is not None)
        if actual is None:
            assert row == {"role": role, "available": False}
            continue
        assert row["program_id"] == actual["program_id"]
        assert row["source_sha256"] == actual["source_sha256"]
        assert row["selection_score"] == actual["selection_score"]
        saved = pointer_value(inputs, base + paths["fields"]["saved_evaluation"])
        assert row["saved_evaluation_status"] == saved['status']
        assert row["saved_evaluation_sha256"] == (saved['reference']['sha256']
            if saved['reference'] else None)
        assert pointer_value(inputs, base + paths["fields"]["source"]) == actual["source"]
        status = pointer_value(inputs, base + paths["fields"]["measurement_status"])
        record = pointer_value(inputs, base + paths["fields"]["measurement_record"])
        assert row["measurement_status"] == status
        if status == "recorded":
            observation = pointer_value(inputs, base + paths["fields"]["child_observation"])
            score = pointer_value(inputs, base + paths["fields"]["score"])
            assert score == record["child_observation"]["score"]
            assert row["observation_summary"] == {
                key: observation[key] for key in ("status", "score", "reached", "confirmed")}
            assert row["measurement_record_id"] == record["id"]
        else:
            assert status == "unavailable" and record is None
            assert row["observation_summary"] is None
            # Unavailable is explicit, not a numeric zero/default estimate.
            with pytest.raises((TypeError, KeyError)):
                pointer_value(inputs, base + paths["fields"]["score"])


@pytest.mark.parametrize("kind", ["positive", "zero", "unavailable", "failed",
                                  "absent_best", "alias"])
def test_rendered_planner_paths_resolve_against_actual_executor_inputs(tmp_path, kind):
    current = target_fixture(kind)
    before = deepcopy(current.research_targets)
    tools, backend, _, costs = controller(tmp_path, [result({"checked": True})])
    llm = LLM([action()])
    episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    assert len(llm.prompts) == len(backend.calls) == 1 and costs == [.03]
    inputs = backend.calls[0]["inputs"]
    view = rendered_view(llm.prompts[0])
    assert_view_matches_inputs(view, inputs)
    # Execute only the controller's four documented access lines, never tool or
    # candidate source. This catches aliases and falsey-zero guard mistakes.
    accessor = llm.prompts[0].split(
        "Guarded access (role is selected or population_best):\n", 1)[1].split(
        "parent_source/parent_program_id alias selected input", 1)[0]
    assert len(accessor.strip().splitlines()) == 4
    for role in ("selected", "population_best"):
        scope = {"payload": inputs, "role": role}
        exec(compile(accessor, "<documented-input-accessor>", "exec"),
             {"__builtins__": {}}, scope)
        target = inputs["research_targets"][role]
        record = target["measurement"]["record"] if target else None
        expected = record["child_observation"]["score"] if record else None
        assert scope["score"] == expected
    assert inputs == tool_inputs(current, tools.task_identity, {})
    assert current.research_targets == inputs["research_targets"] == before
    assert "references" not in inputs["research_targets"]  # View is not the runtime bundle.
    assert "observation_summary" not in inputs["research_targets"]["selected"]
    selected = inputs["research_targets"]["selected"]
    if kind == "zero":
        assert inputs["parent_score"] == 0.0
        assert selected["selection_score"] == 23.0
    elif kind == "failed":
        assert selected["kind"] == "failed_candidate"
        assert inputs["parent_program_id"] is inputs["parent_score"] is None
        assert inputs["ancestry_parent_id"] == "p_original"
    elif kind == "unavailable":
        assert inputs["parent_score"] is None and selected["selection_score"] == 23.0
    elif kind == "alias":
        measure = selected["measurement"]
        assert selected["program_id"] == "p_alias"
        assert measure["record"]["child_program_id"] == "p_original"
        assert measure["aliases"][0]["program_id"] == "p_alias"
    tools.commit(episode)
    saved = tools.state()
    tools.load_state(saved)
    assert tools.state() == saved


@pytest.mark.parametrize("background_enabled", [False, True])
def test_minimum_budget_keeps_complete_view_in_initial_and_repair_prompts(
        tmp_path, background_enabled):
    current = target_fixture("zero", huge=True)
    current.task_prompt = "Long task detail " * 5000
    current.kb_lessons = ["Retained research note " * 3000]
    current.artifacts = {"old_error": "Old context " * 3000}
    before = deepcopy(current.research_targets)
    error = "Use the documented measurement record path, not measurement['score']."
    tools, backend, _, costs = controller(tmp_path,
        [result(status="tool_error", error=error + " More failure diagnostics." * 3000),
         result({"checked": True})],
        background=background_enabled, max_prompt_chars=8000)
    # Explicit foreground actions exercise repair even with job instructions enabled.
    llm = LLM([action(), action(revision=True)])
    episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    assert len(llm.prompts) == len(backend.calls) == 2 and costs == [.03, .03]
    assert not episode.get("submitted_jobs")
    for prompt, call in zip(llm.prompts, backend.calls):
        assert len(prompt) <= 8000
        assert_view_matches_inputs(rendered_view(prompt), call["inputs"])
        assert call["inputs"]["research_targets"] == before
        assert len(call["inputs"]["research_targets"]["selected"]["source"]) > 50000
        assert "source_excerpt_omitted" in prompt
    assert error in llm.prompts[1]
    assert current.research_targets == before
