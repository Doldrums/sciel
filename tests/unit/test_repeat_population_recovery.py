"""Behavioral checks for repeat population recovery."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import pytest
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from repeat_population_fixtures import (
    ROOT, sha, step_of, code_for,
    settings, Provider, Draws, task,
    numeric, draws, construct, projection,
    normalized_calls, interrupted_bootstrap,
)


def test_terminal_transition_replays_once_after_crash_before_checkpoint(
        task, tmp_path, monkeypatch, numeric, draws):
    path = tmp_path / "crash"
    engine, first_provider = construct(task, path, cfg=settings(budget=2, max_iterations=2))
    original = engine.repeat_population.apply

    def crash_after_projection(packet):
        transition = original(packet)
        if packet["kind"] == "comparison":
            raise KeyboardInterrupt("fixture crash after pure transition")
        return transition

    monkeypatch.setattr(engine.repeat_population, "apply", crash_after_projection)
    with pytest.raises(KeyboardInterrupt):
        engine.run()
    assert len(draws.calls) == 9 and len(first_provider.calls) == 1
    engine.log.close()
    resumed, provider = construct(task, path, cfg=settings(budget=2, max_iterations=2), start=1)
    checkpoint.load(resumed, str(path / "checkpoint.json"))
    resumed.run()
    assert len(draws.calls) == 15 and len(provider.calls) == 1
    assert resumed.B_evals == resumed.research.outcomes == 2
    assert resumed.cost_total == pytest.approx(.002)
    assert [r["value"] for r in projection(resumed)["rewards"]] == [0, 1]
    assert len(resumed.research.knowledge.comparisons) == 2


@pytest.mark.parametrize("field", ["identity", "membership", "reward"])
def test_force_cannot_accept_changed_repeat_identity_or_projection(
        task, tmp_path, numeric, draws, field):
    path = tmp_path / field
    first, _ = construct(task, path)
    first._stop_at = 1
    assert first.run()["reason"] == "paused"
    saved_path = path / "checkpoint.json"
    saved = json.loads(saved_path.read_text())
    if field == "identity":
        saved["identity"].pop("repeat_population")
    elif field == "membership":
        saved["repeat_population"]["projection"]["active"]["0"].append("p3")
    else:
        saved["repeat_population"]["projection"]["rewards"][0]["value"] = 1
    saved_path.write_text(json.dumps(saved))
    resumed, provider = construct(task, path, start=1)
    with pytest.raises(checkpoint.IdentityMismatch):
        checkpoint.load(resumed, str(saved_path), allow_identity_mismatch=True)
    assert not provider.calls and len(draws.calls) == 9
    resumed.log.close()


def test_continuous_resume_preserves_membership_rng_receipts_prompts_and_rewards(
        task, tmp_path, numeric, draws):
    whole, all_calls = construct(task, tmp_path / "whole")
    whole.run()
    expected_draws = copy.deepcopy(draws.calls)
    draws.calls.clear()
    first, first_calls = construct(task, tmp_path / "split")
    first._stop_at = 2
    assert first.run()["reason"] == "paused"
    resumed, later_calls = construct(task, tmp_path / "split", start=2)
    checkpoint.load(resumed, str(tmp_path / "split/checkpoint.json"))
    resumed.run()
    assert draws.calls == expected_draws
    for key in ("db", "rng", "schedule", "reward", "op_bandit", "selection"):
        assert getattr(resumed, key).state() == getattr(whole, key).state()
    assert resumed.B_evals == whole.B_evals == 5
    assert resumed.iteration_no == whole.iteration_no == 5
    assert resumed.cost_total == whole.cost_total == pytest.approx(.005)
    left, right = projection(resumed), projection(whole)
    for key in ("history", "active", "evidence_heads", "visits", "pending", "selection",
                "attempted_sources", "selection_counter", "publication_source_sha256", "stopped_reason"):
        assert left[key] == right[key]
    for l, r in zip(left["transitions"], right["transitions"]):
        l, r = copy.deepcopy(l), copy.deepcopy(r)
        for value in (l, r):
            value.pop("packet_sha256", None)
        assert l == r
    assert [v["value"] for v in left["rewards"]] == [v["value"] for v in right["rewards"]]
    left_groups, right_groups = resumed.research.knowledge.comparisons, whole.research.knowledge.comparisons
    assert resumed.research.knowledge.experiments == whole.research.knowledge.experiments
    for l, r in zip(left_groups, right_groups):
        l, r = copy.deepcopy(l), copy.deepcopy(r)
        for group in (l, r):
            group.pop("id")
            group.pop("namespace")
            group["provenance"].pop("run_id")
            group["provenance"]["decision_anchor"].pop("sha256")
        assert l == r
    # This helper verifies each actual full comparison card before normalizing
    # only run-specific identities; source code, metrics and ordinary facts stay.
    assert normalized_calls(first_calls.calls + later_calls.calls, left_groups,
                            resumed.research.knowledge.experiments) == \
        normalized_calls(all_calls.calls, right_groups, whole.research.knowledge.experiments)
    from test_repeat_engine_integration import _observed_feedback
    shown = _observed_feedback(first_calls.calls + later_calls.calls,
        resumed.observed_changes.state()['records'], left_groups, resumed.db._all)
    assert shown == _observed_feedback(all_calls.calls, whole.observed_changes.state()['records'],
                                       right_groups, whole.db._all)
    assert any(record.get('pairs') for rows in shown for record in rows)
    # Normalization must authenticate actual measured directions before touching
    # independent-run group identifiers. A matching shape is not sufficient.
    from sciel.hypothesis.rendering import decode_evidence
    forged = copy.deepcopy(all_calls.calls)
    changed = False
    for call in forged:
        prefix, suffix = call['prompt'].split('RESEARCH_CONTEXT_JSON:\n', 1)
        raw, end = json.JSONDecoder().raw_decode(suffix)
        context = decode_evidence(raw)
        if not context.get('comparison_facts', {}).get('paired'):
            continue
        paired = next(iter(context['comparison_facts']['paired'].values()))
        pair = paired['pairs'][0]
        assert pair['raw_physical_sum_direction'] in ('positive', 'negative')
        pair['raw_physical_sum_direction'] = ('negative' if pair['raw_physical_sum_direction'] == 'positive'
                                             else 'positive')
        call['prompt'] = prefix + 'RESEARCH_CONTEXT_JSON:\n' + json.dumps(context) + suffix[end:]
        changed = True
        break
    assert changed
    with pytest.raises(ValueError, match='comparison facts differ'):
        normalized_calls(forged, right_groups, whole.research.knowledge.experiments)
    # The separate added-feedback check must also reject altered measurements,
    # even though normalization intentionally removes that block from prompts.
    from sciel.engine.observed_changes import _PREFIX, _SUFFIX
    forged_feedback = copy.deepcopy(all_calls.calls)
    for call in forged_feedback:
        if _PREFIX not in call['prompt']:
            continue
        prefix, suffix = call['prompt'].split(_PREFIX, 1)
        records, end = json.JSONDecoder().raw_decode(suffix)
        row = next((r for r in records if r.get('pairs')), None)
        if row is None:
            continue
        row['pairs'][0]['child_raw_sum'] += .01
        assert suffix[end:].startswith(_SUFFIX)
        call['prompt'] = prefix + _PREFIX + json.dumps(records) + suffix[end:]
        break
    else:
        pytest.fail('fixture must actually present paired observed feedback')
    with pytest.raises(AssertionError):
        _observed_feedback(forged_feedback, whole.observed_changes.state()['records'],
                           right_groups, whole.db._all)


def test_unknown_draw_is_retained_and_never_dispatched_again_on_resume(
        task, tmp_path, numeric, draws):
    path = tmp_path / "unknown"
    first, provider = construct(task, path, cfg=settings(budget=2, max_iterations=2))

    def interrupt(call):
        if len(draws.calls) == 4:
            raise KeyboardInterrupt("fixture process vanished after durable draw intent")

    draws.before_draw = interrupt
    with pytest.raises(KeyboardInterrupt):
        first.run()
    assert len(draws.calls) == 4 and len(provider.calls) == 1
    first.log.close()
    draws.before_draw = None
    resumed, later = construct(task, path, cfg=settings(budget=2, max_iterations=2), start=1)
    checkpoint.load(resumed, str(path / "checkpoint.json"))
    result = resumed.run()
    assert result["reason"] == "repeat_stopped"
    assert len(draws.calls) == 4 and not later.calls
    assert projection(resumed)["rewards"] == []
    group = resumed.research.knowledge.comparisons[0]
    slots = [v for pair in group["pairs"] for v in (pair["comparator"], pair["challenger"])]
    assert sum(v is None for v in slots) == 5
    assert [v["status"] for v in slots if v is not None] == ["unknown"]
    assert resumed.B_evals == resumed.research.outcomes == 1


@pytest.mark.parametrize("change", [{"op_fixed": "diff"}, {"base_temp": .4},
    {"max_iterations": 6}, {"backstop_k": 11}])
def test_force_rejects_changed_generation_or_attempt_identity(
        task, tmp_path, numeric, draws, change):
    first, _ = construct(task, tmp_path / "identity-profile")
    first._stop_at = 1
    first.run()
    resumed, provider = construct(task, tmp_path / "identity-profile", cfg=settings(**change), start=1)
    with pytest.raises(checkpoint.IdentityMismatch):
        checkpoint.load(resumed, str(tmp_path / "identity-profile/checkpoint.json"), allow_identity_mismatch=True)
    assert not provider.calls and len(draws.calls) == 9
    resumed.log.close()


def test_extra_ordinary_history_without_population_event_is_rejected_atomically(
        task, tmp_path, numeric, draws):
    path = tmp_path / "extra-history"
    first, _ = construct(task, path)
    first._stop_at = 1
    first.run()
    saved_path = path / "checkpoint.json"
    saved = json.loads(saved_path.read_text())
    extra = copy.deepcopy(saved["db"]["programs"][-1])
    extra.update(id="p4", code=code_for(2), score=2., cases={"sum_radii": 5.27,
        "validity": 1., "combined_score": 2.})
    saved["db"]["programs"].append(extra)
    saved["db"]["counter"] = 4
    saved["db"]["pop_best_id"] = "p4"
    saved["provenance"]["counters"]["db_counter"] = 4
    saved_path.write_text(json.dumps(saved))
    resumed, provider = construct(task, path, start=1)
    with pytest.raises(checkpoint.IdentityMismatch):
        checkpoint.load(resumed, str(saved_path), allow_identity_mismatch=True)
    assert resumed.B_evals == 0 and resumed.db._all == {}
    assert not provider.calls and len(draws.calls) == 9
    resumed.log.close()


def test_prebootstrap_resume_dispatches_shared_seed_block_once_without_re_evaluation(
        task, tmp_path, monkeypatch, numeric, draws):
    path = tmp_path / "before-bootstrap"
    saved_path, _ = interrupted_bootstrap(task, path, monkeypatch, numeric, draws)
    resumed, provider = construct(task, path, cfg=settings(budget=1, max_iterations=1))
    checkpoint.load(resumed, str(saved_path))
    resumed._stop_at = 0
    assert resumed.run()["reason"] == "paused"
    assert len(draws.calls) == 3 and len(numeric) == 1 and not provider.calls
    assert projection(resumed)["active"] == {"0": ["p1"], "1": ["p2"]}
    assert resumed.B_evals == resumed.research.attempts == 0
    assert resumed.seed_score == numeric[0]["score"]
    assert all(p.cases == numeric[0]["cases"] for p in resumed.db._all.values())
    # The newly materialized bootstrap checkpoint must not execute it again.
    again, again_provider = construct(task, path, cfg=settings(budget=1, max_iterations=1))
    checkpoint.load(again, str(saved_path))
    again._stop_at = 0
    assert again.run()["reason"] == "paused"
    assert len(draws.calls) == 3 and len(numeric) == 1 and not again_provider.calls


@pytest.mark.parametrize("tamper", ["extra_generated_history", "shared_seed_observation",
                                  "bool_validity", "bool_generation"])
def test_prebootstrap_restore_rejects_unbound_history_or_consistent_seed_score_substitution(
        task, tmp_path, monkeypatch, numeric, draws, tamper):
    path = tmp_path / tamper
    saved_path, saved = interrupted_bootstrap(task, path, monkeypatch, numeric, draws)
    if tamper == "extra_generated_history":
        extra = copy.deepcopy(saved["db"]["programs"][0])
        extra.update(id="p3", code=code_for(1), parent_id="p1", origin="gen", gen=1,
                     score=2., cases={"sum_radii": 5.27, "validity": 1., "combined_score": 2.})
        saved["db"]["programs"].append(extra)
        saved["db"]["counter"] = 3
        saved["db"]["pop_best_id"] = "p3"
        saved["provenance"]["counters"]["db_counter"] = 3
    elif tamper == "shared_seed_observation":
        replacement_score = .2
        replacement_cases = {"sum_radii": replacement_score * 2.635,
                             "validity": 1., "combined_score": replacement_score}
        for program in saved["db"]["programs"]:
            program["score"] = replacement_score
            program["cases"] = copy.deepcopy(replacement_cases)
        saved["seed_score"] = replacement_score
        saved["provenance"]["counters"]["seed_score"] = replacement_score
    elif tamper == "bool_validity":
        for program in saved["db"]["programs"]:
            program["cases"]["validity"] = True
    else:
        for program in saved["db"]["programs"]:
            program["gen"] = False
    saved_path.write_text(json.dumps(saved))
    resumed, provider = construct(task, path, cfg=settings(budget=1, max_iterations=1))
    with pytest.raises(checkpoint.IdentityMismatch):
        checkpoint.load(resumed, str(saved_path), allow_identity_mismatch=True)
    assert resumed.db._all == {} and resumed.B_evals == 0
    assert not provider.calls and not draws.calls and len(numeric) == 1
    resumed.log.close()


def test_crash_inside_initial_ordinary_evaluation_cannot_replay_seed_dispatch(
        task, tmp_path, monkeypatch, draws):
    path = tmp_path / "seed-intent"
    attempted = []

    def interrupted_evaluation(task, source_path, **kwargs):
        attempted.append(Path(source_path).read_text())
        raise KeyboardInterrupt("fixture disappeared during first ordinary evaluation")

    monkeypatch.setattr("sciel.engine.loop.cascade_evaluate", interrupted_evaluation)
    engine, provider = construct(task, path, cfg=settings(budget=1, max_iterations=1))
    with pytest.raises(KeyboardInterrupt):
        engine.run()
    assert len(attempted) == 1 and not provider.calls and not draws.calls
    saved = json.loads((path / "checkpoint.json").read_text())
    assert saved["db"]["programs"] == [] and saved["B_evals"] == 0
    engine.log.close()
    resumed, later = construct(task, path, cfg=settings(budget=1, max_iterations=1))
    with pytest.raises(checkpoint.IdentityMismatch):
        checkpoint.load(resumed, str(path / "checkpoint.json"), allow_identity_mismatch=True)
    assert len(attempted) == 1 and not later.calls and not draws.calls
    assert resumed.db._all == {}
    resumed.log.close()
