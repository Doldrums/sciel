"""Stage-0 data pipeline: shared features + RTG hygiene (no negatives, killed-run skip)."""

import importlib.util
import json
import os

import pytest

from sciel.components.policy.mode_features import MODES, N_FEATURES, mode_features

_LOADER = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "dt_dataloader.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("dt_dataloader", _LOADER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mode_features_shape_and_phase_onehot():
    f = mode_features(phase=0, progress=0.1, v_t=0.5, island_best=0.5)
    assert len(f) == N_FEATURES
    assert f[:3] == [1.0, 0.0, 0.0]        # uniform
    assert mode_features(1, 0.6, 0.5, 0.5)[:3] == [0.0, 1.0, 0.0]   # sh
    assert mode_features(-1, 0.9, 0.5, 0.5)[:3] == [0.0, 0.0, 1.0]  # final
    assert mode_features(0, 0.1, 0.5, 0.0)[-1] == 0.0   # no gradient
    assert mode_features(0, 0.1, 0.5, 0.5)[-1] == 1.0   # gradient


def _journal(path, with_end=True):
    ev = [{"event": "run_start", "schema_version": 1, "budget": 4,
           "config": {"budget": 4, "milestone_fracs": [0.5]}}]
    steps = [(1, 0.3, "base", 0.3), (2, 0.6, "exploit", 0.3), (3, 1.0, "explore", 0.4)]
    for b, pop, mode, r in steps:
        ev.append({"event": "iteration", "iteration_no": b, "B_evals": b, "island": 0,
                   "child_score": pop, "pop_best": pop, "island_best": pop,
                   "d_pop": r, "d_isl": r, "r_raw": r, "mode": mode})
    if with_end:
        ev.append({"event": "run_end", "pop_best": 1.0})
    with open(path, "w") as fh:
        fh.writelines(json.dumps(e) + "\n" for e in ev)


def test_rtg_nonnegative_and_features_parity(tmp_path):
    mod = _load_module()
    p = tmp_path / "sciel_events.jsonl"
    _journal(str(p))
    samples, final_v = mod.build_samples(str(p))
    assert final_v == 1.0 and len(samples) == 3
    assert all(s["rtg"] >= 0.0 for s in samples)           # Lemma 1 hygiene
    assert all(len(s["features"]) == N_FEATURES for s in samples)
    # pre-step V_t: step1 sees seed(0.0) -> rtg 1.0; step2 sees 0.3 -> 0.7; step3 -> 0.4
    assert [round(s["rtg"], 3) for s in samples] == [1.0, 0.7, 0.4]
    assert [s["mode"] for s in samples] == ["base", "exploit", "explore"]


def test_killed_run_skipped(tmp_path):
    mod = _load_module()
    p = tmp_path / "sciel_events.jsonl"
    _journal(str(p), with_end=False)
    with pytest.raises(mod.DirtyRun):
        mod.build_samples(str(p))
    # load_corpus reports it rather than crashing
    samples, report = mod.load_corpus([str(tmp_path)])
    assert samples == [] and len(report["skipped"]) == 1


def test_corpus_mode_coverage(tmp_path):
    mod = _load_module()
    p = tmp_path / "sciel_events.jsonl"
    _journal(str(p))
    samples, _ = mod.load_corpus([str(tmp_path)])
    assert {s["mode"] for s in samples} == set(MODES)


def _scheduled_journal(path, *, budget=None, horizon=4, policy="sh", recorded=False):
    cfg = {"budget": budget, "milestone_fracs": [.5], "schedule_policy": policy}
    if horizon is not None:
        cfg["schedule_horizon"] = horizon
    rows = [{"event": "run_start", "schema_version": 1, "budget": budget, "config": cfg}]
    for n in range(1, 7):
        rows.append({"event": "iteration", "iteration_no": n, "B_evals": n, "island": 0,
                     "child_score": n / 10, "pop_best_pre": (n - 1) / 10,
                     "island_best_pre": (n - 1) / 10, "mode": "base",
                     **({"phase_pre": 1, "progress_pre": .125} if recorded else {})})
    rows.append({"event": "run_end", "iteration_no": 6, "B_evals": 6,
                 "reason": "backstop", "pop_best": .6})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("policy", ["sh", "round_robin"])
def test_unlimited_corpus_uses_recorded_horizon_and_capped_progress(tmp_path, policy):
    mod = _load_module()
    _scheduled_journal(tmp_path / "sciel_events.jsonl", policy=policy)
    samples, report = mod.load_corpus([str(tmp_path)])
    assert report["skipped"] == [] and len(samples) == 6
    for n, sample in enumerate(samples):
        phase = 0 if n < 2 or policy == "round_robin" else -1
        assert sample["features"] == mode_features(phase, min(1., n / 4), n / 10, n / 10)


def test_finite_budget_still_controls_features_even_with_schedule_horizon(tmp_path):
    mod = _load_module()
    _scheduled_journal(tmp_path / "sciel_events.jsonl", budget=4, horizon=100)
    samples, _ = mod.load_corpus([str(tmp_path)])
    assert samples[-1]["features"] == mode_features(-1, 1.25, .5, .5)


def test_recorded_pre_step_features_override_reconstruction(tmp_path):
    mod = _load_module()
    _scheduled_journal(tmp_path / "sciel_events.jsonl", recorded=True)
    samples, _ = mod.load_corpus([str(tmp_path)])
    assert samples[-1]["features"] == mode_features(1, .125, .5, .5)


@pytest.mark.parametrize("horizon", [None, 0, True])
def test_unlimited_missing_or_invalid_horizon_is_reported_skip(tmp_path, horizon):
    mod = _load_module()
    _scheduled_journal(tmp_path / "sciel_events.jsonl", horizon=horizon)
    samples, report = mod.load_corpus([str(tmp_path)])
    assert samples == [] and len(report["skipped"]) == 1
    assert "recorded scheduling horizon" in report["skipped"][0][1]
