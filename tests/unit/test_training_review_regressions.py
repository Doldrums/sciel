"""Independent holdout, chronological context, and evaluated-source regressions."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def start():
    return {"event": "run_start", "schema_version": 1, "budget": 100,
            "config": {"budget": 100, "milestone_fracs": [0.5, 0.75]}}


def step(t, island, pop, isl, **extra):
    return {"event": "iteration", "iteration_no": t, "B_evals": t,
            "island": island, "child_id": f"p{t}", "child_score": isl,
            "pop_best": pop, "island_best": isl, "mode": "base", **extra}


def write_journal(tmp_path, rows):
    path = tmp_path / "sciel_events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def end(value, reason="budget"):
    return {"event": "run_end", "pop_best": value, "reason": reason}


def test_context_recovers_positive_seed_and_interleaved_global_record(tmp_path):
    loader = module("dt_dataloader")
    path = write_journal(tmp_path, [start(),
        step(1, 0, .4, .4, d_pop=.1, d_isl=.1),
        step(2, 1, .9, .9, d_pop=.5, d_isl=.7),
        step(3, 0, .9, .4, d_pop=0., d_isl=0.), end(1.)])
    samples, _ = loader.build_samples(path)
    assert [s["island"] for s in samples] == [0, 1, 0]
    assert [s["v_t"] for s in samples] == pytest.approx([.3, .4, .9])
    assert [s["rtg"] for s in samples] == pytest.approx([.7, .6, .1])
    assert samples[1]["features"][5] == pytest.approx(.2)


def test_explicit_context_wins_and_unknown_baselines_are_not_invented(tmp_path):
    loader = module("dt_dataloader")
    path = write_journal(tmp_path, [start(),
        step(1, 0, .4, .4),  # no evidence of what the seed scored
        step(2, 0, .8, .7, pop_best_pre=.75, island_best_pre=.6,
             phase_pre=-1, progress_pre=.72, effective_mode="explore"), end(.8)])
    samples, _ = loader.build_samples(path)
    assert len(samples) == 1
    assert samples[0]["features"] == [0., 0., 1., .72, .75, .6, 1.]
    assert samples[0]["mode"] == "explore"


def test_legacy_global_state_advances_across_untrainable_modes(tmp_path):
    loader = module("dt_dataloader")
    path = write_journal(tmp_path, [start(),
        step(1, 0, .4, .4, d_pop=.1, d_isl=.1),
        step(2, 1, .9, .9, mode="offline"),
        step(3, 0, .9, .4), end(1.)])
    samples, _ = loader.build_samples(path)
    assert len(samples) == 2
    assert samples[1]["v_t"] == .9
    assert samples[1]["features"][5] == .4


def test_resume_uses_last_completed_end_and_rejects_open_or_paused_tail(tmp_path):
    loader = module("dt_dataloader")
    rows = [start(), step(1, 0, .4, .4, d_pop=.1, d_isl=.1), end(.4, "paused")]
    path = write_journal(tmp_path, rows)
    with pytest.raises(loader.DirtyRun):
        loader.build_samples(path)
    rows += [start(), step(2, 0, .9, .9, d_pop=.5, d_isl=.5), end(.9)]
    write_journal(tmp_path, rows)
    samples, final = loader.build_samples(path)
    assert final == .9
    assert [s["rtg"] for s in samples] == pytest.approx([.6, .5])
    write_journal(tmp_path, rows + [start()])
    with pytest.raises(loader.DirtyRun, match="resumed segment"):
        loader.build_samples(path)


def test_journal_and_lineage_use_evaluated_source_or_skip(tmp_path):
    pytest.importorskip("torch")
    trainer, lineage = module("train_code_embedder"), module("lineage_labels")
    from sciel.components.policy.code_embed import featurize

    logged, grafted = "def run():\n    return 2\n", "def run():\n    return 3\n"
    raw = "```python\ndef renamed():\n    return 99\n```"
    path = write_journal(tmp_path, [start(),
        step(1, 0, .4, .4, child_code=logged, response=raw, cases={}),
        step(2, 0, .5, .5, response=raw, cases={}, meta={"grafted": ["run"]}),
        step(3, 0, .6, .6, response=raw, cases={}), end(.6)])
    (tmp_path / "checkpoint.json").write_text(json.dumps({"db": {"programs": [
        {"id": "p1", "code": "stale checkpoint source"},
        {"id": "p2", "code": grafted}]}}))
    X, _, _, _ = trainer.load([str(tmp_path)])
    np.testing.assert_array_equal(X, [featurize(logged, {}), featurize(grafted, {})])
    records = lineage.labels_for(path, with_code=True)
    assert {r["id"]: r["code"] for r in records} == {"p1": logged, "p2": grafted}
    # A paused corpus cannot supply complete lineage-credit targets.
    write_journal(tmp_path, [start(), step(1, 0, .4, .4, child_code=logged),
                             end(.4, "paused")])
    assert len(trainer.load([str(tmp_path)])[0]) == 0
    assert lineage.labels_for(path, with_code=True) == []


def test_program_split_keeps_run_groups_and_excludes_copied_programs():
    pytest.importorskip("torch")
    trainer = module("train_code_embedder")
    groups = ["a", "a", "b", "b", "c", "c"]
    codes = ["shared", "a-only", "shared", "b-only", "shared", "c-only"]
    train, test = trainer.split_programs(groups, codes, holdout_frac=1/3)
    assert {groups[i] for i in train}.isdisjoint({groups[i] for i in test})
    assert {codes[i] for i in train}.isdisjoint({codes[i] for i in test})
    assert len(test) == 2
    train, test = trainer.split_programs(["one-run"] * 4, ["a", "a", "b", "c"])
    assert {(["a", "a", "b", "c"])[i] for i in train}.isdisjoint(
        {(["a", "a", "b", "c"])[i] for i in test})


def test_holdout_features_and_labels_cannot_change_training_or_threshold():
    torch = pytest.importorskip("torch")
    trainer = module("train_code_embedder")
    rng = np.random.default_rng(9)
    X = rng.normal(size=(8, 6)).astype(np.float32)
    Y = np.array([.1, .1, .9, .9, .2, .2, .8, .8], dtype=np.float32)
    # Cross-boundary pairs must be discarded from both objectives.
    pos = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 4)]
    neg = [(0, 2), (1, 3), (4, 6), (5, 7), (1, 7)]
    train, test = np.arange(4), np.arange(4, 8)
    first, metrics = trainer.fit_and_evaluate(X, Y, pos, neg, train, test, epochs=2)
    changed_X, changed_Y = X.copy(), Y.copy()
    changed_X[test] *= 100
    changed_Y[test] += 1000
    second, changed = trainer.fit_and_evaluate(changed_X, changed_Y, pos, neg,
                                               train, test, epochs=2)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]), key
    assert metrics["distance_threshold"] == changed["distance_threshold"]
    assert metrics["n_train"] == metrics["n_holdout"] == 4
