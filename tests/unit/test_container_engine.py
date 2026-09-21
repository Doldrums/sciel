"""Container, worker, and engine contracts exercised without paid model calls."""

import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.components.code_novelty import CodeNoveltyGate
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.container_db import ContainerDatabase
from sciel.engine.db import Database, normalized_hash
from sciel.engine.llm.client import LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.engine.loop import Engine
from sciel.engine.types import NoChild, Result, Snapshot
from sciel.engine.worker import run_worker
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task


ROOT = Path(__file__).resolve().parents[2]


def _container(kind, **overrides):
    return apply_defaults({"container": {"kind": kind, **overrides}})["container"]


def _vector(code, cases):
    return cases["vector"]


def _db(kind, cap=3, islands=1, **overrides):
    return ContainerDatabase(
        islands, pop_cap=cap, archive_size=20, grid_bins=10,
        descriptor_fn=lambda code, cases: cases["axis"], vector_fn=_vector,
        container=_container(kind, **overrides),
    )


def _add(db, index, *, score=None, island=0, parent_id=None):
    return db.add(
        f"def run_search(): return ({index}, 0)\n",
        index / 100 if score is None else score,
        {"axis": (index % 10 + 0.5) / 10,
         "vector": [math.cos(index * 0.8), math.sin(index * 0.8)]},
        island, "gen", parent_id=parent_id,
    )


@pytest.mark.parametrize("kind", ["grid", "pool", "pool_novelty", "cvt", "vq", "annealed"])
def test_active_capacity_is_strict_while_best_and_history_survive(kind):
    db = _db(kind, cap=3)
    best = _add(db, 0, score=0.95)
    created = [best]
    for index in range(1, 16):
        db.progress = index / 20
        created.append(_add(db, index, score=index / 30, parent_id=created[-1].id))
        assert len(db.islands[0].programs) <= 3
        assert best.id in db.islands[0].programs
        assert set(db.islands[0].grid.values()) <= set(db.islands[0].programs)
    assert db.pop_best == db.island_best(0) == 0.95
    assert db.pop_best_id == db.islands[0].best_id == best.id
    assert len(db._all) == len(created)
    assert db.get(created[-1].id).gen == len(created) - 1
    assert db.eviction_count == len(created) - 3
    assert db.archive == []
    active_ids = set(db.islands[0].programs)
    assert all(p.id in active_ids for p in db.inspirations(0, best.id, k=50))
    assert db.code_hashes(0) == {normalized_hash(db.get(pid).code) for pid in active_ids}


def test_grid_cap_still_holds_when_every_program_is_a_cell_elite():
    db = _db("grid", cap=2)
    for index, score in enumerate((0.4, 0.9, 0.8, 0.6)):
        _add(db, index, score=score)
    assert len(db.islands[0].programs) == len(db.islands[0].grid) == 2
    assert sorted(p.score for p in db.islands[0].programs.values()) == [0.8, 0.9]
    assert db.eviction_count == db.elite_eviction_count == 2
    assert all(event["was_elite"] for event in db.evictions)


@pytest.mark.parametrize("kind", ["pool", "pool_novelty"])
def test_pool_does_not_retain_grid_or_archive_or_call_descriptor(kind):
    db = _db(kind, cap=2)
    db.descriptor_fn = lambda *args: pytest.fail("pool must not compute grid cells")
    db.vector_fn = lambda *args: pytest.fail("pool must not compute geometry")
    for index in range(8):
        _add(db, index)
    assert db.islands[0].grid == {}
    assert db.archive == []
    assert db.vectors == db.geometries == {}
    assert len(db.islands[0].programs) == 2


def test_migration_applies_destination_cap_and_retains_provenance():
    db = _db("pool", cap=2, islands=2)
    source = [_add(db, index, score=0.6 + index / 10, island=0) for index in range(2)]
    old_target = [_add(db, index + 20, score=0.1 + index / 10, island=1) for index in range(2)]
    assert db.migrate_unique_elites(0, [1]) == 2
    assert len(db.islands[1].programs) == 2
    assert {p.code for p in db.islands[1].programs.values()} == {p.code for p in source}
    assert all(db.get(p.id) == p for p in old_target)
    assert db.islands[1].grid == {} and db.archive == []


@pytest.mark.parametrize("kind", ["cvt", "vq", "annealed"])
def test_geometric_database_json_resume_preserves_eviction_and_future_selection(kind):
    original = _db(kind, cap=5, k=3, max_centres=5)
    for index in range(12):
        original.progress = index / 25
        _add(original, index)
    assert original.eviction_count > 0
    saved = json.loads(json.dumps(original.state(), allow_nan=False))
    restored = _db(kind, cap=5, k=3, max_centres=5)
    restored.load_state(saved)
    assert restored.state() == saved
    rng_original, rng_restored = random.Random(317), random.Random(317)
    for index in range(12, 20):
        a = original.select_parent(0, rng_original)
        b = restored.select_parent(0, rng_restored)
        assert a == b
        assert 0 < a.propensity <= 1
        assert a.debug["rho_sum"] == pytest.approx(1)
        original.progress = restored.progress = index / 25
        _add(original, index, parent_id=a.parent_id)
        _add(restored, index, parent_id=b.parent_id)
        assert original.state() == restored.state()
    assert set(restored.vectors) == set(restored.islands[0].programs)
    assert len(restored.islands[0].programs) == 5


@pytest.mark.parametrize("kind", ["cvt", "vq", "annealed"])
def test_resume_does_not_recompute_frozen_vectors(kind):
    original = _db(kind)
    for index in range(6):
        _add(original, index)
    restored = _db(kind)
    restored.vector_fn = lambda *args: pytest.fail("resume must use saved frozen vectors")
    restored.load_state(json.loads(json.dumps(original.state(), allow_nan=False)))
    assert restored.select_parent(0, random.Random(1)).parent_id in restored.islands[0].programs


@pytest.mark.parametrize("change", [{"kind": "grid"}, {"novelty_threshold": 0.9}])
def test_database_rejects_changed_container_before_mutating_population(change):
    source = _db("pool_novelty")
    _add(source, 1)
    cfg = {**source.container, **change}
    target = ContainerDatabase(1, pop_cap=3, container=cfg)
    with pytest.raises(ValueError, match="container contract"):
        target.load_state(source.state())
    assert target._all == {} and target.pop_best_id is None


class _LLM:
    models = ["test-local"]
    weights = [1.0]

    def __init__(self, codes=None):
        self.codes = iter(codes) if codes is not None else None
        self.calls = 0

    def choose_model(self, rng):
        return self.models[0]

    def generate(self, *args, **kwargs):
        self.calls += 1
        code = next(self.codes) if self.codes is not None else (
            f"def run_search(): return ({self.calls}, 0)\n"
        )
        return LLMResponse(text=f"```python\n{code}```", model=self.models[0],
                           cost=0.125, prompt_tokens=10, completion_tokens=20)


def _snapshot(parent):
    return Snapshot(0, "full_rewrite", "test-local", 0, "p1", parent, 0.1,
                    code_hashes={normalized_hash(parent)}, required_entrypoints=["run_search"])


def test_worker_novelty_rejection_never_evaluates_and_keeps_all_costs():
    parent = "def run_search(): return (1, 0)\n"
    variant = "# changed formatting\ndef run_search( ): return ( 1, 0 )\n"
    llm = _LLM([variant] * 3)
    gate = CodeNoveltyGate()
    result = run_worker(
        "full_rewrite", _snapshot(parent), llm,
        lambda code: pytest.fail("rejected novelty must not reach evaluation"), dup_cap=3,
        novelty_check=lambda code: gate.inspect(code, [parent]),
    )
    assert isinstance(result, NoChild)
    assert result.reason == "gate_rejected"
    assert result.cost == pytest.approx(0.375)
    assert result.meta["duplicates"] == 0
    assert result.meta["gate_rejections"] == len(result.meta["novelty_checks"]) == 3
    assert all(not row["accept"] for row in result.meta["novelty_checks"])


def test_worker_can_recover_after_novelty_rejection_and_accounts_both_calls():
    parent = "def run_search(): return (1, 0)\n"
    novel = "import math\ndef run_search():\n    return (math.sin(3), sum(range(7)))\n"
    llm = _LLM(["# formatting\n" + parent, novel])
    evaluated = []

    def evaluate(code):
        evaluated.append(code)
        return CascadeOutcome(EvalResult(True, 0.7), 0.7, "full")

    result = run_worker("full_rewrite", _snapshot(parent), llm, evaluate, dup_cap=3,
                        novelty_check=lambda code: CodeNoveltyGate().inspect(code, [parent]))
    assert isinstance(result, Result) and result.ok
    assert len(evaluated) == 1 and "math.sin" in evaluated[0]
    assert result.cost == pytest.approx(0.25)
    assert [row["accept"] for row in result.meta["novelty_checks"]] == [False, True]


def _engine(path, **overrides):
    cfg = dict(task="test/toy_quadratic", num_islands=1, budget=5, pop_cap=2,
               embed_grid=False, portfolio_seed=False, idea_island=False,
               dynamic_islands=False, policy={"enabled": False}, milestone_fracs=[],
               op_fixed="full_rewrite", llm={"mock": True}, container={"kind": "pool"})
    cfg.update(overrides)
    task = load_task(str(ROOT / "tasks/test/toy_quadratic"))
    return Engine(cfg, task, str(path), _LLM(), python=sys.executable)


def _seed(engine):
    engine.seed_score = 0.1
    engine.db.add("def run_search(): return (100, 0)\n", 0.1, {}, 0, "seed")


def test_engine_gate_records_rejections_and_cost_without_saturation(tmp_path):
    e = _engine(tmp_path, container={"kind": "pool_novelty"}, dup_cap=2)
    try:
        _seed(e)
        parent = e.db.get(e.db.pop_best_id).code
        e.llm = _LLM(["# formatting\n" + parent] * 2)
        e._evaluate = lambda code: pytest.fail("novelty rejection must bypass evaluator")
        e._step(0)
        assert e.B_evals == 0 and e.n_nochild == 1
        assert e.cost_total == pytest.approx(0.25)
        assert e.schedule.saturation_counter[0] == 0
        assert e.schedule.saturated == set()
    finally:
        e.log.close()
    events = [json.loads(line) for line in (tmp_path / "sciel_events.jsonl").read_text().splitlines()]
    checks = [row for row in events if row.get("policy") == "code_novelty"]
    assert len(checks) == 2 and all(row["action"] == "reject" for row in checks)
    assert all(row["descriptor_version"] == CodeNoveltyGate.descriptor_version for row in checks)


@pytest.mark.parametrize("adaptive", [False, True])
def test_engine_preset_update_tracks_attempts_and_static_control_stays_frozen(tmp_path, adaptive):
    e = _engine(tmp_path, preset_selection=True, preset_adaptive=adaptive)
    try:
        _seed(e)
        before = e.preset_weights.state()
        probabilities_before = e.preset_weights.probabilities()
        e._evaluate = lambda code: CascadeOutcome(EvalResult(True, 0.2 + e.llm.calls / 20),
                                                  0.2 + e.llm.calls / 20, "full")
        for _ in range(3):
            e._step(0)
        if adaptive:
            assert e.preset_weights.state()["updates"] == 3
            assert e.preset_weights.probabilities() != probabilities_before
        else:
            assert e.preset_weights.state() == before
        assert e.B_evals == 3
    finally:
        e.log.close()


def test_failed_generation_is_a_zero_reward_preset_attempt(tmp_path):
    e = _engine(tmp_path, preset_selection=True, container={"kind": "pool_novelty"}, dup_cap=1)
    try:
        _seed(e)
        e.llm = _LLM(["# no algorithm change\n" + e.db.get(e.db.pop_best_id).code])
        e._evaluate = lambda code: pytest.fail("rejected candidate reached evaluator")
        before = e.preset_weights.probabilities()
        e._step(0)
        assert e.preset_weights.state()["updates"] == 1
        assert e.preset_weights.probabilities() == before
        assert e.B_evals == 0 and e.n_nochild == 1
    finally:
        e.log.close()


@pytest.mark.parametrize("kind", ["cvt", "vq", "annealed"])
def test_geometry_reports_marginal_parent_probability_not_only_sampled_branch(kind):
    db = _db(kind, cap=6)
    for index in range(6):
        _add(db, index)

    class PickBranch:
        def __init__(self, centre):
            self.centre = centre
            self.draws = []

        def choices(self, population, weights, k):
            self.draws.append(list(weights))
            return [list(population)[self.centre if len(self.draws) == 1 else 0]]

    contributions, propensities = [], []
    for j in range(len(db.geometries[0].centre_ids)):
        rng = PickBranch(j)
        result = db.select_parent(0, rng)
        centre_probability = rng.draws[0][j] / sum(rng.draws[0])
        parent_probability = rng.draws[1][0] / sum(rng.draws[1])
        contributions.append(centre_probability * parent_probability)
        propensities.append(result.propensity)
        assert result.debug["joint_propensity"] == pytest.approx(contributions[-1])
    assert propensities == pytest.approx([sum(contributions)] * len(propensities))


def test_engine_preset_checkpoint_restores_distribution_and_sampling_rng(tmp_path):
    source = _engine(tmp_path / "source", preset_selection=True)
    target = _engine(tmp_path / "target", preset_selection=True)
    try:
        _seed(source)
        for reward in (0.0, 1.0, 0.5, 1.0):
            name, propensity = source.preset_weights.sample(source.rng.stream("preset"))
            source.preset_weights.update(name, reward, propensity)
        path = checkpoint.save(source, "test")
        checkpoint.load(target, path)
        assert source.preset_weights.state() == target.preset_weights.state()
        assert [source.preset_weights.sample(source.rng.stream("preset")) for _ in range(20)] == [
            target.preset_weights.sample(target.rng.stream("preset")) for _ in range(20)
        ]
    finally:
        source.log.close()
        target.log.close()


@pytest.mark.parametrize("overrides", [
    {"container": {"kind": "grid"}},
    {"pop_cap": 3},
    {"preset_selection": True},
])
def test_engine_refuses_changed_experiment_contract_before_state_load(tmp_path, overrides):
    source = _engine(tmp_path / "source")
    target = _engine(tmp_path / "target", **overrides)
    try:
        _seed(source)
        path = checkpoint.save(source, "test")
        with pytest.raises(checkpoint.IdentityMismatch):
            checkpoint.load(target, path)
        assert target.db.pop_best_id is None and target.B_evals == 0
    finally:
        source.log.close()
        target.log.close()


@pytest.mark.parametrize("container", [
    {"kind": "annealed", "split_min_points": 2},
    {"kind": "annealed", "split_min_points": 3},
    {"kind": "cvt", "k": 7, "max_centres": 6},
    {"kind": "pool", "selection": "centres"},
    {"kind": "pool", "strict_cap": False},
])
def test_invalid_container_config_fails_before_engine_or_model_initialization(container):
    with pytest.raises(ValueError):
        apply_defaults({"container": container})


def test_geometric_engine_refuses_missing_embedder_instead_of_length_fallback(tmp_path):
    with pytest.raises(ValueError, match="frozen embedder"):
        _engine(tmp_path, container={"kind": "cvt"})


def test_legacy_engine_keeps_original_database(tmp_path):
    e = _engine(tmp_path, container={"kind": "legacy"})
    try:
        assert type(e.db) is Database
        assert e.novelty_gate is None
    finally:
        e.log.close()


@pytest.mark.parametrize("kind,selection,presets", [
    ("grid", "shared", False), ("pool", "shared", False),
    ("pool_novelty", "shared", False), ("cvt", "shared", False),
    ("vq", "shared", False), ("annealed", "shared", False),
    ("annealed", "centres", False), ("pool", "shared", True),
])
def test_full_engine_resume_matches_continuous_experimental_search(
        tmp_path, monkeypatch, kind, selection, presets):
    """Pause before SH; retained populations and all later draws must match.

    The real engine, scheduling, worker, containers, reward and checkpoint paths
    run here. Only evaluation and frozen embedding are deterministic local stubs;
    MockLLM is stateless and keyed by the actual assembled prompt.
    """
    import numpy as np
    from sciel.components.policy import code_embed

    class FrozenDescriptor:
        embedder = SimpleNamespace(artifact_identity={'sha256': 'f' * 64})

        def __call__(self, code, cases):
            return cases["axis"]

        def describe(self):
            return {"kind": "test_frozen_descriptor", "axis_sha": "fixed-axis-1",
                    "embedder_sha": "fixed-embedder-1"}

    monkeypatch.setattr(code_embed.LearnedDescriptor, "load",
                        lambda *args, **kwargs: FrozenDescriptor())
    monkeypatch.setattr(code_embed, "embed_batch",
                        lambda embedder, rows: np.asarray([cases["vector"] for _, cases in rows]))

    def evaluate(code):
        digest = hashlib.sha256(code.encode()).digest()
        score = 0.1 + 0.8 * int.from_bytes(digest[:4], "big") / (2**32 - 1)
        angle = int.from_bytes(digest[4:8], "big") / (2**32 - 1) * 2 * math.pi
        cases = {"combined_score": score, "axis": digest[8] / 255,
                 "vector": [math.cos(angle), math.sin(angle)]}
        ok = "raise RuntimeError" not in code
        result = EvalResult(ok, score if ok else 0.0, cases, timing=0.25,
                            artifacts={} if ok else {"error": "mock injected failure"})
        return CascadeOutcome(result, result.score, "full")

    class PricedMockLLM(MockLLM):
        def generate(self, *args, **kwargs):
            response = super().generate(*args, **kwargs)
            response.cost = 0.0025
            return response

    def build(path):
        e = _engine(
            path, budget=20, num_islands=3, pop_cap=4, random_seed=817,
            embed_grid=True, embed_axis_path="frozen-test-axis", milestone_fracs=[0.5, 0.75],
            sh_min_fair_evals=0, log_population=True,
            container={"kind": kind, "selection": selection},
            preset_selection=presets, preset_adaptive=True,
            policy={"enabled": not presets, "mode_bandit": not presets,
                    "plateau_shift": not presets},
        )
        e.llm = PricedMockLLM(dup_modulo=37, fail_modulo=29)
        e._evaluate = evaluate
        return e

    continuous = build(tmp_path / "continuous")
    full_result = continuous.run()
    interrupted = build(tmp_path / "split")
    interrupted._stop_at = 8
    partial_result = interrupted.run()
    assert partial_result["reason"] == "paused" and interrupted.B_evals == 8
    resumed = build(tmp_path / "split")
    checkpoint.load(resumed, str(tmp_path / "split/checkpoint.json"))
    resumed_result = resumed.run()
    assert resumed_result == full_result
    assert resumed.B_evals == 20
    assert resumed.db.eviction_count > 0
    assert resumed.schedule.milestones_passed == 2
    assert len(resumed.schedule.survivors) == 1
    for name in ("db", "rng", "schedule", "selection", "controller", "op_bandit", "reward"):
        assert getattr(resumed, name).state() == getattr(continuous, name).state(), name
    for name in ("iteration_no", "B_evals", "n_events", "n_records", "n_dups", "n_nochild",
                 "n_eval_fail", "cost_total", "noop_parents", "lineage_fail", "eval_times"):
        assert getattr(resumed, name) == getattr(continuous, name), name
    if presets:
        assert resumed.preset_weights.state() == continuous.preset_weights.state()
        assert resumed.preset_weights.updates == resumed.iteration_no
        assert resumed.preset_weights.sample(resumed.rng.stream("preset")) == (
            continuous.preset_weights.sample(continuous.rng.stream("preset")))

    def decisions(path):
        rows = [json.loads(line) for line in (path / "sciel_events.jsonl").read_text().splitlines()]
        keys = ("event", "policy", "action", "island", "iteration_no", "B_evals",
                "parent_id", "child_id", "child_score", "propensity", "parent_propensity",
                "mode", "mode_propensity", "mode_dist", "distribution", "reason")
        return [{key: row[key] for key in keys if key in row} for row in rows
                if row["event"] in {"decision", "iteration"}]

    assert decisions(tmp_path / "split") == decisions(tmp_path / "continuous")
    survivor = resumed.schedule.survivors[0]
    if selection == "centres":
        assert resumed.db.select_parent(survivor, resumed.rng.stream("parent")) == (
            continuous.db.select_parent(survivor, continuous.rng.stream("parent")))
    else:
        assert resumed.selection.select(resumed._island_view(survivor), None,
                                        resumed.rng.stream("parent")) == (
            continuous.selection.select(continuous._island_view(survivor), None,
                                         continuous.rng.stream("parent")))
