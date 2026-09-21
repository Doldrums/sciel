"""Behavioral contracts for the geometry-only container alternatives."""
import json

import numpy as np
import pytest

from sciel.components.containers import Geometry


def cloud(seed=7, count=24):
    rng = np.random.default_rng(seed)
    # Interleave the two modes, so even the first eligible chronological batch
    # represents both modes without consulting any future evaluations.
    values = rng.normal(scale=.025, size=(count, 3))
    values[:, 0] += np.where(np.arange(count) % 2, 1, -1)
    return values


def advance(geometry, values, start=1):
    counts = []
    for size in range(start, len(values) + 1):
        geometry.update([str(i) for i in range(size)], values[:size], size / len(values))
        counts.append(len(geometry.centres))
    return counts


@pytest.mark.parametrize("kind", ["cvt", "vq", "annealed"])
def test_probability_and_json_resume_contract(kind):
    values = cloud()
    uninterrupted = Geometry(kind, dim=3, k=3, max_centres=5)
    for size in range(1, 13):
        uninterrupted.update([str(i) for i in range(size)], values[:size], size / len(values))
    checkpoint = json.loads(json.dumps(uninterrupted.state(), allow_nan=False))
    resumed = Geometry("cvt", dim=2, k=2)
    resumed.load_state(checkpoint)
    assert resumed.state() == checkpoint
    for size in range(13, len(values) + 1):
        ids = [str(i) for i in range(size)]
        left = uninterrupted.update(ids, values[:size], size / len(values))
        right = resumed.update(ids, values[:size], size / len(values))
        assert left == right
        assert uninterrupted.state() == resumed.state()
    probabilities = resumed.probabilities(values)
    assert probabilities.shape == (len(values), len(resumed.centre_ids))
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    json.dumps(resumed.diagnostics(), allow_nan=False)
    assert resumed.probabilities(np.empty((0, 3))).shape == (0, len(resumed.centre_ids))


def test_cvt_centres_are_fixed_before_any_search_outcome():
    a = Geometry("cvt", dim=3, k=4, seed=12)
    b = Geometry("cvt", dim=3, k=4, seed=12)
    initial = a.centres.copy()
    a.update(["a", "b"], np.array([[1., 0, 0], [0, 1., 0]]), 1)
    b.update(["unrelated"], np.array([[-200., 71, 8]]), .2)
    np.testing.assert_array_equal(a.centres, initial)
    np.testing.assert_array_equal(b.centres, initial)
    assert a.centre_ids == [0, 1, 2, 3]
    assert a.diagnostics()["min_separation"] > .05
    assert not np.array_equal(initial, Geometry("cvt", dim=3, k=4, seed=13).centres)


def test_vq_updates_each_id_once_and_preserves_ids_at_capacity():
    g = Geometry("vq", dim=2, k=2, max_centres=2, novelty_threshold=.5)
    g.update(["left", "right"], np.array([[-1., 0], [1., 0]]), .1)
    initial = g.centres.copy()
    assert g.centre_ids == [0, 1]
    # Repeated admissions, even if their supplied vectors change, do not train VQ.
    g.update(["left", "right"], np.array([[-8., 0], [8., 0]]), .2)
    np.testing.assert_array_equal(g.centres, initial)
    g.update(["new"], np.array([[0., 1.]]), .3)
    assert g.centre_ids == [0, 1]
    assert not np.array_equal(g.centres, initial)
    assert g.state()["counts"] == [2, 1]
    assert g.diagnostics()["seen_ids"] == 3


def test_vq_identical_vectors_do_not_fill_the_cap():
    g = Geometry("vq", dim=3)
    advance(g, np.ones((30, 3)))
    assert len(g.centres) == 1
    assert g.state()["counts"] == [30]


def test_annealing_resolves_two_modes_without_cap_cascade():
    values = cloud()
    g = Geometry("annealed", dim=3, max_centres=6)
    counts = advance(g, values)
    assert max(np.diff(counts)) <= 1
    assert counts[-1] == 2
    assert g.diagnostics()["min_separation"] > 1.8
    assert all(n >= 10 for n in g.diagnostics()["occupancy"])
    labels = g.update([str(i) for i in range(len(values))], values, 1)
    assert len({labels[str(i)] for i in range(0, len(values), 2)}) == 1
    assert len({labels[str(i)] for i in range(1, len(values), 2)}) == 1
    assert labels["0"] != labels["1"]
    # Reading/updating an unchanged population is not another split opportunity.
    snapshot = g.state()
    for _ in range(10):
        g.update([str(i) for i in range(len(values))], values, 1)
    assert g.state() == snapshot


def test_annealed_t0_uses_first_eligible_batch_and_never_recalibrates():
    initial = cloud(count=4)
    future = initial * 30
    g = Geometry("annealed", dim=3)
    g.update(["a", "b", "c"], initial[:3], .1)
    assert g.t0 is None
    g.update(["a", "b", "c", "d"], initial, .2)
    t0, temperature = g.t0, g.temperature
    variance = np.linalg.eigvalsh(np.cov(initial, rowvar=False, bias=True))[-1]
    assert t0 == pytest.approx(4 * variance)
    g.update(["e", "f", "g", "h"], future, .8)
    assert g.t0 == t0
    assert g.temperature < temperature
    temperature = g.temperature
    g.update(["e", "f", "g", "h"], future, .3)
    assert g.progress == .8
    assert g.temperature == temperature


def test_annealed_degenerate_population_has_one_real_centre():
    g = Geometry("annealed", dim=3)
    advance(g, np.full((20, 3), .75))
    assert len(g.centres) == 1
    assert g.t0 is None
    np.testing.assert_allclose(g.centres, [[.75, .75, .75]])
    np.testing.assert_array_equal(g.probabilities(np.ones((5, 3))), np.ones((5, 1)))
    assert g.diagnostics()["accepted_splits"] == 0


def test_annealed_centres_follow_active_population_and_keep_survivor_id():
    g = Geometry("annealed", dim=3)
    values = cloud()
    advance(g, values)
    assert len(g.centres) == 2
    previous_ids = set(g.centre_ids)
    # An eviction can remove an entire mode; the old centre must not remain as
    # a phantom niche disconnected from the active population.
    remaining = np.tile([2., 1., 0.], (8, 1))
    g.update([f"new{i}" for i in range(8)], remaining, 1)
    assert len(g.centres) == 1
    assert set(g.centre_ids).issubset(previous_ids)
    np.testing.assert_allclose(g.centres, [[2., 1., 0.]])


@pytest.mark.parametrize("values", [np.ones((3, 2)), np.array([[np.nan, 0, 1]]), np.ones(3)])
def test_bad_vectors_fail_before_mutating_geometry(values):
    g = Geometry("annealed", dim=3)
    snapshot = g.state()
    with pytest.raises(ValueError):
        g.update(["a"], values, .5)
    assert g.state() == snapshot


def test_duplicate_ids_and_invalid_progress_are_rejected():
    g = Geometry("vq", dim=3)
    with pytest.raises(ValueError, match="unique"):
        g.update(["a", "a"], np.ones((2, 3)), .1)
    for progress in [-1, 1.1, np.nan]:
        with pytest.raises(ValueError, match="progress"):
            g.update(["a"], np.ones((1, 3)), progress)


def test_checkpoint_rejects_invalid_centres_without_mutation():
    g = Geometry("cvt", dim=3)
    original = g.state()
    damaged = json.loads(json.dumps(original))
    damaged["centre_ids"][1] = damaged["centre_ids"][0]
    with pytest.raises(ValueError):
        g.load_state(damaged)
    assert g.state() == original


def test_online_geometry_cap_does_not_depend_on_unused_fixed_k():
    for kind in ["vq", "annealed"]:
        g = Geometry(kind, dim=3, max_centres=1)
        advance(g, cloud())
        assert len(g.centres) == 1


def test_annealed_near_coincident_centres_are_pruned_despite_hard_ties():
    g = Geometry("annealed", dim=3)
    values = cloud()
    advance(g, values)
    assert len(g.centres) == 2
    remaining = np.tile([.5, 0, 0], (8, 1))
    remaining[::2, 0] += 1e-8
    g.update([f"new{i}" for i in range(8)], remaining, 1)
    assert len(g.centres) == 1


def test_refresh_active_removes_evicted_mode_without_admission_split_or_cooling():
    values = cloud()
    g = Geometry("annealed", dim=3)
    advance(g, values)
    assert len(g.centres) == 2
    before = g.state()
    # The positive mode is rejected by retention. Refitting only retained members
    # must remove its centre before the next parent draw, without splitting again.
    retained = values[::2]
    ids = [str(i) for i in range(0, len(values), 2)]
    labels = g.refresh_active(ids, retained)
    assert len(g.centres) == 1
    assert len(set(labels.values())) == 1
    np.testing.assert_allclose(g.centres[0], retained.mean(axis=0))
    after = g.state()
    for key in ["seen", "splits", "rejected_splits", "t0", "temperature", "progress"]:
        assert before[key] == after[key]
    assert g.diagnostics()["hard_occupied_centres"] == 1
    assert sum(g.diagnostics()["occupancy"]) == len(retained)
    # A bookkeeping call with the retained population is not another admission.
    g.update(ids, retained, g.progress)
    assert g.state() == after


@pytest.mark.parametrize("kind", ["cvt", "vq"])
def test_refresh_active_does_not_train_fixed_or_admission_geometry(kind):
    values = cloud()
    g = Geometry(kind, dim=3)
    advance(g, values)
    before = g.state()
    g.refresh_active(["never_admitted"], np.array([[8., 9., 10.]]))
    after = g.state()
    for key in before:
        if key not in ["last_occupancy", "last_mass"]:
            assert before[key] == after[key]
    assert sum(after["last_occupancy"]) == 1
