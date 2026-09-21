"""Synthetic geometry and clocks only; no optimizer, candidate or model run."""
import math

import numpy as np
import pytest

from sciel.solvers import circle_relocation as relocation


def packing():
    centers = np.zeros((26, 2)); centers[:2] = [[.25, .5], [.75, .5]]
    radii = np.zeros(26); radii[:2] = .1
    return np.concatenate((centers.ravel(), radii))


def noise_for(points):
    noise = np.zeros(78)
    noise[:32] = (np.asarray(points).reshape(32) * .12) - .06
    return noise


def run(base=None, noise=None, index=0, **kwargs):
    base = packing() if base is None else base
    noise = np.zeros(78) if noise is None else noise
    return relocation.relocate_circle(base, noise, np.full(78, .01), index,
                                      deadline=1., clock=lambda: 0., **kwargs)


def assert_geometry(vector):
    centers, radii = vector[:52].reshape(26, 2), vector[52:]
    assert all(math.isfinite(x) for x in vector) and min(radii) >= 0
    for i, ((x, y), radius) in enumerate(zip(centers, radii)):
        assert min(x, y, 1-x, 1-y) - radius >= -1e-6
        for k in range(i):
            assert math.dist((x, y), centers[k]) - radius - radii[k] >= -1e-6


def test_known_relocation_excludes_own_circle_but_preserves_other25_exactly():
    base = packing(); before = base.copy(); stats = {}
    value = run(base, stats=stats)
    assert value[0] == value[1] == .5
    assert value[52] == .25 - .1
    other = [i for i in range(78) if i not in (0, 1, 52)]
    np.testing.assert_array_equal(value[other], base[other])
    np.testing.assert_array_equal(base, before)
    assert not np.shares_memory(value, base)
    assert_geometry(value)
    assert stats['relocation_attempts'] == stats['relocation_accepted'] == 1
    assert stats['relocation_fallbacks'] == stats['relocation_errors'] == 0
    # Original circle at its own center must not cap its proposed new radius.
    own_center = noise_for([[.25, .5]] * 16)
    value = run(base, own_center)
    assert value[52] == .25


def test_best_of16_and_first_tie_use_exact_original_noise_mapping():
    points = [[.5, .5]] * 16
    points[3], points[7] = [.25, .25], [.25, .75]
    noise = noise_for(points)
    value = run(noise=noise)
    reconstructed = ((noise[:32] + .06) / .12).reshape(16, 2)
    np.testing.assert_array_equal(value[:2], reconstructed[3])
    assert value[52] == .25
    assert_geometry(value)


def test_whole_square_endpoints_zero_radius_is_valid_not_clipped():
    value = run(noise=noise_for([[1., 1.]] * 16))
    np.testing.assert_array_equal(value[:2], [1., 1.])
    assert value[52] == 0 and value[52] != .001
    assert_geometry(value)


def test_all26_circle_indices_cycle_and_unselected_parameters_do_not_change():
    base = np.zeros(78)
    for index in range(0, 108, 4):
        chosen = (index // 4) % 26
        value = run(base=base, index=index)
        changed = {2*chosen, 2*chosen+1, 52+chosen}
        assert set(np.flatnonzero(value != base)) == changed
        assert value[52+chosen] == .5
        assert_geometry(value)


@pytest.mark.parametrize('index', [1, 2, 3, 5, 6, 7, 103])
def test_nonrelocation_indices_keep_exact_default_without_attempt(index):
    stats = {}; value = run(index=index, stats=stats)
    np.testing.assert_array_equal(value, np.full(78, .01))
    assert stats == relocation.new_relocation_stats()


def test_unavailable_location_returns_default_not_repaired_incumbent():
    stats = {}; noise = noise_for([[.75, .5]] * 16)
    default = np.arange(78, dtype=float) / 80
    value = relocation.relocate_circle(packing(), noise, default, 0,
        deadline=1., clock=lambda: 0., stats=stats)
    np.testing.assert_array_equal(value, default)
    assert not np.shares_memory(value, default)
    assert stats['relocation_attempts'] == stats['relocation_fallbacks'] == 1
    assert stats['relocation_fallback_reasons'] == {'no_nonnegative_clearance': 1}


def test_lower_sum_proposal_allowed_without_mutating_saved_incumbent():
    base = packing(); stats = {}
    value = run(base, noise_for([[1., 1.]] * 16), stats=stats)
    assert sum(value[52:]) < sum(base[52:])
    assert stats['relocation_accepted'] == 1
    np.testing.assert_array_equal(base, packing())


def test_no_rng_draw_and_no_input_or_output_aliasing():
    base, noise, default = packing(), np.zeros(78), np.full(78, .01)
    copies = [value.copy() for value in (base, noise, default)]
    state = np.random.get_state()
    value = relocation.relocate_circle(base, noise, default, 0, deadline=1., clock=lambda: 0.)
    assert all(np.array_equal(a, b) for a, b in zip(state, np.random.get_state()))
    for original, copy in zip((base, noise, default), copies):
        np.testing.assert_array_equal(original, copy)
        assert not np.shares_memory(value, original)
    value[:] = 99
    for original, copy in zip((base, noise, default), copies):
        np.testing.assert_array_equal(original, copy)


def test_expired_before_work_is_skip_not_attempt_or_fallback():
    stats = {}
    value = relocation.relocate_circle(packing(), np.zeros(78), np.full(78, .01), 0,
        deadline=0., clock=lambda: 0., stats=stats)
    np.testing.assert_array_equal(value, np.full(78, .01))
    assert stats['relocation_deadline_skips'] == 1
    assert stats['relocation_attempts'] == stats['relocation_fallbacks'] == 0


def test_completed_feasible_proposal_survives_deadline_for_caller_to_consider():
    times = iter([0., 0., 2.]); stats = {}
    value = relocation.relocate_circle(packing(), np.zeros(78), np.full(78, .01), 0,
        deadline=1., clock=lambda: next(times), stats=stats)
    assert_geometry(value)
    assert stats['relocation_accepted'] == 1 and stats['relocation_time_seconds'] == 2.


def test_invalid_computed_proposal_falls_back_without_geometry_repair(monkeypatch):
    actual = relocation._relocation_feasible; calls = []
    def check(value):
        calls.append(value.copy())
        return actual(value) if len(calls) == 1 else False
    monkeypatch.setattr(relocation, '_relocation_feasible', check)
    stats = {}; value = run(stats=stats)
    np.testing.assert_array_equal(value, np.full(78, .01))
    assert stats['relocation_fallback_reasons'] == {'infeasible_proposal': 1}
    assert len(calls) == 2


@pytest.mark.parametrize('field,value', [
    ('incumbent', np.full(78, np.nan)), ('incumbent', np.zeros(77)),
    ('incumbent', np.full(78, 1j)), ('incumbent', np.ones(78)),
    ('noise', np.full(78, .061)), ('noise', np.full(78, np.inf)),
    ('default_guess', [[1], [1, 2]]), ('restart_index', True),
    ('restart_index', -1), ('deadline', True), ('deadline', float('inf')),
])
def test_bad_inputs_fail_explicitly_with_error_not_success(field, value):
    kwargs = dict(incumbent=packing(), noise=np.zeros(78), default_guess=np.zeros(78),
                  restart_index=0, deadline=1., clock=lambda: 0., stats={})
    kwargs[field] = value
    with pytest.raises((ValueError, TypeError)):
        relocation.relocate_circle(**kwargs)
    assert kwargs['stats']['relocation_errors'] == 1
    assert kwargs['stats']['relocation_accepted'] == 0
