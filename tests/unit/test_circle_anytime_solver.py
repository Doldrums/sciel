"""H30 calculus and failure-path tests; no real optimisation or task evaluator.

All solve calls replace SciPy's optimiser and monotonic clock. Geometry checks
below are independent scalar checks against the task's numerical contract.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.solvers import circle_anytime as solver


N = 26


def lattice(radius=.01):
    centers = np.array([(.07 + .16 * (i % 6), .07 + .16 * (i // 6))
                        for i in range(N)])
    return np.concatenate([centers.ravel(), np.full(N, radius)])


def geometry(result, tolerance=1e-6):
    centers, radii, total = result
    assert centers.shape == (N, 2) and radii.shape == (N,)
    assert np.isfinite(centers).all() and np.isfinite(radii).all()
    assert math.isfinite(total) and total == pytest.approx(math.fsum(map(float, radii)), abs=1e-15)
    for i, ((x, y), radius) in enumerate(zip(centers, radii)):
        assert radius >= 0
        assert min(x - radius, y - radius, 1 - x - radius, 1 - y - radius) >= -tolerance
        for j in range(i):
            distance = math.hypot(x - centers[j, 0], y - centers[j, 1])
            assert distance >= radius + radii[j] - tolerance


def finite_difference(function, x, step=1e-6):
    columns = []
    for index in range(len(x)):
        left, right = x.copy(), x.copy()
        left[index] -= step
        right[index] += step
        columns.append((np.asarray(function(right)) - np.asarray(function(left))) / (2 * step))
    return np.stack(columns, axis=-1)


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    rng_state = np.random.get_state()
    monkeypatch.setattr(solver.time, 'monotonic', value)
    # Each test must opt into its own fake optimiser. Accidentally calling the
    # real SciPy implementation fails immediately rather than running a search.
    def forbidden(*args, **kwargs):
        pytest.fail('a real optimisation must never run in these tests')
    monkeypatch.setattr(solver, 'minimize', forbidden)
    try:
        yield value
    finally:
        np.random.set_state(rng_state)


def test_analytic_objective_gradient_matches_central_differences():
    x = lattice()
    x[2 * N:] = np.linspace(.004, .012, N)
    assert solver.objective(x) == pytest.approx(-math.fsum(x[2 * N:]))
    analytic = solver.objective_jac(x)
    assert analytic.shape == (3 * N,)
    np.testing.assert_allclose(analytic, finite_difference(solver.objective, x), atol=1e-9)


def test_wall_and_distance_jacobians_match_differences_and_scalar_geometry():
    x = lattice()
    x[2 * N:] = np.linspace(.004, .012, N)
    centers, radii = x[:2 * N].reshape(N, 2), x[2 * N:]
    expected = []
    for (a, b), radius in zip(centers, radii):
        expected.extend([a - radius, 1 - a - radius, b - radius, 1 - b - radius])
    for i in range(N):
        for j in range(i + 1, N):
            expected.append(math.dist(centers[i], centers[j]) - radii[i] - radii[j])
    constraints = solver.packing_constraints(x)
    jacobian = solver.packing_jacobian(x)
    assert constraints.shape == (429,) and jacobian.shape == (429, 78)
    np.testing.assert_allclose(constraints, expected, rtol=1e-14, atol=1e-14)
    numeric = finite_difference(solver.packing_constraints, x)
    np.testing.assert_allclose(jacobian[:4 * N], numeric[:4 * N], rtol=1e-7, atol=2e-9)
    np.testing.assert_allclose(jacobian[4 * N:], numeric[4 * N:], rtol=1e-7, atol=2e-9)
    # Changing only centers cannot change the radius-sum objective.
    direction = np.arange(3 * N, dtype=float)
    direction[2 * N:] = 0
    assert np.dot(solver.objective_jac(x), direction) == 0


def test_coincident_centers_have_finite_documented_zero_subgradient():
    x = lattice()
    x[2:4] = x[:2]
    assert not solver.is_feasible(x)
    jacobian = solver.packing_jacobian(x)
    assert np.isfinite(jacobian).all()
    first_pair = jacobian[4 * N]
    np.testing.assert_array_equal(first_pair[:2 * N], np.zeros(2 * N))
    assert first_pair[2 * N] == first_pair[2 * N + 1] == -1
    assert np.count_nonzero(first_pair) == 2
    assert np.isfinite(solver.packing_constraints(x)).all()


@pytest.mark.parametrize('corruption', ['short', 'matrix', 'nan', 'infinity', 'complex',
                                       'negative_radius', 'wall', 'pair', 'huge_center'])
def test_feasibility_rejects_invalid_shapes_numbers_and_geometries(corruption):
    x = lattice()
    if corruption == 'short': x = x[:-1]
    elif corruption == 'matrix': x = x.reshape(26, 3)
    elif corruption == 'nan': x[0] = np.nan
    elif corruption == 'infinity': x[-1] = np.inf
    elif corruption == 'complex': x = x.astype(complex); x[0] += 1j
    elif corruption == 'negative_radius': x[-1] = -1e-15
    elif corruption == 'wall': x[0] = 0
    elif corruption == 'pair': x[2:4] = x[:2]
    else: x[0] = 1e300
    assert solver.is_feasible(x) is False
    with pytest.raises(ValueError, match='initial incumbent'):
        solver.Incumbent(x)


def test_tolerance_admission_is_explicit_and_does_not_repair_parameters():
    x = lattice()
    x[0] = x[2 * N] - .5e-6
    before = x.copy()
    assert solver.is_feasible(x)
    keeper = solver.Incumbent(x)
    np.testing.assert_array_equal(x, before)
    result = keeper.result()
    np.testing.assert_array_equal(result[0].ravel(), x[:2 * N])
    assert result[0][0, 0] - result[1][0] < 0  # Tolerance-valid, not strict-certified.
    geometry(result)
    x[0] -= 2e-6
    assert not solver.is_feasible(x)


def test_initial_fallback_is_deterministic_feasible_and_independent_of_rng():
    random_state = np.random.get_state()
    first = solver.initial_parameters()
    second = solver.initial_parameters()
    after = np.random.get_state()
    assert random_state[0] == after[0] and random_state[2:] == after[2:]
    np.testing.assert_array_equal(random_state[1], after[1])
    np.testing.assert_array_equal(first, second)
    assert not np.shares_memory(first, second)
    geometry(solver.Incumbent(first).result(), tolerance=0.)
    first[:] = np.nan
    assert np.isfinite(second).all()


def test_restart_uses_declared_global_draw_shape_and_clamps_without_seeding(monkeypatch, clock):
    initial = solver.initial_parameters()
    perturbation = np.zeros(78)
    perturbation[50:52] = (-.04, .03)  # The archived 26th initial center is (0,0).
    perturbation[52:55] = (-.06, .06, .025)
    draws = []
    def uniform(low, high, shape):
        draws.append((low, high, shape))
        return perturbation.copy()
    def forbidden_seed(*args, **kwargs):
        pytest.fail('solver must not silently seed the shared RNG')
    guesses = []
    def fake(fun, x, **kwargs):
        guesses.append(x.copy())
        return SimpleNamespace(x=initial.copy(), success=False)
    monkeypatch.setattr(solver.np.random, 'uniform', uniform)
    monkeypatch.setattr(solver.np.random, 'seed', forbidden_seed)
    monkeypatch.setattr(solver, 'minimize', fake)
    geometry(solver.solve_packing(budget_seconds=1, max_restarts=1))
    assert draws == [(-.06, .06, (78,))]
    assert len(guesses) == 1
    guess = guesses[0]
    np.testing.assert_array_equal(guess[:50], initial[:50])
    np.testing.assert_array_equal(guess[50:52], [.001, .03])
    np.testing.assert_allclose(guess[52:55], [.001, .07, .035], rtol=0, atol=2e-17)
    np.testing.assert_array_equal(guess[55:-1], initial[55:-1])
    assert guess[-1] == .001


def test_incumbent_copies_inputs_outputs_and_never_replaces_with_equal_worse_or_invalid():
    original = lattice(.01)
    keeper = solver.Incumbent(original)
    original[:] = np.nan
    better = lattice(.03)
    assert keeper.consider(better)
    expected = better.copy()
    better[:] = 0
    assert not keeper.consider(lattice(.02))
    equal = lattice(.03)
    equal[:2 * N] += .001
    assert not keeper.consider(equal)
    invalid = lattice(.2)
    assert not keeper.consider(invalid)
    result = keeper.result()
    np.testing.assert_array_equal(result[0].ravel(), expected[:2 * N])
    np.testing.assert_array_equal(result[1], expected[2 * N:])
    geometry(result)
    result[0][:] = np.nan
    result[1][:] = np.nan
    geometry(keeper.result())
    assert keeper.improvements == 1


def test_expired_before_first_solver_keeps_initial_incumbent(clock):
    result = solver.solve_packing(budget_seconds=0)
    expected = solver.Incumbent(solver.initial_parameters()).result()
    for actual, wanted in zip(result, expected):
        np.testing.assert_array_equal(actual, wanted)
    geometry(result)
    assert solver.LAST_STATS['started_starts'] == 0
    assert solver.LAST_STATS['deadline_reached'] is True


def test_unsuccessful_feasible_result_survives_expiration_before_next_restart(monkeypatch, clock):
    improved = lattice(.03)
    def fake(fun, x, **kwargs):
        assert kwargs['method'] == 'SLSQP'
        assert callable(kwargs['jac']) and callable(kwargs['constraints']['jac'])
        assert kwargs['options']['maxiter'] == 123
        clock.now = 2.
        return SimpleNamespace(x=improved.copy(), success=False)
    monkeypatch.setattr(solver, 'minimize', fake)
    result = solver.solve_packing(budget_seconds=1, max_restarts=5, maxiter=123)
    geometry(result)
    np.testing.assert_array_equal(result[1], improved[2 * N:])
    assert solver.LAST_STATS['completed_starts'] == 1
    assert solver.LAST_STATS['successful_starts'] == 0
    assert solver.LAST_STATS['deadline_reached'] is True


def test_callback_keeps_better_iterate_before_raising_deadline(monkeypatch, clock):
    improved = lattice(.025)
    def fake(fun, x, **kwargs):
        clock.now = 2.
        kwargs['callback'](improved)
        pytest.fail('deadline callback must stop this fake solver')
    monkeypatch.setattr(solver, 'minimize', fake)
    result = solver.solve_packing(budget_seconds=1)
    geometry(result)
    np.testing.assert_array_equal(result[1], improved[2 * N:])
    assert solver.LAST_STATS['callback_calls'] == 1
    assert solver.LAST_STATS['completed_starts'] == 0


@pytest.mark.parametrize('entry', ['objective', 'objective_jac', 'constraints', 'constraints_jac'])
def test_deadline_in_each_numerical_entry_keeps_latest_feasible_state(monkeypatch, clock, entry):
    improved = lattice(.025)
    def fake(fun, x, **kwargs):
        functions = {'objective': fun, 'objective_jac': kwargs['jac'],
                     'constraints': kwargs['constraints']['fun'],
                     'constraints_jac': kwargs['constraints']['jac']}
        clock.now = 2.
        functions[entry](improved)
        pytest.fail('timed numerical entry must stop the fake solver')
    monkeypatch.setattr(solver, 'minimize', fake)
    result = solver.solve_packing(budget_seconds=1)
    geometry(result)
    np.testing.assert_array_equal(result[1], improved[2 * N:])
    assert solver.LAST_STATS['deadline_reached'] is True
    assert solver.LAST_STATS['completed_starts'] == 0


def test_later_timed_out_restart_cannot_erase_prior_best(monkeypatch, clock):
    sequence = [lattice(.03), lattice(.02)]
    def fake(fun, x, **kwargs):
        current = sequence.pop(0)
        if not sequence:
            clock.now = 2.
            kwargs['callback'](current)
            pytest.fail('later deadline should escape')
        return SimpleNamespace(x=current, success=True)
    monkeypatch.setattr(solver, 'minimize', fake)
    result = solver.solve_packing(budget_seconds=1, max_restarts=2)
    assert not sequence
    geometry(result)
    np.testing.assert_array_equal(result[1], lattice(.03)[2 * N:])
    assert solver.LAST_STATS['started_starts'] == 2
    assert solver.LAST_STATS['completed_starts'] == 1


def test_infeasible_success_result_never_replaces_fallback(monkeypatch, clock):
    monkeypatch.setattr(solver, 'minimize', lambda *a, **kw:
                        SimpleNamespace(x=lattice(.2), success=True))
    result = solver.solve_packing(budget_seconds=1, max_restarts=1)
    expected = solver.Incumbent(solver.initial_parameters()).result()
    for actual, wanted in zip(result, expected):
        np.testing.assert_array_equal(actual, wanted)
    geometry(result)
    assert solver.LAST_STATS['successful_starts'] == 1
    assert solver.LAST_STATS['improvements'] == 0


def test_unexpected_exception_propagates_with_diagnostic_incumbent(monkeypatch, clock):
    error = RuntimeError('synthetic unexpected solver defect')
    stats_reference = solver.LAST_STATS
    def fake(fun, x, **kwargs):
        kwargs['callback'](lattice(.03))
        raise error
    monkeypatch.setattr(solver, 'minimize', fake)
    with pytest.raises(RuntimeError) as raised:
        solver.solve_packing(budget_seconds=1)
    assert raised.value is error
    assert solver.LAST_STATS is stats_reference
    assert stats_reference['stop_reason'] == 'error'
    assert stats_reference['deadline_reached'] is False
    assert stats_reference['best_sum'] == pytest.approx(.78)


def test_normal_restart_limit_publishes_feasible_best_and_wrapper_defaults(monkeypatch, clock):
    calls = []
    def fake(fun, x, **kwargs):
        calls.append(x.copy())
        return SimpleNamespace(x=lattice(.02), success=False)
    monkeypatch.setattr(solver, 'minimize', fake)
    result = solver.solve_packing(budget_seconds=1, max_restarts=2)
    assert len(calls) == 2
    geometry(result)
    assert solver.LAST_STATS['stop_reason'] == 'restart_limit'
    marker = object()
    def entry(**kwargs):
        assert kwargs == {}
        return marker
    monkeypatch.setattr(solver, 'solve_packing', entry)
    assert solver.run_packing() is marker


@pytest.mark.parametrize('settings', [dict(budget_seconds=-1), dict(budget_seconds=True),
    dict(budget_seconds=float('nan')), dict(budget_seconds=float('inf')),
    dict(max_restarts=-1), dict(max_restarts=True), dict(maxiter=0), dict(maxiter=True)])
def test_invalid_limits_fail_before_any_optimizer_call(clock, settings):
    with pytest.raises(ValueError):
        solver.solve_packing(**settings)
