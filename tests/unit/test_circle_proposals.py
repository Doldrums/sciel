"""Source/AST contracts and synthetic fake-solver paths, never packing search."""
import ast
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.hypothesis import circle_proposals as builder
from sciel.hypothesis import circle_restart_scales as scales


def helper(body):
    return 'def propose_restart(incumbent, initial, noise, default_guess, restart_index):\n' + ''.join(
        '    ' + line + '\n' for line in body.splitlines())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def test_source_receipt_reconstructs_entire_control_and_keeps_interior_bytes():
    submitted = '\r\n' + helper('"""Preserve two  spaces and α."""\nx = default_guess.copy()\nx[:52] += noise[:52] * 0.5\nreturn x') + '\r\n'
    normalized = submitted.strip('\r\n') + '\n'
    built = builder.build(submitted)
    control, child, proof = builder.source_for().encode(), built['code'].encode(), built['proof']
    assert proof['parent_source_sha256'] == digest(control)
    assert proof['child_source_sha256'] == digest(child)
    assert proof['submitted_helper_sha256'] == digest(submitted.encode())
    assert proof['normalized_helper_sha256'] == digest(normalized.encode())
    a, b = proof['after']['byte_span']; c, d = proof['before']['byte_span']
    assert child[a:b] == normalized.encode() == proof['after']['text'].encode()
    assert control[c:d] == builder.DEFAULT_HELPER.encode()
    assert child[:a] == control[:c] and child[b:] == control[d:]
    assert child[:a] + proof['before']['text'].encode() + child[b:] == control
    assert digest(child[:a]) == proof['unchanged_prefix']['sha256']
    assert digest(child[b:]) == proof['unchanged_suffix']['sha256']
    assert proof['plan_key'] == builder.OPERATION + ':' + digest(control) + ':' + digest(normalized.encode())
    assert builder.source_for(submitted) == built['code']
    assert builder.validate_helper(submitted) == normalized
    assert 'two  spaces and α' in proof['after']['text']


def test_scaffold_is_new_and_all_baseline_patches_have_exact_inverse():
    original = scales.source_for('0.06').encode()
    control = builder.source_for().encode()
    identity = builder.identity()
    assert digest(original) == builder.PINNED_BASELINE_SHA256
    assert control != original and digest(control) == identity['control_source_sha256']
    restored = control
    for patch in reversed(identity['scaffold_patches']):
        a, b = patch['child_byte_span']; c, d = patch['parent_byte_span']
        assert control[a:b] == patch['after'].encode()
        assert original[c:d] == patch['before'].encode()
        assert digest(control[a:b]) == patch['after_sha256']
        assert digest(original[c:d]) == patch['before_sha256']
        restored = restored[:a] + patch['before'].encode() + restored[b:]
    assert restored == original
    assert identity['first_restart_index'] == 0 and identity['restart_period'] == 4
    assert identity['stats_marker'] == builder.STATS_MARKER
    assert 'freshly measured' in builder.interface_description()['baseline']


def test_solver_equations_callback_and_minimize_options_unchanged():
    sources = [scales.source_for('0.06'), builder.source_for(), builder.source_for(helper('return incumbent + noise'))]
    trees = [ast.parse(s) for s in sources]
    for name in ('objective', 'objective_jac', 'packing_constraints', 'packing_jacobian',
                 'is_feasible', 'check_deadline', 'timed', 'callback'):
        functions = [next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == name) for t in trees]
        assert len({ast.dump(n) for n in functions}) == 1
    for name in ('minimize', 'np.random.uniform'):
        calls = [[n for n in ast.walk(t) if isinstance(n, ast.Call) and ast.unparse(n.func) == name] for t in trees]
        assert all(len(c) == 1 for c in calls)
        assert len({ast.dump(c[0]) for c in calls}) == 1
    for line in ('guess[2 * N_CIRCLES:] = np.maximum(guess[2 * N_CIRCLES:], 0.001)',
                 'guess[:2 * N_CIRCLES] = np.clip(guess[:2 * N_CIRCLES], 0.001, 0.999)'):
        assert all(s.count(line) == 1 for s in sources)
    assert not any('from sciel' in s for s in sources)


@pytest.mark.parametrize('text', [None, b'x', '', '\n', 'def nope():\n    return None\n',
    '```python\n' + builder.DEFAULT_HELPER + '```',
    'Explanation\n' + builder.DEFAULT_HELPER, ' ' + builder.DEFAULT_HELPER,
    builder.DEFAULT_HELPER + '\nx = 3\n', builder.DEFAULT_HELPER + '\ndef more():\n    pass\n',
    builder.DEFAULT_HELPER.replace('restart_index)', 'restart_index=0)'),
    builder.DEFAULT_HELPER.replace('restart_index)', '*restart_index)'),
    builder.DEFAULT_HELPER.replace('restart_index)', '*, restart_index)'),
    builder.DEFAULT_HELPER.replace('incumbent,', 'incumbent: float,'),
    builder.DEFAULT_HELPER.replace('):', ') -> float:'),
    '@np.copy\n' + builder.DEFAULT_HELPER,
    builder.DEFAULT_HELPER + '# ' + 'x' * 16384,
    builder.DEFAULT_HELPER + '# \ud800',
])
def test_transport_signature_and_size_rejections(text):
    with pytest.raises(ValueError):
        builder.validate_helper(text)


@pytest.mark.parametrize('body', [
    'import os\nreturn None', 'from math import sqrt\nreturn None',
    'global LAST_STATS\nreturn None', 'nonlocal noise\nreturn None',
    'def inner():\n    return None\nreturn None',
    'return (lambda: None)()', 'return propose_restart(incumbent, initial, noise, default_guess, restart_index)',
    'f = np.copy\nreturn f(default_guess)', 'return globals()', 'return eval("None")',
    'return __import__("os")', 'return np.random.uniform(size=78)', 'return np.linalg.solve(initial, noise)',
    'return np.load("x")', 'np.save("x", noise)\nreturn None', 'return getattr(np, "copy")(noise)',
    'return noise.__class__', 'return noise.ctypes', 'return noise.data',
    'np.copy = 0\nreturn None', 'np = noise\nreturn None', 'math = noise\nreturn None',
    'len = 3\nreturn None', 'return np.copy.__call__(noise)', 'return noise[0]()',
    'return np.array(*noise)', 'return np.array(**noise)', 'return [x for np in noise]',
    'return sum(x for math in noise)', 'return [open("x") for x in noise]',
    'return [x async for x in noise]', 'while True:\n    pass\nreturn None',
    'try:\n    return None\nexcept:\n    return None', 'return 1j', 'return 1e999',
    'return LAST_STATS', 'return incumbent._params', 'return (noise[0] := 1)',
    'break\nreturn None', 'continue\nreturn None',
])
def test_forbidden_ast_and_callable_routes_fail_closed(body):
    with pytest.raises(ValueError):
        builder.validate_helper(helper(body))


def test_useful_multicircle_vector_and_loop_function_is_accepted_without_execution():
    body = '''x = incumbent.copy()
centers = x[:52].reshape(26, 2)
radii = x[52:]
for i in range(26):
    distances = np.sqrt(np.sum((centers - centers[i]) ** 2, axis=1))
    distances[i] = np.inf
    j = int(np.argmin(distances - radii))
    direction = centers[i] - centers[j]
    length = math.hypot(direction[0], direction[1])
    if length > 0:
        centers[i] += direction / length * abs(noise[2*i])
return x'''
    source = helper(body)
    assert builder.validate_helper(source) == source
    assert builder.build(source)['proof']['after']['text'] == source


def test_synchronous_comprehensions_and_literal_format_preserved():
    source = helper('x = np.array([noise[i] for i in range(78)])\ns = sum(abs(v) for v in noise)\nreturn default_guess + x / max(1.0, s)')
    assert builder.validate_helper(source) == source
    assert 'synchronous list comprehensions/generator expressions' in builder.interface_description()['forbidden']


def test_noop_rejected_but_semantic_equivalence_not_claimed():
    with pytest.raises(ValueError, match='repeats control'):
        builder.build('\n' + builder.DEFAULT_HELPER)
    assert builder.build(helper('if restart_index >= 0:\n    return None\nreturn None'))


def test_baseline_pin_is_checked(monkeypatch):
    original = scales.source_for('0.06')
    monkeypatch.setattr(scales, 'source_for', lambda _: original + '# altered\n')
    with pytest.raises(ValueError, match='baseline'):
        builder.identity()


class Clock:
    now = 0.0
    def __call__(self):
        return self.now


def namespace(source=None):
    ns = {}
    # Only definitions are loaded. Any call to solve uses a supplied fake below.
    exec(compile(builder.source_for(source), '<fake-only-proposal-scaffold>', 'exec'), ns)
    clock = Clock()
    ns['time'] = SimpleNamespace(monotonic=clock)
    def forbidden(*args, **kwargs):
        pytest.fail('real optimizer execution is forbidden')
    ns['minimize'] = forbidden
    return ns, clock


def test_copied_inputs_and_none_preserve_original_default_even_after_mutation():
    ns, _ = namespace()
    inputs = [np.linspace(i, i+1, 78) for i in range(4)]
    originals = [x.copy() for x in inputs]
    seen = []
    def mutate(*args):
        seen.extend(args[:4])
        for value in args[:4]:
            value[:] = -999
        return None
    ns['propose_restart'] = mutate
    stats = ns['_proposal_stats']()
    actual = ns['_proposal_invoke'](*inputs, 0, stats)
    for value, expected in zip(inputs, originals):
        np.testing.assert_array_equal(value, expected)
    np.testing.assert_array_equal(actual, originals[-1])
    assert all(not np.shares_memory(a, b) for a, b in zip(seen, inputs))
    assert not np.shares_memory(actual, inputs[-1])
    assert stats['proposal_attempts'] == stats['proposal_none'] == 1
    assert stats['proposal_accepted'] == stats['proposal_errors'] == 0


@pytest.mark.parametrize('value', [np.zeros(77), np.zeros((26, 3)), np.full(78, np.nan),
    np.full(78, np.inf), np.zeros(78, dtype=complex), np.zeros(78, dtype=bool),
    ['1']*78, np.full(78, 1, dtype=object), 3, {'x': 3}, [[1], [1, 2]]])
def test_bad_results_raise_and_are_not_silently_replaced(value):
    ns, _ = namespace()
    ns['propose_restart'] = lambda *args: value
    stats = ns['_proposal_stats']()
    with pytest.raises((ValueError, TypeError)):
        ns['_proposal_invoke'](*(np.zeros(78) for _ in range(4)), 0, stats)
    assert stats['proposal_attempts'] == stats['proposal_errors'] == 1
    assert stats['proposal_none'] == stats['proposal_accepted'] == 0


def test_finite_negative_or_outside_guess_accepted_without_clipping_or_feasibility():
    ns, _ = namespace()
    proposal = np.linspace(-2, 3, 78)
    ns['propose_restart'] = lambda *args: proposal
    stats = ns['_proposal_stats']()
    actual = ns['_proposal_invoke'](*(np.zeros(78) for _ in range(4)), 0, stats)
    np.testing.assert_array_equal(actual, proposal)
    assert not np.shares_memory(actual, proposal)
    assert stats['proposal_accepted'] == 1


def test_helper_exception_propagates_with_time_and_error_count():
    ns, clock = namespace()
    def failed(*args):
        clock.now = .04
        raise RuntimeError('synthetic helper failure')
    ns['propose_restart'] = failed
    stats = ns['_proposal_stats']()
    with pytest.raises(RuntimeError, match='synthetic'):
        ns['_proposal_invoke'](*(np.zeros(78) for _ in range(4)), 0, stats)
    assert stats['proposal_errors'] == 1 and stats['proposal_time_seconds'] == .04


def test_fake_solve_same_full_rng_draws_and_ordinary_guesses_with_scheduled_0_4_8():
    saved_rng = np.random.get_state()
    all_guesses, rng_states = [], []
    try:
        for source in (None, helper('return default_guess.copy()')):
            ns, _ = namespace(source)
            called, guesses = [], []
            original = ns['propose_restart']
            def observe(incumbent, initial, noise, default, index):
                called.append((index, incumbent.copy(), initial.copy(), noise.copy(), default.copy()))
                return original(incumbent, initial, noise, default, index)
            ns['propose_restart'] = observe
            def fake(objective, guess, **kwargs):
                guesses.append(guess.copy())
                assert kwargs['method'] == 'SLSQP'
                assert kwargs['options'] == {'maxiter':15000, 'ftol':1e-7, 'disp':False}
                return SimpleNamespace(x=ns['initial_parameters'](), success=True)
            ns['minimize'] = fake
            np.random.seed(71)
            ns['solve_packing'](max_restarts=9)
            stats = ns['LAST_STATS']
            assert [c[0] for c in called] == [0, 4, 8]
            assert stats['proposal_scheduled_starts'] == stats['proposal_scheduled_completed'] == 3
            assert stats['proposal_ordinary_starts'] == stats['proposal_ordinary_completed'] == 6
            assert stats['proposal_attempts'] == 3
            assert stats['proposal_none'] == (3 if source is None else 0)
            assert stats['proposal_accepted'] == (0 if source is None else 3)
            rng = np.random.RandomState(71)
            initial = ns['initial_parameters']()
            for i, guess in enumerate(guesses):
                noise = rng.uniform(-.06, .06, 78)
                expected = initial+noise
                expected[52:] = np.maximum(expected[52:], .001)
                expected[:52] = np.clip(expected[:52], .001, .999)
                np.testing.assert_array_equal(guess, expected)
                if i % 4 == 0:
                    np.testing.assert_array_equal(called[i//4][3], noise)
                    np.testing.assert_array_equal(called[i//4][4], expected)
            all_guesses.append(guesses)
            rng_states.append(np.random.get_state())
        np.testing.assert_array_equal(all_guesses[0], all_guesses[1])
        assert all(np.array_equal(a, b) for a, b in zip(*rng_states))
    finally:
        np.random.set_state(saved_rng)


def test_improvements_classified_by_restart_role_with_real_geometry_but_fake_solver():
    ns, _ = namespace()
    count = 0
    def fake(objective, guess, **kwargs):
        nonlocal count
        x = ns['initial_parameters']()
        x[52:77] = .02 + count * .005
        kwargs['callback'](x)
        count += 1
        return SimpleNamespace(x=x, success=True)
    ns['minimize'] = fake
    ns['solve_packing'](max_restarts=6)
    stats = ns['LAST_STATS']
    assert stats['improvements'] == 6
    assert stats['proposal_scheduled_improvements'] == 2
    assert stats['proposal_ordinary_improvements'] == 4
    assert stats['proposal_scheduled_gain']+stats['proposal_ordinary_gain'] == pytest.approx(stats['best_sum']-.25)


def test_helper_finishes_after_deadline_latest_feasible_proposal_survives_without_slsqp():
    ns, clock = namespace()
    def late(incumbent, initial, noise, default, index):
        x = initial.copy(); x[52:77] = .02
        clock.now = .05
        return x
    ns['propose_restart'] = late
    _, _, total = ns['solve_packing'](budget_seconds=.02)
    assert total == pytest.approx(.5)
    stats = ns['LAST_STATS']
    assert stats['deadline_reached'] and stats['started_starts'] == 0
    assert stats['proposal_attempts'] == stats['proposal_accepted'] == 1
    assert stats['proposal_scheduled_improvements'] == stats['improvements'] == 1
    assert stats['proposal_scheduled_starts'] == stats['proposal_scheduled_completed'] == 0


def test_partial_solver_start_role_survives_deadline():
    ns, clock = namespace()
    def fake(objective, guess, **kwargs):
        x = ns['initial_parameters'](); x[52:77] = .02
        clock.now = .05
        objective(x)  # cooperative guard retains x and raises before returning
        pytest.fail('expired fake solver should not return')
    ns['minimize'] = fake
    _, _, total = ns['solve_packing'](budget_seconds=.02)
    assert total == pytest.approx(.5)
    stats = ns['LAST_STATS']
    assert stats['proposal_scheduled_starts'] == 1 and stats['proposal_scheduled_completed'] == 0
    assert stats['proposal_scheduled_improvements'] == stats['improvements'] == 1
    assert stats['deadline_reached']


def test_common_publisher_retains_error_diagnostics_without_masking(monkeypatch, capsys):
    ns, _ = namespace()
    def fail(*args):
        raise ValueError('fixture failure')
    ns['propose_restart'] = fail
    with pytest.raises(ValueError, match='fixture failure'):
        ns['run_packing']()
    out = capsys.readouterr()
    assert out.out == ''
    lines = out.err.splitlines()
    assert len(lines) == 1 and lines[0].startswith(builder.STATS_MARKER)
    stats = json.loads(lines[0][len(builder.STATS_MARKER):])
    assert stats['proposal_errors'] == 1 and stats['stop_reason'] == 'error'
    assert stats['started_starts'] == 0 and stats['best_sum'] == pytest.approx(.25)


def test_zero_budget_invokes_neither_rng_helper_nor_solver(monkeypatch):
    ns, _ = namespace()
    def forbidden(*args, **kwargs):
        pytest.fail('zero budget cannot dispatch anything')
    monkeypatch.setattr(np.random, 'uniform', forbidden)
    ns['propose_restart'] = forbidden
    _, _, total = ns['solve_packing'](budget_seconds=0)
    assert total == pytest.approx(.25)
    assert ns['LAST_STATS']['proposal_attempts'] == ns['LAST_STATS']['started_starts'] == 0


def test_prompt_and_identity_are_detached_consistent_and_noncausal():
    a = builder.interface_description(); a['numpy_calls'].append('random')
    b = builder.interface_description()
    assert 'random' not in b['numpy_calls']
    instruction = builder.response_instruction()
    assert 'not a policy name' in instruction and 'bounded allocation' in instruction
    assert 'overlaps and out-of-bounds starts are allowed' in instruction
    identity = builder.identity()
    assert identity['interface_sha256'] == digest(json.dumps(b, sort_keys=True, separators=(',', ':')).encode())
    assert 'not causal effects' in identity['scope']
