"""Exact source transformation and fake SLSQP integration; no packing search."""
import ast
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.hypothesis import circle_relocations as builder
from sciel.hypothesis import circle_restart_scales as scales

sha = lambda value: hashlib.sha256(value).hexdigest()


def test_exact_control_and_all_four_edits_reconstruct_parent_byte_for_byte():
    control = builder.source_for('control').encode()
    built = builder.build(); child = built['code'].encode(); proof = built['proof']
    assert control == scales.source_for('0.06').encode()
    assert sha(control) == builder.PINNED_BASELINE_SHA256
    assert sha(child) == proof['child_source_sha256']
    assert proof['helper_source_sha256'] == sha(builder._HELPER_PATH.read_bytes())
    assert [item['name'] for item in proof['edits']] == ['helper', 'stats', 'noise_assignment', 'initializer']
    restored = child
    for item in reversed(proof['edits']):
        a, b = item['child_byte_span']; c, d = item['parent_byte_span']
        assert child[a:b] == item['after'].encode() and sha(child[a:b]) == item['after_sha256']
        assert control[c:d] == item['before'].encode() and sha(control[c:d]) == item['before_sha256']
        restored = restored[:a] + item['before'].encode() + restored[b:]
    assert restored == control
    for item in proof['unchanged_segments']:
        a, b = item['parent_byte_span']; assert sha(control[a:b]) == item['sha256']
    assert child.count(builder._HELPER_PATH.read_bytes()) == 1
    assert builder.source_for('relocate') == built['code']


def test_unchanged_solver_and_clamps_and_one_identical_rng_call():
    control, treatment = [builder.source_for(p) for p in ('control', 'relocate')]
    def function(source, name):
        return next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
    for name in ('objective', 'objective_jac', 'packing_constraints', 'packing_jacobian',
                 'is_feasible', 'run_packing'):
        assert ast.dump(function(control, name)) == ast.dump(function(treatment, name))
    for name in ('check_deadline', 'callback', 'timed'):
        assert ast.dump(function(control, name)) == ast.dump(function(treatment, name))
    def call(source, name):
        return [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and ast.unparse(n.func) == name]
    for name in ('minimize', 'np.random.uniform'):
        a, b = call(control, name), call(treatment, name)
        assert len(a) == len(b) == 1 and ast.dump(a[0]) == ast.dump(b[0])
    for line in ('guess[2 * N_CIRCLES:] = np.maximum(guess[2 * N_CIRCLES:], 0.001)',
                 'guess[:2 * N_CIRCLES] = np.clip(guess[:2 * N_CIRCLES], 0.001, 0.999)'):
        assert control.count(line) == treatment.count(line) == 1
    assert 'from sciel' not in treatment
    identity = builder.identity()
    assert identity['restart_period'] == 4 and identity['first_restart_index'] == 0
    assert identity['candidate_points'] == 16
    assert identity['relocate_source_sha256'] == sha(treatment.encode())


@pytest.mark.parametrize('policy', [None, True, 1, 'lp', '0.06', 'Relocate', 'unknown'])
def test_unknown_policy_rejected(policy):
    with pytest.raises(ValueError): builder.source_for(policy)


def test_modified_helper_or_parent_cannot_build_family(tmp_path, monkeypatch):
    bad = tmp_path / 'helper.py'; bad.write_bytes(builder._HELPER_PATH.read_bytes() + b'# changed\n')
    monkeypatch.setattr(builder, '_HELPER_PATH', bad)
    with pytest.raises(ValueError, match='helper'): builder.source_for('relocate')
    monkeypatch.setattr(scales, 'source_for', lambda policy: 'changed source')
    with pytest.raises(ValueError, match='control'): builder.source_for('control')


class Clock:
    now = 0.
    def __call__(self): return self.now


def namespace(policy):
    # Definitions only; the actual minimize is replaced before any solve call.
    ns = {}; exec(compile(builder.source_for(policy), '<fake-only-relocation>', 'exec'), ns)
    clock = Clock(); ns['time'] = SimpleNamespace(monotonic=clock)
    if policy == 'relocate': ns['_relocation_time'] = SimpleNamespace(monotonic=clock)
    def forbidden(*args, **kwargs): pytest.fail('actual optimizer forbidden')
    ns['minimize'] = forbidden
    return ns, clock


def test_same_rng_sequence_full78_draws_and_only_indices_zero_four_eight_relocate():
    saved = np.random.get_state()
    try:
        starts, rngs, calls = {}, {}, []
        for policy in ('control', 'relocate'):
            ns, _ = namespace(policy); starts[policy] = []
            if policy == 'relocate':
                helper = ns['relocate_circle']
                def record_helper(base, noise, default, index, **kwargs):
                    calls.append((index, base.copy(), noise.copy(), default.copy()))
                    return helper(base, noise, default, index, **kwargs)
                ns['relocate_circle'] = record_helper
            def fake(fn, guess, **kwargs):
                starts[policy].append(guess.copy())
                assert kwargs['method'] == 'SLSQP'
                assert kwargs['options'] == dict(maxiter=15000, ftol=1e-7, disp=False)
                return SimpleNamespace(x=ns['initial_parameters'](), success=True)
            ns['minimize'] = fake
            np.random.seed(537); ns['solve_packing'](budget_seconds=27., max_restarts=9)
            rngs[policy] = np.random.get_state()
            assert ns['LAST_STATS']['started_starts'] == ns['LAST_STATS']['completed_starts'] == 9
        assert [row[0] for row in calls] == [0, 4, 8]
        assert all(np.array_equal(a, b) for a, b in zip(rngs['control'], rngs['relocate']))
        ns, _ = namespace('control'); initial = ns['initial_parameters']()
        rng = np.random.RandomState(537)
        for index in range(9):
            noise = rng.uniform(-.06, .06, (78,)); expected = initial + noise
            expected[52:] = np.maximum(expected[52:], .001)
            expected[:52] = np.clip(expected[:52], .001, .999)
            np.testing.assert_array_equal(starts['control'][index], expected)
            if index % 4:
                np.testing.assert_array_equal(starts['relocate'][index], expected)
            else:
                record = calls[index // 4]
                np.testing.assert_array_equal(record[1], initial)
                np.testing.assert_array_equal(record[2], noise)
                np.testing.assert_array_equal(record[3], expected)
    finally: np.random.set_state(saved)


def test_all_fallbacks_give_bit_identical_control_starts_and_draw_sequence():
    saved = np.random.get_state()
    try:
        all_starts = {}
        for policy in ('control', 'relocate'):
            ns, _ = namespace(policy); all_starts[policy] = []
            if policy == 'relocate':
                def fallback(base, noise, default, index, *, stats, **kwargs):
                    stats['relocation_attempts'] += 1; stats['relocation_fallbacks'] += 1
                    stats['relocation_fallback_reasons']['no_nonnegative_clearance'] = stats['relocation_fallbacks']
                    return default.copy()
                ns['relocate_circle'] = fallback
            def fake(fn, guess, **kwargs):
                all_starts[policy].append(guess.copy())
                return SimpleNamespace(x=ns['initial_parameters'](), success=True)
            ns['minimize'] = fake
            np.random.seed(539); ns['solve_packing'](budget_seconds=27., max_restarts=6)
        np.testing.assert_array_equal(all_starts['control'], all_starts['relocate'])
    finally: np.random.set_state(saved)


def test_feasible_relocation_after_deadline_survives_without_slsqp(monkeypatch):
    ns, clock = namespace('relocate'); helper = ns['relocate_circle']
    def draw(low, high, shape):
        assert (low, high, shape) == (-.06, .06, (78,))
        noise = np.zeros(78); noise[:32] = -.054; return noise
    monkeypatch.setattr(np.random, 'uniform', draw)
    def finish_late(*args, **kwargs):
        value = helper(*args, **kwargs); clock.now = .03; return value
    ns['relocate_circle'] = finish_late
    centers, radii, total = ns['solve_packing'](budget_seconds=.02, max_restarts=10000)
    assert total == pytest.approx(.29)
    assert ns['is_feasible'](np.concatenate((centers.ravel(), radii)))
    stats = ns['LAST_STATS']
    assert stats['relocation_attempts'] == stats['relocation_accepted'] == stats['improvements'] == 1
    assert stats['started_starts'] == stats['completed_starts'] == 0
    assert stats['deadline_reached'] and stats['stop_reason'] == 'deadline'


def test_bad_internal_noise_propagates_and_final_diagnostics_survive(monkeypatch):
    ns, _ = namespace('relocate')
    monkeypatch.setattr(np.random, 'uniform', lambda *args: np.full(78, .061))
    with pytest.raises(ValueError, match='original'): ns['solve_packing'](budget_seconds=1., max_restarts=1)
    stats = ns['LAST_STATS']
    assert stats['stop_reason'] == 'error' and stats['relocation_errors'] == 1
    assert stats['started_starts'] == 0 and stats['best_sum'] == pytest.approx(.25)


def test_zero_budget_does_not_draw_or_dispatch(monkeypatch):
    ns, _ = namespace('relocate')
    def forbidden(*args, **kwargs): pytest.fail('zero budget cannot draw')
    monkeypatch.setattr(np.random, 'uniform', forbidden)
    _, _, total = ns['solve_packing'](budget_seconds=0., max_restarts=10000)
    assert total == pytest.approx(.25)
    assert all(ns['LAST_STATS'][key] == 0 for key in
        ('relocation_attempts', 'relocation_accepted', 'relocation_fallbacks',
         'relocation_deadline_skips', 'relocation_errors', 'relocation_time_seconds'))
    assert ns['LAST_STATS']['relocation_fallback_reasons'] == {}
