"""ModeBandit (§6.3 step 2): warm-start, per-phase learning + isolation, window, state."""

from collections import Counter
from types import SimpleNamespace

from sciel.components.policy.mode_bandit import MODES, ModeBandit


def _ev(r):
    return SimpleNamespace(r_norm=r)


def test_warm_start_follows_base_mix_ordering():
    b = ModeBandit({"base": 0.8, "exploit": 0.2}, temp=0.5)
    d = b.dist(0)
    assert set(d) == set(MODES)
    assert abs(sum(d.values()) - 1.0) < 1e-9
    assert d["base"] > d["exploit"] > d["explore"]  # empty -> prior ordering


def test_learns_high_reward_mode():
    b = ModeBandit({"base": 0.5, "exploit": 0.5}, temp=0.3, prior_w=1.0)
    for _ in range(40):
        b.observe(0, "exploit", _ev(1.0))
        b.observe(0, "base", _ev(0.0))
    d = b.dist(0)
    assert d["exploit"] > d["base"]  # exploit shown to pay off in phase 0


def test_per_phase_isolation():
    b = ModeBandit({"base": 0.5, "explore": 0.5}, temp=0.3)
    for _ in range(40):
        b.observe(1, "explore", _ev(1.0))   # explore pays only in phase 1
    assert b.dist(1)["explore"] > b.dist(0)["explore"]  # phase 0 unaffected


def test_zero_prior_no_crash():
    # prior_w=0 (pure empirical, no warm-start) must not divide by zero on empty windows.
    b = ModeBandit({"base": 0.8, "exploit": 0.2}, prior_w=0.0)
    d = b.dist(0)
    assert abs(sum(d.values()) - 1.0) < 1e-9
    assert all(abs(d[m] - 1 / 3) < 1e-9 for m in MODES)  # no data -> uniform


def test_window_maxlen_enforced():
    b = ModeBandit({"base": 1.0}, window=5)
    for i in range(20):
        b.observe(0, "base", _ev(float(i)))
    assert len(b.events[(0, "base")]) == 5


def test_state_roundtrip():
    b = ModeBandit({"base": 0.7, "exploit": 0.3}, temp=0.4, window=10, prior_w=2.0)
    b.observe(0, "exploit", _ev(0.9))
    b.observe(-1, "base", _ev(0.1))   # negative phase code (final) round-trips
    d = b.state()
    b2 = ModeBandit({"base": 1.0})
    b2.load_state(d)
    assert b2.state() == d
    assert b2.dist(0) == b.dist(0)
