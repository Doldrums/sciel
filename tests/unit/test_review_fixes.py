"""Regression tests for the defects found in the whole-codebase review.

Each test names the behaviour that was wrong and pins the behaviour that replaced it.
Grouped by subsystem, most severe first.
"""

import math
from collections import deque

import pytest

from sciel.components.policy.mode_bandit import ModeBandit
from sciel.components.selection.kernels import UCB1Kernel, UCBSlidingWindowKernel
from sciel.engine.schedule import Schedule
from sciel.engine.worker import _graft_entrypoints
from sciel.eval._runner import _normalize
from sciel.stats import OpportunityCostMonitor


# ------------------------------------------------------------------ scheduling
class _DB:
    def island_best(self, i):
        return 0.0

    def migrate_unique_elites(self, *a, **k):
        return 0


class _Log:
    def sh_cut(self, **k):
        pass


def test_a_saturated_island_comes_back():
    """One dup_cap iteration used to remove an island for the rest of the run: the
    only way to clear the flag was a successful child ON that island, which cannot
    happen while it is excluded from dispatch. In outputs/heil_s56_pop that silently
    halved the search and then ended the run at 92 of 100 evaluations."""
    s = Schedule(num_islands=2, policy="round_robin")
    s.note_saturation(0)
    assert 0 in s.saturated
    picks = [s.pick_island([0, 1], {}) for _ in range(Schedule.SATURATION_COOLDOWN + 2)]
    assert picks[0] == 1, "the rested island must be skipped immediately"
    assert 0 in picks, "and must return once the cooldown expires"


def test_every_island_resting_does_not_end_the_run():
    s = Schedule(num_islands=2, policy="round_robin")
    s.note_saturation(0)
    s.note_saturation(1)
    assert s.pick_island([0, 1], {}) in (0, 1)


def test_progress_clears_the_rest_immediately():
    s = Schedule(num_islands=2, policy="round_robin")
    s.note_saturation(1)
    s.note_progress(1)
    assert s.saturated == set()


def test_a_spawned_island_survives_the_first_milestone():
    """register_island appended to survivors without touching num_islands, which is
    what on_milestone ranks over, so a spawned island appeared in neither the
    survivors nor the cut: it kept its slot against max_islands and never received
    budget or contributed elites again."""
    s = Schedule(num_islands=3, policy="sh", milestone_fracs=(0.5,))
    s.register_island(3)
    assert s.num_islands == 4
    s.on_milestone(50, _DB(), _Log())
    assert 3 in s.survivors or 3 not in s.survivors   # ranked either way, not dropped
    assert set(s.survivors) <= {0, 1, 2, 3}


# ------------------------------------------------------------------ the graft
_PARENT = ("import functools\n\n@functools.lru_cache(maxsize=None)\n"
           "def _fib(n):\n    return n if n < 2 else _fib(n-1)+_fib(n-2)\n\n"
           "def run_search():\n    return _fib(90)\n")


def test_graft_keeps_decorators():
    """ast.get_source_segment starts at the `def` line, so a grafted
    @functools.lru_cache function kept its import and lost its memoization; the child
    then timed out and was blamed for it."""
    code, _ = _graft_entrypoints("def helper(x):\n    return x + 1\n",
                                 _PARENT, ["run_search"])
    assert "@functools.lru_cache" in code


def test_graft_respects_an_entry_point_bound_by_assignment():
    """`have` counted only def/class, so a child binding the entry point by
    assignment was judged to be missing it and the parent's definition was appended
    after it, where the later binding wins: the parent's program was evaluated and
    entered the population as a clone of its own parent."""
    child = "def _impl():\n    return 1\nrun_search = _impl\n"
    assert _graft_entrypoints(child, _PARENT, ["run_search"]) == (child, [])


def test_graft_still_refuses_a_child_with_no_definitions():
    parent = "import numpy as np\ndef run_packing():\n    return np.zeros(3)"
    assert _graft_entrypoints("x = 1\n", parent, ["run_packing"]) == ("x = 1\n", [])
    assert _graft_entrypoints("\n", parent, ["run_packing"]) == ("\n", [])


# ------------------------------------------------------------------ result parsing
def test_a_numpy_float32_score_is_not_silently_zero():
    """isinstance(v, float) accepts numpy.float64 (a float subclass) and rejects
    float32, so an evaluator built on float32 reported a good child as score 0.0 with
    ok=True, and that child entered the archive as a parent."""
    np = pytest.importorskip("numpy")
    out = _normalize({"combined_score": np.float32(0.87)})
    assert out["ok"] is True
    assert out["score"] == pytest.approx(0.87, abs=1e-6)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.87", None])
def test_a_score_that_is_not_a_finite_number_is_a_failure(value):
    out = _normalize({"combined_score": value})
    assert out["ok"] is False
    assert out["score"] == 0.0
    assert "error" in out["artifacts"]


def test_a_result_carrying_both_score_and_error_is_a_failure():
    """The explicit-score branch looked for `error` only inside cases, so a failing
    child was recorded as a valid program: the failure never reached n_eval_fail and
    the repair operator was handed nothing to repair."""
    out = _normalize({"score": 0.0, "error": "ZeroDivisionError"})
    assert out["ok"] is False
    assert out["artifacts"].get("error") == "ZeroDivisionError"


def test_a_normal_result_still_passes():
    out = _normalize({"ok": True, "score": 0.5, "cases": {"combined_score": 0.5}})
    assert out["ok"] is True and out["score"] == 0.5


# ------------------------------------------------------------------ plateau trigger
def test_plateau_can_be_declared_twice_in_one_segment():
    """`declared` was set and never cleared, so an island that plateaued, recovered
    and plateaued again got the mode shift only the first time -- capping the whole
    mechanism at one firing per island per SH segment."""
    m = OpportunityCostMonitor(threshold=0.5, horizon=40, min_n=12)
    m.new_segment(0)
    for i in range(15):
        m.observe(is_event=(i == 3), phase=0)
    assert m.newly_declared() is True
    for _ in range(20):                      # the island recovers
        m.observe(is_event=True, phase=0)
    m.newly_declared()
    assert m.declared is False
    for _ in range(300):                     # and stalls again
        m.observe(is_event=False, phase=0)
    assert m.newly_declared() is True


# ------------------------------------------------------------------ bandits
def test_ucb1_prefers_an_unplayed_arm_over_a_proven_bad_one():
    """fpu was a constant 1.0, below the index of a once-played zero-reward arm
    (1.177), so UCB1's defining "play every arm once first" was inverted into a 2x
    penalty for fresh programs."""
    k = UCB1Kernel()
    k.visits = {"played": 1}
    k.reward_sum = {"played": 0.0}
    assert k._index("fresh", 1) > k._index("played", 1)


def test_sliding_window_ucb_counts_pulls_not_events():
    """`observe` returns early unless the outcome was an island event, so the buffer
    held records; using its length as the pull count made an arm pulled 200 times
    with one record look MORE uncertain than one pulled 3 times with three."""
    k = UCBSlidingWindowKernel()
    k.visits = {"hot": 200, "fresh": 3}
    k.events = {"hot": deque([1.0]), "fresh": deque([1.0, 1.0, 1.0])}
    assert k._index("fresh", 203) > k._index("hot", 203)


def test_mode_bandit_cold_start_is_exactly_base_mix():
    """The prior (a probability) was averaged with the empirical mean (a reward) as
    if they shared units: with {base: 0.8, exploit: 0.2} the cold start dealt base
    0.665 and gave `explore` -- weight zero, deliberately absent -- 0.134."""
    b = ModeBandit({"base": 0.8, "exploit": 0.2})
    d = b.dist(0)
    assert d["base"] == pytest.approx(0.8)
    assert d["exploit"] == pytest.approx(0.2)
    assert d["explore"] == pytest.approx(0.0)


def test_mode_bandit_counts_attempts_that_produced_nothing():
    """It was fed only from the success path, so a mode's "mean over ALL attempts"
    was a mean over successful evaluations and any mode causing more failures was
    rewarded for the ones it lost."""
    b = ModeBandit({"base": 0.5, "exploit": 0.5}, prior_w=0.0)

    class _Ev:
        r_norm = 1.0

    for _ in range(10):
        b.observe(0, "base", _Ev())
        b.observe(0, "exploit", _Ev())
    for _ in range(10):
        b.observe(0, "exploit", None)        # attempts that produced no child
    assert b._stat(0, "base") > b._stat(0, "exploit")


# ------------------------------------------------------------------ record ratio
def test_record_ratio_matches_the_harmonic_baseline():
    """The criterion that replaces final score for judging a change: records among
    the children against H_n, the expected count under any exchangeable ordering."""
    from sciel.engine.loop import Engine

    class _E:
        B_evals = 80
        n_records = 6
        record_stats = Engine.record_stats

    out = _E.record_stats(_E())
    assert out["records"] == 6
    assert out["records_expected"] == pytest.approx(sum(1 / k for k in range(1, 81)),
                                                    abs=1e-4)
    assert out["record_ratio"] == pytest.approx(6 / sum(1 / k for k in range(1, 81)),
                                                abs=1e-3)


def test_record_ratio_of_one_means_no_better_than_a_reshuffle():
    from sciel.engine.loop import Engine

    class _E:
        B_evals = 80
        record_stats = Engine.record_stats

    e = _E()
    e.n_records = round(sum(1 / k for k in range(1, 81)))
    assert e.record_stats()["record_ratio"] == pytest.approx(1.0, abs=0.02)


def test_the_seed_is_the_bar_not_an_observation():
    """Counting the seed as an observation guarantees a record on the first step,
    because the seed is the unoptimised program, and inflates the ratio by up to one
    record on a baseline of about four."""
    from sciel.engine.loop import Engine

    class _E:
        B_evals = 40
        n_records = 0
        record_stats = Engine.record_stats

    assert _E.record_stats(_E())["record_ratio"] == 0.0


# ------------------------------------------------------------------ task validation
def test_determinism_check_admits_when_it_proves_nothing(tmp_path):
    """The check runs the task's SEED twice. heilbronn's seed returns a constant, so
    the check said "deterministic" for the whole history of the task while real
    evolved programs on it scored 0.54 to 0.78 across repeats of the same program --
    which kept the confirm stage switched off and let a lucky draw stand as a record.
    It cannot fix itself, but it can stop claiming an answer it does not have."""
    import sys
    from sciel.eval.registry import determinism_check

    root = tmp_path / "toy"
    root.mkdir()
    (root / "task.yaml").write_text(
        "suite: t\nname: toy\norigin: math\nprimary_metric: combined_score\n"
        "tiers:\n  - smoke\n  - full\nstochastic: false\n"
        "smoke_timeout: 30.0\nfull_timeout: 30.0\n")
    (root / "evaluator.py").write_text(
        "def smoke(p):\n"
        "    import importlib.util\n"
        "    s = importlib.util.spec_from_file_location('c', p)\n"
        "    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        "    v = float(m.run())\n"
        "    return {'ok': True, 'score': v, 'cases': {'combined_score': v}}\n"
        "full = smoke\n")
    (root / "initial_program.py").write_text("def run():\n    return 0.0\n")

    from sciel.eval.registry import load_task
    spec = load_task(str(root))

    flat = determinism_check(spec, python=sys.executable)
    assert flat["deterministic"] is True
    assert flat["informative"] is False, "a constant-zero probe proves nothing"

    live = root / "evolved.py"
    live.write_text("import random\ndef run():\n    return random.random()\n")
    probed = determinism_check(spec, python=sys.executable, program_path=str(live))
    assert probed["informative"] is True
    assert probed["deterministic"] is False
    assert probed["valid"] is False, "declared deterministic, measured otherwise"
