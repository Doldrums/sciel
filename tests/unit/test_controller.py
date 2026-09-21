"""pi Controller (§5.13): mode-mix draw, tilt targets, non-tilt guard, state roundtrip."""

import random
from collections import Counter

import pytest

from sciel.components.policy.controller import Controller
from sciel.components.selection.novelty_chain import IslandView, NoveltyChainKernel


def _view():
    ids = [f"p{i}" for i in range(5)]
    scores = {pid: 0.1 * i for i, pid in enumerate(ids)}  # p4 best, p3 next
    return IslandView(ids=ids, scores=scores, inspiration_ids={})


def test_disabled_does_not_consume_rng():
    c = Controller({"enabled": False}, num_islands=3)
    rng = random.Random(0)
    before = rng.getstate()
    mode, prop, tilt, _ = c.pick_mode(0, 0, rng)
    assert (mode, prop, tilt) == ("base", 1.0, None)
    assert rng.getstate() == before


def test_mode_mix_frequencies():
    c = Controller({"enabled": True, "base_mix": {"base": 0.8, "exploit": 0.2}},
                   num_islands=1)
    rng = random.Random(1)
    modes = Counter(c.pick_mode(0, 0, rng)[0] for _ in range(2000))
    assert 0.15 < modes["exploit"] / 2000 < 0.25


def test_exploit_tilt_targets_elite():
    c = Controller({"enabled": True, "base_mix": {"exploit": 1.0},
                    "tilt_beta": 1.0, "elite_k": 2}, num_islands=1)
    mode, prop, tilt, _ = c.pick_mode(0, 0, random.Random(0))
    assert mode == "exploit"
    assert set(tilt(_view())) == {"p3", "p4"}
    kernel = NoveltyChainKernel(floor=1e-3)
    picks = {kernel.select(_view(), tilt, random.Random(s)).parent_id for s in range(30)}
    assert picks <= {"p3", "p4"}


def test_exploit_warmup_gate():
    # Force exploit every draw; with no score gradient the tilt must be dropped.
    c = Controller({"enabled": True, "base_mix": {"exploit": 1.0}, "tilt_beta": 1.0},
                   num_islands=1)
    flat = IslandView(ids=["p0", "p1", "p2"], scores={"p0": 0.0, "p1": 0.0, "p2": 0.0},
                      inspiration_ids={})
    mode, _, tilt, _ = c.pick_mode(0, 0, random.Random(0), view=flat)
    assert mode == "exploit" and tilt is None  # gated: nothing to exploit
    # Epsilon "elite" (degenerate candidate at ~1e-16) is still not exploitable.
    eps = IslandView(ids=["p0", "p1", "p2"],
                     scores={"p0": 8.77e-16, "p1": 0.0, "p2": 0.0}, inspiration_ids={})
    mode, _, tilt, _ = c.pick_mode(0, 0, random.Random(0), view=eps)
    assert mode == "exploit" and tilt is None  # gated: spread below _GRAD_EPS
    # A real gradient -> tilt applies.
    mode, _, tilt, _ = c.pick_mode(0, 0, random.Random(0), view=_view())
    assert mode == "exploit" and tilt is not None
    # No view passed (e.g. legacy call) -> gate inert, tilt applies.
    mode, _, tilt, _ = c.pick_mode(0, 0, random.Random(0))
    assert mode == "exploit" and tilt is not None


def test_needs_tilt_gate():
    assert Controller({"enabled": True, "base_mix": {"base": 0.8, "exploit": 0.2}},
                      num_islands=1).needs_tilt() is True
    assert Controller({"enabled": True, "base_mix": {"base": 1.0}},
                      num_islands=1).needs_tilt() is False
    assert Controller({"enabled": False, "base_mix": {"exploit": 1.0}},
                      num_islands=1).needs_tilt() is False


def test_mode_bandit_learns_and_replaces_static_mix():
    from types import SimpleNamespace
    c = Controller({"enabled": True, "base_mix": {"base": 0.5, "exploit": 0.5},
                    "mode_bandit": True}, num_islands=1)
    assert c.mode_bandit is not None and c.needs_tilt() is True
    for _ in range(60):  # teach: exploit pays, base does not (phase 0)
        c.observe(0, "exploit", SimpleNamespace(r_norm=1.0))
        c.observe(0, "base", SimpleNamespace(r_norm=0.0))
    modes = Counter(c.pick_mode(0, 0, random.Random(s), view=_view(), phase=0)[0]
                    for s in range(500))
    assert modes["exploit"] > modes["base"]  # learned mix favours exploit


def test_mode_source_mlp(tmp_path):
    from sciel.components.policy.mode_features import mode_features
    from sciel.components.policy.mode_bandit import MODES
    from sciel.components.policy.mode_policy_mlp import ReturnConditionedModeMLP
    ctx = mode_features(phase=0, progress=0.3, v_t=0.4, island_best=0.4)
    samples = ([{"features": ctx, "rtg": 1.0, "mode": "exploit"}] * 100
               + [{"features": ctx, "rtg": 0.0, "mode": "base"}] * 100)
    p = str(tmp_path / "m.npz")
    ReturnConditionedModeMLP(seed=0).fit(samples, epochs=200).save(p)
    c = Controller({"enabled": True, "base_mix": {"base": 0.8, "exploit": 0.2},
                    "mode_source": "mlp", "mode_policy_path": p, "mode_beta": 1.0,
                    "mode_target_rtg": 1.0}, num_islands=1)
    assert c.mode_mlp is not None and c.needs_tilt() is True
    mode, _, _, mix = c.pick_mode(0, 0, random.Random(0), view=_view(), phase=0,
                                  features=ctx)
    assert set(mix) == set(MODES) and abs(sum(mix.values()) - 1.0) < 1e-6
    assert mode in MODES


def test_mode_bandit_off_is_static():
    c = Controller({"enabled": True, "base_mix": {"base": 1.0}}, num_islands=1)
    assert c.mode_bandit is None
    mode, _, _, _ = c.pick_mode(0, 0, random.Random(0), phase=0)
    assert mode == "base"


def test_unknown_mode_and_key_rejected():
    with pytest.raises(ValueError):
        Controller({"enabled": True, "base_mix": {"nonsense": 1.0}}, num_islands=1)
    with pytest.raises(ValueError):
        Controller({"enabled": True, "bogus_key": 1}, num_islands=1)


def test_plateau_shift_gated():
    c = Controller({"enabled": True, "base_mix": {"base": 1.0}, "plateau_hold": 10},
                   num_islands=2)
    c.on_plateau(0, t=5)  # plateau_shift not set -> inert
    assert c.in_plateau(0, t=6) is False
    c2 = Controller({"enabled": True, "base_mix": {"base": 1.0},
                     "plateau_shift": True, "plateau_hold": 10}, num_islands=2)
    c2.on_plateau(0, t=5)
    assert c2.in_plateau(0, t=6) is True
    assert c2.in_plateau(0, t=20) is False


def test_state_roundtrip():
    c = Controller({"enabled": True, "base_mix": {"base": 0.5, "exploit": 0.5},
                    "plateau_shift": True, "plateau_hold": 10}, num_islands=2)
    c.on_plateau(1, t=5)
    d = c.state()
    c2 = Controller({"enabled": True, "base_mix": {"base": 0.5, "exploit": 0.5}},
                    num_islands=2)
    c2.load_state(d)
    assert c2.in_plateau(1, t=10) is True
    assert c2.in_plateau(0, t=10) is False
