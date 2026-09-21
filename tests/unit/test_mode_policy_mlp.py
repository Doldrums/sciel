"""Stage-A return-conditioned mode MLP: valid dist, return-conditioning, artifact roundtrip."""

from sciel.components.policy.mode_features import MODES, mode_features
from sciel.components.policy.mode_policy_mlp import ReturnConditionedModeMLP


def _dataset():
    # Learnable pattern: high RTG -> exploit was taken; low RTG -> base.
    ctx = mode_features(phase=0, progress=0.3, v_t=0.4, island_best=0.4)
    samples = []
    for _ in range(150):
        samples.append({"features": ctx, "rtg": 1.0, "mode": "exploit"})
        samples.append({"features": ctx, "rtg": 0.0, "mode": "base"})
    return ctx, samples


def test_dist_is_valid_distribution():
    ctx, samples = _dataset()
    m = ReturnConditionedModeMLP(seed=0).fit(samples, epochs=300)
    d = m.dist(ctx, 1.0)
    assert set(d) == set(MODES)
    assert abs(sum(d.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in d.values())


def test_return_conditioning_shifts_distribution():
    ctx, samples = _dataset()
    m = ReturnConditionedModeMLP(seed=0).fit(samples, epochs=400)
    hi, lo = m.dist(ctx, 1.0), m.dist(ctx, 0.0)
    assert hi["exploit"] > lo["exploit"]   # high target-RTG recalls the exploit action
    assert lo["base"] > hi["base"]


def test_artifact_roundtrip(tmp_path):
    ctx, samples = _dataset()
    m = ReturnConditionedModeMLP(seed=1).fit(samples, epochs=200)
    p = str(tmp_path / "m.npz")
    m.save(p)
    m2 = ReturnConditionedModeMLP.load(p)
    d1, d2 = m.dist(ctx, 0.7), m2.dist(ctx, 0.7)
    assert all(abs(d1[k] - d2[k]) < 1e-9 for k in MODES)
