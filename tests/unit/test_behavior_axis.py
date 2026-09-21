"""The fitted behavior axis and the behavior readout it is fed from.

Covers the two defects the axis work was done to fix:
  1. evaluator readouts that the featurizer silently dropped because the evaluator
     spelled the key differently (`value_score` vs `value`);
  2. a descriptor that projected on a random direction WITHOUT standardizing, so
     tanh stayed in its linear region and the grid collapsed to two bins.
"""

import numpy as np
import pytest

from sciel.components.policy.code_embed import (
    BEHAVIOR_EXTRA_KEYS, BEHAVIOR_KEYS, BehaviorAxis, DIM_HASH, behavior_value,
    featurize)

torch = pytest.importorskip("torch")

CODE = "import numpy as np\ndef run():\n    return 1\n"
V2_IN_DIM = len(BEHAVIOR_KEYS) + DIM_HASH + 6        # what code_embed_v2 was trained on


# --------------------------------------------------------------- behavior keys
def test_alias_resolves_the_evaluator_spelling():
    # function_minimization emits *_score; before the alias table these read 0.0
    assert behavior_value({"value_score": 0.8}, "value") == pytest.approx(0.8)
    assert behavior_value({"distance_score": 0.3}, "distance") == pytest.approx(0.3)
    assert behavior_value({"reliability_score": 1.0}, "reliability") == pytest.approx(1.0)


def test_canonical_name_wins_over_alias():
    assert behavior_value({"value": 0.1, "value_score": 0.9}, "value") == pytest.approx(0.1)


def test_missing_and_unparseable_read_zero():
    assert behavior_value({}, "value") == 0.0
    assert behavior_value({"value": "nope"}, "value") == 0.0
    assert behavior_value(None, "value") == 0.0


# --------------------------------------------------------------- feature layout
def test_extra_behavior_is_appended_at_the_tail():
    """The block MUST be last: an older embedder is fed by truncation, and anything
    inserted earlier would shift every token-hash bin it was trained on."""
    plain = featurize(CODE, {"combined_score": 0.5})
    withb = featurize(CODE, {"combined_score": 0.5, "b_dispersion": 0.7})
    assert len(plain) == V2_IN_DIM + len(BEHAVIOR_EXTRA_KEYS)
    assert (plain[:V2_IN_DIM] == withb[:V2_IN_DIM]).all()      # head untouched
    assert withb[V2_IN_DIM] == pytest.approx(0.7)              # first tail slot


def test_featurize_is_deterministic():
    assert (featurize(CODE, {"validity": 1.0}) == featurize(CODE, {"validity": 1.0})).all()


def test_older_embedder_is_fed_by_truncation_not_padding():
    from sciel.components.policy.code_embed import Embedder, embed_batch
    narrow = Embedder(V2_IN_DIM)
    z = embed_batch(narrow, [(CODE, {"b_dispersion": 0.9})])
    assert z.shape == (1, 64)
    wide = Embedder(V2_IN_DIM + len(BEHAVIOR_EXTRA_KEYS) + 1)
    with pytest.raises(ValueError, match="newer than this code"):
        embed_batch(wide, [(CODE, {})])


# --------------------------------------------------------------- the axis itself
def _axis(dim=64, component=1):
    rng = np.random.default_rng(0)
    a = rng.standard_normal(dim)
    return BehaviorAxis(a, np.zeros(dim), center=0.0, scale=1.0, component=component,
                        version="test", embed_sha="deadbeef", corr_score=-0.28,
                        explained=0.2, n_fit=1000)


def test_axis_value_is_bounded_and_centred():
    ax = _axis()
    assert ax.value(np.zeros(64)) == pytest.approx(0.5)
    for scale in (1e-3, 1.0, 1e3):
        ax.scale = scale
        v = ax.value(np.ones(64))
        assert 0.0 <= v <= 1.0


def test_axis_rejects_degenerate_parameters():
    with pytest.raises(ValueError):
        BehaviorAxis(np.zeros(64), np.zeros(64))                 # zero direction
    with pytest.raises(ValueError):
        BehaviorAxis(np.ones(64), np.zeros(64), scale=0.0)       # zero scale
    with pytest.raises(ValueError):
        BehaviorAxis(np.ones(64), np.zeros(32))                  # shape disagreement


def test_axis_roundtrips_through_disk(tmp_path):
    ax = _axis()
    p = str(tmp_path / "axis.npz")
    ax.save(p)
    back = BehaviorAxis.load(p)
    assert back.component == ax.component
    assert back.version == "test"
    assert back.embed_sha == "deadbeef"
    assert back.n_fit == 1000
    z = np.random.default_rng(1).standard_normal(64)
    assert back.value(z) == pytest.approx(ax.value(z))


def test_wrong_format_is_refused(tmp_path):
    p = str(tmp_path / "bad.npz")
    np.savez(p, fmt=np.array("something/else"))
    with pytest.raises(ValueError, match="behavior_axis/1"):
        BehaviorAxis.load(p)


# --------------------------------------------------------------- spread, the point
def _occupancy(values, bins=10):
    idx = np.clip((np.asarray(values) * bins).astype(int), 0, bins - 1)
    return np.bincount(idx, minlength=bins) / len(idx)


def test_standardizing_is_what_spreads_the_descriptor():
    """The shipped bug in one test. At the measured projection scale (std 0.065) a
    raw tanh is a straight line, so essentially all mass piles into the two bins
    either side of 0.5. Dividing by the projection's own std is the entire fix."""
    rng = np.random.default_rng(7)
    u = rng.standard_normal(4000) * 0.065               # measured projection scale
    raw = _occupancy((np.tanh(u) + 1.0) / 2.0)
    std = _occupancy((np.tanh(u / u.std()) + 1.0) / 2.0)
    assert raw[4] + raw[5] > 0.99                       # two bins hold everything
    assert (raw > 0.01).sum() == 2
    assert (std > 0.01).sum() >= 8                      # spread across the axis
    assert std.max() < 0.35


def test_axis_uses_the_fitted_scale_not_the_raw_projection():
    """A BehaviorAxis fitted on a narrow projection must still fill the range."""
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((3000, 64)) * 0.02
    direction = rng.standard_normal(64)
    proj = (Z - Z.mean(0)) @ (direction / np.linalg.norm(direction))
    ax = BehaviorAxis(direction, Z.mean(0), center=float(proj.mean()),
                      scale=float(proj.std()), component=1, version="t")
    occ = _occupancy([ax.value(z) for z in Z])
    assert (occ > 0.01).sum() >= 8


# --------------------------------------------------------------- descriptor wiring
@pytest.fixture
def authored_weights(tmp_path):
    """Known feature routing and disk axis metadata, never campaign weights."""
    from sciel.components.policy.code_embed import Embedder, file_sha
    embed = tmp_path / "authored-embedder.pt"
    axis_dir = tmp_path / "authored-axes"
    axis_dir.mkdir()
    model = Embedder(len(featurize(CODE, {})))
    # The first latent coordinate receives b_dispersion; the second is a fixed
    # nonzero reference so normalization cannot erase the behavior difference.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.trunk[0].weight[0, V2_IN_DIM] = 1.
        model.proj.weight[0, 0] = 1.
        model.proj.bias[1] = 1.
    torch.save({"in_dim": model.in_dim, "state": model.state_dict()}, embed)
    for task in ("oe/circle_packing", "oe/heilbronn_triangle", "oe/function_minimization"):
        BehaviorAxis(np.eye(64)[0], np.zeros(64), center=.1, scale=.2,
            component=0, version="authored", embed_sha=file_sha(embed),
            fit_task=task, corr_score=.1, corr_worst=.2, corr_worst_task=task,
            explained=.3, n_fit=80).save(axis_dir / (task.replace('/', '_') + '.npz'))
    return embed, axis_dir


def test_missing_configured_axis_is_not_a_legacy_projection(tmp_path):
    from test_loaded_descriptor_identity import artifacts
    from sciel.components.policy.code_embed import LearnedDescriptor
    embed, _, _, _ = artifacts(tmp_path)
    with pytest.raises(FileNotFoundError):
        LearnedDescriptor.load(str(embed), axis_path=str(tmp_path / "absent.npz"))


def test_explicit_null_axis_selects_legacy_and_says_so(tmp_path):
    from test_loaded_descriptor_identity import artifacts
    from sciel.components.policy.code_embed import LearnedDescriptor
    embed, _, _, _ = artifacts(tmp_path)
    d = LearnedDescriptor.load(str(embed), axis_path=None)
    assert d.kind == "legacy_random_raw"
    info = d.describe()
    assert info["kind"] == "legacy_random_raw"
    assert set(info) == {"kind", "behavior_in_latent"}, info
    assert isinstance(info["behavior_in_latent"], bool)


def test_the_configured_embedder_receives_the_behavior_block(authored_weights):
    """An encoder trained with appended behavior inputs must receive that tail."""
    import numpy as np
    from sciel.components.policy.code_embed import Embedder, embed_batch
    embed, _ = authored_weights
    m = Embedder.load(str(embed))
    cases = {"combined_score": 0.5, "b_dispersion": 0.7, "b_contact": 0.3}
    stripped = {k: v for k, v in cases.items() if not k.startswith("b_")}
    a = embed_batch(m, [(CODE, cases)])[0]
    b = embed_batch(m, [(CODE, stripped)])[0]
    np.testing.assert_allclose(a[:2], np.array([.7, 1.]) / np.linalg.norm([.7, 1.]), rtol=1e-6)
    np.testing.assert_allclose(b[:2], [0., 1.], rtol=1e-6)
    moved = float(np.linalg.norm(a - b))
    assert moved > 0.01, (
        f"the behaviour block does not reach this latent (moved {moved:.4f}); "
        f"in_dim is {m.in_dim} and the featurizer produces "
        f"{len(featurize(CODE, cases))}")


def test_axis_fitted_against_another_embedder_is_refused(tmp_path, authored_weights):
    """Silently ignoring a mismatched artifact is how a broken descriptor survives a
    campaign. It has to raise."""
    from sciel.components.policy.code_embed import LearnedDescriptor
    embed, _ = authored_weights
    bad = tmp_path / "axis.npz"
    ax = _axis()
    ax.embed_sha = "0000deadbeef"
    ax.save(str(bad))
    with pytest.raises(ValueError, match="refit the axis"):
        LearnedDescriptor.load(str(embed), axis_path=str(bad))


def test_authored_axes_preserve_per_task_binding_and_metadata(authored_weights):
    """Disk loading preserves the declared per-task fit and exact encoder binding."""
    from pathlib import Path
    from sciel.components.policy.code_embed import LearnedDescriptor, file_sha
    embed, axis_dir = authored_weights
    axes = sorted(Path(axis_dir).glob("*.npz")) if Path(axis_dir).is_dir() else []
    assert len(axes) == 3
    sha = file_sha(str(embed))
    for path in axes:
        d = LearnedDescriptor.load(str(embed), axis_path=str(path))
        assert d.kind.startswith("pca"), path
        assert d.axis.embed_sha == sha, f"{path} is bound to a different embedder"
        assert path.stem == d.axis.fit_task.replace("/", "_")
        assert d.axis.corr_worst_task == d.axis.fit_task
        assert abs(d.axis.corr_worst) < 0.35, f"{path}: {d.axis.corr_worst}"
        assert d.axis.dim == 64


def test_descriptor_metadata_is_valid_json(authored_weights):
    """describe() lands in run_start. float('nan') serialises as bare NaN, which is not
    JSON and breaks any strict reader of the journal."""
    import json
    from sciel.components.policy.code_embed import Embedder, LearnedDescriptor
    embed, _ = authored_weights
    d = LearnedDescriptor(Embedder.load(str(embed)), axis=_axis())   # NaN metadata
    text = json.dumps(d.describe())
    assert "NaN" not in text and "Infinity" not in text, text
    # corr_worst is the field _axis() leaves at its NaN default
    assert json.loads(text)["corr_worst"] is None


def test_axis_of_the_wrong_width_is_refused_at_load(tmp_path, authored_weights):
    """A 32-d axis against a 64-d latent used to blow up inside db.add, tens of
    evaluations into the run, as a broadcast error."""
    from sciel.components.policy.code_embed import LearnedDescriptor, file_sha
    embed, _ = authored_weights
    narrow = _axis(dim=32)
    narrow.embed_sha = file_sha(str(embed))
    p = tmp_path / "narrow.npz"
    narrow.save(str(p))
    with pytest.raises(ValueError, match="dimensional"):
        LearnedDescriptor.load(str(embed), axis_path=str(p))


def test_axis_with_no_embedder_binding_is_refused(tmp_path, authored_weights):
    from sciel.components.policy.code_embed import LearnedDescriptor
    embed, _ = authored_weights
    loose = _axis()
    loose.embed_sha = ""
    p = tmp_path / "loose.npz"
    loose.save(str(p))
    with pytest.raises(ValueError, match="no embedder hash"):
        LearnedDescriptor.load(str(embed), axis_path=str(p))
