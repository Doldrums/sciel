"""arch-v4 #4c: shared three-head trunk (promise/eviction/niche) + eviction logging."""

import numpy as np
import pytest

from sciel.engine.factory import build_engine
from sciel.engine.llm.mock import MockLLM

import conftest

torch = pytest.importorskip("torch")
from sciel.components.policy.three_head import ThreeHeadModel   # noqa: E402


def _sets(n_sets=4, n=6, in_dim=8, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, in_dim)).astype(np.float32) for _ in range(n_sets)]


def test_predict_shapes_and_ranges():
    m = ThreeHeadModel(in_dim=8, n_niche=5, d=16, heads=2, blocks=1)
    out = m.predict(_sets(1, n=6)[0])
    assert out["promise"].shape == (6,) and out["eviction"].shape == (6,)
    assert out["niche"].shape == (6,)
    assert out["niche"].min() >= 0 and out["niche"].max() < 5


def test_fit_on_mixed_labels_moves_predictions():
    m = ThreeHeadModel(in_dim=8, n_niche=4, d=16, heads=2, blocks=1)
    sets = _sets(4, n=6)
    # one set carries only promise labels, another only niche -> each head still trains
    labels = [
        {"promise": np.linspace(0, 1, 6)},
        {"niche": np.array([0, 1, 2, 3, 0, 1])},
        {"eviction": np.array([1, 1, 0, 0, 1, 0])},
        {"promise": np.linspace(1, 0, 6), "niche": np.array([1, 1, 1, 2, 2, 2])},
    ]
    before = m.predict(sets[0])["promise"].copy()
    m.fit(sets, labels, epochs=40, lr=1e-2)
    after = m.predict(sets[0])["promise"]
    assert not np.allclose(before, after)          # training changed the promise head


def test_save_load_round_trip(tmp_path):
    m = ThreeHeadModel(in_dim=8, n_niche=5, d=16, heads=2, blocks=1)
    m.fit(_sets(2), [{"promise": np.zeros(6)}, {"niche": np.zeros(6, int)}], epochs=5)
    x = _sets(1, n=7)[0]
    p = tmp_path / "th.pt"
    m.save(str(p))
    m2 = ThreeHeadModel.load(str(p))
    a, b = m.predict(x), m2.predict(x)
    assert np.allclose(a["promise"], b["promise"]) and np.array_equal(a["niche"], b["niche"])


def test_promise_dist_normalized():
    m = ThreeHeadModel(in_dim=8, n_niche=4, d=16, heads=2, blocks=1)
    w = m.promise_dist(_sets(1, n=5)[0])
    assert abs(float(w.sum()) - 1.0) < 1e-5 and (w >= 0).all()


# --- eviction logging (label collection for the eviction head) ---

def _run_evict(tmp_dir, log_evictions):
    cfg = conftest.mock_config(budget=24, seed=42)
    cfg["pop_cap"] = 3                              # force evictions quickly
    cfg["log_evictions"] = log_evictions
    build_engine(cfg, conftest.TASKS_ROOT, tmp_dir, python=conftest.PYTHON,
                 llm=MockLLM()).run()
    return [e for e in conftest.read_journal(tmp_dir)
            if e.get("event") == "decision" and e.get("policy") == "eviction"]


def test_eviction_logging_collects_labels(tmp_path):
    ev = _run_evict(str(tmp_path / "on"), log_evictions=True)
    assert ev, "no eviction events logged with pop_cap=3 and log_evictions=True"
    e = ev[0]
    assert "victim_score" in e and "n_kept" in e and "island" in e


def test_eviction_logging_off_by_default(tmp_path):
    assert _run_evict(str(tmp_path / "off"), log_evictions=False) == []
