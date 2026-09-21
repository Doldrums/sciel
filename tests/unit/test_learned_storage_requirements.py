"""Explicit learned storage cannot start on another representation."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults, load_config, validate
from sciel.engine.run_lock import RunDirectoryLock
from test_niche_load_failure import construct


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("error", [ImportError("torch unavailable"),
    FileNotFoundError("weights absent"), PermissionError("unreadable weights"),
    ValueError("corrupt axis")])
def test_grid_load_failure_stops_before_search(tmp_path, monkeypatch, error):
    from sciel.components.policy import code_embed
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(code_embed, "LearnedDescriptor", SimpleNamespace(load=fail), raising=False)
    with pytest.raises(RuntimeError, match="behavior descriptor.*configured-grid.pt") as caught:
        construct(tmp_path, grid=True)
    assert caught.value.__cause__ is error
    assert "embed_grid_path" in str(caught.value) and "sciel[learning]" in str(caught.value)
    assert not (tmp_path / "sciel_events.jsonl").exists()
    RunDirectoryLock(tmp_path).close()


def test_disabled_portable_storage_does_not_load_models(tmp_path, monkeypatch):
    from sciel.components.policy import code_embed
    def fail(*args, **kwargs):
        pytest.fail("disabled storage tried to load a model")
    for name in ("LearnedDescriptor", "VQNiches"):
        monkeypatch.setattr(code_embed, name, SimpleNamespace(load=fail), raising=False)
    assert apply_defaults({})["embed_grid"] is False
    engine = construct(tmp_path, enabled=False)
    try:
        assert engine.descriptor_info == {"kind": "code_length"}
        assert engine.db.niche_fn is None
    finally:
        engine.close()


@pytest.mark.parametrize("profile,kind", [
    ("circle.yaml", "grid"), ("heilbronn.yaml", "grid"), ("func_min.yaml", "grid"),
    ("circle_parallel_gemini_rl_presets_vq.yaml", "vq"), ("smoke_mock.yaml", "portable"),
])
def test_profiles_preserve_their_declared_storage_and_external_artifact_paths(profile, kind):
    config = validate(load_config(str(ROOT / "configs" / profile)))
    assert config["embed_grid"] is (kind == "grid")
    assert config["embed_niche"] is (kind == "vq")
    if kind != "portable":
        key = "embed_niche_path" if kind == "vq" else "embed_grid_path"
        assert not (ROOT / config[key]).resolve().is_relative_to(ROOT)


@pytest.mark.parametrize("profile", ["heilbronn_embedgrid.yaml", "heilbronn_final_grid.yaml"])
def test_v2_grid_profiles_explicitly_keep_legacy_projection(profile):
    loaded = load_config(str(ROOT / "configs" / profile))
    config = validate(loaded)
    assert config["embed_grid"] is True
    assert config["embed_grid_path"] == "../sciel-artifacts/code_embed_v2.pt"
    assert "embed_axis_path" in loaded
    assert config["embed_axis_path"] is None


@pytest.mark.parametrize("profile", ["circle_seed_flash.yaml", "circle_seed_frontier.yaml"])
def test_seed_circle_profiles_explicitly_reuse_fitted_circle_axis(profile):
    config = validate(load_config(str(ROOT / "configs" / profile)))
    assert config["task"] == "oe/circle_packing_seed"
    assert config["embed_grid"] is True
    assert config["embed_grid_path"] == "../sciel-artifacts/code_embed_v3.pt"
    assert config["embed_axis_path"] == (
        "../sciel-artifacts/behavior_axis/oe_circle_packing_v3.npz")
