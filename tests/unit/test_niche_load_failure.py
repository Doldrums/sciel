"""A damaged configured VQ artifact must not silently change search storage."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.engine.llm.mock import MockLLM
from sciel.engine.loop import Engine
from sciel.engine.run_lock import RunDirectoryLock
from sciel.eval.registry import load_task


ROOT = Path(__file__).resolve().parents[2]


def construct(directory, enabled=True, *, grid=False):
    return Engine(
        dict(task="test/toy_quadratic", budget=1, max_iterations=1,
             num_islands=1, embed_grid=grid, embed_niche=enabled and not grid,
             embed_grid_path="configured-grid.pt", embed_axis_path=None,
             embed_niche_path="configured-vq.pt", portfolio_seed=False,
             idea_island=False, dynamic_islands=False, crossover=False,
             policy={"enabled": False}, llm={"mock": True},
             wandb={"enabled": False}),
        load_task(str(ROOT / "tasks/test/toy_quadratic")),
        str(directory), MockLLM(),
    )


def loader_failure(monkeypatch, error):
    from sciel.components.policy import code_embed

    def fail(*args, **kwargs):
        raise error

    # VQNiches is intentionally unavailable without optional torch. Inject the
    # public loader seam so these startup checks do not require that dependency.
    monkeypatch.setattr(code_embed, "VQNiches", SimpleNamespace(load=fail), raising=False)


@pytest.mark.parametrize("error", [ValueError("invalid weights"),
                                  RuntimeError("wrong tensor shape"),
                                  KeyError("artifact_version"),
                                  ImportError("optional torch unavailable"),
                                  FileNotFoundError("artifact absent")])
def test_invalid_niche_artifact_stops_construction_before_search(tmp_path, monkeypatch, error):
    loader_failure(monkeypatch, error)
    with pytest.raises(RuntimeError, match="VQ niche storage.*configured-vq.pt") as caught:
        construct(tmp_path)
    assert caught.value.__cause__ is error
    assert "embed_niche: false" in str(caught.value)
    assert not (tmp_path / "sciel_events.jsonl").exists()
    # Constructor cleanup still permits another owner to fix and start the run.
    RunDirectoryLock(tmp_path).close()


def test_disabled_niches_never_load_artifact(tmp_path, monkeypatch):
    loader_failure(monkeypatch, AssertionError("disabled loader was called"))
    engine = construct(tmp_path, enabled=False)
    try:
        assert engine.db.niche_fn is None
    finally:
        engine.close()
