"""Niche labels describe a frozen training reference, never arrival order."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")


@pytest.fixture
def trainer():
    path = Path(__file__).resolve().parents[2] / "scripts" / "train_three_head.py"
    spec = importlib.util.spec_from_file_location("three_head_label_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixtures(monkeypatch, trainer, runs):
    monkeypatch.setattr(trainer, "completed_events", lambda _: ([], {}))
    monkeypatch.setattr(trainer, "labels_for", lambda path, **kw: runs[Path(path).parent.name])
    monkeypatch.setattr(trainer, "_embed", lambda _, rows: np.array([row["z"] for row in rows]))


def _rows(vectors, island=0):
    return [{"id": f"p{i}", "island": island, "subtree_max": .1 * i, "z": z}
            for i, z in enumerate(vectors)]


def test_niche_labels_invariant_to_run_island_and_member_order(trainer, monkeypatch):
    vectors = [[1., 0.], [0., 1.], [-1., 0.]]
    runs = {"a": _rows(vectors), "b": _rows(vectors[::-1], island=1)}
    _fixtures(monkeypatch, trainer, runs)
    sets, labels, book = trainer.build_sets(None, ["a", "b"], return_niche_book=True)
    first = {tuple(z): int(n) for z, n in zip(sets[0], labels[0]["niche"])}
    assert len(set(first.values())) == 3
    assert {tuple(z): int(n) for z, n in zip(sets[1], labels[1]["niche"])} == first
    # Reorder both corpus and samples, without changing the training vectors.
    runs["a"] = [runs["a"][1], runs["a"][0], runs["a"][2]]
    sets2, labels2, book2 = trainer.build_sets(None, ["b", "a"], return_niche_book=True)
    assert book2.state() == book.state()
    for z, lab in zip(sets2, labels2):
        assert {tuple(row): int(n) for row, n in zip(z, lab["niche"])} == first


def test_held_out_labels_use_training_book_without_mutating_it(trainer, monkeypatch):
    book = trainer.fit_niche_book([np.array([[1., 0.], [0., 1.], [-1., 0.]])])
    before = book.state()
    held_out = [[0., -1.], [1., 0.], [0., 1.]]
    _fixtures(monkeypatch, trainer, {"holdout": _rows(held_out)})
    sets, labels = trainer.build_sets(None, ["holdout"], niche_book=book)
    assert book.state() == before
    assert labels[0]["niche"].tolist() == [book.assign_fixed(z) for z in held_out]
    assert labels[0]["promise"].tolist() == [0., .5, 1.]


def test_niche_reference_round_trip_preserves_class_meanings(trainer):
    vectors = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    book = trainer.fit_niche_book([vectors])
    restored = trainer.VQBook()
    restored.load_state(book.state())
    assert [restored.assign_fixed(z) for z in vectors] == [book.assign_fixed(z) for z in vectors]


def test_cli_saves_training_reference_next_to_model_without_training(trainer, monkeypatch, tmp_path):
    embedder_path = tmp_path / "embedder.pt"
    embedder_path.write_bytes(b"fixture-embedder")
    monkeypatch.setattr(trainer, "_load_embedder", lambda _: None)
    vectors = [[1., 0.], [0., 1.], [-1., 0.]]
    _fixtures(monkeypatch, trainer, {"train": _rows(vectors)})

    class Model:
        def __init__(self, **kwargs):
            pass

        def fit(self, sets, labels, **kwargs):
            assert len(sets) == 1 and "niche" in labels[0]

        def save(self, path):
            Path(path).write_bytes(b"fixture-model")

    monkeypatch.setattr(trainer, "ThreeHeadModel", Model)
    output = tmp_path / "model.pt"
    trainer.main(str(embedder_path), str(output), ["train"])
    metadata = json.loads(Path(str(output) + ".niche.json").read_text())
    assert metadata["schema"] == "sciel-three-head-niche-reference/1"
    assert metadata["embedder_sha256"] == hashlib.sha256(embedder_path.read_bytes()).hexdigest()
    restored = trainer.VQBook()
    restored.load_state(metadata["book"])
    assert len(set(restored.assign_fixed(z) for z in vectors)) == 3
