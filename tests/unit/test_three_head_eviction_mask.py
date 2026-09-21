"""Only an observed island gets eviction targets; embeddings are inert."""
import numpy as np
import pytest

from test_training_identity_boundaries import legacy_run, module


def labels_for_run(monkeypatch, run):
    pytest.importorskip('torch')
    trainer = module('train_three_head')
    monkeypatch.setattr(trainer, '_embed', lambda _, items: np.array(
        [[float(item['id'][1:]), 1.] for item in items], dtype=np.float32))
    return trainer.build_sets(None, [run])[1]


@pytest.mark.parametrize('eviction', [
    {'island': 0, 'victim_id': 'p1'}, {'victim_id': 'p1'},
])
def test_eviction_supervision_requires_the_same_observed_island(tmp_path, monkeypatch, eviction):
    run = legacy_run(tmp_path / 'run', evictions=[eviction])
    labels = labels_for_run(monkeypatch, run)
    assert labels[0]['eviction'].tolist() == [0., 1., 1.]
    assert 'eviction' not in labels[1]
    assert all('promise' in label and 'niche' in label for label in labels)


def test_both_islands_with_decisions_receive_their_own_targets(tmp_path, monkeypatch):
    run = legacy_run(tmp_path / 'run', evictions=[{'island': 0, 'victim_id': 'p1'},
                                               {'island': 1, 'victim_id': 'p5'}])
    labels = labels_for_run(monkeypatch, run)
    assert labels[0]['eviction'].tolist() == [0., 1., 1.]
    assert labels[1]['eviction'].tolist() == [1., 0., 1.]


def test_unknown_island_and_victim_do_not_create_keep_supervision(tmp_path, monkeypatch):
    run = legacy_run(tmp_path / 'run', evictions=[{'victim_id': 'unknown'}])
    assert all('eviction' not in label for label in labels_for_run(monkeypatch, run))


def test_contradictory_victim_island_is_reported_and_skipped(tmp_path, monkeypatch, capsys):
    run = legacy_run(tmp_path / 'run', evictions=[{'island': 1, 'victim_id': 'p1'}])
    assert labels_for_run(monkeypatch, run) == []
    assert 'eviction' in capsys.readouterr().err
