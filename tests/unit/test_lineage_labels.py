"""Phase-1 foundation: lineage-credit labels (subtree-max propagation over the tree)."""

import importlib.util
import json
import os

import pytest

_MOD = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "lineage_labels.py")


def _load():
    spec = importlib.util.spec_from_file_location("lineage_labels", _MOD)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _journal(path):
    # tree: p1 -> {p2, p4}, p2 -> p3
    steps = [("p1", None, 0.3), ("p2", "p1", 0.6), ("p3", "p2", 0.5), ("p4", "p1", 0.2)]
    ev = [{"event": "run_start", "schema_version": 1}]
    for i, (cid, par, sc) in enumerate(steps, 1):
        ev.append({"event": "iteration", "iteration_no": i, "B_evals": i, "island": 0,
                   "child_score": sc, "child_id": cid, "parent_id": par})
    ev.append({"event": "run_end", "pop_best": 0.6})
    with open(path, "w") as fh:
        fh.writelines(json.dumps(e) + "\n" for e in ev)


def test_subtree_max_and_gain(tmp_path):
    m = _load()
    p = tmp_path / "sciel_events.jsonl"; _journal(str(p))
    recs = {r["id"]: r for r in m.labels_for(str(p))}
    assert recs["p3"]["subtree_max"] == 0.5           # leaf
    assert recs["p2"]["subtree_max"] == 0.6           # max(self 0.6, child p3 0.5)
    assert recs["p4"]["subtree_max"] == 0.2           # leaf
    assert recs["p1"]["subtree_max"] == 0.6           # best in whole subtree
    # lineage_gain = future improvement the lineage produced
    assert abs(recs["p1"]["lineage_gain"] - 0.3) < 1e-9   # 0.6 - 0.3
    assert recs["p2"]["lineage_gain"] == 0.0              # already the peak
    assert recs["p4"]["lineage_gain"] == 0.0


def test_missing_child_id_skipped(tmp_path):
    m = _load()
    p = tmp_path / "sciel_events.jsonl"
    with open(p, "w") as fh:
        fh.write(json.dumps({"event": "iteration", "iteration_no": 1, "B_evals": 1,
                             "island": 0, "child_score": 0.5, "parent_id": None}) + "\n")
    assert m.labels_for(str(p)) == []                 # no child_id -> no edges


@pytest.mark.parametrize('reverse', [False, True])
def test_deep_lineage_labels_are_independent_of_input_order(reverse):
    m = _load()
    count = 2500
    events = [dict(event='iteration', child_id=f'p{i}', parent_id=f'p{i-1}' if i else None,
                   child_score=i / count, iteration_no=i + 1, B_evals=i + 1, island=0)
              for i in range(count)]
    if reverse:
        events.reverse()
    labels = {r['id']: r for r in m.labels_for('in-memory', trajectory=(events, {}))}
    assert len(labels) == count
    assert labels['p0']['subtree_max'] == (count - 1) / count
    assert labels['p0']['lineage_gain'] == (count - 1) / count
    assert labels[f'p{count-1}']['gen'] == count - 1
    assert labels[f'p{count-1}']['lineage_gain'] == 0


def test_shared_descendant_and_cycle_detection(capsys):
    m = _load()
    nodes = {k: {'score': score, 'parent': parent} for k, score, parent in
             [('root', .1, None), ('left', .2, 'root'), ('right', .3, 'root'), ('leaf', .9, 'left')]}
    assert m.subtree_max(nodes, {'root': ['left', 'right'], 'left': ['leaf'], 'right': ['leaf']}) == {
        'root': .9, 'left': .9, 'right': .9, 'leaf': .9}
    nodes['root']['parent'] = 'leaf'
    with pytest.raises(ValueError, match='cycle'):
        m._depths(nodes)
    with pytest.raises(ValueError, match='cycle'):
        m.subtree_max(nodes, {'root': ['left'], 'left': ['leaf'], 'leaf': ['root']})
    events = [dict(event='iteration', child_id=k, parent_id=v['parent'], child_score=v['score'])
              for k, v in nodes.items()]
    assert m.labels_for('cyclic fixture', trajectory=(events, {})) == []
    assert 'cycle in lineage' in capsys.readouterr().err
