"""Resume must refuse a checkpoint written under a different objective or axis.

Both halves of the behaviour-axis work change identity: the evaluators now emit a
behaviour readout (so every task_hash moved) and the descriptor is a fitted principal
axis rather than a seeded random direction (so cells are assigned differently).
Resuming across either change mixes two incomparable populations in one run, and
nothing downstream would ever say so. Hence the guard.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.engine import checkpoint


class _Task:
    def __init__(self, name="oe/x", h="abc"):
        self.qualified_name, self.task_hash = name, h


class _Engine:
    def __init__(self, task=None, descriptor=None):
        self.task = task or _Task()
        self.descriptor_info = descriptor if descriptor is not None else {"kind": "pca2_v1"}


def _state(engine, **over):
    ident = checkpoint._identity(engine)
    ident.update(over)
    return {"identity": ident}


def test_same_identity_passes():
    e = _Engine()
    assert checkpoint.check_identity(e, _state(e)) == []


def test_changed_task_hash_is_caught():
    e = _Engine()
    problems = checkpoint.check_identity(e, _state(e, task_hash="deadbeef"))
    assert len(problems) == 1 and "task_hash" in problems[0]


def test_changed_descriptor_is_caught():
    e = _Engine()
    problems = checkpoint.check_identity(
        e, _state(e, descriptor={"kind": "legacy_random_raw"}))
    assert len(problems) == 1 and "descriptor" in problems[0]


def test_changed_engine_contract_is_caught():
    e = _Engine()
    problems = checkpoint.check_identity(e, _state(e, engine_contract="espalier-2"))
    assert any("engine_contract" in p for p in problems)


def test_pre_identity_checkpoint_is_judged_by_its_stored_config():
    """Historical checkpoints carry no identity block, but their stored config still
    settles the one question that matters: `embed_axis_path` did not exist until the
    fitted axis did, so a config without that key was written by the legacy random
    direction. All 172 checkpoints in outputs/ are of that kind."""
    old = {"milestone": 40, "config": {"embed_grid": True}}
    problems = checkpoint.check_identity(_Engine(descriptor={"kind": "pca2_v1@ab"}), old)
    assert len(problems) == 1 and "predates the fitted axis" in problems[0]

    # ... and it is accepted when the engine is on the legacy descriptor too
    legacy = _Engine(descriptor={"kind": "legacy_random_raw"})
    assert checkpoint.check_identity(legacy, old) == []


def test_checkpoint_with_neither_identity_nor_config_is_accepted():
    """Nothing is knowable; refusing would break resume for no evidence."""
    assert checkpoint.check_identity(_Engine(), {"milestone": 40}) == []


def test_load_refuses_a_mismatch_and_can_be_forced(tmp_path, monkeypatch):
    e = _Engine()
    path = tmp_path / "checkpoint.json"
    state = _state(e, task_hash="deadbeef")
    state.update({"milestones_fired": [], "lineage_fail": {}, "eval_times": {}})
    path.write_text(json.dumps(state))

    with pytest.raises(checkpoint.IdentityMismatch, match="task_hash"):
        checkpoint.load(e, str(path))

    # forcing gets past the guard; everything after it is the normal restore path,
    # which this stub engine does not implement, so only the guard is exercised
    e.kb_on = e.c2_on = e.plateau_shadow = False
    e.plateau_monitors = e.oc_monitors = None
    checkpoint.load(e, str(path), allow_identity_mismatch=True)
    assert e.milestones_fired == set()


@pytest.mark.parametrize("kind", ["legacy_random_raw", "vq_niche"])
def test_legacy_container_binds_embedding_fingerprint(kind):
    descriptor = {"kind": kind, "embed_sha": "embedding-v2"}
    e = _Engine(descriptor=descriptor)
    state = _state(e, descriptor={**descriptor, "embed_sha": "embedding-v3"})
    assert checkpoint.check_identity(e, _state(e)) == []
    assert checkpoint.check_identity(e, state) == [
        "descriptor: checkpoint frozen representation differs"]
    # An old identity with no fingerprint cannot establish compatibility.
    state = _state(e, descriptor={"kind": kind})
    assert any("descriptor" in problem for problem in checkpoint.check_identity(e, state))


def test_legacy_vq_parameters_are_part_of_identity():
    e = _Engine(descriptor={"kind": "vq_niche", "embed_sha": "same", "csc_rate": .08})
    state = _state(e, descriptor={**e.descriptor_info, "csc_rate": .1})
    assert any("descriptor" in problem for problem in checkpoint.check_identity(e, state))


def test_experimental_container_cannot_drop_descriptor_identity():
    e = _Engine()
    e.cfg = {"container": {"kind": "global_elites"}}
    state = _state(e, descriptor={})
    assert any("descriptor" in problem for problem in checkpoint.check_identity(e, state))


def test_rejected_restore_releases_engine_writer(tmp_path):
    e = _Engine()
    e.close = Mock()
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(_state(e, task_hash="different")))
    with pytest.raises(checkpoint.IdentityMismatch, match="task_hash"):
        checkpoint.load(e, str(path))
    e.close.assert_called_once_with()


def test_warm_start_provenance_is_detached_and_resume_bound():
    e = _Engine()
    e.cfg = {"warm_start": {"source_sha256": "seed", "parent_checkpoint_sha256": "parent"}}
    state = _state(e)
    assert checkpoint.check_identity(e, state) == []
    e.cfg["warm_start"]["source_sha256"] = "other-seed"
    assert state["identity"]["warm_start"]["source_sha256"] == "seed"
    assert any("warm_start" in problem for problem in checkpoint.check_identity(e, state))


def test_accounting_stop_marker_is_restored_without_changing_legacy_counters(tmp_path):
    e = _Engine()
    e.kb_on = e.c2_on = e.plateau_shadow = False
    e.plateau_monitors = e.oc_monitors = None
    state = _state(e)
    state.update(milestones_fired=[], lineage_fail={}, eval_times={},
                 _accounting_stopped_reason="generation response cost is unknown")
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(state))
    checkpoint.load(e, str(path))
    assert e._accounting_stopped_reason == state["_accounting_stopped_reason"]
    assert "_accounting_stopped_reason" not in checkpoint.counters(e)


def test_accounting_stop_marker_is_saved_only_when_present(tmp_path):
    e = _Engine()
    e.cfg, e.run_dir = {}, str(tmp_path)
    e.milestones_fired, e.noop_parents = set(), set()
    e.lineage_fail, e.eval_times = {}, {}
    e.kb_on = e.c2_on = e.plateau_shadow = False
    e.plateau_monitors = e.oc_monitors = None
    e.log = SimpleNamespace(run_id="fixture", anchor=lambda: {})
    for name in checkpoint._SCALARS:
        setattr(e, name, 0)
    for name in checkpoint._COMPONENTS:
        setattr(e, name, SimpleNamespace(state=lambda: {}))
    e.db = SimpleNamespace(_counter=0, state=lambda: {"counter": 0})
    checkpoint.save(e, "final")
    path = tmp_path / "checkpoint.json"
    assert "_accounting_stopped_reason" not in json.loads(path.read_text())
    e._accounting_stopped_reason = "postmortem response cost is unknown"
    checkpoint.save(e, "final")
    state = json.loads(path.read_text())
    assert state["_accounting_stopped_reason"] == e._accounting_stopped_reason
    assert "_accounting_stopped_reason" not in state["provenance"]["counters"]


@pytest.mark.parametrize("reason", ["", False, 3, {}])
def test_malformed_accounting_stop_is_rejected(tmp_path, reason):
    e = _Engine()
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"_accounting_stopped_reason": reason}))
    with pytest.raises(checkpoint.IdentityMismatch, match="accounting stop reason"):
        checkpoint.load(e, str(path))
