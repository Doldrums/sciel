"""Distribution overrides have the same meaning in extends and defaults."""

import copy
import hashlib

import pytest
import yaml

from sciel.config import DEFAULTS, apply_defaults, load_config, validate


@pytest.mark.parametrize("field", ["base_mix", "plateau_mix"])
def test_extends_replaces_complete_mode_distribution(tmp_path, field):
    parent = {"task": "oe/x", "policy": {field: {"base": .8, "exploit": .2},
              "plateau_hold": 7}, "llm": {"max_tokens": 123, "temperature": .4}}
    (tmp_path / "parent.yaml").write_text(yaml.safe_dump(parent))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({
        "extends": "parent.yaml", "policy": {field: {"base": 1}},
        "llm": {"temperature": .1}}))
    before = copy.deepcopy(DEFAULTS)
    loaded = load_config(str(tmp_path / "child.yaml"))
    assert loaded["policy"][field] == {"base": 1}
    assert loaded["policy"]["plateau_hold"] == 7
    assert loaded["llm"] == {"max_tokens": 123, "temperature": .1}
    resolved = validate(loaded)
    assert resolved["policy"][field] == {"base": 1}
    assert apply_defaults(resolved) == resolved
    assert DEFAULTS == before
    assert load_config(str(tmp_path / "parent.yaml")) == parent


def test_multi_level_extends_keeps_distribution_from_nearest_override(tmp_path):
    (tmp_path / "root.yaml").write_text("policy:\n  base_mix: {base: 0.8, exploit: 0.2}\n")
    (tmp_path / "middle.yaml").write_text("extends: root.yaml\npolicy:\n  base_mix: {explore: 1}\n")
    (tmp_path / "leaf.yaml").write_text("extends: middle.yaml\npolicy:\n  elite_k: 2\n")
    cfg = validate(load_config(str(tmp_path / "leaf.yaml")))
    assert cfg["policy"]["base_mix"] == {"explore": 1}
    assert cfg["policy"]["elite_k"] == 2


def test_explicit_empty_distribution_is_not_repopulated(tmp_path):
    (tmp_path / "parent.yaml").write_text("policy:\n  base_mix: {base: 1}\n")
    (tmp_path / "child.yaml").write_text("extends: parent.yaml\npolicy:\n  base_mix: {}\n")
    assert load_config(str(tmp_path / "child.yaml"))["policy"]["base_mix"] == {}


def _warm_start():
    source = "def main(): return 1\n"
    return {"schema": "sciel-warm-start/1", "source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "parent": {"run_id": "run", "checkpoint_sha256": "a" * 64,
                       "journal_sha256": "b" * 64, "task": "oe/x",
                       "task_hash": "c" * 64, "program_id": "p1"}}


def test_warm_start_validates_and_preserves_recorded_source():
    supplied = _warm_start()
    assert validate({"warm_start": supplied})["warm_start"] == supplied
    assert validate({"warm_start": None})["warm_start"] is None
    supplied["source"] += "# changed"
    with pytest.raises(ValueError, match="warm_start source identity"):
        validate({"warm_start": supplied})


@pytest.mark.parametrize("route", ["repeat_promotion", "repeat_population",
                                  "circle_generation", "circle_intervention"])
def test_warm_start_rejects_specialized_routes(route):
    with pytest.raises(ValueError, match="warm_start supports ordinary search"):
        validate({"warm_start": _warm_start(), route: {"enabled": True}})


def test_incomplete_auxiliary_is_opt_in_without_legacy_config_drift():
    assert "incomplete_auxiliary" not in validate({"research": {"enabled": True}})["research"]
    for value in ("stop", "skip"):
        cfg = validate({"research": {"enabled": True, "structured_auxiliary": True,
                                     "incomplete_auxiliary": value}})
        assert cfg["research"]["incomplete_auxiliary"] == value
    assert validate({"research": {"incomplete_auxiliary": "stop"}})["research"]["incomplete_auxiliary"] == "stop"
    with pytest.raises(ValueError, match="requires structured_auxiliary"):
        validate({"research": {"incomplete_auxiliary": "skip"}})


@pytest.mark.parametrize("value", [None, True, 1, "retry"])
def test_incomplete_auxiliary_rejects_other_values(value):
    with pytest.raises(ValueError, match="incomplete_auxiliary must be stop or skip"):
        validate({"research": {"structured_auxiliary": True, "incomplete_auxiliary": value}})
