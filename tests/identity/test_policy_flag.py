"""pi policy flag gating (§5.13, §6.3 step 1): off is byte-identical; on mixes modes."""

import os
from collections import Counter

import pytest

from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.mock import MockLLM

import conftest


def _engine(tmp_dir, policy, budget=30, seed=42, selection_policy=None):
    cfg = conftest.mock_config(budget=budget, seed=seed)
    if policy is not None:
        cfg["policy"] = policy
    if selection_policy is not None:
        cfg["selection_policy"] = selection_policy
    return build_engine(cfg, conftest.TASKS_ROOT, tmp_dir, python=conftest.PYTHON,
                        llm=MockLLM())


def _run(tmp_dir, policy, **kw):
    _engine(tmp_dir, policy, **kw).run()
    return conftest.read_journal(tmp_dir)


def _drop_config(journal):
    return [{k: v for k, v in e.items() if k != "config"} for e in journal]


def _decisions(journal):
    return [(e["island"], e.get("mode"), e.get("mode_propensity"),
             e.get("parent_id"), e.get("child_score"))
            for e in journal if e["event"] == "iteration"]


def test_absent_means_espalier_defaults(tmp_path):
    # the engine owns its defaults: no policy block == the explicit espalier
    # policy, byte-identically -- and it is NOT the disabled arm anymore
    import copy

    from sciel.config import DEFAULTS

    absent = _run(str(tmp_path / "absent"), None)
    explicit = _run(str(tmp_path / "explicit"), copy.deepcopy(DEFAULTS["policy"]))
    assert _drop_config(absent) == _drop_config(explicit)
    disabled = _run(str(tmp_path / "disabled"), {"enabled": False})
    assert any(e["event"] == "decision" for e in absent) or \
        _drop_config(absent) != _drop_config(disabled)


def test_disabled_logs_base_mode_no_decisions(tmp_path):
    off = _run(str(tmp_path / "off"), {"enabled": False})
    # the controller logs no decisions when disabled (other default mechanisms
    # -- e.g. dynamic_islands -- may still log their own decision events)
    assert not any(e["event"] == "decision" and e.get("policy") == "controller"
                   for e in off)
    for e in off:
        if e["event"] == "iteration":
            assert e["mode"] == "base"
            assert e["mode_propensity"] == 1.0


def test_enabled_mixes_modes_and_logs_propensity(tmp_path):
    on = _run(str(tmp_path / "on"),
              {"enabled": True, "base_mix": {"base": 0.5, "exploit": 0.5},
               "tilt_beta": 0.5, "elite_k": 2})
    iters = [e for e in on if e["event"] == "iteration"]
    modes = Counter(e["mode"] for e in iters)
    assert modes["exploit"] > 0 and modes["base"] > 0, modes
    for e in iters:
        assert 0.0 < e["mode_propensity"] <= 1.0
    assert any(e["event"] == "decision" and e["policy"] == "controller" for e in on)


def test_enabled_is_deterministic(tmp_path):
    pol = {"enabled": True, "base_mix": {"base": 0.6, "exploit": 0.4}}
    a = _run(str(tmp_path / "a"), pol)
    b = _run(str(tmp_path / "b"), pol)
    assert a == b


def test_nontilt_kernel_with_modes_raises(tmp_path):
    with pytest.raises(ValueError):
        _engine(str(tmp_path / "g"),
                {"enabled": True, "base_mix": {"base": 0.8, "exploit": 0.2}},
                budget=10, selection_policy="ucb1")


def test_enabled_resume_matches_continuous(tmp_path):
    pol = {"enabled": True, "base_mix": {"base": 0.6, "exploit": 0.4}}
    cont = _run(str(tmp_path / "cont"), pol, budget=20)
    continuous = _decisions(cont)

    split_dir = str(tmp_path / "split")
    e1 = _engine(split_dir, pol, budget=20)
    e1._stop_at = 8
    e1.run()
    e2 = _engine(split_dir, pol, budget=20)
    checkpoint.load(e2, os.path.join(split_dir, "checkpoint.json"))
    e2.budget = 20
    e2.run()
    split = _decisions(conftest.read_journal(split_dir))
    assert split == continuous


def test_plateau_shadow_only_does_not_shift(tmp_path):
    # plateau_shift defaults ON now -- shadow-only must disable it explicitly
    pol = {"enabled": True, "base_mix": {"base": 1.0}, "plateau_shadow": True,
           "plateau_shift": False, "plateau_threshold": 100.0, "plateau_min_n": 2}
    j = _run(str(tmp_path / "shadow"), pol, budget=40)
    ocs = [e for e in j if e["event"] == "decision" and e["policy"] == "opportunity_cost"]
    assert ocs and all(e["shadow"] is True for e in ocs)
    assert all(e["mode"] == "base" for e in j if e["event"] == "iteration")


def test_plateau_shift_moves_to_plateau_mix(tmp_path):
    pol = {"enabled": True, "base_mix": {"base": 1.0}, "plateau_mix": {"exploit": 1.0},
           "plateau_shift": True, "plateau_threshold": 100.0, "plateau_min_n": 2,
           "plateau_hold": 100, "tilt_beta": 0.5, "elite_k": 2}
    j = _run(str(tmp_path / "shift"), pol, budget=40)
    ocs = [e for e in j if e["event"] == "decision" and e["policy"] == "opportunity_cost"]
    assert ocs and all(e["shadow"] is False for e in ocs)
    iters = [e for e in j if e["event"] == "iteration"]
    assert any(e["mode"] == "exploit" for e in iters)
