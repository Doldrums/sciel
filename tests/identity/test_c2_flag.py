"""C2/C3 flag gating (§5.9-5.11, P4): off is identical to pre-flag; on adds bets."""

from sciel.engine.factory import build_engine
from sciel.engine.llm.mock import MockLLM

import conftest


def _journal(tmp_dir, c2):
    cfg = conftest.mock_config(budget=30, seed=42)
    if c2 is not None:
        cfg["c2_event_bets"] = c2
    build_engine(cfg, conftest.TASKS_ROOT, tmp_dir, python=conftest.PYTHON,
                 llm=MockLLM()).run()
    return conftest.read_journal(tmp_dir)


def _drop_config(journal):
    return [{k: v for k, v in e.items() if k != "config"} for e in journal]


def test_flag_absent_equals_flag_false(tmp_path):
    absent = _journal(str(tmp_path / "absent"), None)
    false = _journal(str(tmp_path / "false"), False)
    assert _drop_config(absent) == _drop_config(false)


def test_flag_off_has_no_c2_artifacts(tmp_path):
    off = _journal(str(tmp_path / "off"), False)
    assert not any(e["event"] == "postmortem" for e in off)
    for e in off:
        if e["event"] == "iteration":
            assert "branch_cal_brier" not in e
            assert e.get("bet") is None


def test_flag_on_produces_bets_and_postmortems(tmp_path):
    on = _journal(str(tmp_path / "on"), True)
    iters = [e for e in on if e["event"] == "iteration"
             and e.get("child_score") is not None]
    assert any(e.get("bet") is not None for e in iters), "no bets under flag on"
    assert any("branch_cal_brier" in e for e in iters), "no branch_cal_brier under flag on"
    events = [e for e in iters if e.get("is_island_event")]
    if events:
        assert any(e["event"] == "postmortem" for e in on), "event but no postmortem"


def test_flag_on_is_deterministic(tmp_path):
    a = _journal(str(tmp_path / "a"), True)
    b = _journal(str(tmp_path / "b"), True)
    assert a == b
