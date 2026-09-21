import copy

import pytest

from sciel.campaign.no_regression import POLICY, evaluate_policy, verify_disposition


def manifest():
    seeds = [81, 82, 83]
    return {"protocol_sha256": "a" * 64, "seeds": seeds,
            "jobs": [{"id": f"{arm}_{seed}", "arm": arm, "seed": seed}
                     for seed in seeds for pair in POLICY["contrasts"] for arm in pair]}


def observed(arm, seed, score, **kwargs):
    return {f"{arm}_{seed}": {"complete": True, "valid": True,
            "primary_score": score, "mechanism_activated": True,
            "result_sha256": "b" * 64, **kwargs}}


def test_first_loss_stops_only_its_future_contrast_and_binds_evidence():
    m = manifest()
    obs = observed("grid", 81, .8) | observed("pool", 81, .7999999999999999)
    report = evaluate_policy(m, obs)
    assert [r["decision"] for r in report["contrasts"]] == ["rejected_observed_regression", "inconclusive"]
    assert {r["job"]["id"] for r in report["withdrawals"]} == {
        "grid_82", "pool_82", "grid_83", "pool_83"}
    assert all(r["evidence"][1]["result_sha256"] == "b" * 64 for r in report["withdrawals"])


def test_early_wins_cannot_pass_and_one_later_loss_is_not_averaged_away():
    m, obs = manifest(), {}
    for seed in [81, 82]:
        obs.update(observed("grid", seed, .1) | observed("pool", seed, 1.))
    assert evaluate_policy(m, obs)["contrasts"][0]["decision"] == "inconclusive"
    obs.update(observed("grid", 83, .9) | observed("pool", 83, .89))
    assert evaluate_policy(m, obs)["contrasts"][0]["decision"] == "rejected_observed_regression"


def test_all_ties_pass_screen_only_and_inactive_control_blocks_it():
    obs = {}
    for seed in [81, 82, 83]:
        obs.update(observed("grid", seed, .5) | observed("pool", seed, .5))
    result = evaluate_policy(manifest(), obs)["contrasts"][0]
    assert result["decision"] == "screen_pass"
    assert result["automatic_default_promotion"] is False
    obs["grid_82"]["mechanism_activated"] = False
    assert evaluate_policy(manifest(), obs)["contrasts"][0]["decision"] == "inconclusive"


@pytest.mark.parametrize("bad", [
    {"complete": False, "valid": False, "primary_score": None},
    {"complete": True, "valid": False, "primary_score": 0},
])
def test_failed_or_incomplete_draws_are_not_valid_pairs(bad):
    obs = observed("grid", 81, .8) | observed("pool", 81, .7, **bad)
    result = evaluate_policy(manifest(), obs)
    assert result["contrasts"][0]["n_pairs"] == 0
    assert result["withdrawals"] == []


def test_paid_work_after_futility_is_not_hidden_as_withdrawn():
    obs = observed("grid", 81, .8) | observed("pool", 81, .7)
    obs.update(observed("pool", 82, None, complete=False, valid=False))
    with pytest.raises(ValueError, match="continued"):
        evaluate_policy(manifest(), obs)


def test_resume_withdrawals_are_monotonic_and_evidence_cannot_change():
    m = manifest()
    obs = observed("grid", 81, .8) | observed("pool", 81, .7)
    old = evaluate_policy(m, obs)
    obs.update(observed("pool_presets_fixed", 81, .8) | observed("pool_presets", 81, .7))
    current = evaluate_policy(m, obs)
    verify_disposition(old, current)
    damaged = copy.deepcopy(current)
    damaged["withdrawals"][0]["evidence"][0]["result_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="evidence"):
        verify_disposition(old, damaged)
    with pytest.raises(ValueError, match="evidence"):
        verify_disposition(current, old)


@pytest.mark.parametrize("field,value", [("primary_score", float("nan")),
    ("primary_score", True), ("primary_score", -.1), ("result_sha256", "bad"),
    ("valid", 1), ("complete", False)])
def test_malformed_pair_rejected(field, value):
    obs = observed("grid", 81, .8) | observed("pool", 81, .7, **{field: value})
    with pytest.raises(ValueError):
        evaluate_policy(manifest(), obs)


def test_missing_duplicate_and_unknown_jobs_rejected():
    m = manifest()
    m["jobs"].pop()
    with pytest.raises(ValueError):
        evaluate_policy(m, {})
    with pytest.raises(ValueError):
        evaluate_policy(manifest(), {"foreign": {}})


def test_resume_cannot_rewrite_an_earlier_winning_pair():
    m = manifest()
    obs = observed("grid", 81, .7) | observed("pool", 81, .8)
    previous = evaluate_policy(m, obs)
    obs["pool_81"]["result_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="prior compared pair"):
        verify_disposition(previous, evaluate_policy(m, obs))
