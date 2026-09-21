"""Online preset weights stay bounded and checkpoint exactly."""

import json
import math
import random

import pytest

from sciel.components.policy.preset_weights import PresetWeights


def test_normalized_prior_and_explicit_exploration_mixture():
    policy = PresetWeights({"base": 8, "exploit": 2}, exploration=0.12)
    probabilities = policy.probabilities()
    assert policy.prior == {"base": 0.8, "exploit": 0.2, "explore": 0.0}
    assert probabilities["base"] == pytest.approx(0.88 * 0.8 + 0.04)
    assert probabilities["explore"] == pytest.approx(0.04)
    assert sum(probabilities.values()) == pytest.approx(1)


def test_reward_shifts_distribution_without_losing_exploration():
    policy = PresetWeights(exploration=0.15)
    before = policy.probabilities()
    for _ in range(1000):
        policy.update("explore", 1, policy.probabilities()["explore"])
        policy.update("base", 0, policy.probabilities()["base"])
    after = policy.probabilities()
    assert after["explore"] > before["explore"]
    assert all(value >= 0.15 / 3 for value in after.values())
    assert sum(after.values()) == pytest.approx(1)


def test_zero_prior_arm_can_learn_from_exploration():
    policy = PresetWeights({"base": 1.0})
    for _ in range(100):
        policy.update("explore", 1, policy.probabilities()["explore"])
    assert policy.probabilities()["explore"] > policy.probabilities()["base"]


def test_zero_or_negative_reward_does_not_shift_weights():
    policy = PresetWeights()
    initial = policy.probabilities()
    policy.update("exploit", 0, 0.3)
    policy.update("explore", -5, 0.3)
    assert policy.probabilities() == initial
    assert policy.state()["updates"] == 2


def test_rewards_and_importance_credit_are_clipped():
    policy = PresetWeights(learning_rate=0.2, max_importance_reward=4)
    policy.update("base", 999, 1e-300)
    assert policy.log_weights["base"] - policy.log_weights["explore"] == pytest.approx(0.8)
    assert all(math.isfinite(value) for value in policy.probabilities().values())


def test_sample_returns_actual_probability():
    policy = PresetWeights({"base": 3, "exploit": 2, "explore": 1})
    rng = random.Random(42)
    for _ in range(20):
        name, propensity = policy.sample(rng)
        assert propensity == policy.probabilities()[name]


def test_json_roundtrip_continues_identical_sampling_and_updates():
    policy = PresetWeights({"base": 3, "exploit": 2, "explore": 1}, learning_rate=0.3)
    rng = random.Random(317)
    for index in range(20):
        name, propensity = policy.sample(rng)
        policy.update(name, index % 3 / 2, propensity)
    saved = json.loads(json.dumps(policy.state(), allow_nan=False))
    restored = PresetWeights()
    restored.load_state(saved)
    assert restored.state() == saved
    other_rng = random.Random()
    other_rng.setstate(rng.getstate())
    for index in range(30):
        chosen = policy.sample(rng)
        assert restored.sample(other_rng) == chosen
        policy.update(chosen[0], index % 2, chosen[1])
        restored.update(chosen[0], index % 2, chosen[1])
    assert restored.state() == policy.state()


def test_large_finite_prior_does_not_overflow():
    policy = PresetWeights({"base": 1e308, "exploit": 1e308, "explore": 1e308})
    assert policy.probabilities() == pytest.approx({name: 1 / 3 for name in policy.prior})


def test_repeated_resume_preserves_arbitrary_prior_bits():
    rng = random.Random(18)
    for _ in range(20):
        policy = PresetWeights({name: rng.random() for name in ("base", "explore", "exploit")})
        original = policy.state()
        for _ in range(10):
            policy.load_state(json.loads(json.dumps(policy.state())))
        assert policy.state() == original


@pytest.mark.parametrize("name,reward,propensity", [
    ("unknown", 1, 0.5), ("base", float("nan"), 0.5),
    ("base", float("inf"), 0.5), ("base", 1, 0),
    ("base", 1, -0.2), ("base", 1, 1.1), ("base", 1, float("nan")),
])
def test_invalid_update_does_not_mutate_policy(name, reward, propensity):
    policy = PresetWeights()
    before = policy.state()
    with pytest.raises(ValueError):
        policy.update(name, reward, propensity)
    assert policy.state() == before


@pytest.mark.parametrize("kwargs", [
    {"prior": {}}, {"prior": {"unknown": 1}}, {"prior": {"base": -1}},
    {"prior": {"base": float("inf")}}, {"exploration": 0}, {"exploration": 1.1},
    {"learning_rate": -1}, {"learning_rate": float("inf")},
    {"max_importance_reward": 0},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        PresetWeights(**kwargs)


@pytest.mark.parametrize("field,value", [
    ("version", 42), ("updates", -1),
    ("log_weights", {"base": float("nan"), "exploit": 0, "explore": 0}),
    ("log_weights", {"base": -1, "exploit": -2, "explore": -3}),
    ("prior", {"base": 1, "exploit": 1, "explore": 1}),
])
def test_bad_checkpoint_does_not_mutate_live_policy(field, value):
    policy = PresetWeights()
    before = policy.state()
    bad = {**before, field: value}
    with pytest.raises(ValueError):
        policy.load_state(bad)
    assert policy.state() == before
