"""Learning credit must not turn floating-point dust into a dominant policy.

These tests use real reward/selection components and inert local fixtures only.
Raw measurement and strict record facts are independent of learning eligibility.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import math
import sys

import pytest

from sciel.components.policy.mode_bandit import ModeBandit
from sciel.components.selection.kernels import UCB1Kernel, UCBSlidingWindowKernel
from sciel.components.selection.novelty_chain import IslandView
from sciel.engine.operators import OpBandit
from sciel.reward import RewardBus
from test_crossover_eligibility import make_engine  # Real Engine fixture, inert sources.


POLICY = "ulp8-positive-median-cap1/1"
STATE_SCHEMA = "sciel-reward-state/2"
META_SCHEMA = "sciel-reward-normalization/2"


def event(bus, score, *, population=1.0, island=0.0, iteration=1):
    return bus.improvement_event(score,
        {"pop_best": population, "island_best": island}, "gen", iteration)


def actual_v11_pair(bus):
    first = event(bus, .46932143368138723, population=.9935742900059048,
                  island=.469321433681387, iteration=1)
    second = event(bus, .9837090741491017, population=.9935742900059048,
                   island=.9762760002644632, iteration=2)
    return first, second


def assert_finite_credit(ev):
    assert math.isfinite(ev.r_norm) and 0 <= ev.r_norm <= 1
    assert ev.r == ev.r_norm
    # Saved metadata must itself be strict JSON, not merely a finite display.
    json.dumps(asdict(ev), allow_nan=False)


def test_live_four_ulp_then_real_improvement_preserves_facts_without_explosion():
    bus = RewardBus()
    first, second = actual_v11_pair(bus)
    assert first.d_isl == first.r_raw == 2.220446049250313e-16
    assert first.d_isl / math.ulp(.46932143368138723) == 4
    assert first.is_island_event and not first.is_pop_event
    assert first.r_norm == 0.0
    assert first.normalization["channel_eligible"] == {
        "population": False, "island": False}
    assert first.normalization["learning_raw"] == 0.0
    assert first.normalization["ulp_thresholds"]["island"] == 8 * math.ulp(.46932143368138723)
    assert second.d_isl == second.r_raw == .007433073884638541
    assert second.is_island_event and not second.is_pop_event
    assert second.med == 1.0 and second.r_norm == second.r_raw
    assert second.normalization["scale_source"] == "cold_start_unit"
    assert second.normalization["previous_sample_count"] == 0
    assert bus.state()["jumps"] == [second.r_raw]
    assert first.normalization["previous_sample_count"] == 0
    assert first.normalization["schema"] == META_SCHEMA
    assert second.normalization["policy"] == POLICY
    assert_finite_credit(first)
    assert_finite_credit(second)


@pytest.mark.parametrize("steps,eligible", [(7, False), (8, False), (9, True)])
def test_machine_resolution_boundary_is_strictly_above_eight_operand_ulps(steps, eligible):
    score = 1.0
    for _ in range(steps):
        score = math.nextafter(score, math.inf)
    bus = RewardBus(w_pop=0.0, w_isl=1.0)
    ev = event(bus, score, population=2.0, island=1.0)
    assert ev.r_raw == score - 1.0 and ev.is_island_event
    assert ev.normalization["channel_eligible"]["island"] is eligible
    assert ev.normalization["ulp_thresholds"]["island"] == 8 * math.ulp(score)
    assert ev.r_norm == (ev.r_raw if eligible else 0.0)
    assert bus.state()["jumps"] == ([ev.r_raw] if eligible else [])


def test_only_eligible_channel_contributes_to_learning_not_raw_currency():
    population = 1.0
    for _ in range(4):
        population = math.nextafter(population, -math.inf)
    ev = event(RewardBus(w_pop=4.0, w_isl=.5), 1.0,
               population=population, island=.5)
    assert ev.d_pop > 0 and ev.d_isl == .5
    assert ev.is_pop_event and ev.is_island_event
    assert ev.r_raw == 4.0 * (1.0 - population) + .25
    assert ev.normalization["channel_eligible"] == {
        "population": False, "island": True}
    assert ev.normalization["learning_raw"] == ev.r_norm == .25


def test_zero_weight_does_not_erase_raw_record_or_pollute_scale():
    bus = RewardBus(w_pop=0.0, w_isl=0.0)
    ev = event(bus, 2.0, population=1.0, island=1.0)
    assert ev.is_pop_event and ev.is_island_event
    assert ev.d_pop == ev.d_isl == 1.0
    assert ev.normalization["channel_eligible"] == {"population": True, "island": True}
    assert ev.r_raw == ev.r_norm == ev.normalization["learning_raw"] == 0.0
    assert bus.state()["jumps"] == []


def test_previous_only_median_and_cold_start_unit_are_explicit():
    bus = RewardBus(w_pop=0.0, w_isl=1.0)
    first = event(bus, .25)
    first_metadata = deepcopy(first.normalization)
    second = event(bus, .5)
    third = event(bus, .125)
    assert first.med == 1.0 and first.r_norm == .25
    assert second.med == .25 and second.r_norm == 1.0
    assert second.normalization["cap_applied"] is True
    assert third.med == .375 and third.r_norm == pytest.approx(1 / 3)
    assert third.normalization["scale_source"] == "prior_positive_median"
    assert third.normalization["previous_sample_count"] == 2
    assert first.normalization == first_metadata
    assert bus.state()["jumps"] == [.25, .5, .125]


@pytest.mark.parametrize("exponent", [-900, -40, 0, 40, 900])
def test_rescaling_preserves_learning_eligibility_and_warm_ratio(exponent):
    scale = math.ldexp(1.0, exponent)
    bus = RewardBus(w_pop=0.0, w_isl=1.0)
    first = event(bus, 11 * scale, population=20 * scale, island=10 * scale)
    second = event(bus, 10.5 * scale, population=20 * scale, island=10 * scale)
    assert first.normalization["channel_eligible"]["island"] is True
    # Cold-start scale1 is intentional; only the warm ratio is scale invariant.
    assert first.med == 1.0 and first.r_norm == min(1.0, scale)
    assert second.med == scale and second.r_norm == .5
    assert second.normalization["channel_eligible"]["island"] is True
    assert_finite_credit(second)


@pytest.mark.parametrize("previous,score", [(-2.0, -1.0), (-1.0, 0.0), (0.0, .125)])
def test_zero_and_negative_baselines_use_actual_differences(previous, score):
    bus = RewardBus(w_pop=0.0, w_isl=1.0)
    ev = event(bus, score, island=previous)
    assert ev.d_isl == ev.r_raw == score - previous
    assert ev.is_island_event
    assert ev.normalization["channel_eligible"]["island"] is True
    assert_finite_credit(ev)


def test_subnormal_eligible_gains_and_even_median_do_not_underflow():
    unit = math.ulp(0.0)
    bus = RewardBus(w_pop=0.0, w_isl=1.0)
    dust = event(bus, 8 * unit)
    assert dust.r_raw == 8 * unit and dust.r_norm == 0
    a = event(bus, 9 * unit)
    b = event(bus, 10 * unit)
    c = event(bus, 9 * unit)
    assert a.r_norm == 9 * unit
    assert b.med == 9 * unit and b.r_norm == 1
    # Exact midpoint9.5u rounds to10u. a+(b-a)/2 and a/2+b/2 both give9u.
    assert c.med == 10 * unit and c.r_norm == .9
    assert bus.state()["jumps"] == [9 * unit, 10 * unit, 9 * unit]
    assert_finite_credit(c)


def test_huge_equal_jumps_have_finite_even_median_and_unit_credit():
    maximum = sys.float_info.max
    bus = RewardBus(w_pop=0.0, w_isl=1.0)
    first = event(bus, maximum, population=maximum)
    second = event(bus, maximum, population=maximum)
    third = event(bus, maximum / 2, population=maximum)
    assert first.r_norm == second.r_norm == 1.0
    assert second.normalization["cap_applied"] is False
    assert third.med == maximum and third.r_norm == .5
    assert_finite_credit(third)


def test_extreme_finite_ratio_is_capped_before_division_can_overflow():
    bus = RewardBus(w_pop=0.0, w_isl=1.0)
    event(bus, 9 * math.ulp(0.0))
    ev = event(bus, sys.float_info.max, population=sys.float_info.max)
    assert ev.r_norm == 1.0 and ev.normalization["cap_applied"] is True
    assert_finite_credit(ev)


def test_large_negative_non_improvement_is_zero_not_an_overflow_failure():
    bus = RewardBus()
    ev = event(bus, -sys.float_info.max,
               population=sys.float_info.max, island=sys.float_info.max)
    assert ev.r_raw == ev.r_norm == ev.d_isl == ev.d_pop == 0.0
    assert not ev.is_island_event and not ev.is_pop_event
    assert bus.state()["jumps"] == []


@pytest.mark.parametrize("kind", ["difference", "product", "sum"])
def test_unrepresentable_positive_raw_currency_fails_before_history_mutation(kind):
    maximum = sys.float_info.max
    bus = RewardBus(w_pop=0.0 if kind != "sum" else 1.0,
                    w_isl=maximum if kind == "product" else 1.0)
    before = deepcopy(bus.state())
    score = 2.0 if kind == "product" else maximum
    previous = -maximum if kind == "difference" else 0.0
    population = 0.0 if kind == "sum" else maximum
    with pytest.raises(ValueError):
        event(bus, score, population=population, island=previous)
    assert bus.state() == before


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), True, "1"])
def test_invalid_measurement_cannot_mutate_calibration(invalid):
    bus = RewardBus()
    event(bus, .5)
    before = deepcopy(bus.state())
    with pytest.raises(ValueError):
        event(bus, invalid)
    assert bus.state() == before


def test_state_version_roundtrip_preserves_next_reward_and_detaches_history():
    original = RewardBus()
    actual_v11_pair(original)
    state = json.loads(json.dumps(original.state(), allow_nan=False))
    assert state["schema"] == STATE_SCHEMA and state["normalization_policy"] == POLICY
    restored = RewardBus()
    restored.load_state(state)
    state["jumps"].append(1000.0)
    assert restored.state() == original.state()
    a = event(original, .8, population=.99, island=.7, iteration=3)
    b = event(restored, .8, population=.99, island=.7, iteration=3)
    assert asdict(a) == asdict(b) and restored.state() == original.state()


@pytest.mark.parametrize("mutation", ["legacy", "schema", "policy", "zero", "negative", "nan", "bool"])
def test_bad_or_old_state_is_rejected_atomically(mutation):
    bus = RewardBus()
    actual_v11_pair(bus)
    before = deepcopy(bus.state())
    bad = deepcopy(before)
    if mutation == "legacy":
        bad.pop("schema")
        bad.pop("normalization_policy")
    elif mutation == "schema":
        bad["schema"] = "sciel-reward-state/1"
    elif mutation == "policy":
        bad["normalization_policy"] = "unbounded"
    else:
        bad["jumps"] = [{"zero": 0.0, "negative": -1.0, "nan": float("nan"), "bool": True}[mutation]]
    with pytest.raises(ValueError):
        bus.load_state(bad)
    assert bus.state() == before


@pytest.mark.parametrize("name", ["w_pop", "w_isl"])
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), True, "1"])
def test_invalid_weights_rejected_at_constructor_restore_and_event_boundaries(name, invalid):
    with pytest.raises(ValueError):
        RewardBus(**{name: invalid})
    bus = RewardBus()
    actual_v11_pair(bus)
    before = deepcopy(bus.state())
    bad = deepcopy(before)
    bad[name] = invalid
    with pytest.raises(ValueError):
        bus.load_state(bad)
    assert bus.state() == before
    # Public dataclass fields can be assigned after construction; the next event
    # must still reject bad weights before appending a calibration sample.
    setattr(bus, name, invalid)
    with pytest.raises(ValueError):
        event(bus, .9, population=.95, island=.5)
    assert bus._jumps == before["jumps"]


class CaptureChoice:
    def choices(self, population, *, weights, k):
        assert k == 1
        self.distribution = dict(zip(population, weights))
        assert all(math.isfinite(p) and p > 0 for p in weights)
        assert sum(weights) == pytest.approx(1.0)
        return [max(self.distribution, key=self.distribution.get)]


def test_actual_sequence_no_longer_saturates_operator_or_mode_sampling():
    first, second = actual_v11_pair(RewardBus())
    operator = OpBandit(arms=["diff", "full_rewrite", "error_repair", "cross"])
    operator.observe("full_rewrite", first)
    operator.observe("cross", second)
    rng = CaptureChoice()
    choice, propensity, stat = operator.pick(False, rng, cross_available=True)
    assert choice == "cross" and .33 < propensity < .35
    assert min(rng.distribution.values()) > .32
    assert stat["cross"] == second.r_raw and stat["full_rewrite"] == 0
    modes = ModeBandit({"base": 1 / 3, "exploit": 1 / 3, "explore": 1 / 3}, prior_w=0)
    modes.observe(0, "base", first)
    modes.observe(0, "explore", second)
    distribution = modes.dist(0)
    assert all(math.isfinite(p) and .32 < p < .35 for p in distribution.values())
    assert sum(distribution.values()) == pytest.approx(1)


@pytest.mark.parametrize("kernel_class", [UCB1Kernel, UCBSlidingWindowKernel])
def test_actual_sequence_keeps_equal_visit_ucb_probabilities_finite_and_balanced(kernel_class):
    first, second = actual_v11_pair(RewardBus())
    kernel = kernel_class()
    kernel.visits = {"tiny": 1, "real": 1, "other": 1}
    kernel.observe("tiny", first)
    kernel.observe("real", second)
    view = IslandView(ids=["tiny", "real", "other"],
                      scores={"tiny": .4, "real": .98, "other": .6}, inspiration_ids={})
    rng = CaptureChoice()
    chosen = kernel.select(view, None, rng)
    assert chosen.parent_id == "real" and .33 < chosen.propensity < .35
    assert min(rng.distribution.values()) > .32


def test_real_engine_commit_checkpoint_commit_preserves_raw_records_and_learning_credit(
        make_engine, tmp_path, monkeypatch):
    from pathlib import Path

    from sciel.engine import checkpoint
    from sciel.eval.cascade import CascadeOutcome
    from sciel.eval.facade import EvalResult
    from test_crossover_eligibility import DONOR, PARENT, Provider

    first_child = "def run_search():\n    return (10, 0)\n"
    second_child = "def run_search():\n    return (20, 0)\n"
    engine = make_engine(num_islands=3)
    first_parent = engine.db.add(PARENT, .469321433681387, {}, 0, "seed")
    second_parent = engine.db.add(PARENT, .9762760002644632, {}, 1, "seed")
    donor = engine.db.add(DONOR, .3, {}, 1, "gen")
    engine.db.add(PARENT, .9935742900059048, {}, 2, "seed")

    def choose_operator(target, selected):
        stream = target.rng.stream

        class Choose:
            def choices(self, population, *, weights, k):
                assert selected in population and k == 1
                assert all(math.isfinite(p) for p in weights)
                return [selected]

        monkeypatch.setattr(target.rng, "stream",
                            lambda name: Choose() if name == "op" else stream(name))

    def commit(target, island, source, score, op):
        target.llm = Provider(["```python\n" + source + "```"])
        choose_operator(target, op)
        plan = target._prepare_step(island, spawn=False,
            excluded_parents=[donor.id] if island == 1 else [])
        assert plan["snap"].operator_decision["on_policy"] is True

        def evaluate(actual_source):
            assert actual_source.strip() == source.strip()
            return CascadeOutcome(EvalResult(True, score,
                cases={"value": score}, timing=.01), score, "full", False)

        outcome = target._execute_step(plan, evaluate_fn=evaluate)
        target._commit_step(plan, outcome)
        assert len(target.llm.prompts) == 1
        events = [json.loads(line) for line in Path(target.log.path).read_text().splitlines()]
        return next(row for row in reversed(events) if row["event"] == "iteration")

    first = commit(engine, 0, first_child, .46932143368138723, "full_rewrite")
    assert first["parent_id"] == first_parent.id
    assert first["r_raw"] == 2.220446049250313e-16 and first["r_norm"] == 0.0
    assert first["is_island_event"] is True and first["is_pop_event"] is False
    assert first["reward_normalization"]["learning_raw"] == 0.0
    assert engine.db.island_best(0) == first["child_score"]
    saved = checkpoint.save(engine, 0, path=str(tmp_path / "between-events.json"))
    saved_state = json.loads(Path(saved).read_text())
    assert saved_state["reward"]["schema"] == STATE_SCHEMA
    assert saved_state["reward"]["jumps"] == []
    assert saved_state["op_bandit"]["events"]["full_rewrite"] == [0.0]

    resumed = make_engine(num_islands=3)
    checkpoint.load(resumed, saved)
    assert resumed.reward.state() == engine.reward.state()
    assert resumed.db.island_best(0) == first["child_score"]
    second = commit(resumed, 1, second_child, .9837090741491017, "cross")
    assert second["parent_id"] == second_parent.id
    assert second["r_raw"] == second["r_norm"] == .007433073884638541
    assert second["reward_med"] == 1.0
    assert second["reward_normalization"]["previous_sample_count"] == 0
    assert resumed.reward.state()["jumps"] == [second["r_raw"]]
    assert resumed.n_events == 2  # Strict measurement facts remain unchanged.
    assert list(resumed.op_bandit.events["cross"]) == [second["r_norm"]]
    rng = CaptureChoice()
    _, propensity, _ = resumed.op_bandit.pick(False, rng, cross_available=True)
    assert .33 < propensity < .35

    end = checkpoint.save(resumed, 0, path=str(tmp_path / "after-events.json"))
    restored = make_engine(num_islands=3)
    checkpoint.load(restored, end)
    assert restored.reward.state() == resumed.reward.state()
    assert restored.op_bandit.state() == resumed.op_bandit.state()
    assert restored.n_events == 2
