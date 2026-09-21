"""Shared reward calculation for population and island improvements."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


STATE_SCHEMA = "sciel-reward-state/2"
NORMALIZATION_SCHEMA = "sciel-reward-normalization/2"
NORMALIZATION_POLICY = "ulp8-positive-median-cap1/1"
PRECISION_ULPS = 8


def _finite(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"reward {name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"reward {name} must be finite") from error
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"reward {name} must be finite" +
                         (" and nonnegative" if nonnegative else ""))
    return result


def _calibration(values):
    if type(values) is not list:
        raise ValueError("reward calibration jumps must be a list")
    result = [_finite(value, "calibration jump") for value in values]
    if any(value <= 0.0 for value in result):
        raise ValueError("reward calibration jumps must be positive")
    return result


@dataclass
class ImprovementEvent:
    r_raw: float
    r_norm: float
    d_pop: float
    d_isl: float
    is_pop_event: bool
    is_island_event: bool
    origin: str
    iter_ref: int
    med: float
    normalization: dict = field(default_factory=dict)

    @property
    def r(self) -> float:
        return self.r_norm


@dataclass
class RewardBus:
    w_pop: float = 1.0
    w_isl: float = 1.0
    _jumps: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.w_pop = _finite(self.w_pop, "population weight", nonnegative=True)
        self.w_isl = _finite(self.w_isl, "island weight", nonnegative=True)
        self._jumps = _calibration(self._jumps)

    def improvement_event(self, score: float, pre: dict, origin: str,
                          iter_ref: int) -> ImprovementEvent:
        # Validate before changing calibration, including externally modified
        # dataclass fields. Refuse arithmetic overflow, never clip raw evidence.
        w_pop = _finite(self.w_pop, "population weight", nonnegative=True)
        w_isl = _finite(self.w_isl, "island weight", nonnegative=True)
        _calibration(self._jumps)
        score = _finite(score, "score")
        if not isinstance(pre, dict) or not {"pop_best", "island_best"} <= pre.keys():
            raise ValueError("reward requires population and island baselines")
        pop_best = _finite(pre["pop_best"], "population baseline")
        island_best = _finite(pre["island_best"], "island baseline")
        # Non-improvements have exactly zero positive gain; avoid an unused
        # negative subtraction overflowing at opposite finite extremes.
        d_pop = (_finite(score - pop_best, "population delta")
                 if score > pop_best else 0.0)
        d_isl = (_finite(score - island_best, "island delta")
                 if score > island_best else 0.0)
        raw_pop = _finite(w_pop * d_pop, "weighted population delta")
        raw_isl = _finite(w_isl * d_isl, "weighted island delta")
        r_raw = _finite(raw_pop + raw_isl, "weighted delta sum")

        # This resolution heuristic affects learning only. A suppressed gain
        # remains a strict raw record; it is not scientific equivalence or a
        # claim about measurement uncertainty, validity, or task effectiveness.
        pop_threshold = PRECISION_ULPS * max(math.ulp(score), math.ulp(pop_best))
        isl_threshold = PRECISION_ULPS * max(math.ulp(score), math.ulp(island_best))
        pop_eligible = d_pop > pop_threshold
        isl_eligible = d_isl > isl_threshold
        learning_raw = _finite((raw_pop if pop_eligible else 0.0) +
                               (raw_isl if isl_eligible else 0.0), "learning delta sum")
        med = self._median()
        # Branch before dividing so an extreme finite ratio cannot overflow.
        # The cold-start scale remains the explicit legacy convention of 1.0;
        # it is not claimed to be a natural or unit-invariant scale for all tasks.
        r_norm = (0.0 if learning_raw == 0.0 else
                  1.0 if learning_raw >= med else learning_raw / med)
        ev = ImprovementEvent(
            r_raw=r_raw,
            r_norm=r_norm,
            d_pop=d_pop,
            d_isl=d_isl,
            is_pop_event=d_pop > 0.0,
            is_island_event=d_isl > 0.0,
            origin=origin,
            iter_ref=iter_ref,
            med=med,
            normalization={
                "schema": NORMALIZATION_SCHEMA, "policy": NORMALIZATION_POLICY,
                "channel_eligible": {"population": pop_eligible, "island": isl_eligible},
                "ulp_thresholds": {"population": pop_threshold, "island": isl_threshold},
                "learning_raw": learning_raw, "scale": med,
                "scale_source": "prior_positive_median" if self._jumps else "cold_start_unit",
                "cap_applied": learning_raw > med,
                "previous_sample_count": len(self._jumps),
            },
        )
        if learning_raw > 0.0:
            self._jumps.append(learning_raw)
        return ev

    def _median(self) -> float:
        if not self._jumps:
            return 1.0
        s = sorted(self._jumps)
        n = len(s)
        mid = n // 2
        if n % 2:
            return s[mid]
        # Halve the sum when representable to round subnormal midpoints only
        # once. The difference form avoids overflow for large positive values.
        a, b = s[mid - 1], s[mid]
        total = a + b
        return total / 2.0 if math.isfinite(total) else a + (b - a) / 2.0

    def state(self) -> dict:
        return {"schema": STATE_SCHEMA, "normalization_policy": NORMALIZATION_POLICY,
                "w_pop": _finite(self.w_pop, "population weight", nonnegative=True),
                "w_isl": _finite(self.w_isl, "island weight", nonnegative=True),
                "jumps": _calibration(self._jumps)}

    def load_state(self, d: dict) -> None:
        fields = {"schema", "normalization_policy", "w_pop", "w_isl", "jumps"}
        if (type(d) is not dict or set(d) != fields or d["schema"] != STATE_SCHEMA
                or d["normalization_policy"] != NORMALIZATION_POLICY):
            raise ValueError("reward state has an incompatible normalization policy")
        w_pop = _finite(d["w_pop"], "population weight", nonnegative=True)
        w_isl = _finite(d["w_isl"], "island weight", nonnegative=True)
        jumps = _calibration(d["jumps"])
        # Atomic restore: a rejected state cannot partly replace live policy.
        self.w_pop, self.w_isl, self._jumps = w_pop, w_isl, jumps
