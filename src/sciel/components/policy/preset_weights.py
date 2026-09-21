"""Small exponential-weights experiment over existing parent-selection presets.

This is a clipped online heuristic, not a regret guarantee. The caller supplies
the actual probability used to draw a preset and a reward for every attempted
generation, including zero for attempts which did not produce a scored child.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .mode_features import MODES


_STATE_VERSION = 1
_PRIOR_FLOOR = 1e-12
_MIN_LOG_WEIGHT = -700.0


class PresetWeights:
    def __init__(self, prior: Mapping[str, float] | None = None,
                 learning_rate: float = 0.1, exploration: float = 0.1,
                 max_importance_reward: float = 10.0):
        self.learning_rate = float(learning_rate)
        self.exploration = float(exploration)
        self.max_importance_reward = float(max_importance_reward)
        if not math.isfinite(self.learning_rate) or self.learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")
        if not math.isfinite(self.exploration) or not 0 < self.exploration <= 1:
            raise ValueError("exploration must be finite and in (0, 1]")
        if not math.isfinite(self.max_importance_reward) or self.max_importance_reward <= 0:
            raise ValueError("max_importance_reward must be finite and positive")
        if not math.isfinite(self.learning_rate * self.max_importance_reward):
            raise ValueError("learning_rate times max_importance_reward must be finite")
        supplied = dict(prior) if prior is not None else {name: 1.0 for name in MODES}
        if set(supplied) - set(MODES):
            raise ValueError(f"unknown presets: {sorted(set(supplied) - set(MODES))}")
        weights = {name: float(supplied.get(name, 0.0)) for name in MODES}
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("prior weights must be finite and nonnegative")
        largest = max(weights.values())
        if largest <= 0:
            raise ValueError("prior must give positive mass to at least one preset")
        # Scale before summing so valid finite priors such as 1e308 cannot overflow.
        scaled = {name: value / largest for name, value in weights.items()}
        total = sum(scaled.values())
        self.prior = {name: value / total for name, value in scaled.items()}
        # A zero prior is not a permanently disabled arm: exploration can teach it.
        self.log_weights = {name: math.log(max(_PRIOR_FLOOR, value))
                            for name, value in self.prior.items()}
        self._recenter()
        self.updates = 0

    def _recenter(self) -> None:
        peak = max(self.log_weights.values())
        self.log_weights = {
            name: max(_MIN_LOG_WEIGHT, value - peak)
            for name, value in self.log_weights.items()
        }

    def probabilities(self) -> dict[str, float]:
        weights = {name: math.exp(self.log_weights[name]) for name in MODES}
        total = sum(weights.values())
        floor = self.exploration / len(MODES)
        return {name: (1.0 - self.exploration) * weights[name] / total + floor
                for name in MODES}

    def sample(self, rng) -> tuple[str, float]:
        probabilities = self.probabilities()
        name = rng.choices(MODES, weights=[probabilities[name] for name in MODES], k=1)[0]
        return name, probabilities[name]

    def update(self, name: str, reward: float, propensity: float) -> None:
        if name not in MODES:
            raise ValueError(f"unknown preset: {name!r}")
        reward, propensity = float(reward), float(propensity)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if not math.isfinite(propensity) or not 0 < propensity <= 1:
            raise ValueError("propensity must be finite and in (0, 1]")
        reward = min(1.0, max(0.0, reward))
        credit = min(self.max_importance_reward, reward / propensity)
        self.log_weights[name] += self.learning_rate * credit
        self._recenter()
        self.updates += 1

    def state(self) -> dict:
        return {
            "version": _STATE_VERSION,
            "prior": dict(self.prior),
            "learning_rate": self.learning_rate,
            "exploration": self.exploration,
            "max_importance_reward": self.max_importance_reward,
            "log_weights": dict(self.log_weights),
            "updates": self.updates,
        }

    def load_state(self, state: dict) -> None:
        if state.get("version") != _STATE_VERSION:
            raise ValueError("unsupported preset weights state version")
        restored = type(self)(
            state["prior"], state["learning_rate"], state["exploration"],
            state["max_importance_reward"],
        )
        if set(state["prior"]) != set(MODES):
            raise ValueError("state must contain prior mass for exactly the known presets")
        saved_prior = {name: float(state["prior"][name]) for name in MODES}
        if not math.isclose(sum(saved_prior.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("state prior must be normalized")
        # Constructor validation is useful, but normalizing an already-normalized
        # checkpoint again can move the last bit on each successive resume.
        restored.prior = saved_prior
        if set(state["log_weights"]) != set(MODES):
            raise ValueError("state must contain weights for exactly the known presets")
        restored.log_weights = {name: float(state["log_weights"][name]) for name in MODES}
        if any(not math.isfinite(value) or not _MIN_LOG_WEIGHT <= value <= 0
               for value in restored.log_weights.values()):
            raise ValueError("state log weights must be finite and in [-700, 0]")
        if max(restored.log_weights.values()) != 0:
            raise ValueError("state log weights must be centred at zero")
        updates = state["updates"]
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("state updates must be a nonnegative integer")
        restored.updates = updates
        # Validate everything before mutating an existing live policy.
        self.__dict__.update(restored.__dict__)
