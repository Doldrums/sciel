"""Estimators for pi and campaigns. Three tools, distinct jobs:

- IntensityEstimator: EWMA display metric of the improvement-event rate (wandb only).
- PlateauMonitor: betting e-detector (Shin-Ramdas) on the event intensity; anytime-valid,
  segmented by SH phase. Its job is FUTILITY (should a whole campaign arm stop?), not the
  controller's plateau input.
- OpportunityCostMonitor: the controller's plateau input. Not a p-value of
  anomaly -- a forecast of expected budget yield. Models the island-event process as
  diminishing returns K(n) ~ c*ln(n), estimates c from the segment, and declares plateau
  when the expected events over the next `horizon` evals fall below an opportunity-cost
  `threshold`. Segmented by SH phase (a slice zeroes the event stream).
"""

from __future__ import annotations

import math


class IntensityEstimator:
    def __init__(self, halflife: float = 20.0):
        self.halflife = halflife
        self._ewma = 0.0
        self._seen = 0

    def observe(self, is_event: bool) -> None:
        alpha = 1.0 - 0.5 ** (1.0 / self.halflife)
        self._ewma = (1 - alpha) * self._ewma + alpha * (1.0 if is_event else 0.0)
        self._seen += 1

    def upper_bound(self) -> float:
        return self._ewma

    def state(self) -> dict:
        return {"halflife": self.halflife, "ewma": self._ewma, "seen": self._seen}

    def load_state(self, d: dict) -> None:
        self.halflife = d["halflife"]
        self._ewma = d["ewma"]
        self._seen = d["seen"]


class PlateauMonitor:
    """Betting e-detector for a drop in event intensity below threshold (H0: intensity >=
    threshold). Recursion W_t = (1 + lam*(threshold - x)) * W_{t-1}, reset to 1 on any loss
    (Shin-Ramdas e-detector), so a high-then-low stream is detected with ARL(false alarm)
    >= 1/alpha under H0. The bet lam = c/(1-threshold) is fixed and optimal for detecting a
    near-zero rate against the threshold (no estimation lag). Declare plateau when
    W_t >= 1/alpha after min_t clean steps. Segmented by SH phase.
    """

    def __init__(self, alpha: float = 0.1, threshold: float = 0.05, min_t: int = 15,
                 c: float = 0.5):
        self.alpha = alpha
        self.threshold = threshold
        self.min_t = min_t
        self.c = c
        self.t = 0
        self.t_seg = 0
        self.s = 0.0
        self._log_wealth = 0.0
        self.phase = None
        self.declared = False

    def new_segment(self, phase) -> None:
        self.t = 0
        self.t_seg = 0
        self.s = 0.0
        self._log_wealth = 0.0
        self.phase = phase
        self.declared = False

    def _lambda(self) -> float:
        return self.c / (1.0 - self.threshold)

    def observe(self, is_event: bool, phase=None) -> None:
        if phase is not None and phase != self.phase:
            self.new_segment(phase)
        x = 1.0 if is_event else 0.0
        change=self._lambda() * (self.threshold-x)
        raw = self._log_wealth + math.log1p(change) if change>-1 else float('-inf')
        self.t += 1
        self.s += x
        if raw < 0.0:
            self._log_wealth = 0.0
            self.t_seg = 0
        else:
            self._log_wealth = raw
            self.t_seg += 1

    def mean(self) -> float:
        return self.s / self.t if self.t else 0.0

    def log_wealth(self) -> float:
        return self._log_wealth

    @property
    def wealth(self) -> float:
        """Display compatibility only; decisions/checkpoints use uncapped log wealth."""
        try:
            return math.exp(self._log_wealth)
        except OverflowError:
            return float('inf')

    def is_plateau(self) -> bool:
        return self.t_seg >= self.min_t and self._log_wealth >= -math.log(self.alpha)

    def newly_declared(self) -> bool:
        """Edge trigger: fires on each ENTRY into plateau, not once per segment.

        `declared` was only ever set, never cleared, so an island that plateaued,
        recovered and plateaued again got the mode shift only the first time. With
        two milestones that capped the whole step-3 mechanism at three firings per
        island per run. Clearing it when the island recovers restores the intent."""
        if self.is_plateau():
            if not self.declared:
                self.declared = True
                return True
            return False
        self.declared = False          # recovered: arm the trigger again
        return False

    def state(self) -> dict:
        return {"schema":"sciel-plateau-monitor/2", "alpha": self.alpha, "threshold": self.threshold, "min_t": self.min_t,
                "c": self.c, "t": self.t, "t_seg": self.t_seg, "s": self.s,
                "log_wealth": self._log_wealth,
                "wealth": self.wealth if math.isfinite(self.wealth) else None,
                "phase": self.phase, "declared": self.declared}

    def load_state(self, d: dict) -> None:
        if d.get('schema') not in (None,'sciel-plateau-monitor/2'):
            raise ValueError('unsupported plateau monitor state')
        if 'log_wealth' in d:
            value=d['log_wealth']
        else:
            wealth=d.get('wealth')
            if type(wealth) not in (int,float) or not math.isfinite(wealth) or wealth<1:
                raise ValueError('legacy plateau wealth is invalid/nonfinite; lost history cannot be reconstructed')
            value=math.log(wealth)
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError('invalid plateau log wealth')
        parameters={key:d[key] for key in ('alpha','threshold','min_t','c','t','t_seg','s','phase','declared')}
        if (any(type(parameters[k]) not in (int,float) or not math.isfinite(parameters[k])
                for k in ('alpha','threshold','c','s')) or
                not 0<parameters['alpha']<=1 or not 0<=parameters['threshold']<1 or
                not 0<=parameters['c']<=1 or any(type(parameters[k]) is not int or parameters[k]<0
                for k in ('min_t','t','t_seg')) or not math.isfinite(parameters['s'])):
            raise ValueError('invalid plateau monitor state parameters')
        for key,item in parameters.items():
            setattr(self,key,item)
        self._log_wealth=float(value)


class OpportunityCostMonitor:
    """Controller plateau input: expected budget yield, not anomaly.

    Models cumulative island-events in a segment as diminishing returns K(n) ~ c*ln(n),
    estimates c_hat = k / ln(n) from the observed (events k, evals n), and forecasts the
    expected events over the next `horizon` evals as yield = c_hat * ln((n+h)/n). Declares
    plateau once yield < threshold after `min_n` evals -- the tail is expected-empty, so
    continuing base mode there is below the opportunity cost. Segmented by SH phase.
    """

    def __init__(self, threshold: float = 0.5, horizon: int = 50, min_n: int = 15):
        self.threshold = threshold
        self.horizon = horizon
        self.min_n = min_n
        self.n = 0
        self.k = 0
        self.phase = None
        self.declared = False

    def new_segment(self, phase) -> None:
        self.n = 0
        self.k = 0
        self.phase = phase
        self.declared = False

    def observe(self, is_event: bool, phase=None) -> None:
        if phase is not None and phase != self.phase:
            self.new_segment(phase)
        self.n += 1
        if is_event:
            self.k += 1

    def forecast(self, h: int | None = None) -> float:
        h = self.horizon if h is None else h
        if self.n < 2:
            return float("inf")
        c_hat = self.k / math.log(self.n)
        return c_hat * math.log((self.n + h) / self.n)

    def is_plateau(self) -> bool:
        # Require k>=1: an island must have proven productive (landed >=1 record)
        # before it can be "stagnating". With k=0, c_hat=k/ln(n)=0 makes forecast=0,
        # which is "never started", not a diminishing-returns signal -- firing there
        # cut fresh/late-blooming islands into a harmful explore shift.
        return self.n >= self.min_n and self.k >= 1 and self.forecast() < self.threshold

    def newly_declared(self) -> bool:
        """Edge trigger, same reasoning as the shadow detector above: an island that
        plateaus, recovers and plateaus again must be able to shift mode twice."""
        if self.is_plateau():
            if not self.declared:
                self.declared = True
                return True
            return False
        self.declared = False
        return False

    def state(self) -> dict:
        return {"threshold": self.threshold, "horizon": self.horizon,
                "min_n": self.min_n, "n": self.n, "k": self.k,
                "phase": self.phase, "declared": self.declared}

    def load_state(self, d: dict) -> None:
        for key, val in d.items():
            setattr(self, key, val)
