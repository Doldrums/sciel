"""Mode bandit: online contextual bandit over selection modes.

Per-phase, mean-reward Boltzmann. Replaces the controller's static base_mix in the
non-plateau regime: it learns, per SH phase, which selection mode {base, exploit,
explore} yields the most normalized reward. Warm-started from base_mix (via a prior
pseudo-count) so early behaviour matches the tuned static mix, then adapts online as
real reward accrues.

Value semantics differ from OpBandit on purpose: mean over ALL attempts (r_norm=0 on
non-improvements), so a mode's value = record-rate x mean-record-size -- the right
quantity for choosing a mode, not OpBandit's windowed max (which measures best-case).
"""

from __future__ import annotations

from collections import deque

from .base import boltzmann
from .mode_features import MODES


class ModeBandit:
    def __init__(self, base_mix: dict, temp: float = 0.5, window: int = 50,
                 prior_w: float = 3.0):
        if temp <= 0.0:
            raise ValueError("mode_bandit_temp must be > 0")
        if prior_w < 0.0:
            raise ValueError("mode_bandit_prior must be >= 0")
        self.prior = {m: float(base_mix.get(m, 0.0)) for m in MODES}
        self.temp = temp
        self.window = window
        self.prior_w = prior_w
        self.events: dict = {}

    def _win(self, phase: int, mode: str) -> deque:
        buf = self.events.get((phase, mode))
        if buf is None:
            buf = self.events[(phase, mode)] = deque(maxlen=self.window)
        return buf

    def _stat(self, phase: int, mode: str) -> float:
        """Empirical mean reward for this arm. A VALUE, in reward units."""
        buf = self._win(phase, mode)
        return sum(buf) / len(buf) if buf else 0.0

    def dist(self, phase: int) -> dict:
        """Mixture of the base_mix prior and the learned Boltzmann policy.

        The two used to be averaged inside `_stat`, which added a probability from
        base_mix to a mean r_norm as if they had the same units. Two consequences,
        both measured: the cold start was NOT base_mix (with {base: 0.8, exploit: 0.2}
        it dealt base 0.665 and `explore` -- weight zero, deliberately absent -- 0.134),
        and the learned half was inert, because r_norm means differ by ~0.1 against
        temp 0.5 while the prior term dominated.

        Mixing two DISTRIBUTIONS instead is unit-correct: at n=0 it is exactly
        base_mix, and the learned policy takes over as evidence accrues."""
        counts = {m: len(self._win(phase, m)) for m in MODES}
        n = sum(counts.values())
        learned = boltzmann({m: self._stat(phase, m) for m in MODES}, self.temp)
        w = n / (self.prior_w + n) if (self.prior_w + n) > 0 else 1.0
        mixed = {m: (1.0 - w) * self.prior.get(m, 0.0) + w * learned[m] for m in MODES}
        total = sum(mixed.values())
        if total <= 0:
            return {m: 1.0 / len(MODES) for m in MODES}
        return {m: v / total for m, v in mixed.items()}

    def observe(self, phase: int, mode: str, ev) -> None:
        """One ATTEMPT under `mode`. `ev` may be None for an attempt that produced no
        scored child; that counts as reward 0.

        The denominator matters. This was fed only from the success path, so a mode's
        "mean over ALL attempts" was really a mean over successful evaluations, and
        any mode that causes more failures was rewarded for the ones it lost: over a
        60-eval run exploit dropped 13.6% of its attempts to failure against base's
        7.1%, inflating exploit's value by ~7% relative from censoring alone."""
        if mode in MODES:
            self._win(phase, mode).append(float(getattr(ev, "r_norm", 0.0) or 0.0))

    def state(self) -> dict:
        return {"temp": self.temp, "window": self.window, "prior_w": self.prior_w,
                "prior": dict(self.prior),
                "events": {f"{p}|{m}": list(buf) for (p, m), buf in self.events.items()}}

    def load_state(self, d: dict) -> None:
        self.temp = d["temp"]
        self.window = d["window"]
        self.prior_w = d["prior_w"]
        self.prior = {m: d["prior"].get(m, 0.0) for m in MODES}
        self.events = {}
        for k, vals in d["events"].items():
            p, m = k.rsplit("|", 1)
            self.events[(int(p), m)] = deque(vals, maxlen=self.window)
