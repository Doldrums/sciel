"""pi controller: static mode-mix over selection modes.

pick_mode samples a selection mode from a static mixture and returns the tilt that
realizes it (base = the active kernel unchanged). Off (enabled=False) draws no RNG and
returns the base mode, so a run without policy is byte-identical. on_plateau shifts the
mixture for a hold window (step 3, gated by plateau_shift).
"""

from __future__ import annotations

from .base import floor_clip, sample
from .mode_bandit import MODES, ModeBandit
from .mode_policy_mlp import ReturnConditionedModeMLP
from .tilt import Tilt, elite_target, uniform_target


# Score spread below this is not a real gradient: real scores are O(0.01-1) across
# tasks, while degenerate candidates land at ~1e-16 (machine epsilon). Without this
# floor, `max > min` treats a 1e-16 "elite" as exploitable and traps parent selection
# on a degenerate basin.
_GRAD_EPS = 1e-9


def _has_gradient(view) -> bool:
    """True if the island has a score spread worth exploiting (task-agnostic:
    catches an all-zero elite, an all-equal population, and an all-epsilon
    'elite' of degenerate candidates)."""
    s = [view.scores[i] for i in view.ids]
    return len(s) >= 1 and (max(s) - min(s)) > _GRAD_EPS

_KEYS = {"enabled", "base_mix", "plateau_mix", "mode_floor", "tilt_beta", "elite_k",
         "plateau_hold", "plateau_shift", "plateau_shadow", "plateau_threshold",
         "plateau_horizon", "plateau_min_n",
         "mode_bandit", "mode_bandit_temp", "mode_bandit_window", "mode_bandit_prior",
         "mode_source", "mode_policy_path", "mode_beta", "mode_target_rtg"}


class Controller:
    def __init__(self, cfg_policy: dict, num_islands: int):
        unknown = set(cfg_policy) - _KEYS
        if unknown:
            raise ValueError(f"unknown policy keys: {sorted(unknown)}")
        self.enabled = bool(cfg_policy.get("enabled"))
        self.base_mix = dict(cfg_policy.get("base_mix", {"base": 1.0}))
        self.plateau_mix = dict(cfg_policy.get("plateau_mix", self.base_mix))
        self.floor = cfg_policy.get("mode_floor", 1e-3)
        self.beta = cfg_policy.get("tilt_beta", 0.3)
        self.elite_k = cfg_policy.get("elite_k", 3)
        self.hold_windows = cfg_policy.get("plateau_hold", 50)
        self.plateau_shift = bool(cfg_policy.get("plateau_shift"))
        for mix in (self.base_mix, self.plateau_mix):
            bad = set(mix) - set(MODES)
            if bad:
                raise ValueError(f"unknown selection modes {sorted(bad)}; known {MODES}")
        # Step 2 (online RL): a per-phase bandit that learns the non-plateau mode mix,
        # warm-started from base_mix. None -> static base_mix (byte-identical when off).
        self.mode_bandit = (
            ModeBandit(self.base_mix,
                       temp=cfg_policy.get("mode_bandit_temp", 0.5),
                       window=cfg_policy.get("mode_bandit_window", 50),
                       prior_w=cfg_policy.get("mode_bandit_prior", 3.0))
            if cfg_policy.get("mode_bandit") else None)
        # Stage A: a frozen return-conditioned MLP as the non-plateau mode source,
        # blended into base_mix as a tilt (beta). Trained offline; no in-run updates.
        self.mode_source = cfg_policy.get("mode_source")
        self.mode_mlp = (ReturnConditionedModeMLP.load(cfg_policy["mode_policy_path"])
                         if self.mode_source == "mlp" else None)
        self.mode_beta = cfg_policy.get("mode_beta", 0.5)
        self.target_rtg = cfg_policy.get("mode_target_rtg", 1.0)
        self.plateau_until = {i: -1 for i in range(num_islands)}

    def needs_tilt(self) -> bool:
        """Can a non-base mode ever be drawn? The engine uses this to refuse a
        selection kernel that ignores tilt.

        It must ask about the FLOOR-CLIPPED mix, which is what pick_mode samples from.
        Testing the raw mix said "no" for base_mix {base: 1.0}, while floor_clip still
        handed every other mode `mode_floor` of the mass -- so a tilt-blind kernel was
        accepted, those draws ran plain selection, and the journal recorded a mode and
        a propensity that did not match the distribution actually sampled."""
        if not self.enabled:
            return False
        if self.mode_bandit is not None or self.mode_mlp is not None:
            return True  # a learned source can draw exploit/explore in any phase
        for mix in (self.base_mix, self.plateau_mix):
            clipped = floor_clip(mix, self.floor)
            if any(w > 0 for m, w in clipped.items() if m != "base"):
                return True
        return False

    def in_plateau(self, island: int, t: int) -> bool:
        return self.plateau_until.get(island, -1) > t

    def pick_mode(self, island: int, t: int, rng, view=None, phase=None,
                  features=None, adjust=None) -> tuple:
        """Returns (mode, propensity, tilt, mix) -- mix is the floor-clipped
        distribution actually sampled from (for honest observability)."""
        if not self.enabled:
            return "base", 1.0, None, None
        if self.in_plateau(island, t):
            mix = self.plateau_mix
        elif self.mode_mlp is not None and features is not None:
            # step 3 (Stage A): return-conditioned MLP, blended into base_mix as a
            # tilt of strength mode_beta (base_mix is the index prior; I-5 floor below).
            mlp_mix = self.mode_mlp.dist(features, self.target_rtg)
            b = self.mode_beta
            mix = {m: (1 - b) * self.base_mix.get(m, 0.0) + b * mlp_mix[m]
                   for m in MODES}
        elif self.mode_bandit is not None and phase is not None:
            mix = self.mode_bandit.dist(phase)          # step 2: learned mix
        else:
            mix = self.base_mix                          # step 1: static mix
        mix = floor_clip(mix, self.floor)
        if adjust is not None:
            mix = adjust(mix)
        mode = sample(mix, rng)
        tilt = self._tilt_for(mode)
        # Warmup gate: exploit tilts parent selection toward the island's top-k,
        # but with no score gradient (a fresh island whose elite is all-zero) that
        # top-k is garbage -- concentrating on it starves the island of the diverse
        # sampling it needs to land its first record, and SH then cuts it. Drop the
        # tilt so selection stays base-like (rho untouched -> honest propensity).
        if mode == "exploit" and view is not None and not _has_gradient(view):
            tilt = None
        return mode, mix[mode], tilt, mix

    def _tilt_for(self, mode: str):
        if mode == "exploit":
            k = self.elite_k
            return Tilt(self.beta, lambda view: elite_target(view, k))
        if mode == "explore":
            return Tilt(self.beta, uniform_target)
        return None

    def observe(self, phase: int, mode: str, ev, on_policy: bool = True) -> None:
        """Feed the mode bandit the reward from an iteration that used `mode`
        (the effective mode -- a warmup-gated exploit is reported as base).

        `on_policy=False` drops the sample. pick_mode short-circuits to plateau_mix
        when the island is holding, so those iterations were never drawn from the
        bandit's own distribution; crediting them anyway fed the non-plateau estimator
        the run's systematically lowest-yield steps, under a mix that over-weights
        exploit, and dragged exploit's learned value down for choices it never made."""
        if self.mode_bandit is not None and on_policy:
            self.mode_bandit.observe(phase, mode, ev)

    def on_plateau(self, island: int, t: int, islands: int = 1) -> None:
        """Hold this island in the plateau mix for `plateau_hold` of ITS OWN steps.

        `t` is the global iteration counter, so a hold of 40 expressed in global steps
        is ~13 steps of the island's own with three islands and ~7 with six. The hold
        is documented and configured per island, so the window is scaled by how many
        islands share the iteration stream."""
        if self.plateau_shift:
            self.plateau_until[island] = t + self.hold_windows * max(1, int(islands))

    def state(self) -> dict:
        d = {"plateau_until": {str(i): v for i, v in self.plateau_until.items()}}
        if self.mode_bandit is not None:
            d["mode_bandit"] = self.mode_bandit.state()
        return d

    def load_state(self, d: dict) -> None:
        self.plateau_until = {int(i): v for i, v in d["plateau_until"].items()}
        if self.mode_bandit is not None and "mode_bandit" in d:
            self.mode_bandit.load_state(d["mode_bandit"])
