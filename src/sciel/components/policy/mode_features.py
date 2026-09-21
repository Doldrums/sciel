"""Shared selection-mode feature builder -- the single source of truth used by BOTH
offline training (scripts/dt_dataloader.py) and live inference (controller.pick_mode),
so the feature vector cannot drift between train and serve.

Features use ONLY pre-step state known at pick_mode time (before parent/op are chosen):
    [uniform, sh, final,        # SH-phase one-hot
     progress,                  # B_evals / budget in [0,1)
     v_t,                       # pop_best before this step (the value V_t)
     island_best,               # this island's best before this step
     has_gradient]              # 1.0 if the island has a positive elite to exploit
"""

from __future__ import annotations

MODES = ("base", "exploit", "explore")
N_FEATURES = 7
_GRAD_EPS = 1e-9


def mode_features(phase: int, progress: float, v_t: float,
                  island_best: float) -> list:
    return [
        1.0 if phase == 0 else 0.0,     # uniform
        1.0 if phase > 0 else 0.0,      # sh_k
        1.0 if phase == -1 else 0.0,    # final
        float(progress),
        float(v_t),
        float(island_best),
        1.0 if island_best > _GRAD_EPS else 0.0,
    ]
