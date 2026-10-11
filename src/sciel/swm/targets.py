"""Goal behaviors for new islands: dense regions of elite behavior that no island occupies yet."""

from __future__ import annotations

import numpy as np

from ..components.policy.code_embed import BEHAVIOR_EXTRA_KEYS, behavior_value
from .diffusion import sample

DESCRIPTIONS = {
    "b_dispersion": "spread of the solution's parts (sizes, areas, positions)",
    "b_boundary_frac": "share of the solution lying on the domain boundary",
    "b_contact": "average contact or coupling between neighbouring parts",
    "b_offset": "displacement of the solution's centroid from the domain centre",
    "b_concentration": "share of the solution carried by its dominant parts",
    "b_structure": "structural diversity of the parts (distinct sizes, rank, discreteness)",
}
MIN_ELITES = 8
MIN_DIFFUSION = 24
CANDIDATES = 256
BANDWIDTH = 0.08
TEMPERATURE = 0.1
NOVELTY = 2.0


def _behavior(program) -> np.ndarray:
    return np.array([behavior_value(program.cases, k) for k in BEHAVIOR_EXTRA_KEYS])


def _log_density(points: np.ndarray, centres: np.ndarray, weights: np.ndarray) -> np.ndarray:
    d2 = ((points[:, None, :] - centres[None]) ** 2).sum(-1)
    kernel = np.exp(-d2 / (2 * BANDWIDTH ** 2)) * weights[None]
    return np.log(kernel.sum(1) + 1e-12)


def propose(db, seed: int, method: str = "kde") -> dict | None:
    """Candidate behaviors from an elite-weighted KDE or a rank-guided diffusion model, scored by density and novelty."""
    programs = [p for p in db._all.values() if np.isfinite(p.score) and _behavior(p).any()]
    if len(programs) < (MIN_DIFFUSION if method == "diffusion" else MIN_ELITES):
        return None
    rng = np.random.default_rng(seed)
    behaviors = np.stack([_behavior(p) for p in programs])
    scores = np.array([p.score for p in programs])
    ranks = scores.argsort().argsort() / max(len(scores) - 1, 1)
    weights = np.exp((ranks - 1) / TEMPERATURE)
    weights /= weights.sum()
    if method == "diffusion":
        candidates = sample(behaviors, ranks, CANDIDATES, seed)
    else:
        picks = rng.choice(len(programs), size=CANDIDATES, p=weights)
        candidates = np.clip(behaviors[picks] + rng.normal(0, BANDWIDTH, size=(CANDIDATES, behaviors.shape[1])), 0, 1)
    occupied = [np.stack([_behavior(p) for p in island.programs.values()]).mean(0) for island in db.islands
                if island.programs]
    distance = (np.sqrt(((candidates[:, None] - np.stack(occupied)[None]) ** 2).sum(-1)).min(1) if occupied
                else np.zeros(len(candidates)))
    best = int(np.argmax(_log_density(candidates, behaviors, weights) + NOVELTY * distance / BANDWIDTH))
    return {k: round(float(v), 3) for k, v in zip(BEHAVIOR_EXTRA_KEYS, candidates[best])}


def directive(target: dict) -> str:
    lines = "\n".join(f"- {DESCRIPTIONS[k]}: about {v:.2f} on a 0-1 scale" for k, v in target.items())
    return ("an approach whose solution reaches this target behavior, a region near strong solutions "
            "that no current island explores:\n" + lines)
