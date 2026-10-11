"""Long-term memory: a CVT archive over every program of the run, evicted ones included, in the world model latent.

The number of cells is the most bootstrap-stable k by adjusted Rand index, or the smallest k when no k is stable;
each cell keeps its best program.
"""

from __future__ import annotations

import numpy as np

KS = (4, 6, 8, 12)
BOOTSTRAPS = 4
ITERATIONS = 25
NEIGHBOURS = 3
STABILITY_SAMPLE = 800
STABLE = 0.5


def _kmeans(z: np.ndarray, k: int, rng) -> np.ndarray:
    centres = z[rng.choice(len(z), size=k, replace=False)]
    for _ in range(ITERATIONS):
        labels = ((z[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        centres = np.stack([z[labels == j].mean(0) if (labels == j).any() else centres[j] for j in range(k)])
    return centres


def _labels(z: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return ((z[:, None] - centres[None]) ** 2).sum(-1).argmin(1)


def _agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index: pair agreement corrected for chance, so it does not grow with k on unstructured data."""
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    pairs = lambda x: (x * (x - 1) / 2).sum()
    both, rows, cols, total = pairs(table), pairs(table.sum(1)), pairs(table.sum(0)), len(a) * (len(a) - 1) / 2
    expected = rows * cols / total
    top = (rows + cols) / 2
    return float((both - expected) / (top - expected)) if top > expected else 1.0


class LatentArchive:
    def __init__(self):
        self.centres = None
        self.elites: list[str] = []
        self.k = 0

    @property
    def ready(self) -> bool:
        return self.centres is not None

    def rebuild(self, world, programs: list, seed: int) -> None:
        programs = [p for p in programs if world.ready and np.isfinite(p.score) and world.latent(p.id) is not None]
        if len(programs) < 2 * max(KS):
            self.centres, self.elites, self.k = None, [], 0
            return
        z = np.stack([world.latent(p.id) for p in programs])
        rng = np.random.default_rng(seed)
        probe = z[rng.choice(len(z), size=min(len(z), STABILITY_SAMPLE), replace=False)]
        best_k, best_stability = None, -1.0
        for k in KS:
            runs = [_labels(probe, _kmeans(z[rng.choice(len(z), size=len(z))], k, rng)) for _ in range(BOOTSTRAPS)]
            stability = np.mean([_agreement(runs[i], runs[j]) for i in range(BOOTSTRAPS) for j in range(i + 1, BOOTSTRAPS)])
            if stability > best_stability:
                best_k, best_stability = k, stability
        if best_stability < STABLE:
            best_k = KS[0]
        self.centres = _kmeans(z, best_k, rng)
        self.k = best_k
        labels = _labels(z, self.centres)
        self.elites = [max((p for p, l in zip(programs, labels) if l == j), key=lambda p: p.score).id
                       for j in range(best_k) if (labels == j).any()]

    def neighbours(self, world, parent_id: str, exclude: set) -> list[str]:
        """Elites of the cells nearest the parent, excluding the parent's own cell."""
        z = world.latent(parent_id)
        if z is None:
            return []
        own = int(_labels(z[None], self.centres)[0])
        elite_z = np.stack([world.latent(e) for e in self.elites])
        cells = _labels(elite_z, self.centres)
        ranked = sorted((float(((world.latent(e) - z) ** 2).sum()), e)
                        for e, c in zip(self.elites, cells) if c != own and e not in exclude)
        return [e for _, e in ranked[:NEIGHBOURS]]

    def state(self) -> dict:
        return {"centres": None if self.centres is None else self.centres.tolist(), "elites": self.elites, "k": self.k}

    def load_state(self, data: dict) -> None:
        self.centres = None if data["centres"] is None else np.asarray(data["centres"], dtype=np.float32)
        self.elites, self.k = list(data["elites"]), int(data["k"])
