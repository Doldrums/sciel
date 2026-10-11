"""Working memory: niches are a VQ codebook in the world model's latent, rebuilt whenever the latent is refitted.

Before the first fit the codebook lives in the task's behavior descriptors.
"""

from __future__ import annotations

import numpy as np

from ..components.policy.code_embed import BEHAVIOR_EXTRA_KEYS, VQBook, behavior_value


class LatentNiches:
    def __init__(self, world, d_min: float = 0.35, target_k: int = 6, k_max: int = 12):
        self.world = world
        self.options = {"d_min": d_min, "target_k": target_k, "k_max": k_max}
        self.book = VQBook(**self.options)
        self.space = "behavior"

    def _vector(self, code: str, cases: dict) -> np.ndarray:
        if self.space == "behavior":
            return np.array([behavior_value(cases, k) for k in BEHAVIOR_EXTRA_KEYS], dtype=np.float32)
        return self.world.encode(self.world.features(code, cases))

    def assign(self, code: str, cases: dict) -> int:
        return self.book.assign(self._vector(code, cases))

    def assign_fixed(self, code: str, cases: dict) -> int:
        return self.book.assign_fixed(self._vector(code, cases))

    @property
    def k(self) -> int:
        return len(self.book.centroids)

    def rebuild(self, programs: list) -> None:
        self.space = "latent" if self.world.ready else "behavior"
        self.book = VQBook(**self.options)
        for program in programs:
            self.book.assign(self._vector(program.code, program.cases))

    def state(self) -> dict:
        return {"space": self.space, "book": self.book.state()}

    def load_state(self, data: dict) -> None:
        self.space = data["space"]
        self.book.load_state(data["book"])
