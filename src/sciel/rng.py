"""Named RNG streams derived from one master seed."""

from __future__ import annotations

import hashlib
import random

STREAMS = ("parent", "op", "escalation", "policy", "prompt", "preset", "decision")


def _seed_for(master: int, name: str) -> int:
    h = hashlib.sha256(f"{master}:{name}".encode()).digest()
    return int.from_bytes(h[:8], "big")


class RngHub:
    def __init__(self, master_seed: int):
        self.master_seed = int(master_seed)
        self._streams = {name: random.Random(_seed_for(self.master_seed, name))
                         for name in STREAMS}

    def stream(self, name: str) -> random.Random:
        if name not in self._streams:
            raise KeyError(f"unknown rng stream '{name}'; declared: {STREAMS}")
        return self._streams[name]

    def derive_seed(self, name: str) -> int:
        return self.stream(name).getrandbits(63)

    def state(self) -> dict:
        return {"master_seed": self.master_seed,
                "streams": {name: r.getstate() for name, r in self._streams.items()}}

    def load_state(self, d: dict) -> None:
        self.master_seed = int(d["master_seed"])
        for name, st in d["streams"].items():
            restored = tuple(tuple(x) if isinstance(x, list) else x for x in st)
            self._streams[name].setstate(restored)
