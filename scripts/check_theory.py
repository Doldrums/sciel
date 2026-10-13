"""Synthetic checks of the sibling-pair and programme e-processes: false discoveries under the null, power at the
effect sizes a run can afford, and the theorist-versus-control test.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sciel.causal.sequential import e_bh, mixture
from sciel.theory.trials import ALPHA, MAX_PAIRS, PROMOTE

REPEATS = 200


def pairs(rng, effect: float, n: int) -> np.ndarray:
    """Sign outcomes of sibling pairs with mean `effect`: win with probability (1 + effect) / 2, else lose."""
    return np.where(rng.random(n) < (1 + effect) / 2, 1.0, -1.0)


def family(rng, effects: list[float]) -> tuple[set, list]:
    """Items stopped at promotion or at the pair budget; e-BH over all of them at the report time."""
    evalues = {}
    for k, effect in enumerate(effects):
        d = pairs(rng, effect, MAX_PAIRS)
        e = 1.0
        for n in range(1, MAX_PAIRS + 1):
            e = mixture(d[:n], 1.0, ALPHA)["e"]
            if e >= PROMOTE:
                break
        evalues[k] = e
    return e_bh(evalues, ALPHA), [k for k, effect in enumerate(effects) if effect == 0]


def main():
    rng = np.random.default_rng(0)
    false, any_false = [], 0
    for _ in range(REPEATS):
        rejected, nulls = family(rng, [0.0] * 8 + [0.2] * 2)
        bad = len(rejected & set(nulls))
        false.append(bad / max(len(rejected), 1))
        any_false += bad > 0
    power = {}
    for effect in (0.1, 0.2, 0.3):
        hits = sum(mixture(pairs(rng, effect, MAX_PAIRS), 1.0, ALPHA)["e"] >= PROMOTE for _ in range(REPEATS))
        power[effect] = hits / REPEATS
    programme = {}
    for p in (0.5, 0.7):
        for n in (20, 37):
            w = rng.random((REPEATS, n)) < p
            programme[f"p={p} pairs={n}"] = float(np.mean([mixture(2.0 * row - 1.0, 1.0, ALPHA)["e"] >= 20
                                                           for row in w.astype(float)]))
    print(json.dumps({"fdr": float(np.mean(false)), "any_false_discovery": any_false / REPEATS,
                      f"power_at_{MAX_PAIRS}_pairs": power, "programme_e_at_least_20": programme}, indent=2))


if __name__ == "__main__":
    main()
