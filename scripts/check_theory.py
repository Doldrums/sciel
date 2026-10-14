"""Synthetic checks of the sibling trials and the programme e-process, through the code a run uses: false verdicts
and false discoveries under the null with ties and a verdict checked after every pair, the same with wealth inherited
from a null generation, power at the effect sizes a run can afford, and the theorist-versus-control test.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sciel.causal.sequential import mixture
from sciel.theory.trials import ALPHA, MAX_PAIRS, Trials

REPEATS = 200
TIES = 0.2


def outcome(rng, effect: float) -> float:
    """A sibling pair: a tie with probability TIES, else a win with probability (1 + effect) / 2."""
    if rng.random() < TIES:
        return 0.0
    return 1.0 if rng.random() < (1 + effect) / 2 else -1.0


def generation(rng, effects: list[float], trials: Trials) -> dict:
    """Pairs arrive one at a time and every open trial is updated after each; the run ends at the pair budget."""
    pairs = []
    for _ in range(MAX_PAIRS):
        for k, effect in enumerate(effects):
            pairs.append({"item": f"i{k}", "kind": "strategy", "d": outcome(rng, effect), "applied": True})
        trials.update(pairs)
    statuses = {k: trials.status(f"i{k}") for k in range(len(effects))}
    return {"statuses": statuses, "rejected": set(trials.report()["rejected"]), "wealth": trials.wealth(pairs)}


def errors(runs: list, effects: list[float]) -> dict:
    nulls = {k for k, effect in enumerate(effects) if effect == 0}
    wrong = [sum(run["statuses"][k] in ("culture", "retired") for k in nulls) / len(nulls) for run in runs]
    fdp = [len(run["rejected"] & {f"i{k}" for k in nulls}) / max(len(run["rejected"]), 1) for run in runs]
    power = [np.mean([run["statuses"][k] == "culture" for k in range(len(effects)) if effects[k] > 0]) for run in runs]
    return {"null_verdict_rate": float(np.mean(wrong)), "fdr": float(np.mean(fdp)),
            "power_effect_items": float(np.mean(power))}


def main():
    rng = np.random.default_rng(0)
    effects = [0.0] * 8 + [0.3] * 2
    first, second = [], []
    for _ in range(REPEATS // 4):
        donor = generation(rng, effects, Trials())
        first.append(donor)
        heir = Trials()
        for k in range(len(effects)):
            heir.inherit(f"i{k}", "strategy", donor["wealth"][f"i{k}"])
        second.append(generation(rng, effects, heir))
    power = {}
    for effect in (0.1, 0.2, 0.3):
        runs = [generation(rng, [effect], Trials()) for _ in range(REPEATS // 4)]
        power[effect] = float(np.mean([run["statuses"][0] == "culture" for run in runs]))
    programme = {}
    for p in (0.5, 0.7):
        for n in (20, 37):
            w = rng.random((REPEATS, n)) < p
            programme[f"p={p} pairs={n}"] = float(np.mean([mixture(2.0 * row - 1.0, 1.0, ALPHA)["e"] >= 1 / ALPHA
                                                           for row in w.astype(float)]))
    print(json.dumps({"first_generation": errors(first, effects), "inherited_generation": errors(second, effects),
                      f"culture_rate_within_{MAX_PAIRS}_pairs": power, "programme_e_at_least_1_over_alpha": programme},
                     indent=2))


if __name__ == "__main__":
    main()
