"""Prospective zero-margin screening. No finite screen guarantees future quality."""
from __future__ import annotations

import copy
import math
import re


POLICY = {
    "margin": 0,
    "min_pairs": 3,
    "contrasts": [["grid", "pool"], ["pool_presets_fixed", "pool_presets"]],
    "futility": "first_complete_valid_pair_below_control",
    "success": "all_planned_pairs_valid_nonnegative_and_mechanisms_active",
    "automatic_default_promotion": False,
}


def evaluate_policy(manifest, observations):
    """Derive decisions and future withdrawals from verified, started jobs.

    The caller validates file/job/protocol identity and supplies an observation
    for *every* started job (even an interrupted or ledger-only job). ``valid``
    means all independent numeric draws passed, not merely that a score exists.
    Missing observations are never scores of zero. Failed draws remain in the
    separate descriptive analysis, but cannot establish a valid paired result.
    """
    jobs = manifest["jobs"]
    seeds = manifest["seeds"]
    expected_arms = {arm for pair in POLICY["contrasts"] for arm in pair}
    if (len(seeds) != POLICY["min_pairs"] or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("strict screening requires three unique non-negative seeds")
    by_id = {job["id"]: job for job in jobs}
    by_pair = {(job["arm"], job["seed"]): job for job in jobs}
    if (len(by_id) != len(jobs) or len(by_pair) != len(jobs)
            or set(by_pair) != {(arm, seed) for arm in expected_arms for seed in seeds}):
        raise ValueError("strict screening requires exactly the two frozen contrasts")
    if set(observations) - set(by_id):
        raise ValueError("observation does not belong to the frozen jobs")
    for ident, observation in observations.items():
        if any(type(observation.get(key)) is not bool
               for key in ("complete", "valid", "mechanism_activated")):
            raise ValueError("observation states must be explicit booleans")
        if observation["valid"] and not observation["complete"]:
            raise ValueError("an incomplete observation cannot be valid")
        if observation["complete"] and observation["valid"]:
            score = observation.get("primary_score")
            digest = observation.get("result_sha256")
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or score < 0):
                raise ValueError("valid pairs require finite non-negative scores")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("valid pairs require the verified result SHA256")
    contrasts, withdrawals = [], []
    for control, candidate in POLICY["contrasts"]:
        pairs, first_loss = [], None
        for seed in seeds:
            pair_jobs = [by_pair[(arm, seed)] for arm in (control, candidate)]
            pair_obs = [observations.get(job["id"]) for job in pair_jobs]
            if not all(obs and obs["complete"] and obs["valid"] for obs in pair_obs):
                continue
            base, test = pair_obs
            pair = {"seed": seed, "control": base["primary_score"],
                    "candidate": test["primary_score"],
                    "difference": test["primary_score"] - base["primary_score"],
                    "mechanisms_active": all(obs["mechanism_activated"] for obs in pair_obs),
                    "evidence": [{"job_id": job["id"],
                                  "result_sha256": obs["result_sha256"],
                                  "primary_score": obs["primary_score"]}
                                 for job, obs in zip(pair_jobs, pair_obs)]}
            pairs.append(pair)
            # Deliberately no epsilon or averaging a loss against another seed.
            if test["primary_score"] < base["primary_score"] and first_loss is None:
                first_loss = pair
        decision = "inconclusive"
        if first_loss is not None:
            decision = "rejected_observed_regression"
            later_seeds = seeds[seeds.index(first_loss["seed"]) + 1:]
            for job in jobs:
                if job["arm"] in (control, candidate) and job["seed"] in later_seeds:
                    if job["id"] in observations:
                        raise ValueError("a contrast continued after its prospective futility stop")
                    withdrawals.append({"job": copy.deepcopy(job), "control": control,
                        "candidate": candidate, "trigger_seed": first_loss["seed"],
                        "evidence": copy.deepcopy(first_loss["evidence"])})
        elif len(pairs) == len(seeds) and all(pair["mechanisms_active"] for pair in pairs):
            decision = "screen_pass"
        contrasts.append({"control": control, "candidate": candidate, "decision": decision,
            "n_pairs": len(pairs), "pairs": pairs,
            "first_regression_seed": first_loss["seed"] if first_loss else None,
            "automatic_default_promotion": False})
    return {"schema": "sciel-zero-regression-disposition/1",
            "protocol_sha256": manifest["protocol_sha256"], "policy": copy.deepcopy(POLICY),
            "contrasts": contrasts, "withdrawals": withdrawals,
            "interpretation": "Prospective futility screening on one task; screen_pass is not a guarantee of no future regression and never promotes defaults."}


def verify_disposition(previous, current):
    """Resume may add pairs/withdrawals, never undo or rewrite their evidence."""
    for key in ("schema", "protocol_sha256", "policy"):
        if previous.get(key) != current.get(key):
            raise ValueError("strict disposition identity changed")
    old = previous.get("withdrawals")
    new = current.get("withdrawals")
    if not isinstance(old, list) or not isinstance(new, list):
        raise ValueError("strict disposition withdrawals must be lists")
    if len({row["job"]["id"] for row in old}) != len(old):
        raise ValueError("duplicate historical withdrawal")
    lookup = {row["job"]["id"]: row for row in new}
    if len(lookup) != len(new) or any(lookup.get(row["job"]["id"]) != row for row in old):
        raise ValueError("strict disposition lost or modified prior futility evidence")
    pairs = {(row["control"], row["candidate"], pair["seed"]): pair
             for row in current["contrasts"] for pair in row["pairs"]}
    for row in previous["contrasts"]:
        for pair in row["pairs"]:
            if pairs.get((row["control"], row["candidate"], pair["seed"])) != pair:
                raise ValueError("strict disposition lost or modified a prior compared pair")
