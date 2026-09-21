"""Exact restart-amplitude edits of the pinned incumbent-reference Circle source.

Only source bytes are read. No solver is imported or executed. A proof binds a
controlled source transformation, not a causal mechanism or a quality gain.
"""
from __future__ import annotations

import hashlib
from types import MappingProxyType

from . import circle_interventions as references


CONTRACT = "sciel-circle-restart-scales/1"
OPERATION = "circle_restart_scale/1"
PINNED_SOLVER_SHA256 = "eb529ba2164b329b56ed9789453e430d0351df4c2bd925015a2f1cedca2f978f"
PINNED_BASELINE_SHA256 = "78915475a194be9b871f1fde96eafec12561c87c72bef1aa18370154f770e06f"
BASELINE_POLICY = "0.06"

_EFFECTS = {
    "0.015": "Quarter-scale perturbations may refine nearby packings or revisit the same basin; neither effect is established.",
    "0.03": "Half-scale perturbations may trade exploration for refinement, with uncertain effects on final quality and completed restarts.",
    "0.06": "This is the historical amplitude, not evidence of an optimal scale or a new improvement.",
    "0.12": "Double-scale perturbations may reach other basins or increase constraint work and clipping; quality is not guaranteed.",
}
POLICIES = MappingProxyType({
    policy: MappingProxyType({
        "policy": policy,
        "change": (f"Each restart adds one uniform draw with bounds [-{policy}, +{policy}] "
                   "to the retained incumbent's centers and radii. Reference, draw shape, "
                   "clamps and solver limits remain unchanged."),
        "uncertain_effect": effect,
    }) for policy, effect in _EFFECTS.items()
})


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _policy(policy: str) -> str:
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("unknown Circle restart scale policy")
    return policy


def policy_descriptor(policy: str) -> dict:
    """Return a detached factual operation and an explicitly uncertain effect."""
    return dict(POLICIES[_policy(policy)])


def _assignment(policy: str) -> str:
    policy = _policy(policy)
    return ("            guess = incumbent._params + "
            f"np.random.uniform(-{policy}, {policy}, initial.shape)\n")


def baseline_source() -> str:
    """Read exactly the historical 0.06 incumbent source, without solver import."""
    source = references.source_for("incumbent")
    if (_sha(source.encode("utf-8")) != PINNED_BASELINE_SHA256
            or references.PINNED_SOLVER_SHA256 != PINNED_SOLVER_SHA256):
        raise ValueError("Circle incumbent baseline differs from the pinned historical source")
    if source.count(_assignment(BASELINE_POLICY)) != 1:
        raise ValueError("pinned Circle restart assignment is not unique")
    return source


def _family() -> dict[str, str]:
    baseline = baseline_source()
    return {policy: baseline.replace(_assignment(BASELINE_POLICY), _assignment(policy), 1)
            for policy in POLICIES}


def source_for(policy: str) -> str:
    """Return one canonical full source; numeric aliases are not accepted."""
    policy = _policy(policy)
    return _family()[policy]


def _identify(source: str, family: dict[str, str]) -> str:
    if type(source) is not str:
        raise ValueError("Circle restart scale source must be a string")
    for policy, known in family.items():
        if source == known:
            return policy
    raise ValueError("source is outside the exact Circle restart scale family")


def policy_of(source: str) -> str:
    """Recognize exact complete source bytes, not an AST or matching substring."""
    return _identify(source, _family())


def apply(source: str, target_policy: str) -> dict:
    """Change only uniform bounds; return the same proof shape as reference edits.

    Assignment spans are UTF-8 zero-based half-open offsets. Prefix and suffix
    receipts bind every outside byte. Plan keys distinguish all directed edits.
    """
    target_policy = _policy(target_policy)
    family = _family()
    parent_policy = _identify(source, family)
    if parent_policy == target_policy:
        raise ValueError("Circle restart scale intervention must change the policy")
    before, after = _assignment(parent_policy), _assignment(target_policy)
    prefix, found, suffix = source.partition(before)
    if not found or before in suffix:
        raise ValueError("Circle restart scale assignment is not unique")
    child = prefix + after + suffix
    if child != family[target_policy]:
        raise ValueError("Circle restart scale edit changed bytes outside its assignment")
    parent_bytes, child_bytes = source.encode("utf-8"), child.encode("utf-8")
    prefix_bytes, suffix_bytes = prefix.encode("utf-8"), suffix.encode("utf-8")
    before_bytes, after_bytes = before.encode("utf-8"), after.encode("utf-8")
    proof = {
        "schema": "source-bound-edit/1", "contract": CONTRACT, "operation": OPERATION,
        "parent_policy": parent_policy, "target_policy": target_policy,
        "kernel_source_sha256": PINNED_SOLVER_SHA256,
        "baseline_source_sha256": PINNED_BASELINE_SHA256,
        "parent_source_sha256": _sha(parent_bytes), "child_source_sha256": _sha(child_bytes),
        "plan_key": f"{OPERATION}:{_sha(parent_bytes)}:{target_policy}",
        "before": {"text": before, "sha256": _sha(before_bytes),
                   "byte_span": [len(prefix_bytes), len(prefix_bytes) + len(before_bytes)]},
        "after": {"text": after, "sha256": _sha(after_bytes),
                  "byte_span": [len(prefix_bytes), len(prefix_bytes) + len(after_bytes)]},
        "unchanged_prefix": {"bytes": len(prefix_bytes), "sha256": _sha(prefix_bytes)},
        "unchanged_suffix": {"bytes": len(suffix_bytes), "sha256": _sha(suffix_bytes)},
        "unchanged_outside_assignment": True,
        "scope": "Exact controlled source edit only; no semantic, causal or quality guarantee.",
    }
    return {"code": child, "proof": proof}
