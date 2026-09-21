"""Exact, opt-in restart-reference edits of the measured H30 Circle kernel.

This module reads source bytes; it never imports or executes the solver. The
proof describes a controlled textual operation, not semantic compliance by
arbitrary Python, improvement, determinism, or a causal effect.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import MappingProxyType


CONTRACT = "sciel-circle-interventions/1"
OPERATION = "circle_restart_reference/1"
PINNED_SOLVER_SHA256 = "eb529ba2164b329b56ed9789453e430d0351df4c2bd925015a2f1cedca2f978f"
_SOLVER_PATH = Path(__file__).resolve().parents[1] / "solvers" / "circle_anytime.py"
_OLD_ENTRY = "    return solve_packing()\n"
_NEW_ENTRY = ("    return solve_packing(budget_seconds=27.0, "
              "max_restarts=10000, maxiter=15000)\n")
_DRAW = "np.random.uniform(-0.06, 0.06, initial.shape)"
_REFERENCES = MappingProxyType({
    "initial": "initial",
    "incumbent": "incumbent._params",
    "alternating": "(initial if _ % 2 == 0 else incumbent._params)",
})

# Both mapping levels and all values are immutable. Use policy_descriptor for
# a detached, JSON-serializable prompt value rather than exposing shared state.
POLICIES = MappingProxyType({
    "initial": MappingProxyType({
        "policy": "initial",
        "change": "Every restart adds the unchanged uniform draw to the original grid parameters.",
        "uncertain_effect": "Repeated exploration near the grid may revisit similar basins; quality is not guaranteed.",
    }),
    "incumbent": MappingProxyType({
        "policy": "incumbent",
        "change": "Every restart adds the unchanged uniform draw to the current retained tolerance-feasible incumbent parameters.",
        "uncertain_effect": "Local search around the incumbent may improve refinement or reduce basin diversity; neither effect is established.",
    }),
    "alternating": MappingProxyType({
        "policy": "alternating",
        "change": "Even zero-based restarts use the original grid; odd restarts use the current incumbent, with the unchanged uniform draw.",
        "uncertain_effect": "Alternating references may balance exploration and refinement, but may also spend time on unproductive starts.",
    }),
})


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _policy(policy: str) -> str:
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("unknown Circle restart policy")
    return policy


def policy_descriptor(policy: str) -> dict:
    """Return detached factual edit text and an explicitly uncertain mechanism."""
    return dict(POLICIES[_policy(policy)])


def _assignment(policy: str) -> str:
    return f"            guess = {_REFERENCES[_policy(policy)]} + {_DRAW}\n"


def baseline_source() -> str:
    """Pin the measured kernel and explicitly activate its 27s time allocation.

    The kernel's public default remains 25 restarts. Only this returned copy has
    the H30 fast_time invocation; neither the file nor its other bytes change.
    """
    raw = _SOLVER_PATH.read_bytes()
    if _sha(raw) != PINNED_SOLVER_SHA256:
        raise ValueError("Circle kernel source differs from the pinned H30 source")
    source = raw.decode("utf-8")
    if source.count(_OLD_ENTRY) != 1 or source.count(_assignment("initial")) != 1:
        raise ValueError("pinned Circle source anchors are not unique")
    return source.replace(_OLD_ENTRY, _NEW_ENTRY, 1)


def _family() -> dict[str, str]:
    baseline = baseline_source()
    return {policy: baseline.replace(_assignment("initial"), _assignment(policy), 1)
            for policy in POLICIES}


def source_for(policy: str) -> str:
    """Return one exact complete source from the three-member family."""
    return _family()[_policy(policy)]


def _identify(source: str, family: dict[str, str]) -> str:
    if type(source) is not str:
        raise ValueError("Circle intervention source must be a string")
    for policy, known in family.items():
        if source == known:
            return policy
    raise ValueError("source is outside the exact Circle intervention family")


def policy_of(source: str) -> str:
    """Recognize exact full bytes only, never comments, syntax or a substring."""
    return _identify(source, _family())


def apply(source: str, target_policy: str) -> dict:
    """Change only the restart-reference assignment; reject unknown/no-op edits.

    Byte spans are UTF-8, zero-based, half-open. Prefix and suffix receipts let
    consumers independently reconstruct and verify the unchanged bytes. A plan
    key binds the parent source, versioned operation and canonical target policy,
    so different directed comparisons remain distinct plans.
    """
    target_policy = _policy(target_policy)
    family = _family()
    parent_policy = _identify(source, family)
    if parent_policy == target_policy:
        raise ValueError("Circle intervention must change the restart policy")
    before, after = _assignment(parent_policy), _assignment(target_policy)
    prefix, found, suffix = source.partition(before)
    if not found or before in suffix:
        raise ValueError("Circle intervention assignment is not unique")
    child = prefix + after + suffix
    if child != family[target_policy]:
        raise ValueError("Circle intervention changed bytes outside its assignment")
    parent_bytes, child_bytes = source.encode("utf-8"), child.encode("utf-8")
    prefix_bytes, suffix_bytes = prefix.encode("utf-8"), suffix.encode("utf-8")
    before_bytes, after_bytes = before.encode("utf-8"), after.encode("utf-8")
    baseline_sha = _sha(family["initial"].encode("utf-8"))
    proof = {
        "schema": "source-bound-edit/1", "contract": CONTRACT,
        "operation": OPERATION, "parent_policy": parent_policy,
        "target_policy": target_policy,
        "kernel_source_sha256": PINNED_SOLVER_SHA256,
        "baseline_source_sha256": baseline_sha,
        "parent_source_sha256": _sha(parent_bytes),
        "child_source_sha256": _sha(child_bytes),
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
