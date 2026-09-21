"""Pure, exact insertion builder for one preregistered LP initializer contrast.

No solver is imported or executed. This separate family is deliberately outside
controlled_changes' recognized model-edit operations. Proofs bind bytes, not
packing quality or a causal mechanism.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import circle_restart_scales as scales

CONTRACT = "sciel-circle-lp-initialization/1"
OPERATION = "circle_lp_radius_initialization/1"
PINNED_BASELINE_SHA256 = "78915475a194be9b871f1fde96eafec12561c87c72bef1aa18370154f770e06f"
PINNED_HELPER_SHA256 = "d8c31889d50402c1cf75e684fb3b93242711084ed8b258813259d5b36f247620"
_HELPER_PATH = Path(__file__).resolve().parents[1] / "solvers" / "circle_lp_initializer.py"
POLICIES = ("control", "lp")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _baseline():
    source = scales.source_for("0.06")
    if _sha(source.encode()) != PINNED_BASELINE_SHA256:
        raise ValueError("LP control differs from pinned Circle source")
    return source


def _helper():
    raw = _HELPER_PATH.read_bytes()
    if _sha(raw) != PINNED_HELPER_SHA256:
        raise ValueError("LP initializer helper differs from pinned source")
    return raw.decode("utf-8")


def build():
    """Return complete standalone treatment and ordered exact insertion receipts."""
    baseline, helper = _baseline(), _helper()
    patches = [
        ("helper", "from scipy.optimize import minimize\n",
         "\n\n# BEGIN frozen Circle LP initializer\n" + helper + "# END frozen Circle LP initializer\n"),
        ("stats", "    LAST_STATS.clear()\n", "    stats.update(new_lp_stats())\n"),
        ("initializer", "            guess[:2 * N_CIRCLES] = np.clip(guess[:2 * N_CIRCLES], 0.001, 0.999)\n",
         "            guess = initialize_radii(guess, deadline=started_at + budget_seconds,\n"
         "                                     stats=stats)\n"
         "            check_deadline(guess)\n"),
    ]
    raw = baseline.encode(); edits = []
    for name, anchor, text in patches:
        if baseline.count(anchor) != 1:
            raise ValueError("LP insertion anchor is not unique: " + name)
        at = raw.index(anchor.encode()) + len(anchor.encode())
        edits.append((at, name, text.encode()))
    edits.sort()
    pieces, receipts, previous, delta = [], [], 0, 0
    unchanged = []
    for at, name, inserted in edits:
        segment = raw[previous:at]
        pieces.extend((segment, inserted))
        unchanged.append({"parent_byte_span": [previous, at], "sha256": _sha(segment)})
        receipts.append({"name": name, "parent_byte_span": [at, at],
            "child_byte_span": [at + delta, at + delta + len(inserted)],
            "text": inserted.decode(), "sha256": _sha(inserted)})
        delta += len(inserted)
        previous = at
    tail = raw[previous:]
    pieces.append(tail)
    unchanged.append({"parent_byte_span": [previous, len(raw)], "sha256": _sha(tail)})
    child = b"".join(pieces)
    proof = {"schema": "sciel-circle-lp-source-edit/1", "contract": CONTRACT, "operation": OPERATION,
        "parent_policy": "control", "target_policy": "lp", "parent_source_sha256": _sha(raw),
        "child_source_sha256": _sha(child), "helper_source_sha256": PINNED_HELPER_SHA256,
        "kernel_source_sha256": scales.PINNED_SOLVER_SHA256, "insertions": receipts,
        "unchanged_segments": unchanged, "unchanged_outside_declared_insertions": True,
        "scope": "Exact source insertion only; no certificate, semantic, causal or quality guarantee."}
    return {"code": child.decode(), "proof": proof}


def source_for(policy):
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("unknown Circle LP initialization policy")
    return _baseline() if policy == "control" else build()["code"]


def identity():
    return {"contract": CONTRACT, "operation": OPERATION,
        "control_source_sha256": PINNED_BASELINE_SHA256,
        "lp_source_sha256": build()["proof"]["child_source_sha256"],
        "helper_source_sha256": PINNED_HELPER_SHA256, "source_proof": build()["proof"],
        "method": "highs", "options": {"threads": 1, "parallel": False, "random_seed": 0},
        "time_limit_seconds": 0.05, "time_limit_scope": "min(0.05, remaining shared inner time)",
        "presolve": "SciPy/HiGHS default", "lp_variables": 26, "lp_pair_inequalities": 325,
        "candidate_clock_scope": "LP and SLSQP share original27s cooperative deadline; outer watchdog authoritative"}
