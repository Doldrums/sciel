"""Exact standalone source family for one fixed single-circle neighborhood.

No candidate or helper is imported. Source receipts bind the declared edit;
they do not establish semantic adherence of arbitrary code or packing quality.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import circle_restart_scales as scales

CONTRACT = "sciel-circle-relocations/1"
OPERATION = "circle_single_relocation/1"
POLICIES = ("control", "relocate")
PINNED_BASELINE_SHA256 = "78915475a194be9b871f1fde96eafec12561c87c72bef1aa18370154f770e06f"
PINNED_HELPER_SHA256 = "6f95012069beaa4bd7720b295fd0b87f9b9e92c6a2ed2cc87049f947667f5f80"
_HELPER_PATH = Path(__file__).resolve().parents[1] / "solvers" / "circle_relocation.py"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _baseline():
    source = scales.source_for("0.06")
    if _sha(source.encode()) != PINNED_BASELINE_SHA256:
        raise ValueError("relocation control differs from pinned Circle source")
    return source


def _helper():
    raw = _HELPER_PATH.read_bytes()
    if _sha(raw) != PINNED_HELPER_SHA256:
        raise ValueError("relocation helper differs from pinned source")
    return raw.decode("utf-8")


def build():
    """Return treatment code and invertible receipts for every changed byte."""
    baseline, helper = _baseline(), _helper()
    changes = [
        ("helper", "from scipy.optimize import minimize\n", True,
         "\n\n# BEGIN frozen Circle relocation helper\n" + helper +
         "# END frozen Circle relocation helper\n"),
        ("stats", "    LAST_STATS.clear()\n", True,
         "    stats.update(new_relocation_stats())\n"),
        ("noise_assignment", "            guess = incumbent._params + np.random.uniform(-0.06, 0.06, initial.shape)\n", False,
         "            noise = np.random.uniform(-0.06, 0.06, initial.shape)\n"
         "            guess = incumbent._params + noise\n"),
        ("initializer", "            guess[:2 * N_CIRCLES] = np.clip(guess[:2 * N_CIRCLES], 0.001, 0.999)\n", True,
         "            if _ % 4 == 0:\n"
         "                check_deadline()\n"
         "                guess = relocate_circle(incumbent._params, noise, guess, _,\n"
         "                    deadline=started_at + budget_seconds, stats=stats)\n"
         "                check_deadline(guess)\n"),
    ]
    raw = baseline.encode()
    patches = []
    for name, anchor, insert_after, replacement in changes:
        if baseline.count(anchor) != 1:
            raise ValueError("relocation source anchor is not unique: " + name)
        start = raw.index(anchor.encode())
        end = start + len(anchor.encode())
        if insert_after:
            start = end
        patches.append((start, end, name, replacement.encode()))
    patches.sort()
    pieces, receipts, unchanged, previous, delta = [], [], [], 0, 0
    for start, end, name, replacement in patches:
        if start < previous:
            raise ValueError("overlapping relocation edits")
        segment = raw[previous:start]
        unchanged.append({"parent_byte_span": [previous, start], "sha256": _sha(segment)})
        pieces.extend((segment, replacement))
        before = raw[start:end]
        receipts.append({"name": name, "parent_byte_span": [start, end],
            "child_byte_span": [start + delta, start + delta + len(replacement)],
            "before": before.decode(), "after": replacement.decode(),
            "before_sha256": _sha(before), "after_sha256": _sha(replacement)})
        delta += len(replacement) - len(before)
        previous = end
    tail = raw[previous:]
    pieces.append(tail)
    unchanged.append({"parent_byte_span": [previous, len(raw)], "sha256": _sha(tail)})
    child = b"".join(pieces)
    proof = {"schema": "sciel-circle-relocation-source-edit/1", "contract": CONTRACT,
        "operation": OPERATION, "parent_policy": "control", "target_policy": "relocate",
        "parent_source_sha256": _sha(raw), "child_source_sha256": _sha(child),
        "helper_source_sha256": PINNED_HELPER_SHA256,
        "kernel_source_sha256": scales.PINNED_SOLVER_SHA256,
        "edits": receipts, "unchanged_segments": unchanged,
        "unchanged_outside_declared_edits": True,
        "scope": "Exact source edit only; no causal, quality or feasibility certificate."}
    return {"code": child.decode(), "proof": proof}


def source_for(policy):
    if type(policy) is not str or policy not in POLICIES:
        raise ValueError("unknown Circle relocation policy")
    return _baseline() if policy == "control" else build()["code"]


def identity():
    built = build()
    return {"contract": CONTRACT, "operation": OPERATION,
        "control_source_sha256": PINNED_BASELINE_SHA256,
        "relocate_source_sha256": built["proof"]["child_source_sha256"],
        "helper_source_sha256": PINNED_HELPER_SHA256, "source_proof": built["proof"],
        "restart_period": 4, "first_restart_index": 0, "candidate_points": 16,
        "circle_index": "(restart_index // 4) % 26",
        "point_mapping": "((original_uniform_noise[:32] + 0.06) / 0.12).reshape(16,2)",
        "tie_policy": "first maximal nonnegative clearance",
        "rng": "same existing full78 uniform draw; no extra randomness",
        "candidate_clock_scope": "relocation and SLSQP share original27s cooperative deadline; outer watchdog authoritative"}
