"""Evaluator for the Shannon capacity of C_7 from below; smoke/full/confirm on the sciel contract.

Theta(C_7) = sup_d alpha(C_7^(box d))^(1/d) (Tao's optimization constants, 9). A candidate returns a certificate: a
recipe for an independent set of size M in C_7^(box d), built from the 367-word base gadget in C_7^(box 5) by Gao's
product lemma, Tandon's heterogeneous product (Theorem 1 of arXiv:2608.30273) with left patterns whose counts
q = |J minus N(X_L^0)| are exact block counts over the base data, and the combining rules and terminal codes of Buys,
Polak and Zuiddam (arXiv:2607.29681). The evaluator recomputes (M, d) in integers with the asset c7 (base gadget
axioms, J^+ exchanges, BPZ admissibility and block sizes are checked on the way) and scores M^(1/d) against L. Tang's
3.258838705444608 (PR 296, d = 680), which it reproduces exactly; the comparison of two bounds is M^d' against
M'^d.
"""

from sciel.candidate_modules import load_candidate
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

_c7 = load_candidate(str(Path(__file__).resolve().parent / "assets" / "c7.py"), "c7")
_frontier = load_candidate(str(Path(__file__).resolve().parent / "assets" / "frontier.py"), "c7_frontier")

# L. Tang, teorth/optimizationproblems PR 296 (commit b5d6c2c of gomyway/c9-shannon-c7-lower-bound-certificate).
RECORD_M, RECORD_D = _frontier.M, _frontier.DIMENSION


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _plain(value):
    """The certificate as plain JSON data (str keys, lists, ints, strings)."""
    return json.loads(json.dumps(value, allow_nan=False))


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the certificate instead, so two recipes with the same bound
# can land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(cert, d):
    """Six bounded descriptors: dimension, number of gadget steps, share of steps using new (S or N) patterns,
    share of sigma steps, nested rule steps, and the top-level arity. Never raises."""
    try:
        steps = cert.get("gadgets", [])
        n = max(len(steps), 1)
        new = sum(1 for s in steps if str(s.get("pattern", ""))[:2] in ("S<", "N<"))
        sig = sum(1 for s in steps if s.get("op") == "sigma")
        k = len(_c7.CODES[cert["top"]["terminal_code"]][0])
        return {"b_dispersion": min(d / 2000.0, 1.0), "b_boundary_frac": min(len(steps) / 40.0, 1.0),
                "b_contact": new / n, "b_offset": sig / n,
                "b_concentration": min(len(cert.get("vectors", [])) / 20.0, 1.0), "b_structure": k / 4.0}
    except (AttributeError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluate(program_path, stage):
    source_sha = _source_hash(program_path)
    evaluator_sha = _source_hash(__file__)
    invocation_id = uuid.uuid4().hex
    program = _load(program_path, "program_" + stage)
    if not hasattr(program, "certificate"):
        raise AttributeError("missing certificate")
    start_time = time.time()
    cert = _plain(program.certificate())
    eval_time = time.time() - start_time
    M, d = _c7.value(cert)
    if M < 1:
        raise ValueError("empty independent set")
    combined = float(_c7.ratio(M, d, RECORD_M, RECORD_D))
    bound = _c7.rate(M, d)
    cases = {"lower_bound": bound, "dimension": float(d), "log_size": float(len(str(M))),
             "record": float(M ** RECORD_D > RECORD_M ** d), "combined_score": combined, "validity": 1.0}
    cases.update(_behavior(cert, d))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"certificate": cert, "M": str(M), "dimension": d},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "full_output": _canonical(witness)}
    if source_sha != _source_hash(program_path) or evaluator_sha != _source_hash(__file__):
        raise ValueError("source or evaluator changed during invocation")
    return {"ok": True, "score": combined, "cases": cases, "artifacts": artifacts}


def smoke(program_path):
    try:
        return _evaluate(program_path, "smoke")
    except Exception as e:
        return _fail(str(e))


def full(program_path):
    try:
        return _evaluate(program_path, "full")
    except Exception as e:
        return _fail(str(e))


def confirm(program_path, n=3):
    """Re-run the program n times and keep the worst result; a record should survive its own
    repetition."""
    n = max(1, int(n or 1))
    runs = [full(program_path) for _ in range(n)]
    failures = [r for r in runs if not r.get("ok")]
    selected = (next(i for i, result in enumerate(runs) if not result.get("ok"))
                if failures else min(range(n), key=lambda i: runs[i]["score"]))
    worst = dict(runs[selected])
    worst["cases"] = {**worst["cases"], "n_trials": float(n)}
    worst["artifacts"] = dict(worst.get("artifacts", {}))
    selection = {"trials": n, "selected_trial": selected,
                 "selection": "first_failure_else_lowest_score"}
    worst["artifacts"]["confirmation_selection"] = _canonical(selection)
    if "full_output" in worst["artifacts"]:
        witness = json.loads(worst["artifacts"]["full_output"])
        witness["confirmation"] = selection
        worst["artifacts"]["full_output"] = _canonical(witness)
    return worst


def _fail(msg):
    return {
        "ok": False,
        "score": 0.0,
        "cases": {"combined_score": 0.0, "lower_bound": 0.0, "dimension": 0.0, "log_size": 0.0, "record": 0.0,
                  "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
