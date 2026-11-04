"""Evaluator for the 3-sunflower-free capacity from below; smoke/full/confirm on the sciel contract.

mu_3 = lim f(n)^(1/n), f(n) the largest family of subsets of an n-set with no three distinct members A, B, C with
A & B = A & C = B & C (Tao's optimization constants, 49a). A block is a Sperner, sunflower-free family on n points
with an in-port p and an out-port q. Chaining copies of it, each glued at q to the next copy's p by the gluing lemma of
Deuber, Erdos, Gunderson, Kostochka and Meyer (1997, Lemma 4.4), gives families of size c rho^t on (n - 2) t + 2
points, rho the Perron root of [[N10, N11], [N00, N01]], N_ab the members with [p in D] = a, [q in D] = b; hence
mu_3 >= rho^(1/(n - 2)) (the chain argument of IRSI2026 behind every row of the table since September 2026).

A candidate describes its block as a list of nodes. A leaf is an explicit family; a glue node is the lemma's glue of
two earlier nodes plus explicit extra sets. The lemma makes the glue of two valid nodes valid, so the evaluator checks
exactly what the lemma does not cover: every pair and triple through an explicit set, against the whole node. A
leaf's sets are all explicit, so a leaf is checked exhaustively. The exact checks are integer and bit operations; the
rate is a quadratic surd raised to 1/(n - 2), compared with the record through rho^(n' - 2) against rho'^(n - 2).
"""

from sciel.candidate_modules import load_candidate
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

# Koshta (Kos2026, commit c4ee817): 252292 sets on 28 points, N00 = 33969, N01 = N10 = 64728, N11 = 88867,
# rate (64728 + sqrt(3018723123))^(1/26) = 1.5678599024...
RECORD_N = (33969, 64728, 64728, 88867)
RECORD_POINTS = 28
MAX_POINTS = 62
MAX_NODES = 64
MAX_MEMBERS = 4_000_000
# Sum over nodes of explicit sets times node size: the record needs 2.1e8 and checks in about 75 s.
MAX_WORK = 600_000_000
# Rows of one same-kernel group multiplied at once; larger groups are cut into strips of this many rows.
STRIP = 4096


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _drop(x, a):
    """Remove bit a and close the gap, on int64 arrays or ints."""
    return (x & ((1 << a) - 1)) | ((x >> (a + 1)) << a)


def glue(A, na, a, C, nc, b):
    """Lemma 4.4 of DEGKM: {(X - a) | (Y - b) : [a in X] != [b in Y]} on na + nc - 2 points, the points of A other
    than a first, then those of C other than b, each in order."""
    A = np.asarray(A, dtype=np.int64)
    C = np.asarray(C, dtype=np.int64)
    parts = []
    for side in (0, 1):
        x = _drop(A[(A >> a) & 1 == side], a)
        y = _drop(C[(C >> b) & 1 != side], b) << (na - 1)
        if x.size * y.size > MAX_MEMBERS:
            raise ValueError(f"a glue has more than {MAX_MEMBERS} members")
        parts.append((x[:, None] | y[None, :]).ravel())
    return np.concatenate(parts), na + nc - 2


def through(fam, n, E):
    """None if adding E to fam (which may already hold E) keeps every pair and triple through E free of containment
    and 3-sunflowers, else the reason. Two members A, B make a sunflower with E iff A & E == B & E and their parts
    outside E are disjoint, so the members are grouped by A & E and each group is tested by a 0/1 matrix product."""
    fam = np.asarray(fam, dtype=np.int64)
    others = fam[fam != E]
    if ((others & E) == others).any():
        return "containment: a member inside a new set"
    if ((others & E) == E).any():
        return "containment: a new set inside a member"
    K = others & E
    order = np.argsort(K, kind="stable")
    K = K[order]
    cuts = np.flatnonzero(np.diff(K)) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [K.size]))
    groups = np.flatnonzero(ends - starts >= 2)
    if not groups.size:
        return None
    pos = np.array([x for x in range(n) if not E >> x & 1], dtype=np.int64)
    if not pos.size:
        return "a new set is the whole ground set"
    bits = ((others[order][:, None] >> pos[None, :]) & 1).astype(np.float32)
    for g in groups.tolist():
        rows = bits[starts[g]:ends[g]]
        for lo in range(0, rows.shape[0], STRIP):
            G = rows[lo:lo + STRIP] @ rows.T
            G[np.arange(G.shape[0]), lo + np.arange(G.shape[0])] = 1
            if not G.all():
                return "3-sunflower through a new set"
    return None


def _ints(values, n, what):
    out = []
    for v in values:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{what}: sets must be integer bitmasks")
        v = int(v)
        if not 0 <= v < 1 << n:
            raise ValueError(f"{what}: set {v} is outside the {n} points")
        out.append(v)
    return out


def spec(result):
    """The candidate's nodes and ports, normalized: [{"n", "sets"} | {"glue", "sets"}], [p, q]."""
    if not isinstance(result, dict) or set(result) != {"nodes", "ports"}:
        raise ValueError('block() must return {"nodes": [...], "ports": [p, q]}')
    nodes = result["nodes"]
    if not isinstance(nodes, (list, tuple)) or not 1 <= len(nodes) <= MAX_NODES:
        raise ValueError(f"nodes must be a list of 1..{MAX_NODES} nodes")
    out, sizes = [], []
    for k, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"node {k} must be a dict")
        if "glue" in node:
            if set(node) - {"glue", "sets"}:
                raise ValueError(f"node {k}: a glue node has keys glue and sets")
            g = node["glue"]
            if not isinstance(g, (list, tuple)) or len(g) != 4 or not all(
                    isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in g):
                raise ValueError(f"node {k}: glue must be [i, a, j, b]")
            i, a, j, b = map(int, g)
            if not (0 <= i < k and 0 <= j < k and 0 <= a < sizes[i] and 0 <= b < sizes[j]):
                raise ValueError(f"node {k}: glue must join earlier nodes at their points")
            n = sizes[i] + sizes[j] - 2
            entry = {"glue": [i, a, j, b]}
        else:
            if set(node) != {"n", "sets"}:
                raise ValueError(f"node {k}: a leaf has keys n and sets")
            n = node["n"]
            if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
                raise ValueError(f"node {k}: n must be an integer")
            n = int(n)
            entry = {"n": n}
        if not 3 <= n <= MAX_POINTS:
            raise ValueError(f"node {k} has {n} points, outside 3..{MAX_POINTS}")
        entry["sets"] = _ints(node.get("sets", ()), n, f"node {k}")
        if "n" in entry and not entry["sets"]:
            raise ValueError(f"node {k}: a leaf needs sets")
        out.append(entry)
        sizes.append(n)
    ports = result["ports"]
    if not isinstance(ports, (list, tuple)) or len(ports) != 2:
        raise ValueError("ports must be [p, q]")
    p, q = _ints(ports, sizes[-1], "ports")
    if p == q:
        raise ValueError("ports must be two different points")
    return out, [p, q]


def families(nodes):
    """(n, members) of every node, without checks."""
    built = []
    for node in nodes:
        if "glue" in node:
            i, a, j, b = node["glue"]
            fam, n = glue(built[i][1], built[i][0], a, built[j][1], built[j][0], b)
            fam = np.concatenate((fam, np.array(node["sets"], dtype=np.int64)))
        else:
            n, fam = node["n"], np.array(node["sets"], dtype=np.int64)
        if fam.size > MAX_MEMBERS:
            raise ValueError(f"a node has {fam.size} members, above {MAX_MEMBERS}")
        built.append((n, fam))
    return built


def validate(nodes):
    """(n, members) of the last node after checking every node: explicit sets distinct from each other and from the
    glued part, and every pair and triple through them free of containment and 3-sunflowers."""
    sizes, work = [], 0
    for node in nodes:
        if "glue" in node:
            i, _, j, _ = node["glue"]
            sizes.append(sizes[i] + sizes[j] - 2)
        else:
            sizes.append(node["n"])
    built = families(nodes)
    for k, (node, (n, fam)) in enumerate(zip(nodes, built)):
        work += len(node["sets"]) * fam.size
        if work > MAX_WORK:
            raise ValueError(f"explicit sets times node sizes exceed {MAX_WORK}")
        if np.unique(fam).size != fam.size:
            raise ValueError(f"node {k}: repeated member")
        for E in node["sets"]:
            reason = through(fam, n, E)
            if reason:
                raise ValueError(f"node {k}: {reason} ({E})")
    return built[-1]


def port_counts(fam, p, q):
    fam = np.asarray(fam, dtype=np.int64)
    a, b = (fam >> p) & 1, (fam >> q) & 1
    return tuple(int(((a == x) & (b == y)).sum()) for x, y in ((0, 0), (0, 1), (1, 0), (1, 1)))


def rho(N, digits=80):
    """Perron root of [[N10, N11], [N00, N01]] as a Decimal."""
    n00, n01, n10, n11 = N
    with localcontext() as ctx:
        ctx.prec = digits
        return (Decimal(n10 + n01) + Decimal((n10 - n01) ** 2 + 4 * n11 * n00).sqrt()) / 2


def rate(N, n, digits=80):
    """rho^(1/(n - 2)) as a Decimal."""
    with localcontext() as ctx:
        ctx.prec = digits
        return (rho(N, digits).ln() / (n - 2)).exp()


def beats(N, n, M=RECORD_N, m=RECORD_POINTS):
    """rho(N)^(m - 2) > rho(M)^(n - 2), decided at 400 digits; equal surds count as not beating."""
    if (tuple(N), n) == (tuple(M), m):
        return False
    with localcontext() as ctx:
        ctx.prec = 400
        left, right = rho(N, 400) ** (m - 2), rho(M, 400) ** (n - 2)
        if abs(left - right) <= right * Decimal(10) ** -380:
            raise ValueError("rate too close to the record to decide at 400 digits")
        return left > right


RECORD_RATE = rate(RECORD_N, RECORD_POINTS)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the block instead, so two blocks with the same rate can land
# in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(nodes, n, fam, N):
    """Six bounded descriptors: spread of member sizes, members holding exactly one port, explicit sets at the top
    node, mean member size against n/2, nodes used, and ground size. Never raises."""
    try:
        size = np.zeros(fam.size, dtype=np.int64)
        for x in range(n):
            size += (fam >> x) & 1
        mean = float(size.mean())
        spread = float(size.std() / mean) if mean > 0 else 0.0
        out = {"b_dispersion": spread / (1.0 + spread), "b_boundary_frac": (N[1] + N[2]) / fam.size,
               "b_contact": len(nodes[-1]["sets"]) / fam.size, "b_offset": min(abs(mean / n - 0.5) * 2, 1.0),
               "b_concentration": len(nodes) / MAX_NODES, "b_structure": n / MAX_POINTS}
        return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError):
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
    if not hasattr(program, "block"):
        raise AttributeError("missing block")
    start_time = time.time()
    result = program.block()
    eval_time = time.time() - start_time
    nodes, (p, q) = spec(result)
    check_start = time.time()
    n, fam = validate(nodes)
    check_time = time.time() - check_start
    N = port_counts(fam, p, q)
    if min(N) < 1:
        raise ValueError(f"every port count must be at least 1, got N00, N01, N10, N11 = {N}")
    value = rate(N, n)
    combined = float(value / RECORD_RATE)
    cases = {"lower_bound": float(value), "points": float(n), "members": float(fam.size),
             "record": float(beats(N, n)), "combined_score": combined, "validity": 1.0}
    cases.update(_behavior(nodes, n, fam, N))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"nodes": nodes, "ports": [p, q], "port_counts": list(N), "points": n,
                   "rate": str(rate(N, n, 30))},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "check_time": f"{check_time:.4f}",
                 "full_output": _canonical(witness)}
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
        "cases": {"combined_score": 0.0, "lower_bound": 0.0, "points": 0.0, "members": 0.0, "record": 0.0,
                  "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
