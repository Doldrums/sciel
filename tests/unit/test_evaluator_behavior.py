"""Every evaluator must emit a behavior readout, not just a restatement of its score.

Before this contract the diversity map was fed one number: on heilbronn the two
readout fields were the same quantity (correlation 1.000), and on function
minimization three of four were dropped by a name mismatch. A grid cannot separate
programs on an axis that carries no information, so the behavior slots are checked
here for presence, boundedness and finiteness on every task -- including the failure
path, where they must still be present so a failed program does not land in a
different cell than a scored zero.
"""

import importlib.util
import os
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import pytest

from sciel.components.policy.code_embed import BEHAVIOR_EXTRA_KEYS

ROOT = Path(__file__).resolve().parents[2]
# Every task the engine can actually be pointed at. toy_quadratic is deliberately
# absent: it is a determinism fixture for the tests, not a search task.
TASKS = ["oe/heilbronn_triangle", "oe/circle_packing", "oe/circle_packing_seed",
         "oe/function_minimization", "oe/matmul_tensor",
         "math/beat_the_average", "math/difference_bases",
         "math/edges_vs_triangles", "math/erdos_squarefree"]


def _load(task):
    path = ROOT / "tasks" / task / "evaluator.py"
    spec = importlib.util.spec_from_file_location(f"ev_{task.replace('/', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(src):
    fd, path = tempfile.mkstemp(suffix=".py", prefix="beh_")
    with os.fdopen(fd, "w") as f:
        f.write(textwrap.dedent(src))
    return path


def _check_slots(cases):
    for key in BEHAVIOR_EXTRA_KEYS:
        assert key in cases, f"missing behavior slot {key}"
        v = cases[key]
        assert isinstance(v, float), f"{key} is {type(v)}, not float"
        assert np.isfinite(v), f"{key} is {v}"
        # One scale for every slot. A slot that ranges to 4 while the rest range to 1
        # dominates a linear trunk for no reason but its units.
        assert 0.0 <= v <= 1.0, f"{key} is outside [0,1]: {v}"


@pytest.mark.parametrize("task", TASKS)
def test_seed_program_emits_every_behavior_slot(task):
    ev = _load(task)
    seed = str(ROOT / "tasks" / task / "initial_program.py")
    _check_slots(ev.full(seed)["cases"])


@pytest.mark.parametrize("task", TASKS)
def test_failure_path_emits_every_behavior_slot(task):
    """A program that raises must still produce the slots: otherwise a failure and a
    scored zero occupy different cells for no reason."""
    ev = _load(task)
    path = _write("raise RuntimeError('boom')\n")
    try:
        out = ev.full(path)
    finally:
        os.unlink(path)
    assert out["score"] == 0.0
    _check_slots(out["cases"])


def test_heilbronn_behavior_is_not_a_copy_of_the_score():
    """The readout used to be one degree of freedom. Random valid placements must now
    move the behavior slots without moving in lockstep with the score."""
    ev = _load("oe/heilbronn_triangle")
    rows = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        pts = []
        while len(pts) < 11:
            x, y = rng.random(), rng.random() * np.sqrt(3) / 2
            if y >= 0 and np.sqrt(3) * x <= np.sqrt(3) - y and y <= np.sqrt(3) * x:
                pts.append((x, y))
        path = _write("import numpy as np\n"
                      "def heilbronn_triangle11():\n"
                      f"    return np.array({np.array(pts).tolist()!r})\n")
        try:
            rows.append(ev.full(path)["cases"])
        finally:
            os.unlink(path)

    score = np.array([r["combined_score"] for r in rows])
    varying = 0
    for key in BEHAVIOR_EXTRA_KEYS:
        v = np.array([r[key] for r in rows])
        if v.std() <= 1e-12:
            continue
        varying += 1
        assert abs(np.corrcoef(v, score)[0, 1]) < 0.9, f"{key} tracks the score"
    assert varying >= 4, f"only {varying} behavior slots vary across placements"


def test_readout_has_more_than_one_degree_of_freedom():
    """The gate from the diagnosis: effective dimensionality of the readout, measured
    as the participation ratio of its correlation spectrum. It was 1.00."""
    ev = _load("oe/circle_packing")
    # a lattice keeps the packing VALID; only the radii vary, so the readout has to
    # find its degrees of freedom in the shape, which is exactly the point
    step = 1.0 / 6.0
    lattice = [(step * (0.5 + i % 6), step * (0.5 + i // 6)) for i in range(26)]
    rows = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        centers = np.array(lattice)
        radii = 0.02 + 0.05 * rng.random(26)
        path = _write("import numpy as np\n"
                      "def run_packing():\n"
                      f"    c = np.array({centers.tolist()!r})\n"
                      f"    r = np.array({radii.tolist()!r})\n"
                      "    return c, r, float(r.sum())\n")
        try:
            cases = ev.full(path)["cases"]
        finally:
            os.unlink(path)
        assert cases["validity"] == 1.0, "fixture packing must be valid"
        rows.append(cases)

    keys = sorted(rows[0])
    M = np.array([[float(r[k]) for k in keys] for r in rows])
    keep = M.std(0) > 1e-12
    C = np.corrcoef(M[:, keep].T)
    ev_spectrum = np.clip(np.linalg.eigvalsh(np.nan_to_num(C))[::-1], 0, None)
    p = ev_spectrum / ev_spectrum.sum()
    effective_dims = 1.0 / float((p ** 2).sum())
    assert effective_dims > 2.0, f"readout still nearly one-dimensional: {effective_dims:.2f}"


def test_no_behavior_slot_is_a_restatement_of_the_score():
    """The failure mode this whole readout exists to remove: a slot that is an exact
    function of the score, or of another slot. Both were shipped in the first draft --
    matmul's b_structure was a factor of `combined`, and difference_bases' b_contact
    was b_structure/(n-1) on a task where n IS the score."""
    ev = _load("math/difference_bases")
    rows = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        # a valid Sidon set inside [0, 60], built greedily so sizes and spans vary
        vals, diffs = [], set()
        for cand in rng.permutation(61):
            d = {abs(int(cand) - v) for v in vals}
            if len(d) == len(vals) and not (d & diffs):
                diffs |= d
                vals.append(int(cand))
        path = _write("import numpy as np\n"
                      "def difference_basis():\n"
                      f"    return np.array({sorted(vals)!r})\n")
        try:
            rows.append(ev.full(path)["cases"])
        finally:
            os.unlink(path)

    score = np.array([r["combined_score"] for r in rows])
    cols = {k: np.array([r[k] for r in rows]) for k in BEHAVIOR_EXTRA_KEYS}
    varying = {k: v for k, v in cols.items() if v.std() > 1e-12}
    assert len(varying) >= 4
    for k, v in varying.items():
        assert abs(np.corrcoef(v, score)[0, 1]) < 0.95, f"{k} is a restatement of the score"
    names = sorted(varying)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = abs(np.corrcoef(varying[a], varying[b])[0, 1])
            assert r < 0.999, f"{a} and {b} carry one degree of freedom (|r|={r:.4f})"


def test_four_cycle_count_is_a_four_cycle_count():
    """b_structure on the triangle-free task claims to count 4-cycles. A star and a
    path have none; the first implementation reported 80.75 and 1.75 for them."""
    ev = _load("math/edges_vs_triangles")

    def run(edges, n=16):
        A = np.zeros((n, n), dtype=int)
        for i, j in edges:
            A[i, j] = A[j, i] = 1
        path = _write("import numpy as np\n"
                      "def graph_edges():\n"
                      f"    return np.array({A.tolist()!r})\n")
        try:
            return ev.full(path)["cases"]
        finally:
            os.unlink(path)

    star = run([(0, j) for j in range(1, 16)])
    line = run([(i, i + 1) for i in range(15)])
    square = run([(0, 1), (1, 2), (2, 3), (3, 0)])
    assert star["b_structure"] == 0.0, star["b_structure"]
    assert line["b_structure"] == 0.0, line["b_structure"]
    assert square["b_structure"] > 0.0
