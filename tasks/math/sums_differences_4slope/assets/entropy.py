"""Entropy ratios of finite laws on Z^2, the objective of the sums-differences constants (Tao's 3b and 3c); a program
uses it as: from sciel.task import asset; entropy = asset("entropy").

A candidate is {"laws": [{"points": [[x, y], ...], "weights": [w, ...]}, ...], "mix": [t, ...]}. Each law is a
finite probability law on distinct points of Z^2 (weights are normalised by their sum), and the candidate stands for
the pair (X, Y) built from many independent digits, a share t_k of them drawn from law k ("mix" defaults to [1] for
one law). Entropies of a linear form add over independent digits once the digits are spread far enough apart, so per
digit H(aX + bY) = sum_k t_k H(a x_k + b y_k), and the constant is at least

    ratio = sum_k t_k H_k(x - y) / max over the denominator forms (a, b) of sum_k t_k H_k(a x + b y).

FORMS lists the forms of a task: the numerator (1, -1) first, then the denominator forms. A weight or share is a
binary64 number, read exactly, an int, or a pair [numerator, denominator] of ints; zero weights are dropped.
"""

from collections import defaultdict
from fractions import Fraction
import os
import time

import mpmath
import numpy as np
from scipy.optimize import minimize

FORMS_3B = ((1, -1), (1, 0), (0, 1), (1, 1))
FORMS_3C = ((1, -1), (1, 0), (0, 1), (1, 1), (1, 2))
MAX_LAWS = 4
MAX_POINTS = 512
MAX_COORDINATE = 10 ** 4
MAX_BITS = 1100


def task_forms() -> tuple:
    """The forms of the task evaluating this program: (1, -1) and the denominator forms."""
    return FORMS_3C if os.path.basename(os.environ.get("SCIEL_TASK_ROOT", "")) == "sums_differences_4slope" \
        else FORMS_3B


def _rational(w) -> Fraction:
    if isinstance(w, (list, tuple)) and len(w) == 2:
        n, d = w
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (n, d)):
            raise ValueError("a pair weight must be two integers")
        n, d = int(n), int(d)
        if d <= 0:
            raise ValueError("a weight denominator must be positive")
        value = Fraction(n, d)
    elif isinstance(w, (int, np.integer)) and not isinstance(w, bool):
        value = Fraction(int(w))
    elif isinstance(w, (float, np.floating)):
        if not np.isfinite(w):
            raise ValueError("weights must be finite")
        value = Fraction(float(w))
    else:
        raise ValueError(f"a weight must be a float, an int or a pair of ints, got {type(w).__name__}")
    if value < 0:
        raise ValueError("weights must be non-negative")
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise ValueError(f"a weight has more than {MAX_BITS} bits")
    return value


def parse(candidate) -> tuple:
    """([(points, weights)], mix) exactly: points as tuples of ints, weights and mix as Fractions summing to 1."""
    if not isinstance(candidate, dict) or "laws" not in candidate:
        raise ValueError('the candidate must be a dict with "laws"')
    raw = list(candidate["laws"])
    if not 1 <= len(raw) <= MAX_LAWS:
        raise ValueError(f"1 to {MAX_LAWS} laws, got {len(raw)}")
    mix = [_rational(t) for t in candidate.get("mix", [1] * len(raw))]
    if len(mix) != len(raw):
        raise ValueError("mix needs one share per law")
    laws = []
    for law in raw:
        points, weights = list(law["points"]), [_rational(w) for w in law["weights"]]
        if len(points) != len(weights) or not 1 <= len(points) <= MAX_POINTS:
            raise ValueError(f"a law needs as many weights as points, 1 to {MAX_POINTS}")
        pts = []
        for p in points:
            if len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in p):
                raise ValueError("points must be pairs of integers")
            if max(abs(int(v)) for v in p) > MAX_COORDINATE:
                raise ValueError(f"coordinates must be at most {MAX_COORDINATE} in absolute value")
            pts.append((int(p[0]), int(p[1])))
        if len(set(pts)) != len(pts):
            raise ValueError("the points of a law must be distinct")
        total = sum(weights)
        if total == 0:
            raise ValueError("a law has zero total weight")
        laws.append(([p for p, w in zip(pts, weights) if w], [w / total for w in weights if w]))
    total = sum(mix)
    if total == 0:
        raise ValueError("mix has zero total")
    keep = [k for k, t in enumerate(mix) if t]
    return [laws[k] for k in keep], [mix[k] / total for k in keep]


def canonical(candidate) -> dict:
    """The candidate as recorded: weights and shares as [numerator, denominator] pairs exactly as given (not
    normalised), points of zero weight and laws of zero share dropped."""
    parse(candidate)
    raw = list(candidate["laws"])
    mix = [_rational(t) for t in candidate.get("mix", [1] * len(raw))]
    laws = []
    for law, t in zip(raw, mix):
        if t:
            pairs = [([int(p[0]), int(p[1])], _rational(w)) for p, w in zip(law["points"], law["weights"])]
            laws.append({"points": [p for p, w in pairs if w],
                         "weights": [[w.numerator, w.denominator] for p, w in pairs if w]})
    return {"laws": laws, "mix": [[t.numerator, t.denominator] for t in mix if t]}


def _interval_entropy(points, weights, form, iv):
    a, b = form
    image = defaultdict(Fraction)
    for (x, y), w in zip(points, weights):
        image[a * x + b * y] += w
    h = iv.mpf(0)
    for q in image.values():
        qi = iv.mpf(q.numerator) / iv.mpf(q.denominator)
        h -= qi * iv.log(qi)
    return h


def certify(candidate, forms=None, dps=50) -> tuple:
    """(lower, entropies): a rigorous lower bound on the ratio as an mpmath mpf, from exact rational image laws and
    outward-rounded interval logarithms, and the per-digit entropy of each form as an interval."""
    laws, mix = parse(candidate)
    forms = forms or task_forms()
    ctx = mpmath.iv
    old = ctx.dps
    ctx.dps = dps
    try:
        # Endpoints are read out at the same precision, so they are not rounded again.
        with mpmath.workdps(dps):
            entropies = []
            for form in forms:
                h = ctx.mpf(0)
                for (points, weights), t in zip(laws, mix):
                    h += ctx.mpf(t.numerator) / ctx.mpf(t.denominator) * _interval_entropy(points, weights, form,
                                                                                         ctx)
                entropies.append(h)
            top = max(mpmath.mpf(h.b) for h in entropies[1:])
            lower = mpmath.mpf((entropies[0] / ctx.mpf(top)).a)
    finally:
        ctx.dps = old
    return lower, entropies


def exact_value(candidate, forms=None) -> float:
    """The certified lower bound as scored, rounded down to binary64; about 0.1 s for 100 points."""
    return down(certify(candidate, forms or task_forms())[0])


def down(x) -> float:
    """An mpf rounded down to binary64."""
    f = float(x)
    return float(np.nextafter(f, -np.inf)) if mpmath.mpf(f) > x else f


# --- floating point -------------------------------------------------------------------------------------------


def _images(points, form):
    v = points @ np.asarray(form)
    values, index = np.unique(v, return_inverse=True)
    return values.size, index


def entropies_and_jacobian(points, p, forms=None) -> tuple:
    """(E, J) in binary64 for one law: E[f] = H(form_f) in nats, J[f, i] = dE[f]/dp[i] = -(log q_f(image of i) + 1)
    for unnormalised p (sum p = 1 is assumed for E). About 30 us per form for 100 points."""
    points = np.asarray(points, dtype=np.int64).reshape(-1, 2)
    p = np.asarray(p, dtype=float)
    E, J = [], []
    for form in forms or task_forms():
        n, index = _images(points, form)
        q = np.bincount(index, weights=p, minlength=n)
        logq = np.log(np.maximum(q, 1e-300))
        E.append(-(q * logq).sum())
        J.append(-(logq[index] + 1))
    return np.array(E), np.array(J)


def ratio(candidate, forms=None) -> float:
    """The ratio in binary64, relative error about 1e-15."""
    laws, mix = as_float(candidate)
    E = sum(t * entropies_and_jacobian(points, p, forms)[0] for (points, p), t in zip(laws, mix))
    return float(E[0] / E[1:].max())


def value_and_gradient(points, p, forms=None) -> tuple:
    """Ratio of one law and its gradient in p (sum p = 1 assumed), through the largest denominator form; the max is
    not smooth where denominators tie, which is where optima sit, so polish() works with constraints instead."""
    E, J = entropies_and_jacobian(points, p, forms)
    f = 1 + int(np.argmax(E[1:]))
    return float(E[0] / E[f]), (J[0] * E[f] - E[0] * J[f]) / E[f] ** 2


def as_float(candidate) -> tuple:
    """([(points int array (n, 2), p float array)], mix float array) of a candidate, normalised."""
    laws, mix = parse(candidate)
    out = [(np.array(points, dtype=np.int64), np.array([float(w) for w in weights])) for points, weights in laws]
    return [(points, p / p.sum()) for points, p in out], np.array([float(t) for t in mix])


def make(laws, mix=None) -> dict:
    """A candidate from [(points, p)] and mix in floats or ints, as the evaluator accepts it."""
    out = {"laws": [{"points": [[int(x), int(y)] for x, y in np.asarray(points).reshape(-1, 2)],
                     "weights": [float(w) for w in p]} for points, p in laws]}
    if mix is not None:
        out["mix"] = [float(t) for t in mix]
    return out


def load(output) -> dict:
    """The candidate a scored program returned, from its recorded output (a population entry's "output")."""
    return {"laws": [{"points": law["points"], "weights": [list(w) for w in law["weights"]]}
                     for law in output["laws"]], "mix": [list(t) for t in output["mix"]]}


def polish(candidate, forms=None, seconds=10.0) -> dict:
    """Weights and shares re-optimised on the fixed supports by SLSQP on the epigraph problem: maximise the numerator
    entropy over s subject to every denominator entropy <= s, in softmax coordinates (weights stay positive).
    Returns a float candidate, the input when SLSQP finds nothing better."""
    forms = forms or task_forms()
    laws, mix = as_float(candidate)
    K = len(laws)
    cuts = np.cumsum([0] + [p.size for _, p in laws] + [K])

    def softmax(u):
        e = np.exp(u - u.max())
        return e / e.sum()

    def state(z):
        """Entropy vector E, its Jacobian in z (s excluded), and the weights and shares."""
        ws = [softmax(z[cuts[k]:cuts[k + 1]]) for k in range(K + 1)]
        ts = ws[K]
        EJ = [entropies_and_jacobian(laws[k][0], ws[k], forms) for k in range(K)]
        E = sum(t * e for t, (e, _) in zip(ts, EJ))
        G = np.zeros((len(forms), cuts[-1]))
        for k in range(K):
            g = ts[k] * EJ[k][1]
            G[:, cuts[k]:cuts[k + 1]] = ws[k] * (g - (g @ ws[k])[:, None])
        T = np.array([e for e, _ in EJ]).T
        G[:, cuts[K]:] = ts * (T - (T @ ts)[:, None])
        return E, G, ws

    def objective(z):
        E, G, _ = state(z[:-1])
        return -E[0] / z[-1], np.append(-G[0] / z[-1], E[0] / z[-1] ** 2)

    def bound(z):
        return z[-1] - state(z[:-1])[0][1:]

    def bound_jac(z):
        G = state(z[:-1])[1]
        return np.hstack([-G[1:], np.ones((len(forms) - 1, 1))])

    z = np.concatenate([np.log(np.maximum(p, 1e-300)) for _, p in laws] + [np.log(np.maximum(mix, 1e-300))])
    z = np.append(z, state(z)[0][1:].max())
    best, best_value = candidate, ratio(candidate, forms)
    deadline = time.time() + seconds
    while time.time() < deadline:
        result = minimize(objective, z, jac=True, method="SLSQP",
                          constraints=[{"type": "ineq", "fun": bound, "jac": bound_jac}],
                          options={"maxiter": 500, "ftol": 1e-16})
        z = result.x
        ws = state(z[:-1])[2]
        trial = make([(laws[k][0], ws[k]) for k in range(K)], ws[K] if K > 1 else None)
        value = ratio(trial, forms)
        if value <= best_value * (1 + 1e-15):
            break
        best, best_value = trial, value
        z[-1] = state(z[:-1])[0][1:].max()
    return best


def neighbours(points, radius=1) -> np.ndarray:
    """Lattice points within radius (max norm) of the support and not in it: where a support extension starts."""
    points = np.asarray(points, dtype=np.int64).reshape(-1, 2)
    steps = np.array([(i, j) for i in range(-radius, radius + 1) for j in range(-radius, radius + 1) if i or j])
    near = np.unique((points[:, None, :] + steps[None]).reshape(-1, 2), axis=0)
    have = {tuple(p) for p in points}
    return np.array([p for p in near if tuple(p) not in have], dtype=np.int64).reshape(-1, 2)


def extend(candidate, new_points, law=0, weight=1e-6) -> dict:
    """The candidate with new_points added to one law at a small weight each (others scaled down to keep sum 1)."""
    laws, mix = as_float(candidate)
    points, p = laws[law]
    new_points = np.asarray(new_points, dtype=np.int64).reshape(-1, 2)
    p = np.concatenate([p * (1 - weight * len(new_points)), np.full(len(new_points), weight)])
    laws[law] = (np.concatenate([points, new_points]), p)
    return make(laws, mix if len(laws) > 1 else None)
