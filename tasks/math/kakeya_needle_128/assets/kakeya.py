"""The task's objective for a candidate to build on: from sciel.task import asset; kakeya = asset("kakeya").

Offsets x are 128 values, x[j - 1] the offset of triangle j = conv{(x_j, 0), (x_j + 1/128, 0), (x_j + j/128, 1)}; every
function takes decimal strings or numbers. At height y the section of triangle j is [x_j + j y/128, x_j + 1/128 +
(j - 1) y/128]. The area is unchanged by a common shift of all offsets, so its gradient sums to zero. Between heights
where two section endpoints cross, the union length is affine in y, so the area is exact on those slabs; moving x_j
changes the area by the heights over which the left end of j is the left end of a piece of the union (area falls)
and its right end is the right end of one (area grows).
"""

from pathlib import Path

import numpy as np

from sciel.candidate_modules import load_candidate

N = 128
_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")
_J = np.arange(1, N + 1)
_CHUNK = 2048


def values(x) -> np.ndarray:
    """Offsets as binary64, from decimal strings or numbers."""
    return np.array([float(v) for v in x], dtype=float)


def decimals(x) -> list:
    """Offsets as the decimal strings the evaluator scores: strings as written, numbers rounded to 25 places."""
    return _evaluator.offsets_of(list(x))


def exact_area(x) -> float:
    """The area exactly as the evaluator scores it, after its validity checks; about a second."""
    return float(_evaluator.area(_evaluator.offsets_of(list(x))))


def _heights(x):
    """The crossing heights of section endpoints in (0, 1), with 0 and 1, sorted."""
    slope = np.concatenate([_J, _J - 1]) / N
    intercept = np.concatenate([x, x + 1 / N])
    ds = slope[:, None] - slope[None, :]
    db = intercept[None, :] - intercept[:, None]
    upper = np.triu(ds != 0, 1)
    y = db[upper] / ds[upper]
    y = np.unique(y[(y > 0) & (y < 1)])
    return np.concatenate([[0.0], y, [1.0]])


def _slabs(x):
    """Per slab between consecutive crossing heights: its width, and per triangle whether its left or right end is
    a boundary of the union there, evaluated at the slab's middle height."""
    x = values(x)
    y = _heights(x)
    width = np.diff(y)
    keep = width > 0
    mid, width = ((y[:-1] + y[1:]) / 2)[keep], width[keep]
    for lo in range(0, mid.size, _CHUNK):
        m, w = mid[lo:lo + _CHUNK, None], width[lo:lo + _CHUNK]
        left = x + _J / N * m
        right = x + 1 / N + (_J - 1) / N * m
        order = np.argsort(left, axis=1, kind="stable")
        ls = np.take_along_axis(left, order, 1)
        rs = np.take_along_axis(right, order, 1)
        reach = np.maximum.accumulate(rs, axis=1)
        prev = np.concatenate([np.full((m.shape[0], 1), -np.inf), reach[:, :-1]], axis=1)
        length = (reach - np.maximum(ls, prev)).sum(axis=1)
        start = ls > prev
        end = np.concatenate([start[:, 1:], np.ones((m.shape[0], 1), bool)], axis=1)
        # The interval giving a piece its right end is the last one that raised the running maximum.
        owner = np.maximum.accumulate(np.where(rs >= reach, np.arange(N), 0), axis=1)
        yield w, length, order, start, np.take_along_axis(order, owner, 1), end


def area(x) -> float:
    """The union area in floating point; agrees with exact_area to about 1e-15."""
    return float(sum(w @ length for w, length, *_ in _slabs(x)))


def _exposure(x):
    total, left, right = 0.0, np.zeros(N), np.zeros(N)
    for w, length, order, start, owner, end in _slabs(x):
        total += w @ length
        left += np.bincount(order[start], np.broadcast_to(w[:, None], start.shape)[start], N)
        right += np.bincount(owner[end], np.broadcast_to(w[:, None], end.shape)[end], N)
    return float(total), left, right


def active(x) -> tuple:
    """Per triangle, the measure of heights over which its left end is a left boundary of the union and over which
    its right end is a right boundary. A triangle with both zero is hidden and moves freely."""
    return _exposure(x)[1:]


def subgradient(x) -> tuple:
    """The area and its gradient in the offsets, right exposure minus left exposure; exact where the boundary set is
    unique, one subgradient at ties."""
    total, left, right = _exposure(x)
    return total, right - left


def slide(x, start: int, stop: int, delta: float) -> np.ndarray:
    """Triangles start + 1 .. stop (offsets x[start:stop]) moved together by delta. A block keeps its own overlaps
    and changes only those with its neighbours, so it crosses barriers single offsets cannot."""
    x = values(x).copy()
    x[start:stop] += delta
    return x
