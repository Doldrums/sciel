"""The task's objective for a candidate to build on: from sciel.task import asset; cover = asset("cover").

A periodic measure is ((P_x, P_y), [(x, y, mass), ...]) with the atoms in [0, P_x) x [0, P_y). A point z has
M mu(z) >= 1 when some atoms of total mass m fit in the square of side sqrt(m) centred at z; for the atoms in a box
[x_lo, x_hi] x [y_lo, y_hi] these z form the witness rectangle [x_hi - L/2, x_lo + L/2] x [y_hi - L/2, y_lo + L/2],
L = sqrt(m). The bound is the area of the union of the witness rectangles of the 3 x 3 block of periods inside the
period box, over the mass of the box; it is invariant to scaling space by s and the masses by s^2.
"""

from fractions import Fraction
import math
from pathlib import Path

from sciel.candidate_modules import load_candidate

_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")
MAX_ATOMS = _evaluator.MAX_ATOMS
GRID = _evaluator.GRID


def load(output) -> tuple:
    """((P_x, P_y), atoms) as Fractions, from the output a scored program recorded (a population entry's
    "output"), where every number is kept exactly as the string 'p/q'."""
    return (tuple(Fraction(v) for v in output["period"]),
            [tuple(Fraction(v) for v in atom) for atom in output["atoms"]])


def rational(value, bits: int = 48) -> Fraction:
    """value rounded to the nearest multiple of 2**-bits, as a Fraction."""
    return Fraction(round(Fraction(value) * (1 << bits)), 1 << bits)


def clean(period, atoms, bits: int = 48) -> tuple:
    """A valid measure near the given one: numbers rounded to 2**-bits, positions wrapped into the period box,
    atoms at one position merged."""
    px, py = (rational(v, bits) for v in period)
    merged = {}
    for x, y, m in atoms:
        key = (rational(x, bits) % px, rational(y, bits) % py)
        merged[key] = merged.get(key, Fraction(0)) + rational(m, bits)
    return (px, py), [(x, y, m) for (x, y), m in merged.items() if m > 0]


def bound(period, atoms) -> Fraction:
    """The certified lower bound exactly as the evaluator scores it, after its validity checks."""
    return _evaluator.bound(*_evaluator.measure((period, atoms)))


def value(period, atoms) -> float:
    """bound(period, atoms) as a float (it takes milliseconds for a few atoms)."""
    return float(bound(period, atoms))


def witnesses(period, atoms) -> list:
    """The maximal witness rectangles inside the period box: [((x0, x1, y0, y1), (x_lo, x_hi, y_lo, y_hi)), ...] as
    floats, each with the box of window atoms whose square certifies it."""
    period, atoms = _evaluator.measure((period, atoms))
    found = _evaluator.rectangles(period, atoms)
    kept = set(_evaluator._dominated([rect for rect, _ in found]))
    scale = float(1 << GRID)
    seen, out = set(), []
    for rect, box in found:
        if rect in kept and rect not in seen:
            seen.add(rect)
            out.append((tuple(v / scale for v in rect), tuple(float(v) for v in box)))
    return out


def gaps(period, atoms, min_area: float = 1e-12) -> list:
    """The parts of the period box no witness covers, as rectangles (x0, x1, y0, y1) in floats, largest first;
    slivers of area below min_area (left where witnesses abut, by the inward rounding) are dropped."""
    period, atoms = _evaluator.measure((period, atoms))
    _, _, found = _evaluator.cover(period, atoms)
    scale = float(1 << GRID)
    out = [tuple(v / scale for v in g) for g in found]
    out = [g for g in out if (g[1] - g[0]) * (g[3] - g[2]) >= min_area]
    return sorted(out, key=lambda g: -(g[1] - g[0]) * (g[3] - g[2]))


def area_union(rects) -> Fraction:
    """Exact area of the union of rectangles (x0, x1, y0, y1), given as ints, Fractions or floats (read exactly),
    by a sweep over the x-coordinates."""
    rs = [tuple(Fraction(v) for v in r) for r in rects]
    rs = [r for r in rs if r[0] < r[1] and r[2] < r[3]]
    xs = sorted({v for r in rs for v in r[:2]})
    area = Fraction(0)
    for a, b in zip(xs[:-1], xs[1:]):
        live = sorted((r[2], r[3]) for r in rs if r[0] <= a and r[1] >= b)
        covered, top = Fraction(0), None
        for y0, y1 in live:
            if top is None or y0 > top:
                covered += y1 - y0
                top = y1
            elif y1 > top:
                covered += y1 - top
                top = y1
        area += (b - a) * covered
    return area


def perturb(period, atoms, rng, scale: float = 1e-3, bits: int = 48) -> tuple:
    """A random nearby measure: one number, a period side, an atom coordinate or a mass, moved by scale N(0, 1)
    relative to its period side or itself; cleaned."""
    numbers = [float(v) for v in period] + [float(v) for atom in atoms for v in atom]
    t = int(rng.integers(len(numbers)))
    if t < 2 or (t - 2) % 3 == 2:
        numbers[t] *= math.exp(scale * rng.normal())
    else:
        numbers[t] += scale * numbers[(t - 2) % 3] * rng.normal()
    period = numbers[:2]
    return clean(period, [tuple(numbers[i:i + 3]) for i in range(2, len(numbers), 3)], bits)
