"""Pure JSON Circle geometry; no candidate import or execution.

Copied numerical policies from trusted verifier/certifier below. Keep raw
floating tolerance validity separate from exact radius-shrink certificates.
No zero-valued failure estimand is introduced by this module.
"""
from fractions import Fraction
import hashlib
import json
import math

N_CIRCLES = 26
TOLERANCE = 1e-6
TARGET_VALUE = 2.635
_TOL = Fraction.from_float(TOLERANCE)
_SQRT_BITS = 1074
_SQRT_UNIT = Fraction(1, 1 << _SQRT_BITS)
CERTIFICATE_POLICY = "sciel-circle-common-decrement-binary64/1"

# Numerical provenance (source bytes, before package extraction):
# scripts/verify_packing_candidate.py SHA256 c70ffddbc0afc1deb88381e9af4c735920ae38ee5b905045eb69688166948ba3
# scripts/certify_circle_packing.py SHA256 fe365d59e6bf31eeec51f9733ed66cede8b501870829198efc24afb6dbc2c0ac

def _failure(error):
    return {"centers": None, "radii": None, "valid": False, "strict_valid": False,
            "score": 0.0, "strict_score": 0.0, "sum_radii": None,
            "min_pair_slack": None, "min_wall_slack": None, "min_radius": None,
            "error": str(error)}

def _number(value):
    if type(value) not in (int, float):
        raise ValueError("coordinates and radii must be JSON numbers")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError("coordinates and radii must be finite") from None
    if not math.isfinite(result):
        raise ValueError("coordinates and radii must be finite")
    return result

def validate_packing(payload):
    """Validate numeric JSON data, with tolerance 1e-6 and strict-zero diagnostics.

    ``sum_radii`` records the actual numeric sum even for an overlapping packing;
    ``score`` is zero unless all geometric checks pass. Malformed data are rejected
    rather than coerced into a potentially valid packing.
    """
    if not isinstance(payload, dict):
        return _failure("packing payload must be a JSON object")
    centers, radii = payload.get("centers"), payload.get("radii")
    if (not isinstance(centers, list) or len(centers) != N_CIRCLES or
            any(not isinstance(row, list) or len(row) != 2 for row in centers) or
            not isinstance(radii, list) or len(radii) != N_CIRCLES):
        return _failure("expected centers shape (26, 2) and radii shape (26,)")
    try:
        centers = [[_number(x), _number(y)] for x, y in centers]
        radii = [_number(radius) for radius in radii]
    except ValueError as exc:
        return _failure(exc)
    minimum_radius = min(radii)
    wall = min(min(x - r, y - r, 1 - x - r, 1 - y - r)
               for (x, y), r in zip(centers, radii))
    pair = min(math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
               - radii[i] - radii[j] for i in range(N_CIRCLES) for j in range(i + 1, N_CIRCLES))
    try:
        total = math.fsum(radii)
    except OverflowError:
        return _failure("sum of radii is not finite")
    if not all(math.isfinite(v) for v in (wall, pair, total)):
        return _failure("packing geometry is outside the finite numeric range")
    errors = []
    if minimum_radius < 0:
        errors.append("negative radius")
    if wall < -TOLERANCE:
        errors.append("circle outside unit square")
    if pair < -TOLERANCE:
        errors.append("overlapping circles")
    valid = not errors
    strict = minimum_radius >= 0 and wall >= 0 and pair >= 0
    return {"centers": centers, "radii": radii, "valid": valid, "strict_valid": strict,
            "score": total / TARGET_VALUE if valid else 0.0,
            "strict_score": total / TARGET_VALUE if strict else 0.0,
            "sum_radii": total, "min_pair_slack": pair, "min_wall_slack": wall,
            "min_radius": minimum_radius, "error": "; ".join(errors) if errors else None}

def _ratio(value):
    return {"numerator": value.numerator, "denominator": value.denominator}

def _float_down(value):
    number = float(value)
    if Fraction.from_float(number) > value:
        number = math.nextafter(number, -math.inf)
    return number

def _float_up(value):
    number = float(value)
    if Fraction.from_float(number) < value:
        number = math.nextafter(number, math.inf)
    return number

def _sqrt_lower(value):
    scaled = (value.numerator << (2 * _SQRT_BITS)) // value.denominator
    return Fraction(math.isqrt(scaled), 1 << _SQRT_BITS)

def _geometry(centers, radii):
    walls = [min(x, y, 1 - x, 1 - y) for x, y in centers]
    pairs = [(i, j, (centers[i][0] - centers[j][0]) ** 2
                       + (centers[i][1] - centers[j][1]) ** 2)
             for i in range(N_CIRCLES) for j in range(i + 1, N_CIRCLES)]
    feasible = (all(r >= 0 and r <= w for r, w in zip(radii, walls))
                and all(d2 >= (radii[i] + radii[j]) ** 2 for i, j, d2 in pairs))
    return walls, pairs, feasible

def _numeric_sha(centers, radii):
    data = json.dumps({"centers": centers, "radii": radii}, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()

def certify_packing(payload):
    """Return a strict certificate or a rejection; never trust saved valid flags.

    Radius/wall tolerance is exactly the binary64 value 1e-6. For pairs the
    tolerance test is d² >= max(0, ri+rj-tolerance)², without rounded hypot.
    Consequently a boundary result from the floating verifier can disagree.
    """
    result = {"schema": "sciel-circle-packing-certificate/1", "status": "rejected",
              "error": None, "n_circles": N_CIRCLES, "tolerance": TOLERANCE,
              "original_exact_feasible": None, "tolerance_valid": False,
              "input_numeric_sha256": None, "original_sum_exact": None,
              "centers": None, "radii": None, "sum_exact": None,
              "sum_lower_bound": None, "sum_upper_bound": None,
              "uniform_decrement_exact": None, "uniform_decrement_upper_bound": None,
              "max_radius_reduction_exact": None, "max_radius_reduction_upper_bound": None,
              "certified_numeric_sha256": None,
              "method": "common absolute decrement; clip at zero; round each radius down; exact recheck",
              "scope": "Exact feasibility of the represented binary64 coordinates/radii; not an optimizer bound."}
    try:
        if not isinstance(payload, dict):
            raise ValueError("packing must be a JSON object")
        c, r = payload.get("centers"), payload.get("radii")
        if (not isinstance(c, list) or len(c) != N_CIRCLES or
                any(not isinstance(x, list) or len(x) != 2 for x in c) or
                not isinstance(r, list) or len(r) != N_CIRCLES):
            raise ValueError("expected centers (26,2) and radii (26,)")
        c = [[_number(x), _number(y)] for x, y in c]
        r = [_number(x) for x in r]
        centers = [[Fraction.from_float(x), Fraction.from_float(y)] for x, y in c]
        radii = [Fraction.from_float(x) for x in r]
        result["input_numeric_sha256"] = _numeric_sha(c, r)
        result["original_sum_exact"] = _ratio(sum(radii, Fraction()))
        walls, pairs, feasible = _geometry(centers, radii)
        result["original_exact_feasible"] = feasible
        if any(x < 0 or x > 1 or y < 0 or y > 1 for x, y in centers):
            raise ValueError("a center lies outside the unit square")
        if any(x < 0 for x in radii):
            raise ValueError("negative radius")
        if any(radius > wall + _TOL for radius, wall in zip(radii, walls)):
            raise ValueError("wall violation exceeds tolerance")
        if any(d2 < max(Fraction(), radii[i] + radii[j] - _TOL) ** 2 for i, j, d2 in pairs):
            raise ValueError("pair overlap exceeds tolerance")
        result["tolerance_valid"] = True
        decrement = Fraction()
        if not feasible:
            decrement = max([Fraction()] + [radius - wall for radius, wall in zip(radii, walls)])
            for i, j, d2 in pairs:
                distance = _sqrt_lower(d2)
                # The second term handles clipping the smaller radius to zero.
                decrement = max(decrement, (radii[i] + radii[j] - distance) / 2,
                                max(radii[i], radii[j]) - distance)
            if decrement > _TOL + _SQRT_UNIT:
                raise ValueError("required decrement exceeds the tolerance repair bound")
        certified = [_float_down(max(Fraction(), radius - decrement)) for radius in radii]
        exact = [Fraction.from_float(x) for x in certified]
        if not _geometry(centers, exact)[2] or any(a > b for a, b in zip(exact, radii)):
            raise ValueError("exact final feasibility verification failed")
        total = sum(exact, Fraction())
        reduction = max(old - new for old, new in zip(radii, exact))
        result.update(status="certified", centers=c, radii=certified, sum_exact=_ratio(total),
            sum_lower_bound=_float_down(total), sum_upper_bound=_float_up(total),
            uniform_decrement_exact=_ratio(decrement), uniform_decrement_upper_bound=_float_up(decrement),
            max_radius_reduction_exact=_ratio(reduction), max_radius_reduction_upper_bound=_float_up(reduction),
            certified_numeric_sha256=_numeric_sha(c, certified))
    except ValueError as exc:
        result["error"] = str(exc)
    return result
