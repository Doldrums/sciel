"""Certify saved n=26 packing numbers without importing or running a candidate.

Every input binary64 value is interpreted exactly using Fraction.from_float.
This certifies one recorded packing, not an optimizer or a global upper bound.
Only tolerance-valid packings with centers inside the square can be repaired:
a common absolute decrement is subtracted from every radius, clipped at zero,
then rounded downward to binary64. Coordinates never move. All final constraints
are checked again with exact rational arithmetic, including squared distances.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


N_CIRCLES = 26
TOLERANCE = 1e-6
_TOL = Fraction.from_float(TOLERANCE)
# One binary64 subnormal unit. Exact dyadic lower bounds avoid a rounded sqrt
# accidentally overstating separation, including for coincident/tiny centers.
_SQRT_BITS = 1074
_SQRT_UNIT = Fraction(1, 1 << _SQRT_BITS)
_SCHEMAS = {"sciel-independent-packing-validation/1", "sciel-circle-research-numeric/1"}


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


def _number(value):
    if type(value) not in (int, float):
        raise ValueError("coordinates and radii must be JSON numbers")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError("coordinates and radii must be finite") from None
    if not math.isfinite(value):
        raise ValueError("coordinates and radii must be finite")
    return value


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


def certify_validation(document):
    """Certify every saved draw, retaining failures with zero in the mean.

    No repeat is selected or omitted. This is an arithmetic recheck of saved
    numbers, not another independent draw or a repeatability demonstration.
    """
    if not isinstance(document, dict) or document.get("schema") not in _SCHEMAS:
        raise ValueError("unsupported saved numeric validation schema")
    samples, repeats = document.get("samples"), document.get("repeats")
    if (type(repeats) is not int or not 1 <= repeats <= 3 or
            not isinstance(samples, list) or len(samples) != repeats):
        raise ValueError("expected every requested saved repeat (1 to 3)")
    if document.get("n_circles", N_CIRCLES) != N_CIRCLES:
        raise ValueError("certificate requires n=26")
    source_hash = document.get("source_sha256")
    output, total = [], Fraction()
    for index, sample in enumerate(samples, 1):
        if not isinstance(sample, dict) or type(sample.get("repeat")) is not int or sample["repeat"] != index:
            raise ValueError("saved repeats must be unique and ordered 1..repeats")
        if sample.get("source_sha256") != source_hash:
            raise ValueError("sample source SHA does not match saved validation")
        if sample.get("status") == "ok":
            certificate = certify_packing(sample)
        else:
            certificate = certify_packing(None)
            certificate["error"] = "saved draw did not finish with status ok"
        certificate.update(repeat=index, source_sha256=source_hash, saved_status=sample.get("status"))
        output.append(certificate)
        if certificate["status"] == "certified":
            q = certificate["sum_exact"]
            total += Fraction(q["numerator"], q["denominator"])
    mean = total / repeats
    count = sum(x["status"] == "certified" for x in output)
    return {"schema": "sciel-circle-validation-certificate/1", "source_sha256": source_hash,
            "source_protocol_sha256": document.get("protocol_sha256"),
            "repeats": repeats, "samples": output, "certified_repeats": count,
            "all_certified": count == repeats, "mean_sum_exact": _ratio(mean),
            "mean_sum_lower_bound": _float_down(mean), "mean_sum_upper_bound": _float_up(mean),
            "mean_scope": "all requested repeats; rejected samples contribute zero",
            "candidate_executions": 0, "paid_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output already exists; preserve earlier certificates")
    raw = args.input.read_bytes()
    if len(raw) > 1_048_576:
        raise ValueError("saved numeric JSON exceeds size limit")
    payload = json.loads(raw)
    result = certify_validation(payload) if isinstance(payload, dict) and "samples" in payload else certify_packing(payload)
    result.update(input_sha256=hashlib.sha256(raw).hexdigest(),
                  input_path=str(args.input.resolve()),
                  certificate_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()),
                      "status": result.get("status"), "all_certified": result.get("all_certified")}))
    return 0 if result.get("status") == "certified" or result.get("all_certified") else 2


if __name__ == "__main__":
    raise SystemExit(main())
