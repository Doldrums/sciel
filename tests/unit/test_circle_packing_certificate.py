"""Exact certificates use saved numbers only and retain every failed draw."""
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/certify_circle_packing.py"
spec = importlib.util.spec_from_file_location("packing_certificate", SCRIPT)
certificate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(certificate)


def packing():
    return {"centers": [[.25, .25], [.75, .75]] + [[0.0, 0.0] for _ in range(24)],
            "radii": [.125, .125] + [0.0] * 24}


def exact(value):
    return Fraction(value["numerator"], value["denominator"])


def assert_certificate(result, original):
    assert result["status"] == "certified" and result["error"] is None
    assert result["centers"] == original["centers"]
    centers = [[Fraction.from_float(float(v)) for v in p] for p in result["centers"]]
    radii = [Fraction.from_float(v) for v in result["radii"]]
    for i, ((x, y), radius) in enumerate(zip(centers, radii)):
        assert 0 <= radius <= Fraction.from_float(float(original["radii"][i]))
        assert radius <= min(x, y, 1-x, 1-y)
        for j in range(i):
            distance2 = (x-centers[j][0])**2 + (y-centers[j][1])**2
            assert distance2 >= (radius+radii[j])**2
    total = sum(radii, Fraction())
    assert exact(result["sum_exact"]) == total
    assert Fraction.from_float(result["sum_lower_bound"]) <= total
    assert Fraction.from_float(result["sum_upper_bound"]) >= total


def test_exact_feasible_numbers_are_preserved_and_certificate_is_deterministic():
    p = packing()
    before = copy.deepcopy(p)
    result = certificate.certify_packing(p)
    assert_certificate(result, p)
    assert result["original_exact_feasible"] is True
    assert result["radii"] == p["radii"]
    assert exact(result["uniform_decrement_exact"]) == 0
    assert result == certificate.certify_packing(json.loads(json.dumps(p)))
    assert p == before


def test_rounded_hypot_contact_is_not_an_exact_feasibility_certificate():
    p = packing()
    p["centers"][1] = [.5, .5]
    radius = math.hypot(.25, .25) / 2
    p["radii"][:2] = [radius, radius]
    assert math.hypot(.25, .25) - radius - radius == 0.0
    assert (Fraction.from_float(radius) * 2)**2 > Fraction(1, 8)
    result = certificate.certify_packing(p)
    assert result["original_exact_feasible"] is False and result["tolerance_valid"] is True
    assert 0 < exact(result["uniform_decrement_exact"]) < Fraction(1, 10**14)
    assert_certificate(result, p)


def test_small_wall_violation_has_small_common_decrement_and_downward_rounding():
    p = packing()
    p["centers"][0] = [.1, .5]
    p["radii"][0] = .1000001
    result = certificate.certify_packing(p)
    assert result["original_exact_feasible"] is False
    assert exact(result["uniform_decrement_exact"]) == Fraction.from_float(.1000001)-Fraction.from_float(.1)
    assert 0 < result["max_radius_reduction_upper_bound"] < 1.001e-7
    assert_certificate(result, p)


def test_clipping_the_smaller_radius_needs_more_than_half_the_overlap():
    p = packing()
    p["centers"][:2] = [[.5, .5], [.5, .5]]
    p["radii"][:2] = [5e-7, 1e-7]
    result = certificate.certify_packing(p)
    assert exact(result["uniform_decrement_exact"]) == Fraction.from_float(5e-7)
    assert result["radii"] == [0.0] * 26
    assert_certificate(result, p)


@pytest.mark.parametrize("damage", [
    lambda p: p.update(centers=[]),
    lambda p: p["radii"].pop(),
    lambda p: p["centers"][0].append(0),
    lambda p: p["radii"].__setitem__(0, True),
    lambda p: p["radii"].__setitem__(0, "0.1"),
    lambda p: p["radii"].__setitem__(0, -1e-15),
    lambda p: p["centers"][0].__setitem__(0, float("nan")),
    lambda p: p["radii"].__setitem__(0, float("inf")),
    lambda p: p["radii"].__setitem__(0, 10**1000),
    lambda p: p["radii"].__setitem__(0, .3),
    lambda p: p["centers"].__setitem__(1, [.25, .25]),
    lambda p: p["centers"][0].__setitem__(0, -1e-15),
])
def test_invalid_numbers_or_geometry_are_rejected_without_a_certificate(damage):
    p = packing()
    damage(p)
    result = certificate.certify_packing(p)
    assert result["status"] == "rejected" and result["error"]
    assert result["radii"] is None and result["sum_exact"] is None


def validation():
    source = "a" * 64
    return {"schema": "sciel-circle-research-numeric/1", "source_sha256": source,
            "repeats": 3, "samples": [
                {**packing(), "repeat": 1, "status": "ok", "source_sha256": source},
                {"repeat": 2, "status": "timeout", "centers": None, "radii": None, "source_sha256": source},
                {**packing(), "repeat": 3, "status": "ok", "source_sha256": source}]}


def test_all_draws_remain_in_mean_with_failure_zero_and_no_cherry_picking():
    doc = validation()
    result = certificate.certify_validation(doc)
    assert result["repeats"] == 3 and result["certified_repeats"] == 2
    assert result["all_certified"] is False
    assert [s["repeat"] for s in result["samples"]] == [1, 2, 3]
    assert [s["status"] for s in result["samples"]] == ["certified", "rejected", "certified"]
    assert exact(result["mean_sum_exact"]) == Fraction(1, 6)
    assert Fraction.from_float(result["mean_sum_lower_bound"]) <= Fraction(1, 6)
    assert result["candidate_executions"] == result["paid_calls"] == 0


@pytest.mark.parametrize("damage", [
    lambda d: d["samples"].pop(),
    lambda d: d["samples"][1].update(repeat=1),
    lambda d: d["samples"][0].update(source_sha256="b"*64),
    lambda d: d.update(schema="unknown"),
    lambda d: d.update(repeats=True),
])
def test_malformed_repeat_envelopes_fail_instead_of_omitting_draws(damage):
    doc = validation()
    damage(doc)
    with pytest.raises(ValueError):
        certificate.certify_validation(doc)


def test_cli_only_reads_saved_numbers_binds_input_and_never_overwrites(tmp_path):
    marker = tmp_path / "candidate_executed"
    program = tmp_path / "candidate.py"
    program.write_text(f"open({str(marker)!r}, 'w').write('must never happen')\n")
    doc = validation()
    doc["source_path"] = str(program)
    source = tmp_path / "numbers.json"
    source.write_text(json.dumps(doc))
    output = tmp_path / "certificate.json"
    assert certificate.main([str(source), "--output", str(output)]) == 2  # retained timeout
    result = json.loads(output.read_text())
    assert result["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert not marker.exists()
    before = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        certificate.main([str(source), "--output", str(output)])
    assert output.read_bytes() == before
