import copy
import hashlib
import importlib.util
import json
import math
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.eval.trusted_scoring.adapter import HostBinding, aggregate_completed, score_payload
from sciel.eval.trusted_scoring.transport import MAX_BYTES, SCHEMA, decode

ROOT = Path(__file__).resolve().parents[2]


def wire(kind, values):
    return json.dumps({"schema": SCHEMA, "kind": kind, "values": values},
                      separators=(",", ":"), allow_nan=False).encode()


def circle_values(dtype="float64", radius=.03):
    centers = np.array([[(i % 6 + .5) / 6, (i // 6 + .5) / 5] for i in range(26)], dtype=dtype)
    radii = np.full(26, radius, dtype=dtype)
    return {"centers": centers.astype(float).tolist(), "radii": radii.astype(float).tolist(),
            "centers_dtype": dtype, "radii_dtype": dtype, "reported_sum": "123.000000"}


def heil_values():
    x = np.linspace(.1, .9, 11)
    return {"points": np.column_stack((x, .03 + .15*x*x)).tolist()}


def binding(stage="full", invocation="c"*32):
    return HostBinding("a"*64, "b"*64, invocation, stage, 0.0)


def legacy(kind):
    path = ROOT / "tasks" / {"circle":"oe/circle_packing", "heil":"oe/heilbronn_triangle", "beat":"math/beat_the_average"}[kind] / "evaluator.py"
    spec = importlib.util.spec_from_file_location("legacy_" + kind, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_geometry(monkeypatch, tmp_path, kind, values, stage):
    module = legacy(kind)
    source = tmp_path / "never_imported.txt"
    source.write_text("This is deliberately not valid Python; no candidate is imported.")
    if kind == "circle":
        arrays = (np.array(values["centers"], dtype=values["centers_dtype"]),
                  np.array(values["radii"], dtype=values["radii_dtype"]), 123.0)
        candidate = SimpleNamespace(run_packing=lambda: arrays)
    else:
        candidate = SimpleNamespace(heilbronn_triangle11=lambda: values["points"])
    seen = []
    def inert_loader(path, _name):
        assert path == str(source)
        seen.append(path)
        return candidate
    monkeypatch.setattr(module, "_load", inert_loader)
    result = getattr(module, stage)(str(source))
    assert seen == [str(source)]
    return result


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
@pytest.mark.parametrize("stage", ["smoke", "full"])
def test_circle_exact_legacy_metric_cases_artifacts(monkeypatch, tmp_path, dtype, stage):
    values = circle_values(dtype)
    old = legacy_geometry(monkeypatch, tmp_path, "circle", values, stage)
    artifact = json.loads(old["artifacts"]["native_geometry"])
    context = HostBinding(artifact["source_sha256"], artifact["evaluator_sha256"],
                          artifact["invocation_id"], stage,
                          float(old["artifacts"].get("eval_time", "0")))
    new = score_payload("circle", wire("circle", values), context)
    assert new == old
    radii = np.array(values["radii"], dtype=dtype)
    assert new["cases"]["sum_radii"].hex() == float(np.sum(radii)).hex()
    if dtype != "float64":
        assert float(np.sum(radii)) != float(np.sum(radii.astype(np.float64)))


@pytest.mark.parametrize("case", ["ordinary", "duplicate", "collinear", "tolerance"])
@pytest.mark.parametrize("stage", ["smoke", "full"])
def test_heil_exact_legacy_metric_cases_artifacts(monkeypatch, tmp_path, case, stage):
    values = heil_values()
    if case == "duplicate":
        values["points"][1] = values["points"][0][:]
    elif case == "collinear":
        values["points"] = [[float(x), 0.0] for x in np.linspace(0, 1, 11)]
    elif case == "tolerance":
        values["points"][0] = [.2, -0.5e-6]
    old = legacy_geometry(monkeypatch, tmp_path, "heil", values, stage)
    witness = json.loads(old["artifacts"]["full_output"])
    context = HostBinding(witness["source_sha256"], witness["evaluator_sha256"],
                          witness["invocation_id"], stage, float(old["artifacts"]["eval_time"]))
    new = score_payload("heil", wire("heil", values), context)
    assert new == old
    assert new["ok"] and new["cases"]["validity"] == 1.0
    if case in ("duplicate", "collinear"):
        assert new["score"] == 0.0


@pytest.mark.parametrize("case", ["negative_radius", "overlap", "outside"])
def test_infeasible_circle_remains_successful_zero_penalty(monkeypatch, tmp_path, case):
    values = circle_values()
    if case == "negative_radius":
        values["radii"][0] = -.1
    elif case == "overlap":
        values["centers"][1] = values["centers"][0][:]
    else:
        values["centers"][0] = [-.5, -.5]
    old = legacy_geometry(monkeypatch, tmp_path, "circle", values, "full")
    new = score_payload("circle", wire("circle", values), binding())
    assert new["ok"] is old["ok"] is True
    assert new["cases"] == old["cases"]
    assert new["score"] == 0.0
    assert "native_geometry" in new["artifacts"]


def test_heil_outside_is_failed_with_no_borrowed_geometry(monkeypatch, tmp_path):
    values = heil_values()
    values["points"][0] = [.2, -2e-6]
    old = legacy_geometry(monkeypatch, tmp_path, "heil", values, "full")
    new = score_payload("heil", wire("heil", values), binding())
    assert new["ok"] is old["ok"] is False
    assert new["cases"] == old["cases"]
    assert set(new["artifacts"]) == set(old["artifacts"]) == {"error", "traceback"}


def test_heil_dyadic_geometry_matches_independent_fraction_areas():
    points = [(Fraction(i, 16), Fraction(i*i, 1024)+Fraction(1, 64)) for i in range(1, 12)]
    areas = [abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))/2
             for a, b, c in combinations(points, 3)]
    assert len(areas) == 165
    values = {"points": [[float(x), float(y)] for x, y in points]}
    result = score_payload("heil", wire("heil", values), binding())
    assert result["ok"]
    assert result["cases"]["min_area"] == float(min(areas))
    assert result["cases"]["min_area_normalized"] == float(min(areas))/(math.sqrt(3)/4)


def test_dtype_round_trip_and_signed_zero_are_exact():
    values = circle_values("float32")
    values["radii"][0] = -0.0
    result = score_payload("circle", wire("circle", values), binding())
    restored = json.loads(result["artifacts"]["native_geometry"])
    assert restored["numeric"]["radii"][0].hex() == (-0.0).hex()
    values["radii"][0] = .1  # This binary64 value is not exact float32.
    assert score_payload("circle", wire("circle", values), binding())["ok"] is False


def test_circle_binary64_boundary_uses_the_existing_tolerance():
    values = circle_values()
    values["radii"] = [0.0]*26
    values["centers"][0] = [-1e-6, .5]
    assert score_payload("circle", wire("circle", values), binding())["cases"]["validity"] == 1.0
    values["centers"][0][0] = float(np.nextafter(-1e-6, -math.inf))
    result = score_payload("circle", wire("circle", values), binding())
    assert result["ok"] and result["cases"]["validity"] == 0.0


def test_integer_worker_normalization_matches_legacy(monkeypatch, tmp_path):
    original = {"centers": [[0, 0]]*26, "radii": [0]*26,
                "centers_dtype": "uint8", "radii_dtype": "uint8", "reported_sum": "123.000000"}
    old = legacy_geometry(monkeypatch, tmp_path, "circle", original, "full")
    normalized = {**original, "centers": [[0.0, 0.0]]*26, "radii": [0.0]*26,
                  "centers_dtype": "float64", "radii_dtype": "float64"}
    new = score_payload("circle", wire("circle", normalized), binding())
    assert new["ok"] is old["ok"] is True
    assert new["cases"] == old["cases"]
    assert json.loads(new["artifacts"]["native_geometry"])["radii_dtype"] == "float64"
    with pytest.raises(ValueError):
        decode("circle", wire("circle", original))


@pytest.mark.parametrize("corruption", ["score", "hash", "dtype", "bool", "int", "string", "shape", "extra", "kind"])
def test_untrusted_output_cannot_inject_authority_or_coercions(corruption):
    values = circle_values()
    data = json.loads(wire("circle", values))
    if corruption in ("score", "hash"):
        data[corruption] = 1.0
    elif corruption == "dtype":
        data["values"]["radii_dtype"] = "object"
    elif corruption in ("bool", "int", "string"):
        data["values"]["radii"][0] = {"bool": True, "int": 1, "string": "0.1"}[corruption]
    elif corruption == "shape":
        data["values"]["radii"].pop()
    elif corruption == "extra":
        data["values"]["path"] = "/private/never-open"
    else:
        data["kind"] = "beat"
    with pytest.raises(ValueError):
        decode("circle", json.dumps(data).encode())


@pytest.mark.parametrize("payload", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}',
                                    b'[[[[[[0]]]]]]', b'{} trailing', b'\xff',
                                    b' ' * (MAX_BYTES+1)])
def test_strict_json_bounds(payload):
    with pytest.raises(ValueError):
        decode("circle", payload)


def test_numeric_overflow_and_nested_duplicate_keys_are_rejected():
    payload = wire("circle", circle_values())
    with pytest.raises(ValueError):
        decode("circle", payload.replace(b'0.03', b'1e999', 1))
    with pytest.raises(ValueError):
        decode("circle", payload.replace(b'"radii_dtype":"float64"',
                                          b'"radii_dtype":"float64","radii_dtype":"float32"'))


def test_callbacks_are_never_invoked():
    class Hostile:
        def __bytes__(self):
            raise AssertionError("callback reached")
        def __float__(self):
            raise AssertionError("callback reached")
        def __deepcopy__(self, _memo):
            raise AssertionError("callback reached")
    with pytest.raises(ValueError):
        decode("circle", Hostile())
    result = score_payload("circle", wire("circle", circle_values()), binding())
    result["cases"]["sum_radii"] = Hostile()
    with pytest.raises(ValueError):
        aggregate_completed("circle", [result], 1)


@pytest.mark.parametrize("kind", ["circle", "heil", "beat"])
@pytest.mark.parametrize("case", ["worst", "tie", "first_failure", "middle_failure"])
def test_confirmation_matches_all_calls_legacy_selection(monkeypatch, kind, case):
    module = legacy(kind)
    payloads = []
    for i in range(3):
        if kind == "circle":
            values = circle_values(radius=.01*(i+1))
        elif kind == "heil":
            values = heil_values()
            values["points"] = [[x, y*(i+1)/3] for x, y in values["points"]]
        else:
            values = {"support": [0.0, 1.0], "weights": [1.0, float(i+1)]}
        payloads.append(values)
    runs = [score_payload(kind, wire(kind, values), binding(invocation=f"{i:032x}"))
            for i, values in enumerate(payloads)]
    ordered = sorted(runs, key=lambda result: result["score"])
    assert ordered[0]["score"] < ordered[-1]["score"]
    runs = [ordered[-1], ordered[0], ordered[1]]
    if case == "tie":
        runs[0] = copy.deepcopy(runs[1])
    elif case.endswith("failure"):
        index = 0 if case == "first_failure" else 1
        runs[index] = score_payload(kind, b'invalid', binding())
    original = copy.deepcopy(runs)
    called = []
    def full(_path):
        index = len(called)
        called.append(index)
        return copy.deepcopy(runs[index])
    monkeypatch.setattr(module, "full", full)
    expected = module.confirm("never-opened", n=3)
    assert called == [0, 1, 2]  # Legacy selection still observes all calls after failure.
    actual = aggregate_completed(kind, runs, 3)
    assert actual == expected
    assert runs == original
    with pytest.raises(ValueError):
        aggregate_completed(kind, runs[:2], 3)
    if not actual["ok"]:
        assert "native_geometry" not in actual["artifacts"]
        assert "full_output" not in actual["artifacts"]


def test_maximum_legacy_numeric_payload_fits_cap():
    # A finite binary64 shortest decimal representation is <=24 ASCII chars
    # including a possible sign. Arrays add one delimiter per number. Use a
    # 24-character representable float to exercise the conservative 4000 bound.
    number = -1.7976931348623157e308
    assert len(repr(number)) == 24
    payload = wire("beat", {"support": [number]*2000, "weights": [number]*2000})
    assert len(payload) < 4000*25 + 256 < MAX_BYTES
    decode("beat", payload)  # Domain rejects negatives only at scoring.
    values = circle_values()
    values["reported_sum"] = f"{number:.6f}"
    assert len(values["reported_sum"]) == 317
    decode("circle", wire("circle", values))


@pytest.fixture(autouse=True)
def forbid_candidate_import(monkeypatch):
    from sciel import candidate_modules
    def forbidden(*args, **kwargs):
        raise AssertionError("Only inert evaluator parity fixtures may run")
    monkeypatch.setattr(candidate_modules, "load_candidate", forbidden)
