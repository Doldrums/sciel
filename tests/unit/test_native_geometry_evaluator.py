"""Same-invocation native geometry using tiny deterministic array fixtures.

No optimizer, provider, Docker, or packing search is invoked. Subprocess cases
exercise the real evaluator transport rather than parsing candidate stdout.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import textwrap

import numpy as np
import pytest

from sciel.eval._runner import _normalize
from sciel.eval.facade import EvalResult
from sciel.eval.sandbox import run_stage


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "tasks/oe/circle_packing/evaluator.py"
FIELDS = {
    "schema", "source_sha256", "evaluator_sha256", "invocation_id", "stage",
    "confirmation", "numeric", "numeric_sha256", "centers_dtype", "radii_dtype",
    "native_validity", "native_sum_radii", "native_score",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def load_evaluator(path=EVALUATOR):
    spec = importlib.util.spec_from_file_location("native_geometry_fixture_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def evaluator():
    return load_evaluator()


def fixture_program(tmp_path, *, dtype="float64", edit="", prelude="",
                    radii="np.full(26, 0.01, dtype=DTYPE)", reported="-123456.0"):
    """The counter proves one array production for each evaluator invocation."""
    path = tmp_path / "program.py"
    count = tmp_path / "invocations.txt"
    lines = [
        "import numpy as np", "from pathlib import Path",
        f"DTYPE = np.{dtype}", f"COUNTER = Path({str(count)!r})",
        textwrap.dedent(prelude).strip(),
        "def run_packing():",
        "    invocation = int(COUNTER.read_text()) + 1 if COUNTER.exists() else 1",
        "    COUNTER.write_text(str(invocation))",
        "    centers = np.array([(0.1 + 0.16*(i%6), 0.1 + 0.2*(i//6)) for i in range(26)], dtype=DTYPE)",
        f"    radii = {radii}",
        "    print('native_geometry: forged stdout; sum_radii=999999')",
    ]
    lines.extend("    " + line for line in textwrap.dedent(edit).strip().splitlines())
    lines.append(f"    return centers, radii, {reported}")
    path.write_text("\n".join(lines) + "\n")
    return path, count


def result_dict(result):
    return result.to_json() if isinstance(result, EvalResult) else result


def capture(result, source, *, stage, evaluator_path=EVALUATOR):
    result = result_dict(result)
    wire = result["artifacts"]["native_geometry"]
    assert isinstance(wire, str)
    value = json.loads(wire)
    assert wire == canonical(value)
    assert set(value) == FIELDS
    assert value["schema"] == "sciel-native-circle-geometry/1"
    assert value["source_sha256"] == digest(source.read_bytes())
    assert value["evaluator_sha256"] == digest(evaluator_path.read_bytes())
    assert re.fullmatch("[0-9a-f]{32}", value["invocation_id"])
    assert value["stage"] == stage
    numeric = value["numeric"]
    assert set(numeric) == {"centers", "radii"}
    assert len(numeric["centers"]) == len(numeric["radii"]) == 26
    assert all(len(row) == 2 for row in numeric["centers"])
    numbers = [x for row in numeric["centers"] for x in row] + numeric["radii"]
    assert all(type(x) is float and math.isfinite(x) for x in numbers)
    assert value["numeric_sha256"] == digest(canonical(numeric).encode())
    assert type(value["native_validity"]) is bool
    assert value["native_validity"] == bool(result["cases"]["validity"])
    assert value["native_sum_radii"] == result["cases"]["sum_radii"]
    assert value["native_score"] == result["score"]
    return value


@pytest.mark.parametrize("stage", ["smoke", "full"])
def test_direct_capture_binds_one_actual_invocation_and_ignores_claim(evaluator, tmp_path, stage):
    source, count = fixture_program(tmp_path, reported="None")
    before = source.read_bytes()
    result = getattr(evaluator, stage)(str(source))
    value = capture(result, source, stage=stage)
    assert result["ok"] and result["cases"]["validity"] == 1.0
    assert count.read_text() == "1" and source.read_bytes() == before
    assert value["confirmation"] is None
    assert value["centers_dtype"] == value["radii_dtype"] == "float64"
    assert value["numeric"]["radii"] == [0.01] * 26
    assert value["native_sum_radii"] == float(np.sum(np.full(26, .01)))
    assert value["native_score"] == value["native_sum_radii"] / 2.635
    assert _normalize(result)["artifacts"]["native_geometry"] == result["artifacts"]["native_geometry"]
    assert EvalResult.from_json(_normalize(result)).artifacts["native_geometry"] == result["artifacts"]["native_geometry"]


def test_smoke_full_are_distinct_actual_invocations(evaluator, tmp_path):
    source, count = fixture_program(tmp_path, radii="np.full(26, invocation * .01)")
    smoke = capture(evaluator.smoke(str(source)), source, stage="smoke")
    full = capture(evaluator.full(str(source)), source, stage="full")
    assert count.read_text() == "2"
    assert smoke["invocation_id"] != full["invocation_id"]
    assert smoke["numeric"]["radii"] == [.01] * 26
    assert full["numeric"]["radii"] == [.02] * 26
    assert smoke["numeric_sha256"] != full["numeric_sha256"]


@pytest.mark.parametrize("stage", ["smoke", "full"])
def test_float32_sum_is_native_reduction_not_recomputed_float64(evaluator, tmp_path, stage):
    source, count = fixture_program(tmp_path, dtype="float32")
    value = capture(getattr(evaluator, stage)(str(source)), source, stage=stage)
    native = float(np.sum(np.full(26, .01, dtype=np.float32)))
    assert native != math.fsum(value["numeric"]["radii"]), "fixture must expose precision distinction"
    assert value["native_sum_radii"] == native
    assert value["native_score"] == native / 2.635
    assert value["centers_dtype"] == value["radii_dtype"] == "float32"
    assert value["numeric"]["radii"] == [float(np.float32(.01))] * 26
    assert count.read_text() == "1"


@pytest.mark.parametrize("stage", ["smoke", "full"])
@pytest.mark.parametrize("edit", [
    "centers[1] = centers[0]",
    "radii[0] = -0.01",
    "centers[0, 0] = 1.5",
])
def test_finite_invalid_geometry_retained_as_penalty_not_physical_zero(evaluator, tmp_path, stage, edit):
    source, count = fixture_program(tmp_path, edit=edit)
    result = getattr(evaluator, stage)(str(source))
    value = capture(result, source, stage=stage)
    assert result["ok"] and result["score"] == 0.0
    assert value["native_validity"] is False
    assert value["native_sum_radii"] == 0.0
    assert math.fsum(value["numeric"]["radii"]) > 0
    assert count.read_text() == "1"


@pytest.mark.parametrize("stage", ["smoke", "full"])
@pytest.mark.parametrize("edit", [
    "centers = centers[:25]", "radii = radii.reshape(26, 1)",
    "centers[0, 0] = np.nan", "radii[0] = np.inf",
    "centers = centers.astype(np.complex128)",
    "radii = radii.astype(object)", "centers = centers.astype(bool)",
    "centers = np.full((26, 2), 2**53 + 1, dtype=np.uint64)",
])
def test_malformed_or_nonreal_capture_fails_closed(evaluator, tmp_path, stage, edit):
    source, count = fixture_program(tmp_path, edit=edit)
    result = getattr(evaluator, stage)(str(source))
    assert result["ok"] is False and result["score"] == 0.0
    assert result["artifacts"].get("error")
    assert "native_geometry" not in result["artifacts"]
    assert count.read_text() == "1"
    json.dumps(_normalize(result), allow_nan=False)


def test_exactly_representable_integer_arrays_use_float64_native_arithmetic(evaluator, tmp_path):
    source, _ = fixture_program(tmp_path, dtype="int64", radii="np.zeros(26, dtype=DTYPE)")
    result = evaluator.full(str(source))
    value = capture(result, source, stage="full")
    assert result["ok"] and value["native_validity"]
    assert value["centers_dtype"] == value["radii_dtype"] == "float64"
    assert value["native_sum_radii"] == 0.0


@pytest.mark.parametrize("stage", ["smoke", "full"])
def test_uint8_overflow_cannot_admit_impossible_packing(evaluator, tmp_path, stage):
    source, count = fixture_program(tmp_path, dtype="uint8",
        radii="np.full(26, 128, dtype=DTYPE)",
        edit="centers = np.full((26, 2), 128, dtype=DTYPE)")
    result = getattr(evaluator, stage)(str(source))
    value = capture(result, source, stage=stage)
    assert result["ok"] and result["cases"]["validity"] == 0.0
    assert result["score"] == value["native_sum_radii"] == 0.0
    assert value["native_validity"] is False
    assert value["centers_dtype"] == value["radii_dtype"] == "float64"
    assert value["numeric"]["centers"] == [[128., 128.]] * 26
    assert value["numeric"]["radii"] == [128.] * 26
    assert count.read_text() == "1"


@pytest.mark.parametrize("stage", ["smoke", "full"])
def test_missing_source_is_not_executed_or_given_capture(evaluator, tmp_path, monkeypatch, stage):
    def forbidden(*args, **kwargs):
        pytest.fail("source file must be bound before candidate execution")
    monkeypatch.setattr(evaluator, "_run", forbidden)
    result = getattr(evaluator, stage)(str(tmp_path / "absent.py"))
    assert result["ok"] is False
    assert "native_geometry" not in result["artifacts"]


@pytest.mark.parametrize("stage", ["smoke", "full"])
@pytest.mark.parametrize("target", ["source", "evaluator"])
def test_source_or_evaluator_changed_during_invocation_fails_closed(tmp_path, stage, target):
    local_evaluator = tmp_path / "evaluator.py"
    local_evaluator.write_bytes(EVALUATOR.read_bytes())
    module = load_evaluator(local_evaluator)
    changed = tmp_path / "program.py" if target == "source" else local_evaluator
    edit = f"p = Path({str(changed)!r})\np.write_text(p.read_text() + '\\n# mutated during invocation\\n')"
    source, count = fixture_program(tmp_path, edit=edit)
    before = changed.read_bytes()
    result = getattr(module, stage)(str(source))
    assert changed.read_bytes() != before and count.read_text() == "1"
    assert result["ok"] is False and result["score"] == 0.0
    assert "native_geometry" not in result["artifacts"]


@pytest.mark.parametrize("target", ["source", "evaluator"])
@pytest.mark.parametrize("callback", ["__float__", "__repr__"])
def test_reported_sum_callback_after_run_cannot_invalidate_hash_binding(tmp_path, target, callback):
    local_evaluator = tmp_path / "evaluator.py"
    local_evaluator.write_bytes(EVALUATOR.read_bytes())
    module = load_evaluator(local_evaluator)
    changed = tmp_path / "program.py" if target == "source" else local_evaluator
    methods = ("    def __float__(self):\n        raise TypeError('use repr')\n"
               if callback == "__repr__" else "")
    methods += (f"    def {callback}(self):\n"
        f"        p = Path({str(changed)!r})\n"
        "        p.write_text(p.read_text() + '\\n# mutation during reported_sum formatting\\n')\n"
        + ("        return 'claim'\n" if callback == "__repr__" else "        return 999.0\n"))
    source, count = fixture_program(tmp_path, prelude="class Reported:\n" + methods,
                                    reported="Reported()")
    before = changed.read_bytes()
    result = module.full(str(source))
    assert count.read_text() == "1" and changed.read_bytes() != before
    assert result["ok"] is False and result["score"] == 0.0
    assert "changed during native invocation" in result["artifacts"]["error"]
    assert "native_geometry" not in result["artifacts"]


def test_capture_detaches_from_candidate_owned_arrays(evaluator, tmp_path, monkeypatch):
    source, _ = fixture_program(tmp_path)
    loaded = []
    original_load = evaluator._load

    def record_load(*args, **kwargs):
        module = original_load(*args, **kwargs)
        original = module.run_packing
        def retain():
            result = original()
            loaded.append(result)
            return result
        module.run_packing = retain
        return module

    monkeypatch.setattr(evaluator, "_load", record_load)
    result = evaluator.full(str(source))
    wire = result["artifacts"]["native_geometry"]
    loaded[0][0][:] = np.nan
    loaded[0][1][:] = 999
    value = capture(result, source, stage="full")
    assert result["artifacts"]["native_geometry"] == wire
    assert value["numeric"]["radii"] == [.01] * 26


@pytest.mark.parametrize("radii", [[.03, .01, .02], [.03, .01, .01]])
def test_confirm_retains_selected_full_invocation_and_first_tie(evaluator, tmp_path, monkeypatch, radii):
    source, count = fixture_program(tmp_path, radii=f"np.full(26, {radii!r}[invocation - 1])")
    original = evaluator.full
    full_results = []
    def tracked(path):
        result = original(path)
        full_results.append(result)
        return result
    monkeypatch.setattr(evaluator, "full", tracked)
    result = evaluator.confirm(str(source), n=3)
    value = capture(result, source, stage="full")
    originals = [capture(item, source, stage="full") for item in full_results]
    assert count.read_text() == "3" and len({x["invocation_id"] for x in originals}) == 3
    assert result["score"] == full_results[1]["score"]
    assert result["cases"]["n_trials"] == 3.0
    expected = {"trials": 3, "selected_trial": 1, "selection": "first_failure_else_lowest_score"}
    assert value["confirmation"] == expected
    assert {k: v for k, v in value.items() if k != "confirmation"} == {
        k: v for k, v in originals[1].items() if k != "confirmation"}
    assert all(item["confirmation"] is None for item in originals)
    assert json.loads(result["artifacts"]["confirmation_selection"]) == expected


def test_confirm_first_failed_trial_does_not_borrow_success_geometry(evaluator, tmp_path):
    source, count = fixture_program(tmp_path, edit="if invocation in (2, 3):\n    raise ValueError('failed selected trial')")
    result = evaluator.confirm(str(source), n=3)
    assert count.read_text() == "3"
    assert result["ok"] is False and result["cases"]["n_trials"] == 3.0
    assert "native_geometry" not in result["artifacts"]
    assert "failed selected trial" in result["artifacts"]["error"]
    assert json.loads(result["artifacts"]["confirmation_selection"]) == {
        "trials": 3, "selected_trial": 1, "selection": "first_failure_else_lowest_score"}


@pytest.mark.parametrize("stage", ["smoke", "full"])
def test_real_subprocess_capture_survives_normalization_and_noisy_stdout(tmp_path, stage):
    source, count = fixture_program(tmp_path, dtype="float32")
    result = run_stage(str(EVALUATOR), stage, str(source), timeout=10, python=sys.executable)
    value = capture(result, source, stage=stage)
    assert result.ok and count.read_text() == "1"
    assert value["native_sum_radii"] == float(np.sum(np.full(26, .01, dtype=np.float32)))
    assert "forged stdout" not in result.artifacts["native_geometry"]
    restored = EvalResult.from_json(json.loads(json.dumps(result.to_json(), allow_nan=False)))
    assert restored.artifacts["native_geometry"] == result.artifacts["native_geometry"]


def test_real_subprocess_confirm_keeps_worst_trial_geometry(tmp_path):
    source, count = fixture_program(tmp_path, radii="np.full(26, [.03, .01, .02][invocation - 1])")
    result = run_stage(str(EVALUATOR), "confirm", str(source), timeout=10, n=3, python=sys.executable)
    value = capture(result, source, stage="full")
    assert result.ok and result.confirmed_sampled and count.read_text() == "3"
    assert value["numeric"]["radii"] == [.01] * 26
    assert value["confirmation"] == {
        "trials": 3, "selected_trial": 1, "selection": "first_failure_else_lowest_score"}


def test_real_subprocess_failed_confirm_has_no_success_geometry(tmp_path):
    source, count = fixture_program(tmp_path, edit="if invocation == 2:\n    raise RuntimeError('fixture trial failure')")
    result = run_stage(str(EVALUATOR), "confirm", str(source), timeout=10, n=3, python=sys.executable)
    assert not result.ok and not result.confirmed_sampled and count.read_text() == "3"
    assert "native_geometry" not in result.artifacts
    assert json.loads(result.artifacts["confirmation_selection"])["selected_trial"] == 1
