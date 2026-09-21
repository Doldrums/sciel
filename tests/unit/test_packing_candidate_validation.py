"""Packing scores are recomputed outside candidate code, with bounded execution."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "packing_candidate_validation", ROOT / "scripts/verify_packing_candidate.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def packing():
    return {
        "centers": [[0.1 + 0.16 * (i % 6), 0.1 + 0.16 * (i // 6)] for i in range(26)],
        "radii": [0.04] * 26,
        "reported_sum": 999999.0,
    }


def candidate(tmp_path, prefix="", suffix=""):
    payload = packing()
    source = tmp_path / "candidate.py"
    source.write_text(
        prefix
        + "\ndef run_packing():\n"
        + f"    return {payload['centers']!r}, {payload['radii']!r}, 999999.0\n"
        + suffix,
        encoding="utf-8",
    )
    return source


def test_valid_geometry_ignores_reported_sum_and_reports_slack():
    result = verifier.validate_packing(packing())
    assert result["valid"] is True
    assert result["strict_valid"] is True
    assert result["error"] is None
    assert result["sum_radii"] == pytest.approx(1.04)
    assert result["score"] == pytest.approx(1.04 / 2.635)
    assert result["min_wall_slack"] == pytest.approx(0.06)
    assert result["min_pair_slack"] == pytest.approx(0.08)
    assert result["centers"] == packing()["centers"]
    assert result["radii"] == packing()["radii"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["centers"].__setitem__(1, p["centers"][0][:]),
        lambda p: p["centers"][0].__setitem__(0, -0.1),
        lambda p: p["radii"].__setitem__(0, -0.001),
        lambda p: p["centers"].pop(),
        lambda p: p["radii"].pop(),
        lambda p: p["centers"].__setitem__(0, [0.1]),
        lambda p: p["centers"][0].__setitem__(0, math.nan),
        lambda p: p["radii"].__setitem__(0, math.inf),
        lambda p: p["radii"].__setitem__(0, True),
        lambda p: p["radii"].__setitem__(0, "0.04"),
    ],
    ids=["overlap", "outside", "negative-radius", "wrong-count", "wrong-radii",
         "wrong-dimension", "nan-center", "infinite-radius", "bool-radius", "text-radius"],
)
def test_invalid_geometry_cannot_receive_a_score(mutate):
    payload = packing()
    mutate(payload)
    result = verifier.validate_packing(payload)
    assert result["valid"] is False
    assert result["strict_valid"] is False
    assert result["score"] == 0
    assert isinstance(result["error"], str) and result["error"]
    json.dumps(result, allow_nan=False)


def test_tolerance_validity_is_distinct_from_strict_validity():
    payload = packing()
    payload["centers"][0][0] = 0.04 - 0.5e-6
    tolerated = verifier.validate_packing(payload)
    assert tolerated["valid"] is True
    assert tolerated["strict_valid"] is False
    assert tolerated["min_wall_slack"] == pytest.approx(-0.5e-6)
    payload["centers"][0][0] = 0.04 - 2e-6
    rejected = verifier.validate_packing(payload)
    assert rejected["valid"] is False
    assert rejected["score"] == 0


def test_numpy_sum_monkeypatch_stays_in_child_and_cannot_inflate_score(tmp_path):
    import numpy as np

    real_sum = np.sum
    source = candidate(tmp_path, "import numpy as np\nnp.sum = lambda *a, **k: 999999.0\n")
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    result = verifier.verify_candidate(source, repeats=2, timeout=5.0, expected_sha256=expected_sha)
    assert np.sum is real_sum
    assert result["source_sha256"] == expected_sha
    assert result["new_candidate_executions"] is True
    assert result["repeats"] == 2
    assert result["timeout_seconds"] == 5.0
    assert result["valid_repeats"] == 2
    assert result["all_valid"] is True
    assert result["mean_score"] == pytest.approx(1.04 / 2.635)
    assert len(result["samples"]) == 2
    for sample in result["samples"]:
        assert sample["status"] == "ok"
        assert sample["valid"] is True and sample["strict_valid"] is True
        assert sample["score"] == pytest.approx(1.04 / 2.635)
        assert sample["centers"] == packing()["centers"]
        assert sample["radii"] == packing()["radii"]
        assert sample["elapsed_seconds"] >= 0
        assert sample["error"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("prefix,status", [
    ("raise RuntimeError('fixture crash')\n", "error"),
    ("", "invalid"),
])
def test_candidate_failure_is_reported_without_parent_exception(tmp_path, prefix, status):
    source = candidate(tmp_path, prefix=prefix)
    if status == "invalid":
        source.write_text("def run_packing():\n    return [[0.5, 0.5]] * 26, [0.1] * 26, 2.6\n")
    result = verifier.verify_candidate(source, repeats=1, timeout=5.0)
    assert result["valid_repeats"] == 0
    assert result["all_valid"] is False
    assert result["mean_score"] == 0
    sample = result["samples"][0]
    assert sample["status"] == status
    assert sample["valid"] is False
    assert sample["score"] == 0
    assert sample["error"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"repeats": 0}, {"repeats": 4}, {"repeats": 1.5},
    {"timeout": 0}, {"timeout": 5.01}, {"timeout": math.inf}, {"timeout": math.nan},
    {"expected_sha256": "0" * 64},
])
def test_invalid_limits_or_source_identity_fail_before_execution(tmp_path, kwargs):
    marker = tmp_path / "executed"
    source = candidate(tmp_path, f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    with pytest.raises(ValueError):
        verifier.verify_candidate(source, **kwargs)
    assert not marker.exists()


def test_source_change_during_execution_prevents_publication(tmp_path):
    source = tmp_path / "candidate.py"
    candidate(tmp_path, f"from pathlib import Path\nPath({str(source)!r}).write_text('# changed')\n")
    with pytest.raises(ValueError, match="source changed"):
        verifier.verify_candidate(source, repeats=1, timeout=5.0)


def test_candidate_receives_no_api_key_or_arbitrary_sensitive_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key-not-a-secret")
    monkeypatch.setenv("SCIEL_TEST_PRIVATE_TOKEN", "fake-test-private-value")
    source = candidate(tmp_path, "import os\n"
        "assert 'OPENROUTER_API_KEY' not in os.environ\n"
        "assert 'SCIEL_TEST_PRIVATE_TOKEN' not in os.environ\n")
    result = verifier.verify_candidate(source, repeats=1, timeout=5.0)
    assert result["all_valid"] is True
    assert result["samples"][0]["status"] == "ok"


def test_cli_saves_numeric_artifact_and_refuses_to_overwrite_it(tmp_path):
    source = candidate(tmp_path)
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "verification" / "result.json"
    command = [sys.executable, str(ROOT / "scripts/verify_packing_candidate.py"),
               str(source), "--output", str(output), "--repeats", "1",
               "--expected-sha256", expected_sha]
    first = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    assert first.returncode == 0, first.stderr
    original_bytes = output.read_bytes()
    original_mtime = output.stat().st_mtime_ns
    artifact = json.loads(original_bytes)
    assert artifact["schema"] == "sciel-independent-packing-validation/1"
    assert artifact["source_sha256"] == expected_sha
    assert artifact["all_valid"] is True
    assert artifact["samples"][0]["centers"] == packing()["centers"]
    assert artifact["samples"][0]["radii"] == packing()["radii"]
    second = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    assert second.returncode != 0
    assert "already exists" in second.stderr
    assert output.read_bytes() == original_bytes
    assert output.stat().st_mtime_ns == original_mtime


def process_running(pid):
    """An unreaped zombie is terminated; it no longer consumes candidate compute."""
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                            capture_output=True, text=True, check=False)
    state = result.stdout.strip()
    return bool(state) and not state.startswith("Z")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup contract")
@pytest.mark.parametrize("hang", [False, True], ids=["success", "timeout"])
def test_normal_descendant_is_terminated_on_success_and_timeout(tmp_path, hang):
    pidfile = tmp_path / "pids.json"
    prefix = (
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"Path({str(pidfile)!r}).write_text(json.dumps([os.getpid(), child.pid]))\n"
    )
    if hang:
        prefix += "time.sleep(60)\n"
    source = candidate(tmp_path, prefix)
    pids = []
    try:
        result = verifier.verify_candidate(source, repeats=1, timeout=2.0)
        assert pidfile.exists(), "fixture must spawn its descendant before the timeout"
        pids = json.loads(pidfile.read_text())
        sample = result["samples"][0]
        assert sample["status"] == ("timeout" if hang else "ok")
        assert sample["valid"] is (not hang)
        if hang:
            assert sample["score"] == 0
        deadline = time.monotonic() + 2.0
        while any(process_running(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.03)
        assert not any(process_running(pid) for pid in pids)
    finally:
        if not pids and pidfile.exists():
            pids = json.loads(pidfile.read_text())
        for pid in pids:
            if process_running(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
