"""Actual process allocation/cleanup and deterministic telemetry coverage checks."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import sys

import pytest

from sciel.campaign import memory


class FakeError(Exception):
    pass


class AccessDenied(FakeError):
    pass


class NoSuchProcess(FakeError):
    pass


class FakeProcess:
    def __init__(self, pid, rss, uss, children=()):
        self.pid, self.rss, self.uss, self.descendants = pid, rss, uss, children
        self.birth, self.running = float(pid), True

    def create_time(self):
        return self.birth

    def is_running(self):
        return self.running

    def status(self):
        return "running"

    def kill(self):
        self.running = False

    def children(self, recursive):
        return self.descendants

    def memory_info(self):
        return SimpleNamespace(rss=self.rss)

    def memory_full_info(self):
        if isinstance(self.uss, Exception):
            raise self.uss
        return SimpleNamespace() if self.uss is None else SimpleNamespace(uss=self.uss)


def sampler_for(monkeypatch, process):
    backend = SimpleNamespace(Process=lambda pid: process, Error=FakeError,
                              NoSuchProcess=NoSuchProcess, STATUS_ZOMBIE="zombie", __version__="fixture")
    monkeypatch.setattr(memory, "_psutil", lambda: backend)
    return memory.ProcessTreeSampler(process.pid)


def test_tree_peak_is_peak_of_observed_sums_not_sum_of_individual_peaks(monkeypatch):
    child = FakeProcess(2, 30, 24)
    root = FakeProcess(1, 10, 8, [child])
    s = sampler_for(monkeypatch, root)
    s.sample()
    root.rss, root.uss, child.rss, child.uss = 30, 24, 10, 8
    s.sample()
    r = s.report()
    assert r["sampled_peak_root_rss_bytes"] == 30
    assert r["sampled_peak_observed_tree_rss_bytes"] == 40  # Not 30 + 30.
    assert r["sampled_peak_observed_tree_uss_bytes"] == 32
    assert r["max_observed_processes"] == 2
    assert r["complete_tree_coverage_guaranteed"] is False


@pytest.mark.parametrize("missing", [None, AccessDenied()])
def test_missing_child_uss_is_null_and_root_uss_remains_available(monkeypatch, missing):
    child = FakeProcess(2, 30, missing)
    s = sampler_for(monkeypatch, FakeProcess(1, 10, 8, [child]))
    sample = s.sample()
    r = s.report()
    assert sample["observed_tree_uss_bytes"] is None
    assert r["sampled_peak_observed_tree_uss_bytes"] is None
    assert r["sampled_peak_root_uss_bytes"] == 8
    assert r["sampled_peak_observed_tree_rss_bytes"] == 40
    assert r["errors"] == ({"uss:AccessDenied": 1} if missing else {})


def test_reparented_child_kept_but_reused_pid_excluded(monkeypatch):
    child = FakeProcess(2, 30, 24)
    root = FakeProcess(1, 10, 8, [child])
    s = sampler_for(monkeypatch, root)
    s.sample()
    root.running = False
    child.rss = 40
    assert s.sample()["observed_tree_rss_bytes"] == 40
    child.birth += 1
    child.rss = 1000
    assert s.sample()["observed_tree_rss_bytes"] is None
    assert s.report()["sampled_peak_observed_tree_rss_bytes"] == 40


def test_discovery_errors_are_exposed(monkeypatch):
    root = FakeProcess(1, 10, 8)
    def denied(**kwargs):
        raise AccessDenied()
    root.children = denied
    s = sampler_for(monkeypatch, root)
    assert s.sample()["discovery_ok"] is False
    assert s.report()["sample_counts"]["incomplete_observed_tree_samples"] == 1
    assert s.report()["errors"] == {"discover:AccessDenied": 1}


@pytest.mark.parametrize("field,value", [("timeout", 0), ("timeout", float("inf")),
    ("interval_seconds", -1), ("interval_seconds", float("nan")), ("cleanup_timeout", 0)])
def test_invalid_bounds_rejected_before_process_launch(monkeypatch, field, value):
    monkeypatch.setattr(memory.subprocess, "Popen", lambda *a, **k: pytest.fail("process launched"))
    kwargs = {"timeout": 1, field: value}
    with pytest.raises(ValueError, match="finite and positive"):
        memory.run_measured([sys.executable, "-c", "pass"], **kwargs)


def real_psutil():
    return pytest.importorskip("psutil")


def not_running(pid, psutil):
    try:
        p = psutil.Process(pid)
        return not p.is_running() or p.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_real_allocations_are_observed_without_monitor_self_memory(tmp_path):
    real_psutil()
    child = "import time; x=bytearray(24*1024*1024); time.sleep(.35)"
    root = ("import subprocess,sys,time; x=bytearray(12*1024*1024); "
            f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); p.wait()")
    r = memory.run_measured([sys.executable, "-c", root], timeout=3, interval_seconds=.01,
                            stdout_path=tmp_path / "out", stderr_path=tmp_path / "err")
    assert r["returncode"] == 0 and not r["timed_out"]
    assert r["memory"]["sampled_peak_root_rss_bytes"] >= 12*1024*1024
    assert r["memory"]["sampled_peak_observed_tree_rss_bytes"] >= 36*1024*1024
    assert r["memory"]["max_observed_processes"] == 2
    assert os.getpid() not in [i[0] for i in r["memory"]["observed_process_identities"]]
    assert r["cleanup"]["leader_reaped"] and not r["cleanup"]["errors"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group contract")
def test_timeout_kills_leader_and_child(tmp_path):
    psutil = real_psutil()
    pidfile = tmp_path / "pids.json"
    code = ("import subprocess,sys,time,json,os; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            f"open({str(pidfile)!r},'w').write(json.dumps([os.getpid(),p.pid])); time.sleep(30)")
    r = memory.run_measured([sys.executable, "-c", code], timeout=.4, interval_seconds=.01)
    assert r["timed_out"] and r["returncode"] != 0
    cleanup = r["cleanup"]
    waitid_api = ("waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    if all(hasattr(os, name) for name in waitid_api):
        assert cleanup["group_killed"]
    else:
        assert not cleanup["group_killed"]
        assert cleanup["group_skip_reason"] == "direct_child_unsupported"
    assert cleanup["leader_reaped"] and cleanup["observed_termination_verified"]
    assert not cleanup["errors"] and not cleanup["unverified_observed_identities"]
    pids = json.loads(pidfile.read_text())
    observed = {pid for pid, _ in r["memory"]["observed_process_identities"]}
    assert set(pids) <= observed
    assert all(not_running(pid, psutil) for pid in pids)


def test_callback_failure_cleans_up_before_propagating(tmp_path):
    psutil = real_psutil()
    seen = []
    original = memory.ProcessTreeSampler
    class Capture(original):
        def __init__(self, pid, **kwargs):
            seen.append(pid)
            super().__init__(pid, **kwargs)
    from unittest.mock import patch
    def fail(sample):
        raise RuntimeError("observer stopped")
    with patch.object(memory, "ProcessTreeSampler", Capture):
        with pytest.raises(RuntimeError, match="observer stopped"):
            memory.run_measured([sys.executable, "-c", "import time; time.sleep(30)"],
                                timeout=1, on_sample=fail)
    assert len(seen) == 1 and not_running(seen[0], psutil)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group contract")
def test_reaped_leader_never_signals_a_reused_process_group(monkeypatch):
    proc = SimpleNamespace(pid=1234, returncode=0, poll=lambda: 0, wait=lambda **kw: 0)
    sampler = SimpleNamespace(processes={})
    monkeypatch.setattr(memory.os, "killpg", lambda *a: pytest.fail("unowned process group signalled"))
    result = memory._cleanup(proc, sampler, 1)
    assert not result["group_killed"]
    assert result["group_skip_reason"] == "direct_child_reaped"
    assert result["leader_reaped"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group contract")
def test_measurement_keeps_exited_child_waitable_until_group_cleanup(monkeypatch):
    calls = []
    class Proc:
        pid, returncode = 1234, None
        def poll(self):
            calls.append("poll")
            self.returncode = 0
            return 0
        def wait(self, **kwargs):
            calls.append("wait")
            self.returncode = 0
            return 0
    proc = Proc()
    def signal_group(pid, signal):
        assert proc.returncode is None  # PID remains reserved, even after exit.
        calls.append("group_signal")
    def waitid(kind, pid, flags):
        assert flags & os.WNOWAIT
        return SimpleNamespace(si_pid=pid)
    monkeypatch.setattr(memory, "_psutil", lambda: None)
    monkeypatch.setattr(memory.subprocess, "Popen", lambda *a, **kw: proc)
    monkeypatch.setattr(memory, "ProcessTreeSampler", lambda *a, **kw: SimpleNamespace(
        processes={}, sample=lambda: {}, report=lambda: {}))
    # This deterministic test supplies the whole waitid API; macOS Python can
    # expose WNOWAIT while omitting waitid itself.
    for name, fallback in (("P_PID", 1), ("WEXITED", 2), ("WNOHANG", 4), ("WNOWAIT", 8)):
        monkeypatch.setattr(memory.os, name, getattr(memory.os, name, fallback), raising=False)
    monkeypatch.setattr(memory.os, "waitid", waitid, raising=False)
    monkeypatch.setattr(memory.os, "killpg", signal_group)
    result = memory.run_measured(["stub"], timeout=1)
    assert calls[0] == "group_signal"
    assert "wait" in calls and result["returncode"] == 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group contract")
@pytest.mark.parametrize("members", [False, True, "inaccessible"])
def test_exited_group_permission_error_is_benign_only_with_no_live_members(monkeypatch, members):
    proc = SimpleNamespace(pid=1234, returncode=None, poll=lambda: 0, wait=lambda **kw: 0)
    monkeypatch.setattr(memory, "_child_state", lambda p: "exited")
    monkeypatch.setattr(memory, "_psutil", lambda: SimpleNamespace(Error=FakeError))
    def denied(*args):
        raise PermissionError()
    def scan(pid):
        if members == "inaccessible":
            raise AccessDenied()
        return members
    monkeypatch.setattr(memory.os, "killpg", denied)
    monkeypatch.setattr(memory, "_group_has_live_members", scan)
    result = memory._cleanup(proc, SimpleNamespace(processes={}), 1)
    assert bool(result["errors"]) is (members is not False)
    assert (result["group_skip_reason"] == "no_live_group_members") is (members is False)


@pytest.mark.parametrize("state", ["unowned", "unsupported"])
def test_missing_child_ownership_skips_group_but_checks_observed_identities(monkeypatch, state):
    killed = []
    child = FakeProcess(2, 1, 1)
    def stop():
        killed.append(2)
        child.running = False
    child.kill = stop
    reused = FakeProcess(3, 1, 1)
    reused.birth += 1
    reused.kill = lambda: pytest.fail("reused child PID killed")
    proc = SimpleNamespace(pid=1234, returncode=None, poll=lambda: 0, wait=lambda **kw: 0)
    sampler = SimpleNamespace(processes={(2, 2.): child, (3, 3.): reused},
                              backend=SimpleNamespace(NoSuchProcess=NoSuchProcess, Error=FakeError, STATUS_ZOMBIE="zombie"))
    monkeypatch.setattr(memory, "_child_state", lambda p: state)
    monkeypatch.setattr(memory.os, "killpg", lambda *a: pytest.fail("unowned process group signalled"), raising=False)
    result = memory._cleanup(proc, sampler, 1)
    assert killed == [2]
    assert result["group_skip_reason"] == f"direct_child_{state}"
    assert result["observed_termination_verified"]


@pytest.mark.parametrize("mode", ["zombie", "delayed_exit", "inaccessible", "survives"])
def test_observed_cleanup_verifies_nonexecution_and_keeps_real_failures(monkeypatch, mode):
    proc = SimpleNamespace(pid=1234, returncode=0, poll=lambda: 0, wait=lambda **kw: 0)
    child = FakeProcess(2, 1, 1)
    reads = []
    kills = []
    def status():
        reads.append(1)
        if mode == "inaccessible":
            raise AccessDenied()
        return "zombie" if mode == "zombie" or (mode == "delayed_exit" and len(reads) >= 3) else "running"
    child.status = status
    child.kill = lambda: kills.append(2)  # Signal delivery need not mean exit.
    sampler = SimpleNamespace(processes={(2, 2.): child}, backend=SimpleNamespace(
        NoSuchProcess=NoSuchProcess, Error=FakeError, STATUS_ZOMBIE="zombie"))
    result = memory._cleanup(proc, sampler, .025)
    assert result["observed_termination_verified"] is (mode in {"zombie", "delayed_exit"})
    assert bool(result["errors"]) is (mode in {"inaccessible", "survives"})
    assert result["unverified_observed_identities"] == ([] if mode in {"zombie", "delayed_exit"} else [[2, 2.]])
    assert kills == ([2] if mode in {"delayed_exit", "survives"} else [])


def test_sampler_constructor_failure_still_cleans_up_direct_child(monkeypatch):
    psutil = real_psutil()
    seen = []
    def fail(pid, **kwargs):
        seen.append(pid)
        raise RuntimeError("sampler construction failed")
    monkeypatch.setattr(memory, "ProcessTreeSampler", fail)
    with pytest.raises(RuntimeError, match="sampler construction failed"):
        memory.run_measured([sys.executable, "-c", "import time; time.sleep(30)"], timeout=1)
    assert len(seen) == 1 and not_running(seen[0], psutil)


def test_logs_are_exclusive_before_any_process_starts(tmp_path, monkeypatch):
    real_psutil()
    path = tmp_path / "exists"
    path.write_text("preserve")
    monkeypatch.setattr(memory.subprocess, "Popen", lambda *a, **k: pytest.fail("process launched"))
    with pytest.raises(FileExistsError):
        memory.run_measured([sys.executable, "-c", "pass"], timeout=1, stdout_path=path)
    assert path.read_text() == "preserve"


@pytest.mark.parametrize("missing", ["waitid", "WNOWAIT"])
def test_partial_waitid_api_preserves_unsupported_fallback(monkeypatch, missing):
    monkeypatch.delattr(memory.os, missing, raising=False)
    proc = SimpleNamespace(pid=1234, returncode=None)
    assert memory._child_state(proc) == "unsupported"
