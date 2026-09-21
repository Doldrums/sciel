import json
import os
from pathlib import Path
import signal
import sys

import pytest

from sciel.diagnostics import methods_observer as observer


def _receipt(root, key, cpu, *, invocation=None, status="invoked"):
    value = dict(category="execution", result=dict(invocation_id=invocation or key,
                 dispatch_status=status, resources=dict(cpu_seconds=cpu),
                 resource_measurement=dict(cpu="trusted_cgroup_observation")))
    value["sha256"] = observer._digest(value)
    path = root / "registry" / "dispatch" / key / "receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_observation_counts_only_new_unique_invocations_and_exposes_unknown(tmp_path):
    _receipt(tmp_path, "old", 100)
    _receipt(tmp_path, "new", 2, invocation="one")
    _receipt(tmp_path, "same_call", 2, invocation="one")
    _receipt(tmp_path, "unknown", None, status="unknown")
    pending = tmp_path / "registry" / "dispatch" / "pending" / "intent.json"
    pending.parent.mkdir()
    pending.write_text(json.dumps(dict(category="execution")))
    value = observer.container_cpu(tmp_path, {"old"})
    assert value["container_cpu_seconds"] == 2
    assert value["container_invocations"] == 1
    assert value["container_unknown_receipts"] == 1
    assert value["container_pending_calls"] == 1
    assert value["container_cpu_complete"] is False


def test_tampered_receipt_is_not_measurement(tmp_path):
    path = _receipt(tmp_path, "bad", 1)
    value = json.loads(path.read_text())
    value["result"]["resources"]["cpu_seconds"] = 1000
    path.write_text(json.dumps(value))
    value = observer.container_cpu(tmp_path, set())
    assert value["container_cpu_seconds"] == 0
    assert value["invalid_receipts"] == 1
    assert value["container_cpu_complete"] is False


def test_ledger_scope_excludes_historical_reservations_and_fixed_upfront_allocations(tmp_path):
    from sciel.methods.budget import ResourceLedger
    ledger = ResourceLedger(tmp_path / "resources.jsonl", dict(cpu_seconds=100, wall_seconds=100,
                            dollars=0, gpu_seconds=0, storage_bytes=10000, steps=100))
    ledger.create_account("lab", {})
    def charge(key, category, amount):
        bounds = dict(cpu_seconds=amount, wall_seconds=0, dollars=0, gpu_seconds=0, storage_bytes=0, steps=0)
        row = ledger.reserve("lab", key, bounds, category)
        ledger.settle(row["id"], bounds)
        return row["id"]
    old = charge("old", "execution", 5)
    charge("new", "execution", 2)
    charge("setup", "bootstrap", 1)
    charge("upfront", "upfront_creation", 20)
    value = observer.ledger_cpu(tmp_path, {old})
    assert value["ledger_cpu_seconds_this_invocation"] == 3
    assert value["ledger_current_unresolved"] == 0
    assert value["ledger_read_errors"] == 0


def test_current_regular_file_footprint_does_not_follow_external_symlink(tmp_path):
    outside = tmp_path / "outside"
    outside.write_bytes(b"private" * 100)
    root = tmp_path / "run"
    root.mkdir()
    (root / "data").write_bytes(b"abc")
    (root / "linked").symlink_to(outside)
    assert observer.logical_size(root) == (3, 0)


def test_successful_child_is_observed_without_serializing_command_or_env(tmp_path, monkeypatch):
    output = tmp_path / "run"
    terminal = tmp_path / "worker.json"
    marker = "TEST_PRIVATE_VALUE_not_for_telemetry"
    monkeypatch.setenv("OBSERVER_TEST_SECRET", marker)
    # Simulate the CLI and wrapper; no providers, Docker, or existing runs.
    command = [sys.executable, "-c", """
import json,os,resource,time
from pathlib import Path
root,terminal=map(Path,__import__('sys').argv[1:])
assert os.environ['OBSERVER_TEST_SECRET']
start=time.process_time()
while time.process_time()-start<.08: pass
(root/'run-report.json').write_text(json.dumps({'status':'completed'}))
u=resource.getrusage(resource.RUSAGE_SELF)
terminal.write_text(json.dumps({'schema':'sciel-method-worker-terminal/1','pid':os.getpid(),'exit_code':0,'controller_cpu_seconds':u.ru_utime+u.ru_stime}))
""", str(output), str(terminal)]
    value = observer.observe(command, output, interval=.05, worker_terminal=terminal)
    assert value["status"] == "completed"
    assert value["finished_at"]
    assert value["overhead"]["controller_cpu_seconds"] >= .08
    assert value["overhead"]["controller_cpu_measurement"] == "worker_rusage_terminal"
    assert value["overhead"]["additive_to_ledger"] is False
    assert value["overhead"]["budget_enforced"] is False
    assert value["overhead"]["combined_observed_cpu_seconds"] == value["overhead"]["controller_cpu_seconds"]
    assert marker not in (output / "runtime.json").read_text()
    assert "command" not in value and "env" not in value
    assert observer.process_matches(value) is False
    assert list((output / "observation").glob("*/terminal.json"))


def test_crashed_controller_does_not_reuse_historical_success_or_retry(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "run-report.json").write_text('{"status":"completed"}')
    counter = tmp_path / "dispatches"
    command = [sys.executable, "-c",
               "import os,signal,sys; open(sys.argv[1],'a').write('once\\n'); os.kill(os.getpid(),signal.SIGKILL)",
               str(counter)]
    value = observer.observe(command, output, interval=.05, resume=True)
    assert value["status"] == "unknown"
    assert value["exit_code"] == -signal.SIGKILL
    assert counter.read_text() == "once\n"
    assert value["controller_measurement_terminal"] is False


@pytest.mark.parametrize("matches", [True, None])
def test_live_or_unverifiable_previous_pid_refuses_launch(tmp_path, monkeypatch, matches):
    (tmp_path / "runtime.json").write_text('{"pid":123}')
    monkeypatch.setattr(observer, "process_matches", lambda record: matches)
    monkeypatch.setattr(observer.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="live or its identity"):
        observer.observe([sys.executable, "-c", "pass"], tmp_path, resume=True)


def test_existing_scientific_run_requires_explicit_resume(tmp_path):
    (tmp_path / "registry").mkdir()
    with pytest.raises(ValueError, match="explicit --resume"):
        observer.observe([sys.executable, "-c", "pass"], tmp_path)


def test_worker_failure_saves_typed_error_without_repr(tmp_path, monkeypatch):
    import sciel.methods.__main__ as methods
    secret = "private-provider-payload"
    def broken(argv):
        raise RuntimeError(secret)
    monkeypatch.setattr(methods, "main", broken)
    terminal = tmp_path / "terminal.json"
    assert observer._worker(tmp_path / "config", tmp_path / "out", terminal) == 2
    value = json.loads(terminal.read_bytes())
    assert value["error_type"] == "RuntimeError"
    assert secret not in terminal.read_text()


def test_pid_reuse_does_not_attribute_foreign_cpu(monkeypatch):
    monkeypatch.setattr(observer, "process_matches", lambda record: False)
    monkeypatch.setattr(observer.subprocess, "run", lambda *a, **kw: pytest.fail("must not sample reused PID"))
    assert observer._sample_cpu(dict(pid=123)) is None


def test_budget_completion_is_stopped_not_unexpected_failure():
    assert observer._terminal_status(2, {"pid":123}, {"status":"budget_exhausted"}) == ("stopped", "budget_exhausted")
    assert observer._terminal_status(2, None, {"status":"budget_exhausted"})[0] == "unknown"
    assert observer._terminal_status(2, {"pid":123}, {"status":"accounting_stopped"}) == ("unknown", "accounting_stopped")


def test_invalid_worker_terminal_does_not_certify_controller_success(tmp_path):
    terminal = tmp_path / "worker.json"
    terminal.write_text(json.dumps(dict(schema="sciel-method-worker-terminal/1",pid=1,
                                       exit_code=0,controller_cpu_seconds=999)))
    command = [sys.executable, "-c", "from pathlib import Path; import sys; (Path(sys.argv[1])/'run-report.json').write_text('{\"status\":\"completed\"}')", str(tmp_path)]
    value = observer.observe(command, tmp_path, interval=.05, worker_terminal=terminal)
    assert value["status"] == "unknown"
    assert value["controller_measurement_terminal"] is False
    assert value["overhead"]["controller_cpu_seconds"] != 999
