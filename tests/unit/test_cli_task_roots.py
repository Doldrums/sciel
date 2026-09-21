"""Task-bundle CLI contracts; no provider or evaluator is invoked."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sciel.cli import main as cli
from sciel.eval import registry


PROJECT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(cli.__file__).resolve().parents[2]


def _cli(package_root, *args):
    # Run the real entry point from an explicit package location, independent of
    # editable installs/PYTHONPATH. No shell, auth environment or candidate runs.
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c",
         "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); "
         "runpy.run_module('sciel.cli.main', run_name='__main__')",
         str(package_root), *map(str, args)],
        capture_output=True, text=True, timeout=20,
        env={"PATH": os.defpath},
    )


def _failure(result):
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert result.stdout == ""
    assert "--tasks-root" in result.stderr
    assert "task.yaml" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.fixture
def external_tasks(tmp_path):
    root = tmp_path / "external tasks"
    task = root / "test" / "toy_quadratic"
    task.mkdir(parents=True)
    (task / "task.yaml").write_bytes(
        (PROJECT / "tasks/test/toy_quadratic/task.yaml").read_bytes())
    for name in ("evaluator.py", "initial_program.py"):
        (task / name).write_text("raise AssertionError('CLI discovery must not execute code')\n")
    return root


@pytest.mark.parametrize("action", ["list", "validate", "hash"])
@pytest.mark.parametrize("root_kind", ["missing", "empty", "file", "unrelated_files"])
def test_task_commands_reject_unavailable_bundles(tmp_path, action, root_kind):
    root = tmp_path / root_kind
    if root_kind == "empty":
        root.mkdir()
    elif root_kind == "unrelated_files":
        root.mkdir()
        (root / "README.txt").write_text("No task definitions here.")
    elif root_kind == "file":
        root.write_text("not a task directory")
    args = ["tasks", action]
    if action == "hash":
        args.append("test/toy_quadratic")
    result = _cli(PACKAGE_ROOT, *args, "--tasks-root", root)
    _failure(result)
    assert str(root) in result.stderr


@pytest.fixture(scope="module")
def installed_layout(tmp_path_factory):
    # Model the installed package layout without building/installing a wheel or
    # relying on checkout assets beside it. Dependencies remain in this Python.
    root = tmp_path_factory.mktemp("installed") / "lib/python3.14/site-packages"
    root.mkdir(parents=True)
    shutil.copytree(PACKAGE_ROOT / "sciel", root / "sciel",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return root


@pytest.mark.parametrize("action", ["list", "validate", "hash"])
def test_installed_layout_missing_default_is_actionable(installed_layout, action):
    args = ["tasks", action]
    if action == "hash":
        args.append("test/toy_quadratic")
    result = _cli(installed_layout, *args)
    _failure(result)
    assert "not included in the installed wheel" in result.stderr


def test_installed_layout_accepts_external_bundle_without_executing_it(
        installed_layout, external_tasks):
    listed = _cli(installed_layout, "tasks", "list", "--tasks-root", external_tasks)
    assert listed.returncode == 0, listed.stderr
    assert "test/toy_quadratic\t" in listed.stdout
    hashed = _cli(installed_layout, "tasks", "hash", "test/toy_quadratic",
                  "--tasks-root", external_tasks)
    assert hashed.returncode == 0, hashed.stderr
    assert hashed.stdout.strip() == registry.load_task(
        str(external_tasks / "test/toy_quadratic")).task_hash


def test_source_checkout_default_still_lists_tasks():
    result = _cli(PROJECT / "src", "tasks", "list")
    assert result.returncode == 0, result.stderr
    assert "test/toy_quadratic\t" in result.stdout
    assert "oe/circle_packing\t" in result.stdout


def test_empty_library_discovery_remains_compatible(tmp_path):
    assert registry.discover(str(tmp_path)) == []
    assert registry.discover(str(tmp_path / "missing")) == []


def test_explicit_validation_dispatches_only_discovered_tasks(monkeypatch, external_tasks):
    seen = []
    def check(task, **kwargs):
        seen.append((task.qualified_name, kwargs))
        return {"valid": True, "informative": True}
    monkeypatch.setattr(registry, "determinism_check", check)
    assert cli.main(["tasks", "validate", "--tasks-root", str(external_tasks)]) == 0
    assert seen == [("test/toy_quadratic", {"python": sys.executable, "program_path": None})]


@pytest.mark.parametrize("command", ["run", "resume"])
def test_pinned_remote_source_keeps_factory_resolution(
        monkeypatch, tmp_path, external_tasks, command):
    cfg = {"task": "test/toy_quadratic", "random_seed": 42,
           "sciel_tasks_repo": "example/tasks", "sciel_tasks_revision": "pinned-revision"}
    root = tmp_path / "missing-default"
    monkeypatch.setattr(cli, "_default_tasks_root", lambda: str(root))
    calls = []
    def build(config, tasks_root, run_dir, **kwargs):
        calls.append((config, tasks_root, kwargs["tasks_root_override"]))
        return SimpleNamespace(run=lambda: {"reason": "fixture"})
    monkeypatch.setattr(cli, "build_engine", build)
    if command == "run":
        monkeypatch.setattr(cli, "_resolve_config", lambda *_: dict(cfg))
        args = ["run", "unused-config", "--outputs", str(tmp_path / "outputs")]
    else:
        from sciel.engine import checkpoint
        monkeypatch.setattr(checkpoint, "load", lambda *_args, **_kwargs: None)
        run = tmp_path / "run"
        run.mkdir()
        (run / "checkpoint.json").write_text(json.dumps({"config": cfg}))
        args = ["resume", str(run)]
    assert cli.main(args) == 0
    assert len(calls) == 1
    assert calls[0][0]["sciel_tasks_repo"] == "example/tasks"
    assert calls[0][1:] == (str(root), False)
    assert cli.main([*args, "--tasks-root", str(external_tasks)]) == 0
    assert len(calls) == 2
    assert calls[1][1:] == (str(external_tasks), True)


def test_unrelated_task_validation_errors_are_not_relabelled(monkeypatch, external_tasks):
    def invalid(_root):
        raise ValueError("invalid evaluator-owned metadata")
    monkeypatch.setattr(registry, "discover", invalid)
    with pytest.raises(ValueError, match="invalid evaluator-owned metadata"):
        cli.main(["tasks", "list", "--tasks-root", str(external_tasks)])


def test_hash_missing_name_keeps_usage_precedence(tmp_path):
    result = _cli(PACKAGE_ROOT, "tasks", "hash", "--tasks-root", tmp_path / "missing")
    assert result.returncode == 2
    assert "usage: sciel tasks hash <suite>/<name>" in result.stderr
    assert "Task root" not in result.stderr


def test_hash_does_not_discover_unrelated_malformed_task(external_tasks):
    unrelated = external_tasks / "other" / "malformed"
    unrelated.mkdir(parents=True)
    (unrelated / "task.yaml").write_text("invalid: task metadata\n")
    result = _cli(PACKAGE_ROOT, "tasks", "hash", "test/toy_quadratic",
                  "--tasks-root", external_tasks)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == registry.load_task(
        str(external_tasks / "test/toy_quadratic")).task_hash
