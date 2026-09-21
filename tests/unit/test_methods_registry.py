"""Offline persistence/contract tests; no submitted source is executed."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sciel.methods.models import canonical_bytes, content_hash, validate_manifest, validate_schema, validate_value
from sciel.methods.registry import Registry, RegistryCorruption, RevisionConflict


IMAGE = "sha256:" + "a" * 64


def manifest(**changes):
    return {"name": "bounded tool", "kind": "tool", "author_lab": "author",
            "entrypoint": "tool.py:run", "input_schema": {"type": "number"},
            "output_schema": {"type": "number"}, "protocol": "Apply once to assigned input.",
            "runtime_lock": {"image_id": IMAGE}, "dependencies": [], "parents": [],
            "resources": {"wall_seconds": 2.0, "cpu_seconds": 1.0, "memory_bytes": 67108864,
                          "pids": 4, "max_output_bytes": 8192, "max_stderr_bytes": 4096},
            "scope": {"inputs": "finite scalars"}, "examples": [], **changes}


def register(registry, **changes):
    return registry.register({"tool.py": "def run(value):\n    return value + 1\n"}, manifest(**changes))


def initial(**changes):
    return {"search_procedure": "b" * 64, "runtime_lock": {"image_id": IMAGE},
            "generator": {"model": "fixture", "max_steps": 4}, **changes}


def test_immutable_package_identity_and_detached_views(tmp_path):
    registry = Registry(tmp_path)
    files = {"tool.py": "def run(value):\n    return value + 1\n"}
    spec = manifest()
    method = registry.register(files, spec)
    assert method == content_hash({"schema": "sciel-method/1", "manifest": spec, "files": files})
    files["tool.py"] = "broken"
    spec["scope"]["inputs"] = "changed"
    first = registry.get_method(method)
    first["files"]["tool.py"] = "changed"
    assert registry.get_method(method)["files"]["tool.py"].endswith("value + 1\n")
    assert register(registry) == method
    assert len(registry.events()) == 1
    changed = register(registry, protocol="A genuinely different frozen protocol.", parents=[method])
    assert changed != method
    assert registry.get_method(changed)["manifest"]["parents"] == [method]


@pytest.mark.parametrize("path", ["/tmp/payload.py", "../payload.py", "a/../b.py", "a//b.py",
                                  "C:/b.py", "a\\b.py", "a/./b.py", "a\x00.py"])
def test_unsafe_package_paths_rejected_without_writes(tmp_path, path):
    registry = Registry(tmp_path)
    with pytest.raises(ValueError):
        registry.register({"tool.py": "def run(x): return x", path: "unused"}, manifest())
    assert registry.events() == []
    assert list((tmp_path / "methods").iterdir()) == []


def test_package_conflicts_entrypoint_and_mutable_runtime_rejected():
    for files, spec in [
        ({"tool.py": "def other(x): return x"}, manifest()),
        ({"tool.py": "def run("}, manifest()),
        ({"tool.py": "def run(x): return x", "tool.py/child": "x"}, manifest()),
        ({"tool.py": "def run(x): return x"}, manifest(runtime_lock={"image_id": "python:latest"})),
        ({"tool.py": "def run(x): return x"}, manifest(kind=[])),
        ({"tool.py": "def run(x): return x", "helper.txt": "\ud800"}, manifest()),
    ]:
        with pytest.raises(ValueError):
            validate_manifest(spec, files)


def test_schema_does_not_silently_ignore_constraints_or_bool_numbers():
    schema = {"type": "object", "properties": {
        "values": {"type": "array", "items": {"type": "integer", "minimum": 0}, "minItems": 1, "maxItems": 2},
        "label": {"type": "string", "minLength": 1, "maxLength": 4},
    }, "required": ["values"], "additionalProperties": False}
    validate_value({"values": [0, 2], "label": "yes"}, schema)
    for bad in ({"values": [True]}, {"values": []}, {"values": [1], "extra": 1},
                {"values": [1], "label": "longer"}, {"label": "yes"}):
        with pytest.raises(ValueError):
            validate_value(bad, schema)
    for unsupported in ({"type": "string", "pattern": "x"}, {"$ref": "x"},
                        {"type": "array"}, {"type": "object", "required": ["undeclared"]}):
        with pytest.raises(ValueError):
            validate_schema(unsupported)
    with pytest.raises(ValueError):
        validate_value(float("nan"), {"type": "number"})
    with pytest.raises(ValueError):
        validate_value(True, {"type": "integer", "enum": [1]})
    validate_value("x", {"type": "string", "enum": ["x", "y"]})


def test_dependency_dag_requires_existing_versions(tmp_path):
    registry = Registry(tmp_path)
    with pytest.raises(ValueError, match="unknown method"):
        register(registry, dependencies=["c" * 64])
    parent = register(registry)
    middle = register(registry, name="middle", dependencies=[parent])
    leaf = register(registry, name="leaf", dependencies=[middle], parents=[parent])
    assert registry.get_method(leaf)["manifest"]["dependencies"] == [middle]
    with pytest.raises(ValueError, match="duplicate"):
        register(registry, dependencies=[parent, parent])
    assert len(registry.events()) == 3


def test_status_evidence_separate_from_bytes_and_history_survives_replacement(tmp_path):
    registry = Registry(tmp_path)
    method = register(registry)
    original = registry.get_method(method)
    assert registry.method_status(method) == {"status": "draft", "evidence_refs": []}
    with pytest.raises(ValueError, match="requires evidence"):
        registry.status_method(method, "released")
    with pytest.raises(ValueError, match="unknown evidence"):
        registry.status_method(method, "released", ["c" * 64])
    negative = registry.put_artifact("assessment", {"outcome": "zero", "contract_passed": True})
    registry.status_method(method, "released", [negative])
    revised = register(registry, name="repaired", parents=[method])
    registry.status_method(method, "superseded", [negative])
    assert registry.get_method(method) == original
    assert registry.get_method(revised)["manifest"]["parents"] == [method]
    assert registry.method_status(method)["evidence_refs"] == [negative]
    assert len([e for e in registry.events() if e["kind"] == "method_status"]) == 2
    with pytest.raises(ValueError, match="terminal"):
        registry.status_method(method, "released", [negative])


def test_artifact_kind_binding_and_idempotent_detached_returns(tmp_path):
    registry = Registry(tmp_path)
    body = {"receipts": [{"status": "unknown"}]}
    artifact = registry.put_artifact("execution", body)
    assert registry.put_artifact("execution", body) == artifact
    assert registry.put_artifact("assessment", body) != artifact
    body["receipts"].clear()
    returned = registry.get_artifact("execution", artifact)
    returned["receipts"].clear()
    assert registry.get_artifact("execution", artifact)["receipts"] == [{"status": "unknown"}]
    with pytest.raises(KeyError):
        registry.get_artifact("assessment", artifact)


def test_lab_revision_cas_fork_and_budget_identity(tmp_path):
    registry = Registry(tmp_path)
    method = register(registry)
    lab = registry.create_lab("author", "Can this help?", "account-a", "B", initial(active_methods=[method]))
    first = lab["active_revision"]
    evidence = registry.put_artifact("execution", {"status": "failed"})
    second = registry.revise_lab("author", first, {"evidence_refs": [evidence]}, "use the observed failure")
    assert registry.lab("author")["active_revision"] == second
    assert registry.revision(first)["evidence_refs"] == []
    assert registry.revision(second)["budget_account"] == "account-a"
    with pytest.raises(RevisionConflict):
        registry.revise_lab("author", first, {}, "stale update")
    with pytest.raises(ValueError):
        registry.revise_lab("author", second, {"budget_account": "free-money"}, "reset")
    with pytest.raises(ValueError, match="own budget"):
        registry.create_lab("fork", "New question", "account-a", "C", {"parent_revision": second})
    fork = registry.create_lab("fork", "New question", "account-b", "C", {"parent_revision": second})
    snapshot = registry.revision(fork["active_revision"])
    assert snapshot["parent_revision"] == second
    assert snapshot["active_methods"] == [method]
    assert snapshot["evidence_refs"] == [evidence]
    registry.revise_lab("author", second, {"active_methods": []}, "leave local toolbox")
    assert registry.revision(fork["active_revision"])["active_methods"] == [method]


def test_concurrent_cas_has_exactly_one_winner(tmp_path):
    registry = Registry(tmp_path)
    lab = registry.create_lab("lab", "question", "money", "B", initial())
    first = lab["active_revision"]
    def revise(n):
        try:
            return Registry(tmp_path).revise_lab("lab", first, {"generator": {"choice": n}}, "parallel choice")
        except RevisionConflict:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(revise, range(4)))
    assert sum(result is not None for result in results) == 1
    assert len(registry.events()) == 2


def test_process_shared_append_serializes_and_chains(tmp_path):
    registry = Registry(tmp_path)
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    script = ("from sciel.methods.registry import Registry\nimport sys\n"
              "r=Registry(sys.argv[1])\n"
              "for i in range(6): r.append('receipt', {'worker':sys.argv[2], 'n':i})\n")
    env = {**os.environ, "PYTHONPATH": source_root, "PYTHONDONTWRITEBYTECODE": "1"}
    processes = [subprocess.Popen([sys.executable, "-B", "-c", script, str(tmp_path), str(i)],
                                  env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) for i in range(3)]
    for process in processes:
        _, error = process.communicate(timeout=10)
        assert process.returncode == 0, error.decode()
    events = registry.events()
    assert len(events) == 18
    assert [event["seq"] for event in events] == list(range(1, 19))
    assert len({(event["data"]["worker"], event["data"]["n"]) for event in events}) == 18


def test_checkpoint_validates_prefix_without_rolling_back_later_known_receipts(tmp_path):
    registry = Registry(tmp_path / "registry")
    method = register(registry)
    registry.create_lab("lab", "question", "budget", "B", initial(active_methods=[method]))
    path = tmp_path / "checkpoint.json"
    checkpoint = registry.checkpoint(path)
    event = registry.append("dispatch", {"invocation_id": "i1", "reservation": "held"})
    resumed = Registry(tmp_path / "registry")
    assert resumed.verify_checkpoint(path) == checkpoint
    assert resumed.events()[-1] == event
    assert resumed.lab("lab")["active_revision"] == checkpoint["state"]["labs"]["lab"]["active_revision"]
    # A missing terminal receipt remains a recorded unresolved dispatch, never
    # converted by the persistence layer into a successful/zero-cost execution.
    assert resumed.events()[-1]["kind"] == "dispatch"


@pytest.mark.parametrize("damage", ["edit", "truncate", "partial", "reorder"])
def test_corrupt_journal_or_missing_checkpoint_prefix_fail_closed(tmp_path, damage):
    registry = Registry(tmp_path)
    registry.append("receipt", {"n": 1})
    registry.append("receipt", {"n": 2})
    checkpoint = tmp_path / "checkpoint.json"
    registry.checkpoint(checkpoint)
    journal = tmp_path / "journal.jsonl"
    lines = journal.read_bytes().splitlines(keepends=True)
    if damage == "edit":
        lines[0] = lines[0].replace(b'"n":1', b'"n":9')
    elif damage == "truncate":
        lines.pop()
    elif damage == "partial":
        lines[-1] = lines[-1][:-4]
    else:
        lines.reverse()
    journal.write_bytes(b"".join(lines))
    with pytest.raises(RegistryCorruption):
        registry.verify_checkpoint(checkpoint)


def test_corrupt_package_detected_on_resume_even_if_not_current(tmp_path):
    registry = Registry(tmp_path)
    method = register(registry)
    newer = register(registry, parents=[method], name="newer")
    registry.status_method(method, "superseded")
    path = tmp_path / "methods" / f"{method}.json"
    value = json.loads(path.read_text())
    value["files"]["tool.py"] += "\n# tampered\n"
    path.write_bytes(canonical_bytes(value))
    with pytest.raises(RegistryCorruption, match="content hash"):
        Registry(tmp_path)
    assert newer != method


def test_rehashed_checkpoint_cannot_fabricate_activated_state(tmp_path):
    registry = Registry(tmp_path)
    method = register(registry)
    path = tmp_path / "checkpoint.json"
    value = registry.checkpoint(path)
    value["state"]["methods"][method]["status"] = "released"
    value["id"] = content_hash({key: item for key, item in value.items() if key != "id"})
    path.write_bytes(canonical_bytes(value))
    with pytest.raises(RegistryCorruption, match="anchored journal"):
        registry.verify_checkpoint(path)


def test_symlink_and_checkpoint_alias_cannot_overwrite_registry(tmp_path):
    root = tmp_path / "registry"
    registry = Registry(root)
    register(registry)
    before = (root / "journal.jsonl").read_bytes()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        registry.checkpoint(alias / "journal.jsonl")
    with pytest.raises(RegistryCorruption):
        Registry(alias)
    assert (root / "journal.jsonl").read_bytes() == before
    methods = root / "methods"
    moved = root / "moved"
    methods.rename(moved)
    methods.symlink_to(moved, target_is_directory=True)
    with pytest.raises(RegistryCorruption):
        registry.events()


def test_public_append_cannot_forge_reserved_activation(tmp_path):
    registry = Registry(tmp_path)
    with pytest.raises(ValueError, match="typed registry"):
        registry.append("method_status", {"id": "a" * 64, "status": "released"})
    assert registry.events() == []


def test_artifact_durable_before_registration_is_safe_to_retry(tmp_path, monkeypatch):
    registry = Registry(tmp_path)
    original = registry._append
    def fail(*args):
        raise OSError("fixture crash before journal commit")
    monkeypatch.setattr(registry, "_append", fail)
    with pytest.raises(OSError):
        registry.put_artifact("proposal", {"status": "draft"})
    assert registry.events() == []
    monkeypatch.setattr(registry, "_append", original)
    artifact = registry.put_artifact("proposal", {"status": "draft"})
    assert Registry(tmp_path).get_artifact("proposal", artifact) == {"status": "draft"}
    assert len(registry.events()) == 1
