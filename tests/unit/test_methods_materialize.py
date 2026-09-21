"""Pure source/provenance tests: no submitted or composed code executes here."""
import ast
from copy import deepcopy
import hashlib

import pytest

from sciel.methods.materialize import materialize
from sciel.methods.models import content_hash


IMAGE = "sha256:" + "a" * 64


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def bundle(source="def run(x):\n    return x + 1\n", *, files=None, **changes):
    manifest = dict(name="fixture", kind="tool", author_lab="author", entrypoint="tool.py:run",
                    input_schema={"type": "number"}, output_schema={"type": "number"},
                    protocol="Apply once.", runtime_lock={"image_id": IMAGE},
                    dependencies=[], parents=[], scope="finite inputs", examples=[],
                    resources=dict(wall_seconds=2, cpu_seconds=1, memory_bytes=67108864,
                                   pids=4, max_output_bytes=8192, max_stderr_bytes=4096))
    manifest.update(changes)
    body = dict(schema="sciel-method/1", manifest=manifest, files=files or {"tool.py": source})
    return dict(id=content_hash(body), manifest=manifest, files=body["files"])


def adapter(parent, edits):
    patch = dict(expected_parent_sha256=sha(parent), edits=edits)
    method = bundle(kind="adapter", protocol={"patch": patch})
    return method, dict(version_id=method["id"], **patch)


def payload_from_source(source):
    tree = ast.parse(source)
    bootstrap = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_sciel_bootstrap")
    assignment = next(node for node in bootstrap.body if isinstance(node, ast.Assign)
                      and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "bundles")
    return ast.literal_eval(assignment.value)


def test_exact_bytes_embedded_without_executing_and_future_docstring_preserved(tmp_path):
    forbidden = tmp_path / "must-not-exist"
    source = f"from pathlib import Path\nPath({str(forbidden)!r}).write_text('executed')\ndef run(x): return x + 1\n"
    tool = bundle(source, files={"tool.py": source, "data/readme.txt": "exact data\n", "helper.py": "VALUE = 3\n"})
    parent = '#!/usr/bin/env python3\n"""Original module documentation."""\nfrom __future__ import annotations\n\ndef run(value):\n    return sciel_method(' + repr(tool["id"]) + ', value)\n'
    before = deepcopy(tool)
    result = materialize(parent, [tool])
    assert result["status"] == "ok", result
    assert not forbidden.exists()
    tree = ast.parse(result["code"])
    assert ast.get_docstring(tree) == "Original module documentation."
    assert isinstance(tree.body[1], ast.ImportFrom) and tree.body[1].module == "__future__"
    assert result["code"].startswith(parent.split("\ndef run")[0])
    assert payload_from_source(result["code"])[tool["id"]]["files"] == tool["files"]
    assert tool == before
    receipt = result["receipt"]
    assert receipt["source_sha256"] == sha(result["code"])
    assert receipt["parent_source_sha256"] == sha(parent)
    assert receipt["components"][0]["id"] == tool["id"]
    assert receipt["invocations_attested"] is False
    assert receipt["resources_enforced"] == "outer_candidate_only"
    assert receipt["id"] == content_hash({k: v for k, v in receipt.items() if k != "id"})


def test_dependency_closure_includes_exact_helpers_and_provenance_parents_need_not_be_loaded():
    dependency = bundle(files={"tool.py": "import helper\ndef run(x): return helper.V + x\n", "helper.py": "V = 2\n"})
    primary = bundle("from m_" + dependency["id"] + " import invoke\ndef run(x): return invoke(x)\n",
                     dependencies=[dependency["id"]], parents=["b" * 64])
    parent = "def solve(): return sciel_method(" + repr(primary["id"]) + ", 3)\n"
    missing = materialize(parent, [primary])
    assert missing["reason"] == "missing_dependency" and "code" not in missing
    result = materialize(parent, [primary, dependency])
    assert result["status"] == "ok"
    assert set(payload_from_source(result["code"])) == {primary["id"], dependency["id"]}
    assert [c["id"] for c in result["receipt"]["components"]] == sorted([primary["id"], dependency["id"]])
    assert materialize(parent, [dependency, primary]) == result


def test_meaningful_constant_patch_is_immutable_bound_and_transactional():
    parent = "LIMIT = 2\ndef solve(): return LIMIT\n"
    method, patch = adapter(parent, [{"old": "LIMIT = 2", "new": "LIMIT = 4", "expected_count": 1}])
    result = materialize(parent, [method], [patch])
    assert result["status"] == "ok"
    assert "LIMIT = 4\ndef solve(): return LIMIT\n" in result["code"]
    proof = result["receipt"]["adapters"][0]
    assert proof["input_source_sha256"] == sha(parent)
    assert proof["output_source_sha256"] == sha(parent.replace("LIMIT = 2", "LIMIT = 4"))
    assert proof["edits"] == [dict(old_sha256=sha("LIMIT = 2"), new_sha256=sha("LIMIT = 4"), expected_count=1, applied_count=1)]
    changed = deepcopy(patch)
    changed["edits"][0]["new"] = "LIMIT = 8"
    refusal = materialize(parent, [method], [changed])
    assert refusal["reason"] == "adapter_protocol_mismatch" and "code" not in refusal
    assert parent.startswith("LIMIT = 2")


def test_sequential_adapters_bind_preceding_source_and_edits_rematch_after_each_change():
    parent = "X = 1\nY = 1\n"
    first, a = adapter(parent, [dict(old="X = 1", new="X = 22", expected_count=1),
                                dict(old="X = 22", new="X = 3", expected_count=1)])
    intermediate = "X = 3\nY = 1\n"
    second, b = adapter(intermediate, [dict(old="Y = 1", new="Y = 5", expected_count=1)])
    result = materialize(parent, [first, second], [a, b])
    assert result["status"] == "ok"
    assert result["receipt"]["adapted_parent_source_sha256"] == sha("X = 3\nY = 5\n")
    refused = materialize(parent, [first, second], [b, a])
    assert refused["reason"] == "adapter_parent_mismatch" and "code" not in refused


@pytest.mark.parametrize("edits,reason", [
    ([dict(old="X = 1", new="X = 2", expected_count=2)], "adapter_match_mismatch"),
    ([dict(old="X = 1", new="X = 2", expected_count=1), dict(old="missing", new="x", expected_count=1)], "adapter_match_mismatch"),
    ([dict(old="X = 1", new="X = (", expected_count=1)], "invalid_source"),
    ([dict(old="", new="x", expected_count=1)], "invalid_adapter"),
    ([dict(old="X = 1", new="X = 2", expected_count=True)], "invalid_adapter"),
])
def test_bad_patch_returns_no_partial_candidate(edits, reason):
    parent = "X = 1\n"
    method, patch = adapter(parent, edits)
    result = materialize(parent, [method], [patch])
    assert result["status"] == "refused" and result["reason"] == reason
    assert "code" not in result and "receipt" not in result


def test_registered_tool_cannot_be_relabelled_as_adapter():
    parent = "X = 1\n"
    ordinary = bundle(protocol={"patch": {"expected_parent_sha256": sha(parent), "edits": []}})
    result = materialize(parent, [ordinary], [dict(version_id=ordinary["id"], expected_parent_sha256=sha(parent), edits=[])])
    assert result["reason"] == "unregistered_adapter"


def test_hash_runtime_duplicate_and_reserved_names_fail_closed():
    first = bundle()
    tampered = deepcopy(first)
    tampered["files"]["tool.py"] += "# altered\n"
    assert materialize("X = 1\n", [tampered])["reason"] == "bundle_hash_mismatch"
    different_runtime = bundle(runtime_lock={"image_id": "sha256:" + "b" * 64})
    assert materialize("X = 1\n", [first, different_runtime])["reason"] == "runtime_mismatch"
    assert materialize("X = 1\n", [first, first])["reason"] == "duplicate_bundle"
    for parent in ("sciel_method = None\n", "def _sciel_bootstrap(): pass\n", "import json as sciel_method\n"):
        assert materialize(parent, [first])["reason"] == "reserved_name_collision"


def test_marker_is_a_real_top_level_comment_not_text_inside_strings():
    method = bundle()
    source = '"""# SCIEL_METHODS"""\nfrom __future__ import annotations\n# SCIEL_METHODS\nX = 1\n'
    result = materialize(source, [method])
    assert result["status"] == "ok"
    assert ast.get_docstring(ast.parse(result["code"])) == "# SCIEL_METHODS"
    assert result["receipt"]["insertions"][0]["replaced_marker"] is True
    for parent, reason in [
        ("# SCIEL_METHODS\n# SCIEL_METHODS\nX=1\n", "ambiguous_marker"),
        ("def f():\n    # SCIEL_METHODS\n    return 1\n", "invalid_marker"),
        ("# SCIEL_METHODS\nfrom __future__ import annotations\n", "invalid_marker"),
        ('"doc"; X=1\n', "ambiguous_insertion"),
    ]:
        assert materialize(parent, [method])["reason"] == reason


def test_empty_composition_does_not_rewrite_parent_and_future_errors_are_refused():
    parent = '"doc"\n# unchanged\ndef sciel_method(x): return x\n'
    result = materialize(parent, [])
    assert result["status"] == "ok" and result["code"] == parent
    assert result["receipt"]["insertions"] == []
    assert result["receipt"]["components"] == []
    assert materialize("X = 1\nfrom __future__ import annotations\n", [bundle()])["reason"] == "invalid_source"
