"""Static source and proof checks; never import or execute a packing candidate."""
import ast
import hashlib
import itertools
import json
from pathlib import Path

import pytest

from sciel.hypothesis import circle_interventions as edits


ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "src/sciel/solvers/circle_anytime.py"
POLICIES = ("initial", "incumbent", "alternating")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def function(source, name):
    return next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def test_baseline_is_exact_pinned_kernel_with_only_explicit_time_allocation():
    original = KERNEL.read_bytes()
    assert sha(original) == edits.PINNED_SOLVER_SHA256
    baseline = edits.baseline_source().encode()
    old = b"    return solve_packing()\n"
    new = b"    return solve_packing(budget_seconds=27.0, max_restarts=10000, maxiter=15000)\n"
    assert original.count(old) == baseline.count(new) == 1
    assert baseline.replace(new, old) == original
    assert edits.policy_of(baseline.decode()) == "initial"
    with pytest.raises(ValueError, match="outside"):
        edits.policy_of(original.decode())  # The weak 25-start default is not this baseline.
    assert KERNEL.read_bytes() == original


@pytest.mark.parametrize("policy", POLICIES)
def test_all_variants_preserve_full_source_except_exact_restart_reference(policy):
    baseline = edits.baseline_source()
    source = edits.source_for(policy)
    assert edits.policy_of(source) == policy
    original_assignment = "            guess = initial + np.random.uniform(-0.06, 0.06, initial.shape)\n"
    reference = {"initial": "initial", "incumbent": "incumbent._params",
                 "alternating": "(initial if _ % 2 == 0 else incumbent._params)"}[policy]
    changed_assignment = f"            guess = {reference} + np.random.uniform(-0.06, 0.06, initial.shape)\n"
    assert source.count(changed_assignment) == 1
    assert source.replace(changed_assignment, original_assignment) == baseline
    entry = function(source, "run_packing").body[-1].value
    assert isinstance(entry, ast.Call) and entry.func.id == "solve_packing"
    assert {k.arg: ast.literal_eval(k.value) for k in entry.keywords} == {
        "budget_seconds": 27.0, "max_restarts": 10000, "maxiter": 15000}


@pytest.mark.parametrize("policy", POLICIES)
def test_rng_arguments_clamps_solver_parameters_and_deadline_stay_identical(policy):
    baseline = function(edits.baseline_source(), "solve_packing")
    changed = function(edits.source_for(policy), "solve_packing")
    def guess_assignment(node):
        return next(n for n in ast.walk(node) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "guess" for t in n.targets))
    before, after = guess_assignment(baseline), guess_assignment(changed)
    assert isinstance(after.value, ast.BinOp) and isinstance(after.value.op, ast.Add)
    assert ast.dump(before.value.right) == ast.dump(after.value.right)
    call = after.value.right
    assert ast.unparse(call.func) == "np.random.uniform"
    assert [ast.unparse(a) for a in call.args] == ["-0.06", "0.06", "initial.shape"]
    assert not call.keywords
    # Removing only the approved base-expression delta gives the entire original
    # function AST, including RNG placement, clamps, optimizer options and checks.
    after.value.left = before.value.left
    assert ast.dump(changed) == ast.dump(baseline)


@pytest.mark.parametrize("parent,target", tuple(itertools.permutations(POLICIES, 2)))
def test_every_directed_edit_has_independently_reconstructable_byte_proof(parent, target):
    original = edits.source_for(parent)
    result = edits.apply(original, target)
    child, proof = result["code"], result["proof"]
    assert child == edits.source_for(target)
    assert json.loads(json.dumps(result)) == result
    assert proof["schema"] == "source-bound-edit/1"
    assert proof["operation"] == "circle_restart_reference/1"
    assert proof["contract"] == edits.CONTRACT
    assert proof["parent_policy"] == parent and proof["target_policy"] == target
    assert proof["kernel_source_sha256"] == sha(KERNEL.read_bytes())
    assert proof["baseline_source_sha256"] == sha(edits.baseline_source().encode())
    old, new = original.encode(), child.encode()
    assert proof["parent_source_sha256"] == sha(old)
    assert proof["child_source_sha256"] == sha(new)
    a, b = proof["before"]["byte_span"]
    c, d = proof["after"]["byte_span"]
    assert a == c
    for text, part in ((old[a:b], proof["before"]), (new[c:d], proof["after"])):
        assert text == part["text"].encode() and sha(text) == part["sha256"]
    assert old[:a] == new[:c] and old[b:] == new[d:]
    assert proof["unchanged_prefix"] == {"bytes": a, "sha256": sha(old[:a])}
    assert proof["unchanged_suffix"] == {"bytes": len(old[b:]), "sha256": sha(old[b:])}
    assert old[:a] + proof["after"]["text"].encode() + old[b:] == new
    assert proof["unchanged_outside_assignment"] is True
    assert edits.apply(child, parent)["code"] == original
    assert original == edits.source_for(parent)


def test_plan_key_binds_parent_operation_and_target_for_directed_comparisons():
    first = edits.apply(edits.source_for("initial"), "alternating")["proof"]
    second = edits.apply(edits.source_for("incumbent"), "alternating")["proof"]
    assert first["plan_key"] != second["plan_key"]
    assert first["parent_source_sha256"] != second["parent_source_sha256"]
    assert first["child_source_sha256"] == second["child_source_sha256"]
    assert first["plan_key"] == edits.apply(edits.source_for("initial"), "alternating")["proof"]["plan_key"]
    assert first["plan_key"] == f"{edits.OPERATION}:{first['parent_source_sha256']}:alternating"
    assert first["plan_key"] != edits.apply(edits.source_for("initial"), "incumbent")["proof"]["plan_key"]


@pytest.mark.parametrize("policy", POLICIES)
def test_same_policy_is_rejected_instead_of_charging_for_noop(policy):
    with pytest.raises(ValueError, match="must change"):
        edits.apply(edits.source_for(policy), policy)


@pytest.mark.parametrize("policy", [None, True, 1, {}, [], "", "INITIAL", " initial", "other"])
def test_unknown_policy_is_rejected(policy):
    for invoke in (edits.source_for, edits.policy_descriptor,
                   lambda p: edits.apply(edits.baseline_source(), p)):
        with pytest.raises(ValueError, match="unknown"):
            invoke(policy)


@pytest.mark.parametrize("change", [
    lambda s: s + "\n", lambda s: s.replace("ftol", "FTOL"),
    lambda s: s.replace("27.0", "26.0"), lambda s: s.replace("0.06", "0.07"),
    lambda s: s.replace("\n", "\r\n"),
    lambda s: s + "\ndef run_packing():\n    return None\n",
])
def test_unknown_source_never_passes_by_matching_assignment_or_comment(change):
    foreign = change(edits.baseline_source())
    for invoke in (edits.policy_of, lambda s: edits.apply(s, "incumbent")):
        with pytest.raises(ValueError, match="outside"):
            invoke(foreign)


@pytest.mark.parametrize("value", [None, 1, True, {}, b"source"])
def test_source_must_be_exact_string(value):
    with pytest.raises(ValueError, match="string"):
        edits.policy_of(value)
    with pytest.raises(ValueError, match="string"):
        edits.apply(value, "incumbent")


def test_policy_descriptors_are_immutable_and_prompt_copies_are_detached():
    with pytest.raises(TypeError):
        edits.POLICIES["initial"] = {}
    with pytest.raises(TypeError):
        edits.POLICIES["initial"]["change"] = "changed"
    detached = edits.policy_descriptor("initial")
    assert json.loads(json.dumps(detached)) == detached
    detached["change"] = "changed"
    assert edits.policy_descriptor("initial")["change"] != "changed"
    assert all("uncertain_effect" in edits.policy_descriptor(p) for p in POLICIES)


def test_kernel_tamper_fails_closed_even_after_successful_previous_read(tmp_path, monkeypatch):
    path = tmp_path / "circle_anytime.py"
    original = KERNEL.read_bytes()
    path.write_bytes(original)
    monkeypatch.setattr(edits, "_SOLVER_PATH", path)
    parent = edits.baseline_source()
    path.write_bytes(original + b"\n")
    for invoke in (edits.baseline_source, lambda: edits.source_for("initial"),
                   lambda: edits.policy_of(parent), lambda: edits.apply(parent, "incumbent")):
        with pytest.raises(ValueError, match="pinned"):
            invoke()


def test_builder_does_not_import_solver_or_numeric_runtime():
    # Static proof of the module's direct dependency boundary; AST parsing is
    # sufficient and avoids a candidate or optimizer import as a test side effect.
    tree = ast.parse(Path(edits.__file__).read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module)
    assert modules == ["__future__", "hashlib", "pathlib", "types"]
