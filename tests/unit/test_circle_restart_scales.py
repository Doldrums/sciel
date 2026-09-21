"""Pure source/AST/proof fixtures; never import or execute a packing solver."""
import ast
import hashlib
import itertools
import json
from pathlib import Path

import pytest

from sciel.hypothesis import circle_interventions as references
from sciel.hypothesis import circle_restart_scales as scales

POLICIES = ('0.015', '0.03', '0.06', '0.12')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def assignment(tree):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'guess' for t in n.targets))


def test_baseline_is_exact_historical_incumbent_with_strong_time_allocation():
    before = Path(references.__file__).read_bytes()
    baseline = scales.baseline_source()
    assert sha(baseline.encode()) == '78915475a194be9b871f1fde96eafec12561c87c72bef1aa18370154f770e06f'
    assert baseline == references.source_for('incumbent') == scales.source_for('0.06')
    assert scales.policy_of(baseline) == '0.06'
    assert 'return solve_packing(budget_seconds=27.0, max_restarts=10000, maxiter=15000)' in baseline
    assert Path(references.__file__).read_bytes() == before
    for policy in ('initial', 'alternating'):
        with pytest.raises(ValueError, match='outside'):
            scales.policy_of(references.source_for(policy))


@pytest.mark.parametrize('policy', POLICIES)
def test_only_two_uniform_bounds_change_in_entire_source_and_ast(policy):
    baseline, child = scales.baseline_source(), scales.source_for(policy)
    old = 'np.random.uniform(-0.06, 0.06, initial.shape)'
    new = f'np.random.uniform(-{policy}, {policy}, initial.shape)'
    assert child.count(new) == 1 and child.replace(new, old) == baseline
    assert scales.policy_of(child) == policy
    a, b = ast.parse(baseline), ast.parse(child)
    original, changed = assignment(a), assignment(b)
    assert ast.dump(original.value.left) == ast.dump(changed.value.left)
    call = changed.value.right
    assert ast.unparse(call.func) == 'np.random.uniform' and not call.keywords
    assert [ast.literal_eval(v) for v in call.args[:2]] == [-float(policy), float(policy)]
    assert ast.dump(call.args[2]) == ast.dump(original.value.right.args[2])
    # Restore only the two bounds; all AST nodes including RNG placement, clamps,
    # maxiter/ftol, feasibility/deadline and incumbent retention must be identical.
    call.args[:2] = original.value.right.args[:2]
    assert ast.dump(a) == ast.dump(b)


@pytest.mark.parametrize('parent,target', tuple(itertools.permutations(POLICIES, 2)))
def test_all_twelve_directed_proofs_reconstruct_exact_bytes_and_roundtrip(parent, target):
    source = scales.source_for(parent)
    result = scales.apply(source, target)
    child, proof = result['code'], result['proof']
    assert child == scales.source_for(target)
    assert json.loads(json.dumps(result)) == result
    old, new = source.encode(), child.encode()
    start, end = proof['before']['byte_span']
    new_start, new_end = proof['after']['byte_span']
    assert start == new_start
    assert old[:start] == new[:new_start] and old[end:] == new[new_end:]
    assert old[:start] + proof['after']['text'].encode() + old[end:] == new
    for raw, part in ((old[start:end], proof['before']), (new[new_start:new_end], proof['after'])):
        assert raw == part['text'].encode() and sha(raw) == part['sha256']
    assert proof['unchanged_prefix'] == {'bytes': start, 'sha256': sha(old[:start])}
    assert proof['unchanged_suffix'] == {'bytes': len(old[end:]), 'sha256': sha(old[end:])}
    assert proof['parent_source_sha256'] == sha(old) and proof['child_source_sha256'] == sha(new)
    assert proof['parent_policy'] == parent and proof['target_policy'] == target
    assert proof['baseline_source_sha256'] == scales.PINNED_BASELINE_SHA256
    assert proof['kernel_source_sha256'] == scales.PINNED_SOLVER_SHA256
    assert proof['operation'] == 'circle_restart_scale/1'
    assert proof['contract'] == 'sciel-circle-restart-scales/1'
    assert proof['schema'] == 'source-bound-edit/1'
    assert proof['plan_key'] == f"circle_restart_scale/1:{sha(old)}:{target}"
    assert scales.apply(child, parent)['code'] == source
    assert set(proof) == set(references.apply(references.source_for('initial'), 'incumbent')['proof'])


def test_plan_keys_are_twelve_distinct_parent_bound_comparisons():
    results = [scales.apply(scales.source_for(p), t)['proof'] for p, t in itertools.permutations(POLICIES, 2)]
    assert len({p['plan_key'] for p in results}) == 12
    assert len({p['child_source_sha256'] for p in results}) == 4
    assert {p['target_policy'] for p in results} == set(POLICIES)


@pytest.mark.parametrize('policy', POLICIES)
def test_noop_rejected(policy):
    with pytest.raises(ValueError, match='must change'):
        scales.apply(scales.source_for(policy), policy)


@pytest.mark.parametrize('policy', [None, False, .06, 6, [], {}, '', '0.060', '.06', '6e-2', ' 0.06', 'incumbent'])
def test_only_canonical_policy_strings_accepted(policy):
    for call in (scales.source_for, scales.policy_descriptor,
                 lambda p: scales.apply(scales.baseline_source(), p)):
        with pytest.raises(ValueError, match='unknown'):
            call(policy)


@pytest.mark.parametrize('alter', [
    lambda s: s + '\n', lambda s: s.replace('\n', '\r\n'),
    lambda s: s.replace('27.0', '26.0'), lambda s: s.replace('ftol', 'FTOL'),
    lambda s: s.replace('incumbent._params +', 'initial +'),
    lambda s: s.replace('-0.06, 0.06', '-0.03, 0.06'),
    lambda s: s.replace('-0.06, 0.06', '-.06, .06'),
    lambda s: s.replace('initial.shape)', '(78,))'),
    lambda s: s + '\ndef run_packing(): return None\n',
])
def test_similar_or_valid_python_outside_exact_family_is_refused(alter):
    source = alter(scales.baseline_source())
    for call in (scales.policy_of, lambda s: scales.apply(s, '0.03')):
        with pytest.raises(ValueError, match='outside'):
            call(source)


@pytest.mark.parametrize('source', [None, True, 1, {}, b'code'])
def test_source_must_be_a_full_exact_string(source):
    for call in (scales.policy_of, lambda s: scales.apply(s, '0.03')):
        with pytest.raises(ValueError, match='string'):
            call(source)


def test_pinned_baseline_is_checked_on_every_call(monkeypatch):
    baseline = scales.baseline_source()
    monkeypatch.setattr(scales.references, 'source_for', lambda p: baseline + '\n')
    for call in (scales.baseline_source, lambda: scales.source_for('0.03'),
                 lambda: scales.policy_of(baseline), lambda: scales.apply(baseline, '0.03')):
        with pytest.raises(ValueError, match='pinned'):
            call()


def test_descriptors_are_immutable_factual_and_detached():
    assert tuple(scales.POLICIES) == POLICIES
    with pytest.raises(TypeError):
        scales.POLICIES['0.06'] = {}
    with pytest.raises(TypeError):
        scales.POLICIES['0.06']['change'] = 'changed'
    value = scales.policy_descriptor('0.03')
    assert value['policy'] == '0.03' and 'centers and radii' in value['change']
    assert value['uncertain_effect']
    value['change'] = 'changed'
    assert scales.policy_descriptor('0.03')['change'] != 'changed'


def test_direct_import_boundary_contains_no_solver_or_numeric_runtime():
    tree = ast.parse(Path(scales.__file__).read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.extend([node.module] if node.module else [a.name for a in node.names])
    assert imports == ['__future__', 'hashlib', 'types', 'circle_interventions']
