"""Canonical cascade packaging and per-lab replay-safe source history."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path

import pytest

from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from sciel.methods.budget import BudgetUnknown
from sciel.methods.engine_bridge import EngineCascadeBridge, LabSolutionHistory
from sciel.methods.models import content_hash, validate_manifest, validate_value
from sciel.methods.registry import Registry


RUNTIME = {'image_id': 'sha256:' + 'a' * 64}
LIMITS = dict(wall_seconds=10, cpu_seconds=5, memory_bytes=268435456,
              pids=16, max_output_bytes=131072, max_stderr_bytes=4096)
CODE = 'def solve():\n    return 3\n'


def sha(code):
    return hashlib.sha256(code.encode()).hexdigest()


def bridge(tmp_path):
    root = tmp_path / 'task'
    root.mkdir(exist_ok=True)
    (root / 'task.yaml').write_text('suite: fixture\nname: scalar\norigin: math\nprimary_metric: score\nstochastic: true\nconfirm_n: 3\n')
    (root / 'evaluator.py').write_text('raise RuntimeError("do not import this on the host")\n')
    (root / 'initial_program.py').write_text(CODE)
    return EngineCascadeBridge(load_task(str(root)))


def output(adapter, code=CODE, *, ok=True, confirmed=False):
    result = EvalResult(ok=ok, score=3. if ok else 0., cases={'n_trials': 3.} if confirmed else {},
                        confirmed_sampled=confirmed)
    return {'schema': 'sciel-method-cascade-result/1', 'source_sha256': sha(code),
            'task_identity_sha256': content_hash(adapter.identity),
            'outcome': asdict(CascadeOutcome(result, 3., 'confirm' if confirmed else 'full', confirmed))}


def setup(tmp_path):
    adapter = bridge(tmp_path)
    registry = Registry(tmp_path / 'registry')
    lab = registry.create_lab('lab', 'question', 'account', 'B',
        dict(search_procedure='b' * 64, runtime_lock=RUNTIME))
    return adapter, registry, lab


def plan(registry, adapter, revision, *, code=CODE, lab_id='lab', case='a'):
    bundle = adapter.solution_bundle(code, lab_id, RUNTIME, LIMITS)
    ident = registry.register(bundle['files'], bundle['manifest'])
    plan_id = registry.put_artifact('experiment_plan', dict(lab_id=lab_id, lab_revision_id=revision,
        purpose='solution', treatment=[ident], control=[], inputs=[{}], case=case,
        measurement_protocols={'treatment': adapter.identity}))
    return plan_id, ident


def receipt(registry, adapter, plan_id, method_id, *, code=CODE, ok=True, status='ok'):
    return registry.put_artifact('call_receipt', {'status': 'completed' if status == 'ok' else status,
        'result': {'status': status, 'method_id': method_id, 'plan_id': plan_id, 'input_sha256':content_hash({}),
                   'output': output(adapter, code, ok=ok) if status == 'ok' else None}})


def test_bundle_reuses_actual_cascade_and_task_bytes_without_host_import(tmp_path):
    adapter = bridge(tmp_path)
    bundle = adapter.solution_bundle(CODE, 'lab', RUNTIME, LIMITS)
    validate_manifest(bundle['manifest'], bundle['files'])
    root = Path(__file__).resolve().parents[2] / 'src' / 'sciel' / 'eval'
    for name in ('cascade.py', 'facade.py', 'sandbox.py', '_runner.py', 'native_geometry.py', 'packing_geometry.py'):
        assert bundle['files']['sciel/eval/' + name].encode() == (root / name).read_bytes()
    assert bundle['files']['candidate.py'] == CODE
    assert bundle['files']['_task/evaluator.py'].startswith('raise RuntimeError')
    assert 'cascade_evaluate(' in bundle['files']['adapter.py']
    assert bundle['manifest']['protocol']['evaluation_protocol'] == 'cascade'
    with pytest.raises(ValueError):
        validate_value({'seed': 42}, adapter.input_schema)


def test_extract_real_dataclasses_and_program_exact_source_binding(tmp_path):
    adapter = bridge(tmp_path)
    observed = output(adapter)
    cascade = adapter.extract(observed, expected_source_sha256=sha(CODE))
    assert isinstance(cascade, CascadeOutcome) and isinstance(cascade.result, EvalResult)
    program = adapter.program(CODE, observed, 'p1', island=3, parent_id='p0', gen=4)
    assert program.code == CODE and program.score == 3 and program.parent_id == 'p0'
    assert program.island == 3 and program.gen == 4
    with pytest.raises(ValueError, match='source binding'):
        adapter.program(CODE + '# different', observed, 'p2')
    with pytest.raises(ValueError, match='invalid cascade'):
        adapter.program(CODE, output(adapter, ok=False), 'p2')


def test_confirmation_remains_canonical_and_not_a_fabricated_replication(tmp_path):
    adapter = bridge(tmp_path)
    observed = output(adapter, confirmed=True)
    assert adapter.measure({'record_candidate': True}, observed)['observables']['confirmed'] is True
    with pytest.raises(ValueError, match='not requested'):
        adapter.measure({}, observed)
    observed['outcome']['result']['cases']['n_trials'] = 1
    with pytest.raises(ValueError, match='confirmation flags'):
        adapter.extract(observed)


def test_task_spec_is_frozen_and_binary_assets_refuse_before_execution(tmp_path):
    adapter = bridge(tmp_path)
    spec = load_task(adapter.task.root)
    frozen = EngineCascadeBridge(spec)
    before = frozen.solution_bundle(CODE, 'lab', RUNTIME, LIMITS)
    spec.smoke_timeout = 999
    assert frozen.solution_bundle(CODE, 'lab', RUNTIME, LIMITS) == before
    assets = Path(spec.root) / 'assets'
    assets.mkdir()
    (assets / 'binary.dat').write_bytes(b'\xff')
    with pytest.raises(ValueError, match='binary task assets'):
        EngineCascadeBridge(load_task(spec.root))


def test_solution_history_rejects_ast_duplicate_after_checkpoint_resume(tmp_path):
    adapter, registry, lab = setup(tmp_path)
    history = LabSolutionHistory(registry, 'lab', adapter)
    plan_id, method = plan(registry, adapter, lab['active_revision'])
    assert history.reserve(CODE, 'e1', lab['active_revision'], plan_id=plan_id)['dispatch_allowed']
    history.finalize('e1', receipt(registry, adapter, plan_id, method), program_id='p1')
    registry.checkpoint(tmp_path / 'checkpoint.json')
    resumed_registry = Registry(registry.root)
    resumed_registry.verify_checkpoint(tmp_path / 'checkpoint.json')
    resumed = LabSolutionHistory(resumed_registry, 'lab', EngineCascadeBridge(load_task(adapter.task.root)))
    decision = resumed.inspect(CODE + '# comment-only edit\n')
    assert decision['accept'] is False and decision['match_kind'] == 'ast'
    assert decision['matched_program_id'] == 'p1'
    same = resumed.reserve(CODE, 'e1', lab['active_revision'], plan_id=plan_id)
    assert same['accept'] and same['replayed'] and same['finalized'] and not same['dispatch_allowed']


def test_pending_keeps_original_revision_and_failed_attempt_allows_remeasurement(tmp_path):
    adapter, registry, lab = setup(tmp_path)
    history = LabSolutionHistory(registry, 'lab', adapter)
    old_revision = lab['active_revision']
    plan_id, method = plan(registry, adapter, old_revision)
    history.reserve(CODE, 'e1', old_revision, plan_id=plan_id)
    new_revision = registry.revise_lab('lab', old_revision, {'generator': {'version': 2}}, 'next practice')
    assert history.state()['reservations']['e1']['lab_revision_id'] == old_revision
    assert not history.inspect(CODE)['accept']
    unknown = receipt(registry, adapter, plan_id, method, status='unknown')
    with pytest.raises(BudgetUnknown):
        history.finalize('e1', unknown)
    history.finalize('e1', receipt(registry, adapter, plan_id, method, ok=False))
    candidate = history.inspect(CODE)
    assert candidate['accept'] and candidate['remeasurement']
    new_plan, _ = plan(registry, adapter, new_revision, case='retry')
    retry = history.reserve(CODE, 'e2', new_revision, plan_id=new_plan)
    assert retry['accept'] and retry['remeasurement']


def test_receipt_cannot_bind_another_source_or_plan(tmp_path):
    adapter, registry, lab = setup(tmp_path)
    history = LabSolutionHistory(registry, 'lab', adapter)
    plan_id, method = plan(registry, adapter, lab['active_revision'])
    history.reserve(CODE, 'e1', lab['active_revision'], plan_id=plan_id)
    with pytest.raises(ValueError, match='source binding'):
        history.finalize('e1', receipt(registry, adapter, plan_id, method, code=CODE + '# other'))
    with pytest.raises(ValueError, match='another plan'):
        history.finalize('e1', receipt(registry, adapter, 'f' * 64, method))
    assert history.state()['terminal'] == {}


def test_same_code_isolated_between_labs_and_concurrent_same_lab_claims_are_exclusive(tmp_path):
    adapter, registry, lab = setup(tmp_path)
    recipient = registry.create_lab('recipient', 'independent', 'other-money', 'C',
        dict(search_procedure='b' * 64, runtime_lock=RUNTIME))
    first_plan, _ = plan(registry, adapter, lab['active_revision'])
    second_plan, _ = plan(registry, adapter, lab['active_revision'], case='b')
    def reserve(args):
        ident, plan_id = args
        return LabSolutionHistory(registry, 'lab', adapter).reserve(CODE, ident, lab['active_revision'], plan_id=plan_id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = list(pool.map(reserve, [('e1', first_plan), ('e2', second_plan)]))
    assert sum(bool(item['accept']) for item in decisions) == 1
    other_plan, _ = plan(registry, adapter, recipient['active_revision'], lab_id='recipient')
    assert LabSolutionHistory(registry, 'recipient', adapter).reserve(
        CODE, 'e1', recipient['active_revision'], plan_id=other_plan)['accept']


def test_untrusted_replacement_wrapper_is_not_accepted_as_canonical_cascade(tmp_path):
    adapter, registry, lab = setup(tmp_path)
    bundle = adapter.solution_bundle(CODE, 'lab', RUNTIME, LIMITS)
    bundle['files']['adapter.py'] = 'def run(inputs): return {"pretend": 999}\n'
    method = registry.register(bundle['files'], bundle['manifest'])
    plan_id = registry.put_artifact('experiment_plan', dict(lab_id='lab', lab_revision_id=lab['active_revision'], treatment=[method]))
    with pytest.raises(ValueError, match='exact trusted cascade'):
        LabSolutionHistory(registry, 'lab', adapter).reserve(CODE, 'e1', lab['active_revision'], plan_id=plan_id)


def test_native_measurement_signature_uses_validated_output_not_source_or_invocation_identity():
    import json
    from sciel.methods.tasks import Circle26Task
    from sciel.eval.native_geometry import SCHEMA
    from sciel.engine.repeat_support import EVALUATOR_SHA
    task=Circle26Task()
    adapter=EngineCascadeBridge(task.task)
    numeric={'centers':[[0.,0.] for _ in range(26)],'radii':[0.]*26}
    geometry=dict(schema=SCHEMA,source_sha256=sha(CODE),evaluator_sha256=EVALUATOR_SHA,
        invocation_id='a'*32,stage='full',confirmation=None,numeric=numeric,numeric_sha256=content_hash(numeric),
        centers_dtype='float64',radii_dtype='float64',native_validity=True,native_sum_radii=0.,native_score=0.)
    def observed(g,code=CODE):
        result=EvalResult(ok=True,score=0.,cases={'sum_radii':0.,'validity':1.,'target_ratio':0.,'combined_score':0.},
            artifacts={'native_geometry':json.dumps(g,sort_keys=True,separators=(',',':'))})
        return dict(schema='sciel-method-cascade-result/1',source_sha256=sha(code),
            task_identity_sha256=content_hash(adapter.identity),outcome=asdict(CascadeOutcome(result,0.,'full')))
    first=adapter.measure({},observed(geometry))
    exported=adapter.transfer_payload({},observed(geometry))
    assert set(exported)=={'centers','radii','centers_dtype','radii_dtype'}
    assert exported['centers']==numeric['centers'] and 'source_sha256' not in exported
    exported['radii'][0]=999
    assert geometry['numeric']['radii'][0]==0.
    changed=deepcopy(geometry)
    changed.update(source_sha256=sha(CODE+'# other'),invocation_id='b'*32)
    assert adapter.measure({},observed(changed,CODE+'# other'))==first
    changed['numeric']['centers'][0][0]=.1
    changed['numeric_sha256']=content_hash(changed['numeric'])
    assert adapter.measure({},observed(changed,CODE+'# other'))!=first
    with pytest.raises(ValueError,match='source differs'):
        adapter.measure({},observed(changed))


def test_unknown_task_cannot_export_algorithm_as_a_ready_solution(tmp_path):
    adapter=bridge(tmp_path)
    with pytest.raises(ValueError,match='no trusted ready-output'):
        adapter.transfer_payload({},output(adapter))
