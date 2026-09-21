"""Reuse the Engine's canonical cascade inside the external method executor.

No live Engine, task files, or run checkpoints are changed. Submitted source is
only copied into an immutable bundle; it is never imported on the host.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import fcntl
import hashlib
import os
from pathlib import Path

from ..engine.novelty import CandidateNoveltyRegistry
from ..engine.types import Program
from ..eval.cascade import CascadeOutcome
from ..eval.facade import EvalResult
from ..eval.registry import TaskSpec, load_task
from .budget import BudgetUnknown
from .models import canonical_bytes, content_hash, detached, safe_path, validate_manifest, validate_value
from .tasks import TaskAdapter, _manifest


def _sha(code):
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


class EngineCascadeBridge(TaskAdapter):
    """TaskSpec → actual smoke/full/confirm code → CascadeOutcome/Program.

    Canonical stage semantics are retained, including its confirmation gate.
    The outer Docker CPU/memory/wall contract additionally caps the entire call.
    The existing cascade has no assigned RNG-seed interface; do not label its
    stage repetitions as paired seeded draws.
    """
    def __init__(self, task):
        if not isinstance(task, TaskSpec) or load_task(task.root).task_hash != task.task_hash:
            raise ValueError('bridge requires a current content-bound TaskSpec')
        self.task = deepcopy(task)
        self.metric = task.primary_metric
        self._input_schema = {'type': 'object', 'properties': {'record_candidate': {'type': 'boolean'}},
                              'additionalProperties': False}
        result = {'type': 'object', 'properties': {
            'ok': {'type': 'boolean'}, 'score': {'type': 'number'},
            'cases': {'type': 'object', 'additionalProperties': {'type': 'number'}},
            'stderr': {'type': 'string'}, 'timing': {'type': 'number', 'minimum': 0},
            'cost': {'type': 'number', 'minimum': 0},
            'artifacts': {'type': 'object', 'additionalProperties': {'type': 'string'}},
            'confirmed_sampled': {'type': 'boolean'}},
            'required': ['ok', 'score', 'cases', 'stderr', 'timing', 'cost', 'artifacts', 'confirmed_sampled'],
            'additionalProperties': False}
        outcome = {'type': 'object', 'properties': {'result': result,
            'raw_first_score': {'type': 'number'}, 'reached': {'type': 'string', 'enum': ['smoke', 'full', 'confirm']},
            'confirmed': {'type': 'boolean'}},
            'required': ['result', 'raw_first_score', 'reached', 'confirmed'], 'additionalProperties': False}
        self._output_schema = {'type': 'object', 'properties': {
            'schema': {'type': 'string', 'enum': ['sciel-method-cascade-result/1']},
            'source_sha256': {'type': 'string', 'minLength': 64, 'maxLength': 64},
            'task_identity_sha256': {'type': 'string', 'minLength': 64, 'maxLength': 64},
            'outcome': outcome},
            'required': ['schema', 'source_sha256', 'task_identity_sha256', 'outcome'], 'additionalProperties': False}
        self._files = {'sciel/__init__.py': '', 'sciel/eval/__init__.py': ''}
        source_root = Path(__file__).resolve().parents[1] / 'eval'
        for name in ('cascade.py', 'facade.py', 'sandbox.py', '_runner.py', 'native_geometry.py', 'packing_geometry.py'):
            self._files['sciel/eval/' + name] = (source_root / name).read_bytes().decode('utf-8')
        # _runner and task evaluators load spawn-importable candidates through
        # this package. The bundle must not depend on an ambient SCiEL install.
        self._files['sciel/candidate_modules/__init__.py'] = (
            source_root.parent / 'candidate_modules' / '__init__.py').read_bytes().decode('utf-8')
        root = Path(task.root)
        for name in ('task.yaml', 'evaluator.py', 'initial_program.py'):
            self._files['_task/' + name] = (root / name).read_bytes().decode('utf-8')
        assets = root / 'assets'
        if assets.exists():
            for path in sorted(assets.rglob('*')):
                if '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
                    continue
                if path.is_symlink():
                    raise ValueError('bridge assets cannot be symlinks')
                if path.is_file():
                    relative = path.relative_to(root).as_posix()
                    safe_path(relative)
                    try:
                        self._files['_task/' + relative] = path.read_bytes().decode('utf-8')
                    except UnicodeError as exc:
                        raise ValueError('text method bundles do not yet support binary task assets') from exc
        spec = asdict(task)
        spec.pop('root')
        if load_task(task.root).task_hash != task.task_hash:
            raise ValueError('task changed while packaging the canonical cascade')
        self._identity = {'schema': 'sciel-engine-cascade-bridge/1', 'task_hash': task.task_hash,
            'evaluator_sha256': _sha(self._files['_task/evaluator.py']), 'task_spec': spec,
            'cascade_source_sha256': _sha(self._files['sciel/eval/cascade.py']),
            'runtime_sources_sha256': content_hash(self._files), 'adapter_sha256': _sha(Path(__file__).read_text()),
            'input_schema': self.input_schema, 'output_schema': self.output_schema,
            'metric': self.metric, 'direction': self.direction,
            'scope': 'canonical cascade in isolated container; no assigned-seed or independent-replication claim'}

    def solution_bundle(self, code, author_lab, runtime_lock, resources):
        # Dataclass fields are JSON data, never interpolated executable source.
        spec = asdict(self.task)
        spec.pop('root')
        files = {**self._files, 'candidate.py': code, 'task_spec.json': canonical_bytes(spec).decode()}
        identity_sha = content_hash(self.identity)
        wrapper = '''import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
from sciel.eval.cascade import cascade_evaluate

def run(inputs):
    root = Path(__file__).resolve().parent
    spec = json.loads((root / "task_spec.json").read_text())
    spec["root"] = str(root / "_task")
    spec["evaluator_path"] = str(root / "_task" / "evaluator.py")
    # The unchanged subprocess runner imports its normal package from this exact
    # immutable bundle. It receives no host filesystem or provider environment.
    os.environ["PYTHONPATH"] = str(root)
    path = root / "candidate.py"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    outcome = cascade_evaluate(SimpleNamespace(**spec), str(path),
                               record_candidate=inputs.get("record_candidate", False))
    if hashlib.sha256(path.read_bytes()).hexdigest() != before:
        raise ValueError("candidate source changed during cascade")
    return {"schema": "sciel-method-cascade-result/1", "source_sha256": before,
            "task_identity_sha256": IDENTITY, "outcome": asdict(outcome)}
'''.replace('IDENTITY', repr(identity_sha))
        files['adapter.py'] = wrapper
        manifest = _manifest('canonical_cascade_solution', author_lab, self.input_schema, self.output_schema,
                             runtime_lock, resources, self.identity)
        manifest['protocol'].update(evaluation_protocol='cascade', source_sha256=_sha(code))
        validate_manifest(manifest, files)
        return {'files': files, 'manifest': manifest}

    def extract(self, output, *, expected_source_sha256=None):
        validate_value(output, self.output_schema)
        if output['task_identity_sha256'] != content_hash(self.identity):
            raise ValueError('cascade output uses another frozen task identity')
        if expected_source_sha256 is not None and output['source_sha256'] != expected_source_sha256:
            raise ValueError('cascade source binding differs')
        value = output['outcome']
        if value['confirmed'] and (value['reached'] != 'confirm' or not value['result']['confirmed_sampled']
                or not value['result']['ok'] or not self.task.stochastic
                or value['result']['cases'].get('n_trials', 0) < max(2, self.task.confirm_n)
                or not float(value['result']['cases'].get('n_trials', 0)).is_integer()):
            raise ValueError('cascade confirmation flags contradict its result')
        return CascadeOutcome(result=EvalResult.from_json(value['result']), raw_first_score=value['raw_first_score'],
                              reached=value['reached'], confirmed=value['confirmed'])

    def _valid(self, outcome):
        if not outcome.result.ok:
            return False
        metric = self.task.research_metric or {}
        if metric.get('kind') == 'validity_gated_ratio':
            return outcome.result.cases.get(metric['validity_case']) == metric['valid_value']
        return True

    def measure(self, input_value, output_value):
        validate_value(input_value, self.input_schema)
        outcome = self.extract(output_value)
        if outcome.confirmed and not input_value.get('record_candidate', False):
            raise ValueError('confirmation was not requested by the frozen input')
        observables={'reached':outcome.reached,'confirmed':outcome.confirmed,
                     'raw_first_score':outcome.raw_first_score,'evaluation_ok':outcome.result.ok}
        if 'native_geometry' in outcome.result.artifacts:
            from ..eval.native_geometry import parse_artifact, validate_binding
            from ..hypothesis.measurement import metric_context
            geometry=parse_artifact(outcome.result.artifacts['native_geometry'])
            validate_binding(geometry,dict(ok=outcome.result.ok,score=outcome.result.score,
                cases=outcome.result.cases,reached=outcome.reached,confirmed=outcome.confirmed),
                source_sha256=output_value['source_sha256'],metric=metric_context(self.task))
            observables['native_geometry']={key:geometry[key] for key in
                ('numeric_sha256','centers_dtype','radii_dtype','evaluator_sha256','native_validity')}
        # Source identity belongs to the immutable receipt, not an output
        # novelty signature. Invocation UUIDs and timing are not observations.
        return {'valid':self._valid(outcome),
                'metrics':{**outcome.result.cases,self.metric:outcome.result.score},
                'observables':observables}

    def transfer_payload(self, input_value, output_value):
        if self.task.qualified_name!='oe/circle_packing':
            return super().transfer_payload(input_value,output_value)
        measured=self.measure(input_value,output_value)
        outcome=self.extract(output_value)
        if not measured['valid'] or 'native_geometry' not in measured['observables']:
            raise ValueError('ready packing transfer requires valid source-bound native geometry')
        from ..eval.native_geometry import parse_artifact
        geometry=parse_artifact(outcome.result.artifacts['native_geometry'])
        return {'centers':detached(geometry['numeric']['centers']),
                'radii':detached(geometry['numeric']['radii']),
                'centers_dtype':geometry['centers_dtype'],'radii_dtype':geometry['radii_dtype']}

    def program(self, code, output, program_id, *, island=0, parent_id=None, gen=0):
        outcome = self.extract(output, expected_source_sha256=_sha(code))
        if not self._valid(outcome):
            raise ValueError('invalid cascade outcome cannot become a measured solution Program')
        return Program(id=program_id, code=code, score=outcome.result.score,
            cases=detached(outcome.result.cases), island=island, parent_id=parent_id,
            gen=gen, origin='research_solution', sig_hash=_sha(code))


class LabSolutionHistory:
    """All admitted and failed solution sources per lab, independent of storage.

    Registry journal/checkpoint retention preserves this even if a Program is
    evicted. A replayed reservation permits receipt lookup only, never another
    dispatch; DispatchStore must own the actual invocation.
    """
    def __init__(self, registry, lab_id, bridge):
        registry.lab(lab_id)
        self.registry, self.lab_id, self.bridge = registry, lab_id, bridge

    @contextmanager
    def _locked(self):
        path = self.registry.root / ('.solution-' + content_hash(self.lab_id) + '.lock')
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _load(self):
        novelty, reservations, terminal = CandidateNoveltyRegistry(), {}, {}
        identity = content_hash(self.bridge.identity)
        for event in self.registry.events():
            data = event['data']
            if not event['kind'].startswith('solution_novelty_') or data.get('lab_id') != self.lab_id:
                continue
            if event['kind'] == 'solution_novelty_reserved':
                source = self.registry.get_artifact('solution_candidate', data['source_id'])
                if source['bridge_identity_sha256'] != identity:
                    raise ValueError('lab solution history belongs to another task/bridge identity')
                code, ident = source['code'], source['evaluation_id']
                if ident in reservations or source['lab_id'] != self.lab_id or _sha(code) != source['source_sha256']:
                    raise ValueError('invalid solution reservation binding')
                novelty.register(code, ident, evaluation_ok=None)
                reservations[ident] = source
            elif event['kind'] == 'solution_novelty_finalized':
                ident = data['evaluation_id']
                source = reservations[ident]
                if ident in terminal:
                    raise ValueError('solution evaluation finalized twice')
                observed = self._receipt_outcome(source, data['call_receipt_id'], data.get('program_id'))
                evaluation=observed.result if isinstance(observed,CascadeOutcome) else observed
                if data['evaluation_ok'] != bool(evaluation and evaluation.ok):
                    raise ValueError('solution finalization differs from its actual receipt')
                novelty.register(source['code'], ident, evaluation_ok=data['evaluation_ok'])
                if data.get('program_id'):
                    novelty.bind_program(source['code'], data['program_id'], evaluation_id=ident)
                terminal[ident] = data
        return novelty, reservations, terminal

    def inspect(self, code):
        with self._locked():
            novelty, _, _ = self._load()
            return novelty.inspect(code)

    def reserve(self, code, evaluation_id, lab_revision_id, *, plan_id, remeasurement=False):
        revision = self.registry.revision(lab_revision_id)
        plan = self.registry.get_artifact('experiment_plan', plan_id)
        if revision['lab_id'] != self.lab_id or plan['lab_id'] != self.lab_id or plan['lab_revision_id'] != lab_revision_id:
            raise ValueError('solution must bind its own frozen lab revision and plan')
        treatment = plan['treatment']
        if len(treatment) != 1:
            raise ValueError('solution plan requires one execution bundle')
        method = self.registry.get_method(treatment[0])
        expected = self.bridge.solution_bundle(code, self.lab_id, method['manifest']['runtime_lock'], method['manifest']['resources'])
        if expected != {key: method[key] for key in ('files', 'manifest')}:
            raise ValueError('solution plan does not contain the exact trusted cascade/task bundle')
        if len(plan.get('inputs',[]))!=1:
            raise ValueError('solution plan requires exactly one assigned input')
        source = dict(schema='sciel-lab-solution-candidate/1', lab_id=self.lab_id,
            evaluation_id=evaluation_id, lab_revision_id=lab_revision_id, plan_id=plan_id, method_id=method['id'],
            bridge_identity_sha256=content_hash(self.bridge.identity), source_sha256=_sha(code), code=code,
            remeasurement=remeasurement)
        with self._locked():
            novelty, reservations, terminal = self._load()
            if evaluation_id in reservations:
                if reservations[evaluation_id] != source:
                    raise ValueError('evaluation identity already binds a different solution/plan/revision')
                return {'accept': True, 'replayed': True, 'dispatch_allowed': False,
                        'finalized': evaluation_id in terminal, 'source': detached(source)}
            decision = novelty.reserve(code, evaluation_id, remeasurement=remeasurement)
            if not decision['accept']:
                return decision
            source_id = self.registry.put_artifact('solution_candidate', source)
            self.registry.append('solution_novelty_reserved', dict(lab_id=self.lab_id, source_id=source_id))
            return {**decision, 'replayed': False, 'dispatch_allowed': True, 'source': detached(source)}

    def _receipt_outcome(self, source, call_receipt_id, program_id):
        receipt = self.registry.get_artifact('call_receipt', call_receipt_id)
        execution = receipt['result']
        if receipt['status'] == 'unknown' or execution['status'] == 'unknown':
            raise BudgetUnknown('unknown solution execution must remain pending')
        if execution.get('plan_id') != source['plan_id'] or execution.get('method_id') != source['method_id']:
            raise ValueError('solution receipt belongs to another plan or execution bundle')
        outcome = None
        if execution['status'] == 'ok':
            if receipt['status'] != 'completed':
                raise ValueError('successful execution contradicts its accounting receipt')
            plan=self.registry.get_artifact('experiment_plan',source['plan_id'])
            if execution.get('input_sha256')!=content_hash(plan['inputs'][0]):
                raise ValueError('solution receipt input differs from its frozen plan')
            if isinstance(self.bridge,EngineCascadeBridge):
                outcome = self.bridge.extract(execution['output'], expected_source_sha256=source['source_sha256'])
            else:
                measured=self.bridge.measure(plan['inputs'][0],execution['output'])
                outcome=EvalResult(ok=measured['valid'],score=measured['metrics'][self.bridge.metric],
                    cases=measured['metrics'],artifacts={'call_receipt_id':call_receipt_id})
        elif execution['status'] != 'failed':
            raise ValueError('unsupported solution execution status')
        valid=(self.bridge._valid(outcome) if isinstance(outcome,CascadeOutcome)
               else bool(outcome and outcome.ok))
        if program_id is not None and not valid:
            raise ValueError('failed solution cannot bind an admitted Program')
        return outcome

    def finalize(self, evaluation_id, call_receipt_id, program_id=None):
        with self._locked():
            _, reservations, terminal = self._load()
            source = reservations[evaluation_id]
            outcome = self._receipt_outcome(source, call_receipt_id, program_id)
            evaluation=outcome.result if isinstance(outcome,CascadeOutcome) else outcome
            data = dict(lab_id=self.lab_id, evaluation_id=evaluation_id, call_receipt_id=call_receipt_id,
                        evaluation_ok=bool(evaluation and evaluation.ok), program_id=program_id)
            if evaluation_id in terminal:
                if terminal[evaluation_id] != data:
                    raise ValueError('completed solution measurement cannot be rewritten')
            else:
                self.registry.append('solution_novelty_finalized', data)
            return outcome

    def state(self):
        with self._locked():
            novelty, reservations, terminal = self._load()
            return {'novelty': novelty.state(), 'reservations': detached(reservations), 'terminal': detached(terminal)}
