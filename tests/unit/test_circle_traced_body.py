"""Pure assembly and fixed fake-solver boundaries; never a numerical search."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.hypothesis import circle_restart_body as legacy
from sciel.hypothesis import circle_traced_restart as bound_only
from sciel.hypothesis import circle_traced_body as builder
from sciel.hypothesis.restart_trace import TRACE_SCHEMA, validate_trace


def sha(value):
    return hashlib.sha256(value if type(value) is bytes else value.encode()).hexdigest()


@pytest.mark.parametrize('body', ['return ordinary_params.copy()',
    'x = "a\u2028b"\nreturn ordinary_params\n',
    '# Preserve α and CRLF\r\nreturn best_params.copy()\r\n'])
def test_exact_body_inverse_preserves_lf_crlf_and_unicode_source_bytes(body):
    source = builder.source_for(body)
    recovered = builder.body_of(source)
    assert recovered == body + ('' if body.endswith('\n') else '\n')
    assert builder.source_for(recovered) == source
    assert legacy.validate_body(body) == builder.validate_body(body)


def test_full_traced_source_bound_proof_restores_parent_and_common_scaffold():
    parent = builder.source_for()
    target = 'value = ordinary_params.copy()\nvalue[0] += uniform_noise[0]\nreturn value'
    built = builder.apply(parent, target)
    proof, child = built['proof'], built['code']
    assert proof['schema'] == 'source-bound-edit/1'
    assert proof['operation'] == proof['contract'] == builder.CONTRACT
    assert proof['execution_profile'] == builder.EXECUTION_PROFILE
    assert proof['wire_contract'] == legacy.CONTRACT
    assert proof['parent_source_sha256'] == sha(parent) and proof['child_source_sha256'] == sha(child)
    assert proof['parent_body'] == builder.DEFAULT_BODY and proof['target_body'] == target
    pa, pb = proof['before']['byte_span']; ca, cb = proof['after']['byte_span']
    p, c = parent.encode(), child.encode()
    assert p[pa:pb] == proof['before']['text'].encode()
    assert c[ca:cb] == proof['after']['text'].encode()
    assert c[:ca] == p[:pa] and c[cb:] == p[pb:]
    assert c[:ca] + p[pa:pb] + c[cb:] == p
    identity = builder.identity()
    assert proof['common_scaffold_sha256'] == identity['common_scaffold_sha256']
    assert proof['dependencies'] == identity['dependencies']
    assert proof['dependencies']['traced_builder_sha256'] == sha(Path(bound_only.__file__).read_bytes())
    assert proof['baseline_source_sha256'] == sha(builder.source_for())
    with pytest.raises(ValueError, match='unchanged'):
        builder.apply(child, target+'\n')


def test_legacy_and_damaged_scaffolds_are_not_relabelled_as_traced_body():
    good = builder.source_for()
    for source in (legacy.source_for(), bound_only.source_for(),
                   good.replace('class _NoopTraceRecorder:', 'class _OtherRecorder:', 1),
                   good + '\n'):
        with pytest.raises(ValueError, match='exact traced body family'):
            builder.body_of(source)


def test_wire_schema_is_legacy_but_identity_and_writer_interface_bind_new_source():
    ident, interface = builder.identity(), builder.interface_description()
    assert builder.response_format() == legacy.response_format()
    assert ident['control_source_sha256'] == sha(builder.source_for()) != sha(legacy.source_for())
    assert ident['control_source_sha256'] != sha(bound_only.source_for())
    assert interface['contract'] == ident['contract'] == builder.CONTRACT
    assert interface['wire_contract'] == legacy.CONTRACT
    assert interface['default_transformation']['control_source_sha256'] == ident['control_source_sha256']
    assert interface['default_transformation']['common_scaffold_sha256'] == ident['common_scaffold_sha256']
    canonical = json.dumps(interface, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert ident['interface_sha256'] == sha(canonical)
    assert builder.response_instruction().endswith(canonical)
    assert 'no-op' in interface['tracing']['unbound']
    interface['arrays'].clear()
    assert builder.interface_description()['arrays']


def test_typed_body_extension_preserves_other_guard_restrictions():
    for body in ('import os\nreturn None\n', 'return incumbent\n',
                 'values = best_params\nvalues.append(1)\nreturn None\n',
                 'values = []\nvalues.extend([1])\nreturn None\n'):
        with pytest.raises(ValueError): builder.source_for(body)
        with pytest.raises(ValueError): legacy.source_for(body)


def test_solver_equations_and_single_rng_optimizer_calls_are_unchanged():
    trees = [ast.parse(source) for source in (bound_only.source_for(), builder.source_for())]
    for name in ('objective', 'objective_jac', 'packing_constraints', 'packing_jacobian',
                 'is_feasible', 'check_deadline', 'timed', 'callback', '_proposal_invoke'):
        nodes = [next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == name) for t in trees]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1])
    for name in ('minimize', 'np.random.uniform'):
        nodes = [[n for n in ast.walk(t) if isinstance(n, ast.Call) and ast.unparse(n.func) == name] for t in trees]
        assert len(nodes[0]) == len(nodes[1]) == 1
        assert ast.dump(nodes[0][0]) == ast.dump(nodes[1][0])


def inert_namespace(monkeypatch):
    source = builder.source_for()
    namespace = {}
    exec(compile(source, '<inert-traced-body>', 'exec'), namespace)
    namespace['time'] = SimpleNamespace(monotonic=lambda: 0.)
    calls = {'noise': 0, 'helper': 0, 'optimizer': 0}
    def uniform(low, high, size):
        assert (low, high, size) == (-.06, .06, (78,))
        calls['noise'] += 1
        return np.full(78, .001)
    monkeypatch.setattr(np.random, 'uniform', uniform)
    def helper(*args):
        calls['helper'] += 1
        return None
    def optimizer(*args, **kwargs):
        calls['optimizer'] += 1
        return SimpleNamespace(x=namespace['initial_parameters'](), success=True)
    namespace.update(propose_restart=helper, minimize=optimizer)
    binding = dict(source_sha256=sha(source), seed=31, runtime_sha256='a'*64,
                   invocation_id='b'*32, adapter_identity_sha256='c'*64)
    return namespace, calls, binding


def test_same_source_ordinary_noop_and_bound_trace_each_use_one_numeric_path(monkeypatch):
    namespace, calls, binding = inert_namespace(monkeypatch)
    original_recorder = namespace['RestartTrace']
    def forbidden(*args, **kwargs): pytest.fail('ordinary evaluation must not instantiate collector')
    namespace['RestartTrace'] = forbidden
    ordinary = namespace['solve_packing'](max_restarts=1)
    assert namespace['SCIEL_RESTART_TRACE'] is None
    assert calls == {'noise':1, 'helper':1, 'optimizer':1}
    namespace['RestartTrace'] = original_recorder
    namespace['SCIEL_RESTART_TRACE_BINDING'] = binding
    recorded = namespace['solve_packing'](max_restarts=1)
    assert calls == {'noise':2, 'helper':2, 'optimizer':2}
    for old, new in zip(ordinary, recorded, strict=True):
        np.testing.assert_array_equal(old, new)
    trace = validate_trace(namespace['SCIEL_RESTART_TRACE'], source_sha256=binding['source_sha256'],
        seed=binding['seed'], runtime_sha256=binding['runtime_sha256'], invocation_id=binding['invocation_id'],
        scaffold_sha256=namespace['SCIEL_RESTART_TRACE_SCAFFOLD_SHA256'])
    assert trace['schema'] == TRACE_SCHEMA and trace['total_scheduled_calls'] == 1
    assert trace['calls'][0]['optimizer_result']['ref'] is not None


@pytest.mark.parametrize('damage', ['type', 'missing', 'seed', 'adapter'])
def test_invalid_nonnull_binding_fails_before_helper_rng_or_optimizer(monkeypatch, damage):
    namespace, calls, binding = inert_namespace(monkeypatch)
    if damage == 'type': binding = False
    elif damage == 'missing': binding.pop('seed')
    elif damage == 'seed': binding['seed'] = True
    else: binding['adapter_identity_sha256'] = 'bad'
    namespace['SCIEL_RESTART_TRACE_BINDING'] = binding
    namespace['SCIEL_RESTART_TRACE'] = {'old': 'must not survive'}
    with pytest.raises(ValueError): namespace['solve_packing'](max_restarts=1)
    assert calls == {'noise':0, 'helper':0, 'optimizer':0}
    assert namespace['SCIEL_RESTART_TRACE'] is None


@pytest.mark.parametrize('bound', [False, True])
def test_helper_failure_is_not_suppressed_in_either_recording_mode(monkeypatch, bound):
    namespace, calls, binding = inert_namespace(monkeypatch)
    if bound: namespace['SCIEL_RESTART_TRACE_BINDING'] = binding
    def fail(*args):
        calls['helper'] += 1
        raise RuntimeError('inert helper failure')
    namespace['propose_restart'] = fail
    with pytest.raises(RuntimeError, match='inert helper failure'):
        namespace['solve_packing'](max_restarts=1)
    assert calls == {'noise':1, 'helper':1, 'optimizer':0}
    assert namespace['LAST_STATS']['proposal_errors'] == 1
    trace = namespace['SCIEL_RESTART_TRACE']
    assert (trace is None) if not bound else (trace['schema'] == TRACE_SCHEMA and trace['status'] == 'error')


def test_bound_then_unbound_call_clears_prior_trace_without_extra_work(monkeypatch):
    namespace, calls, binding = inert_namespace(monkeypatch)
    namespace['SCIEL_RESTART_TRACE_BINDING'] = binding
    namespace['solve_packing'](max_restarts=1)
    assert namespace['SCIEL_RESTART_TRACE']['schema'] == TRACE_SCHEMA
    namespace['SCIEL_RESTART_TRACE_BINDING'] = None
    namespace['solve_packing'](max_restarts=1)
    assert namespace['SCIEL_RESTART_TRACE'] is None
    assert calls == {'noise':2, 'helper':2, 'optimizer':2}


def test_collector_method_fault_preserves_result_and_exports_unavailable(monkeypatch):
    namespace, calls, binding = inert_namespace(monkeypatch)
    ordinary = namespace['solve_packing'](max_restarts=1)
    class BrokenRecorder(namespace['RestartTrace']):
        def optimizer_result(self, *args, **kwargs):
            raise RuntimeError('inert diagnostic failure')
    namespace['RestartTrace'] = BrokenRecorder
    namespace['SCIEL_RESTART_TRACE_BINDING'] = binding
    recorded = namespace['solve_packing'](max_restarts=1)
    for before, after in zip(ordinary, recorded, strict=True):
        np.testing.assert_array_equal(before, after)
    assert calls == {'noise':2, 'helper':2, 'optimizer':2}
    assert namespace['SCIEL_RESTART_TRACE'] == {
        'schema':'sciel-restart-trace-unavailable/1',
        'reason':'recorder_error', 'error_type':'RuntimeError'}
