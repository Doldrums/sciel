"""Pure assembly and fake one-call solver boundaries; no numerical search."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.hypothesis import circle_restart_body as typed
from sciel.hypothesis import circle_traced_body as old
from sciel.hypothesis import circle_traced_restart as old_trace
from sciel.hypothesis import restart_trace as old_codec
from sciel.hypothesis import restart_optimizer_trace as codec
from sciel.hypothesis import circle_optimizer_traced_body as builder


def sha(value):
    return hashlib.sha256(value if type(value) is bytes else value.encode()).hexdigest()


@pytest.mark.parametrize('body', ['return None\n', 'return best_params.copy()',
    'label = "a\u2028b"\nreturn ordinary_params\n', '# Preserve α\r\nreturn ordinary_params\r\n'])
def test_exact_inverse_and_same_typed_surface(body):
    source = builder.source_for(body)
    assert builder.body_of(source) == body + ('' if body.endswith('\n') else '\n')
    assert builder.source_for(builder.body_of(source)) == source
    assert builder.validate_body(body) == typed.validate_body(body)


def test_old_profile_sources_and_collector_remain_byte_identical():
    assert sha(Path(old.__file__).read_bytes()) == builder.PINNED_LEGACY['body_builder_sha256']
    assert sha(Path(old_trace.__file__).read_bytes()) == builder.PINNED_LEGACY['traced_builder_sha256']
    assert sha(old_codec.collector_source()) == builder.PINNED_LEGACY['collector_sha256']
    assert sha(old.source_for()) == '2f4bd6cc96eb780a73f547f1e24f28961a3b3a70478a04e4c6fc4a864e76e34d'
    for source in (typed.source_for(), old.source_for(), old_trace.source_for(), builder.source_for()+'\n'):
        with pytest.raises(ValueError, match='exact optimizer traced body family'): builder.body_of(source)


def test_source_bound_proof_only_changes_helper_and_identity_advertises_profile2():
    parent = builder.source_for()
    child = builder.apply(parent, 'return best_params.copy()')
    proof = child['proof']; code = child['code']; identity = builder.identity()
    assert proof['operation'] == 'circle_restart_body_traced/2'
    assert proof['execution_profile'] == 'paired_restart_trace/2'
    assert proof['parent_source_sha256'] == sha(parent) and proof['child_source_sha256'] == sha(code)
    pa,pb = proof['before']['byte_span']; ca,cb = proof['after']['byte_span']
    assert parent.encode()[:pa] == code.encode()[:ca] and parent.encode()[pb:] == code.encode()[cb:]
    assert code.encode()[:ca]+parent.encode()[pa:pb]+code.encode()[cb:] == parent.encode()
    assert proof['dependencies'] == identity['dependencies']
    assert identity['dependencies']['collector_sha256'] == sha(Path(codec.__file__).read_bytes())
    assert identity['dependencies']['legacy_collector_sha256'] == builder.PINNED_LEGACY['collector_sha256']
    assert proof['common_scaffold_sha256'] == identity['common_scaffold_sha256']
    assert sha(parent) == identity['control_source_sha256'] != sha(old.source_for())
    interface = builder.interface_description()
    assert interface['tracing']['schema'] == codec.TRACE_SCHEMA
    assert builder.response_format() == typed.response_format()
    encoded = json.dumps(interface, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert identity['interface_sha256'] == sha(encoded) and builder.response_instruction().endswith(encoded)


def test_original_solver_equations_calls_and_x_read_are_not_changed():
    trees = [ast.parse(source) for source in (old.source_for(), builder.source_for())]
    for name in ('objective','objective_jac','packing_constraints','packing_jacobian','is_feasible',
                 'consider','check_deadline','timed','callback','_proposal_invoke'):
        nodes = [next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name) for t in trees]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1])
    for name in ('minimize','np.random.uniform'):
        calls = [[n for n in ast.walk(t) if isinstance(n,ast.Call) and ast.unparse(n.func)==name] for t in trees]
        assert len(calls[0]) == len(calls[1]) == 1 and ast.dump(calls[0][0]) == ast.dump(calls[1][0])
    for field in ('x','success'):
        for tree in trees:
            attrs = [n for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name)
                     and n.value.id=='result' and n.attr==field]
            assert len(attrs) == 1


def inert(monkeypatch):
    source = builder.source_for()
    ns = {}; exec(compile(source,'<inert-optimizer-profile>','exec'),ns)
    ns['time'] = SimpleNamespace(monotonic=lambda:0.)
    calls = dict(helper=0,optimizer=0,noise=0); reads=[]
    def noise(low,high,size):
        calls['noise'] += 1
        assert (low,high,size)==(-.06,.06,(78,))
        return np.full(78,.001)
    monkeypatch.setattr(np.random,'uniform',noise)
    def helper(*args): calls['helper']+=1; return None
    class Flag:
        def __bool__(self): reads.append('bool'); return False
    class Result:
        @property
        def success(self): reads.append('success'); return Flag()
        @property
        def status(self): reads.append('status'); return 8
        @property
        def x(self): reads.append('x'); return ns['initial_parameters']()
    def optimize(*args,**kwargs): calls['optimizer']+=1; return Result()
    ns.update(propose_restart=helper,minimize=optimize)
    binding=dict(source_sha256=sha(source),seed=17,runtime_sha256='a'*64,invocation_id='b'*32)
    return ns,calls,reads,binding,Result


@pytest.mark.parametrize('bound',[False,True])
def test_success_bool_and_x_are_read_once_in_original_order_and_status_only_when_bound(monkeypatch,bound):
    ns,calls,reads,binding,_ = inert(monkeypatch)
    if bound: ns['SCIEL_RESTART_TRACE_BINDING']=binding
    result=ns['solve_packing'](max_restarts=1)
    assert result[2] == .25 and calls==dict(helper=1,optimizer=1,noise=1)
    assert reads == (['success','bool','status','x'] if bound else ['success','bool','x'])
    assert ns['LAST_STATS']['successful_starts']==0
    trace=ns['SCIEL_RESTART_TRACE']
    if not bound: assert trace is None
    else:
        value=codec.validate_trace(trace,**binding,scaffold_sha256=ns['SCIEL_RESTART_TRACE_SCAFFOLD_SHA256'])
        assert value['calls'][0]['optimizer_outcome']=={
            'success':dict(value=False,reason=None),'status':dict(value=8,reason=None)}


@pytest.mark.parametrize('field',['success','x'])
def test_original_attribute_errors_still_propagate_without_later_reads(monkeypatch,field):
    ns,calls,reads,binding,Result=inert(monkeypatch)
    def broken(self): reads.append(field); raise RuntimeError('original attribute failure')
    setattr(Result,field,property(broken))
    ns['SCIEL_RESTART_TRACE_BINDING']=binding
    with pytest.raises(RuntimeError,match='original attribute failure'): ns['solve_packing'](max_restarts=1)
    assert reads==(['success'] if field=='success' else ['success','bool','status','x'])
    assert calls==dict(helper=1,optimizer=1,noise=1)
    value=codec.validate_trace(ns['SCIEL_RESTART_TRACE'])
    assert value['status']=='error'
    if field=='success':
        assert value['calls'][0]['optimizer_outcome']['success']==dict(value=None,reason='not_observed')
    else:
        assert value['calls'][0]['optimizer_outcome']['success']==dict(value=False,reason=None)


def test_status_getter_failure_preserves_returned_packing_and_retains_explicit_reason(monkeypatch):
    ns,calls,reads,binding,Result=inert(monkeypatch)
    def broken(self): reads.append('status'); raise RuntimeError('diagnostic only')
    Result.status=property(broken)
    ns['SCIEL_RESTART_TRACE_BINDING']=binding
    result=ns['solve_packing'](max_restarts=1)
    assert result[2]==.25 and reads==['success','bool','status','x']
    assert ns['SCIEL_RESTART_TRACE']['calls'][0]['optimizer_outcome']['status']==dict(value=None,reason='read_error')


def test_recorder_fault_preserves_result_and_unbound_call_clears_stale_trace(monkeypatch):
    ns,calls,reads,binding,_=inert(monkeypatch)
    plain=ns['solve_packing'](max_restarts=1)
    class Broken(ns['RestartTrace']):
        def optimizer_outcome(self,*args,**kwargs): raise RuntimeError('recorder failure')
    ns['RestartTrace']=Broken; ns['SCIEL_RESTART_TRACE_BINDING']=binding
    bound=ns['solve_packing'](max_restarts=1)
    for a,b in zip(plain,bound): np.testing.assert_array_equal(a,b)
    assert ns['SCIEL_RESTART_TRACE']==dict(schema='sciel-restart-trace-unavailable/1',reason='recorder_error',error_type='RuntimeError')
    ns['SCIEL_RESTART_TRACE_BINDING']=None
    ns['solve_packing'](max_restarts=1)
    assert ns['SCIEL_RESTART_TRACE'] is None and calls==dict(helper=3,optimizer=3,noise=3)


def test_invalid_nonnull_binding_still_fails_before_any_candidate_work(monkeypatch):
    ns,calls,reads,binding,_=inert(monkeypatch)
    binding['seed']=True; ns['SCIEL_RESTART_TRACE_BINDING']=binding
    with pytest.raises(ValueError): ns['solve_packing'](max_restarts=1)
    assert calls==dict(helper=0,optimizer=0,noise=0) and reads==[]
