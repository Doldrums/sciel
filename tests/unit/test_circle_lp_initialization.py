"""Exact source receipts and fake joint optimizer integration, no packing search."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.hypothesis import circle_lp_initialization as builder
from sciel.hypothesis import circle_restart_scales as scales

sha=lambda raw:hashlib.sha256(raw).hexdigest()


def test_control_is_literal_pinned_source_and_three_insertions_are_invertible():
    baseline=builder.source_for('control').encode();built=builder.build();child=built['code'].encode();proof=built['proof']
    assert baseline==scales.source_for('0.06').encode()
    assert sha(baseline)==builder.PINNED_BASELINE_SHA256
    assert sha(child)==proof['child_source_sha256']
    assert proof['helper_source_sha256']==sha(builder._HELPER_PATH.read_bytes())
    assert len(proof['insertions'])==3 and [p['name']for p in proof['insertions']]==['helper','stats','initializer']
    restored=child
    for p in reversed(proof['insertions']):
        a,b=p['child_byte_span'];assert child[a:b]==p['text'].encode() and sha(child[a:b])==p['sha256']
        assert p['parent_byte_span'][0]==p['parent_byte_span'][1]
        restored=restored[:a]+restored[b:]
    assert restored==baseline
    for segment in proof['unchanged_segments']:
        a,b=segment['parent_byte_span'];assert sha(baseline[a:b])==segment['sha256']
    assert child.count(builder._HELPER_PATH.read_bytes())==1
    assert builder.source_for('lp')==built['code']


def test_rng_draw_clamps_and_slsqp_ast_are_literal_unchanged():
    a,b=[builder.source_for(p)for p in ('control','lp')]
    for line in ('guess = incumbent._params + np.random.uniform(-0.06, 0.06, initial.shape)',
        'guess[2 * N_CIRCLES:] = np.maximum(guess[2 * N_CIRCLES:], 0.001)',
        'guess[:2 * N_CIRCLES] = np.clip(guess[:2 * N_CIRCLES], 0.001, 0.999)',
        'return solve_packing(budget_seconds=27.0, max_restarts=10000, maxiter=15000)'):
        assert a.count(line)==b.count(line)==1
    def node(source,name):return next(n for n in ast.walk(ast.parse(source))if isinstance(n,ast.FunctionDef)and n.name==name)
    for name in ('objective','objective_jac','packing_constraints','packing_jacobian','is_feasible','run_packing'):
        assert ast.dump(node(a,name))==ast.dump(node(b,name))
    def minimize(source):return next(n for n in ast.walk(ast.parse(source))if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='minimize')
    assert ast.dump(minimize(a))==ast.dump(minimize(b))
    assert 'from sciel' not in b
    assert builder.identity()['lp_source_sha256']==sha(b.encode())


@pytest.mark.parametrize('policy',[None,1,True,'LP','0.06','unknown'])
def test_unknown_policy_refused(policy):
    with pytest.raises(ValueError):builder.source_for(policy)


def test_helper_or_baseline_tamper_rejected(tmp_path,monkeypatch):
    bad=tmp_path/'helper.py';bad.write_bytes(builder._HELPER_PATH.read_bytes()+b'# changed\n')
    monkeypatch.setattr(builder,'_HELPER_PATH',bad)
    with pytest.raises(ValueError,match='helper'):builder.source_for('lp')
    monkeypatch.setattr(scales,'source_for',lambda policy:'edited parent')
    with pytest.raises(ValueError,match='control'):builder.source_for('control')


class Clock:
    now=0.
    def __call__(self):return self.now


def fake_namespace(policy):
    # Compile the newly built standalone definitions only. Both optimizers are
    # replaced before any solve function is invoked, so no optimization runs.
    ns={};exec(compile(builder.source_for(policy),'<fake-only-lp-source>','exec'),ns)
    clock=Clock();ns['time']=SimpleNamespace(monotonic=clock)
    def forbidden(*a,**k):pytest.fail('real optimizer forbidden in source fixture')
    ns['minimize']=forbidden
    if policy=='lp':
        ns['_lp_time']=SimpleNamespace(monotonic=clock);ns['_lp_linprog']=forbidden
    return ns,clock


def test_same_three_full_rng_draws_clamps_and_no_extra_rng_on_lp_fallback():
    saved=np.random.get_state()
    try:
        starts={};end_rng={}
        for policy in ('control','lp'):
            ns,clock=fake_namespace(policy);starts[policy]=[]
            def fake_minimize(fn,x,**kw):
                starts[policy].append(x.copy())
                assert kw['method']=='SLSQP' and kw['options']==dict(maxiter=15000,ftol=1e-7,disp=False)
                return SimpleNamespace(x=ns['initial_parameters'](),success=True)
            ns['minimize']=fake_minimize
            if policy=='lp':ns['_lp_linprog']=lambda *a,**k:SimpleNamespace(status=1,success=False,message='Time limit reached',x=None)
            np.random.seed(471);ns['solve_packing'](budget_seconds=27.,max_restarts=3)
            end_rng[policy]=np.random.get_state()
            if policy=='lp':assert ns['LAST_STATS']['lp_fallbacks']==ns['LAST_STATS']['lp_calls']==3
        ns,_=fake_namespace('control');initial=ns['initial_parameters']();rng=np.random.RandomState(471)
        for index in range(3):
            expected=initial+rng.uniform(-.06,.06,(78,))
            expected[52:]=np.maximum(expected[52:],.001);expected[:52]=np.clip(expected[:52],.001,.999)
            np.testing.assert_array_equal(starts['control'][index],expected)
            np.testing.assert_array_equal(starts['lp'][index],expected)
        assert all(np.array_equal(a,b)for a,b in zip(end_rng['control'],end_rng['lp']))
    finally:np.random.set_state(saved)


def test_feasible_lp_after_deadline_retained_without_joint_solver(monkeypatch):
    ns,clock=fake_namespace('lp')
    def perturb(low,high,shape):
        assert (low,high,shape)==(-.06,.06,(78,))
        noise=np.zeros(78);noise[52:]=.04;return noise
    monkeypatch.setattr(np.random,'uniform',perturb)
    def fake_lp(*a,**kw):
        assert kw['options']['time_limit']==.02
        clock.now=.1
        radii=np.full(26,.05);radii[-1]=.001
        return SimpleNamespace(status=0,success=True,x=radii)
    ns['_lp_linprog']=fake_lp
    centers,radii,total=ns['solve_packing'](budget_seconds=.02,max_restarts=10000)
    assert total==pytest.approx(1.251) and ns['is_feasible'](np.concatenate((centers.ravel(),radii)))
    stats=ns['LAST_STATS'];assert stats['deadline_reached'] and stats['stop_reason']=='deadline'
    assert stats['started_starts']==stats['completed_starts']==0
    assert stats['lp_calls']==stats['lp_accepted']==stats['improvements']==1


def test_unexpected_lp_error_propagates_and_final_stats_survive():
    ns,clock=fake_namespace('lp')
    ns['_lp_linprog']=lambda *a,**kw:SimpleNamespace(status=2,success=False,message='impossible infeasible',x=None)
    saved=np.random.get_state()
    try:
        with pytest.raises(ns['LPInitializationError']):ns['solve_packing'](budget_seconds=1.,max_restarts=1)
    finally:np.random.set_state(saved)
    stats=ns['LAST_STATS'];assert stats['stop_reason']=='error' and stats['lp_errors']==1
    assert stats['started_starts']==0 and stats['best_sum']==pytest.approx(.25)


def test_zero_budget_initializes_all_lp_diagnostics_without_dispatch():
    ns,clock=fake_namespace('lp');_,_,total=ns['solve_packing'](budget_seconds=0.,max_restarts=10000)
    assert total==pytest.approx(.25)
    assert all(ns['LAST_STATS'][key]==0 for key in ('lp_calls','lp_accepted','lp_fallbacks','lp_deadline_skips','lp_errors','lp_time_seconds'))
    assert ns['LAST_STATS']['lp_fallback_reasons']=={}
