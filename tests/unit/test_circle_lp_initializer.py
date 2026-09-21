"""Known cheap LPs and fake solver/clock edges; never a packing search."""
import math
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeWarning

from sciel.solvers import circle_lp_initializer as lp


def guess(radius=0.):
    centers=np.zeros((26,2));centers[0]=(.25,.5);centers[1]=(.75,.5)
    radii=np.zeros(26);radii[:2]=radius
    return np.concatenate((centers.ravel(),radii))


def vector(radius=.25):
    value=np.zeros(26);value[:2]=radius;return value


def result(x=None,*,status=0,success=True,message='optimal'):
    return SimpleNamespace(x=vector() if x is None else x,status=status,success=success,message=message)


class Clock:
    now=0.
    def __call__(self):return self.now


def test_known_two_circle_optimum_with_24_zero_wall_circles_real_lp():
    params=guess(.1);before=params.copy();stats=lp.new_lp_stats()
    value=lp.initialize_radii(params,deadline=lp._lp_time.monotonic()+1.,stats=stats)
    np.testing.assert_array_equal(params,before)
    np.testing.assert_array_equal(value[:52],params[:52])
    np.testing.assert_allclose(value[52:],vector(),atol=1e-12,rtol=0)
    assert math.fsum(value[52:])==pytest.approx(.5,abs=1e-12)
    assert stats['lp_calls']==stats['lp_accepted']==1 and stats['lp_fallbacks']==stats['lp_errors']==0
    assert stats['lp_time_seconds']>=0


def test_real_lp_duplicate_centers_have_known_zero_optimum():
    params=guess();params[:4]=(.5,.5,.5,.5)
    value=lp.initialize_radii(params,deadline=lp._lp_time.monotonic()+1.)
    assert math.fsum(value[52:])==0
    np.testing.assert_array_equal(value[:52],params[:52])


def test_all26_bounds_and325_unique_pair_constraints_and_pinned_options():
    params=guess();calls=[];clock=Clock()
    def fake(c,**kw):
        calls.append(kw)
        np.testing.assert_array_equal(c,-np.ones(26))
        assert kw['A_ub'].shape==(325,26) and kw['b_ub'].shape==(325,)
        assert kw['bounds'].shape==(26,2)
        pairs=set()
        centers=params[:52].reshape(26,2)
        for row,rhs in zip(kw['A_ub'],kw['b_ub']):
            i,j=np.flatnonzero(row);assert i<j and sum(row)==2 and row[i]==row[j]==1
            pairs.add((i,j));assert rhs==pytest.approx(math.dist(centers[i],centers[j]),abs=1e-15)
        assert len(pairs)==325
        for (low,high),(x,y) in zip(kw['bounds'],centers):assert low==0 and high==min(x,y,1-x,1-y)
        assert kw['method']=='highs' and kw['options']==dict(time_limit=.02,threads=1,parallel=False,random_seed=0)
        return result()
    lp.initialize_radii(params,deadline=.02,clock=clock,linprog_fn=fake)
    assert len(calls)==1


def test_cap_point05_copies_arrays_and_does_not_advance_global_rng():
    params=guess();before=params.copy();provided=vector();rng=np.random.get_state();stats={}
    def fake(c,**kw):
        assert kw['options']['time_limit']==.05
        kw['A_ub'][:]=999;kw['bounds'][:]=999
        return result(provided)
    try:
        value=lp.initialize_radii(params,deadline=10.,clock=Clock(),linprog_fn=fake,stats=stats)
        after_rng=np.random.get_state()
        assert all(np.array_equal(a,b) for a,b in zip(rng,after_rng))
        provided[:]=9
        np.testing.assert_array_equal(value[52:],vector())
        np.testing.assert_array_equal(params,before)
        assert not np.shares_memory(value,params) and not np.shares_memory(value,provided)
        assert np.max(lp._LP_A)==1
    finally:np.random.set_state(rng)


def test_deadline_before_lp_skips_without_solver_or_mutation():
    def forbidden(*a,**k):pytest.fail('expired deadline must not call LP')
    stats=lp.new_lp_stats();params=guess(.1)
    value=lp.initialize_radii(params,deadline=0.,clock=Clock(),linprog_fn=forbidden,stats=stats)
    np.testing.assert_array_equal(value,params)
    assert stats['lp_deadline_skips']==1 and stats['lp_calls']==stats['lp_accepted']==0


def test_deadline_expiring_while_building_problem_still_skips_call():
    times=iter([0.,0.,1.,1.]);stats={}
    def forbidden(*a,**k):pytest.fail('construction consumed remaining budget')
    value=lp.initialize_radii(guess(),deadline=.5,clock=lambda:next(times),linprog_fn=forbidden,stats=stats)
    assert stats['lp_deadline_skips']==1 and stats['lp_calls']==0 and stats['lp_time_seconds']==1.
    np.testing.assert_array_equal(value,guess())


def test_success_computed_after_deadline_remains_available_to_caller():
    clock=Clock();stats={}
    def fake(*a,**k):clock.now=.1;return result()
    value=lp.initialize_radii(guess(),deadline=.02,clock=clock,linprog_fn=fake,stats=stats)
    np.testing.assert_array_equal(value[52:],vector())
    assert stats['lp_accepted']==1 and stats['lp_time_seconds']==.1


@pytest.mark.parametrize('status,message',[(2,'infeasible'),(3,'unbounded'),(4,'numerical error'),(1,'iteration limit')])
def test_impossible_or_unexpected_termination_fails_candidate(status,message):
    stats={}
    with pytest.raises(lp.LPInitializationError):
        lp.initialize_radii(guess(),deadline=1.,clock=Clock(),stats=stats,
            linprog_fn=lambda *a,**k:result(status=status,success=False,message=message))
    assert stats['lp_calls']==stats['lp_errors']==1 and stats['lp_fallbacks']==0


def test_normal_time_limit_returns_exact_original_with_specific_reason():
    params=guess(.1);stats={}
    value=lp.initialize_radii(params,deadline=1.,clock=Clock(),stats=stats,
        linprog_fn=lambda *a,**k:result(status=1,success=False,message='Time limit reached. (HiGHS Status 13)'))
    np.testing.assert_array_equal(value,params)
    assert stats['lp_fallback_reasons']=={'time_limit':1} and stats['lp_accepted']==0


@pytest.mark.parametrize('radii,reason',[([[1],[1,2]],'invalid_vector'),(['invalid']*26,'invalid_vector'),
    (np.ones(25),'invalid_vector'),(np.full(26,np.nan),'invalid_vector'),
    (np.full(26,np.inf),'invalid_vector'),(np.full(26,1j),'invalid_vector'),
    (np.full(26,-1e-12),'infeasible_vector'),(vector(.3),'infeasible_vector')])
def test_invalid_success_vectors_are_not_repaired(radii,reason):
    params=guess(.1);stats={}
    actual=lp.initialize_radii(params,deadline=1.,clock=Clock(),stats=stats,linprog_fn=lambda *a,**k:result(radii))
    np.testing.assert_array_equal(actual,params)
    assert stats['lp_fallback_reasons']=={reason:1} and stats['lp_accepted']==0


def test_pair_overlap_independent_check_even_when_each_wall_bound_satisfied():
    params=guess();params[:4]=(.4,.5,.6,.5);stats={}
    actual=lp.initialize_radii(params,deadline=1.,clock=Clock(),stats=stats,linprog_fn=lambda *a,**k:result(vector(.25)))
    np.testing.assert_array_equal(actual,params)
    assert stats['lp_fallback_reasons']=={'infeasible_vector':1}


def test_tolerance_valid_is_not_repaired_or_claimed_exact():
    stats={};radii=vector(.2500001)
    actual=lp.initialize_radii(guess(),deadline=1.,clock=Clock(),stats=stats,linprog_fn=lambda *a,**k:result(radii))
    np.testing.assert_array_equal(actual[52:],radii)
    assert stats['lp_accepted']==1 and math.fsum(actual[52:])>.5


def test_lower_lp_sum_rejected_only_if_original_guess_is_feasible():
    stats={};original=guess(.2)
    actual=lp.initialize_radii(original,deadline=1.,clock=Clock(),stats=stats,linprog_fn=lambda *a,**k:result(vector(.19)))
    np.testing.assert_array_equal(actual,original)
    assert stats['lp_fallback_reasons']=={'lower_than_feasible_guess':1}
    stats={};actual=lp.initialize_radii(guess(.4),deadline=1.,clock=Clock(),stats=stats,linprog_fn=lambda *a,**k:result(vector(.19)))
    assert stats['lp_accepted']==1 and math.fsum(actual[52:])==.38


def test_unexpected_programming_error_propagates_and_stats_persist():
    stats={}
    def broken(*a,**k):raise RuntimeError('programming defect')
    with pytest.raises(RuntimeError,match='programming defect'):
        lp.initialize_radii(guess(),deadline=1.,clock=Clock(),stats=stats,linprog_fn=broken)
    assert stats['lp_errors']==1 and stats['lp_fallbacks']==0


def test_only_verified_forwarding_warning_is_suppressed():
    def expected(*a,**k):
        warnings.warn("Unrecognized options detected: {'threads': 1, 'parallel': False, 'random_seed': 0}. These will be passed to HiGHS verbatim.",OptimizeWarning)
        return result()
    lp.initialize_radii(guess(),deadline=1.,clock=Clock(),linprog_fn=expected)
    def unsupported(*a,**k):
        warnings.warn('parallel option unsupported; using default',OptimizeWarning)
        return result()
    with pytest.raises(OptimizeWarning,match='unsupported'):
        lp.initialize_radii(guess(),deadline=1.,clock=Clock(),linprog_fn=unsupported)
    def other_warning(*a,**k):
        warnings.warn('native runtime problem',RuntimeWarning)
        return result()
    with pytest.raises(RuntimeWarning,match='runtime problem'):
        lp.initialize_radii(guess(),deadline=1.,clock=Clock(),linprog_fn=other_warning)


@pytest.mark.parametrize('params,deadline',[(np.ones(77),1.),(np.full(78,np.nan),1.),(np.full(78,1j),1.),
    (np.full(78,2.),1.),(guess(),True),(guess(),float('nan'))])
def test_bad_inputs_fail_before_solver(params,deadline):
    def forbidden(*a,**k):pytest.fail('bad input called LP')
    with pytest.raises(ValueError):lp.initialize_radii(params,deadline=deadline,clock=Clock(),linprog_fn=forbidden)
