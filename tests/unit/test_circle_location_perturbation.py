"""Finite helper mathematics only; no scaffold, solver, evaluator, runtime or API.

All vectors are authored fixed fixtures, not numerical packing trials.
Only AST-extracted propose_restart functions are executed on fixed vectors.
"""
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import math
import struct

import numpy as np
import pytest

from sciel.hypothesis import circle_location_perturbation as proposal
from sciel.hypothesis import circle_optimizer_traced_body as builder

BODY_SHA = 'fa572547037b87c415dd92889337b79033ccfad6eb4d9043ced0357855d50328'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def helper(body):
    nodes = [node for node in ast.parse(builder.source_for(body)).body
             if isinstance(node, ast.FunctionDef) and node.name == 'propose_restart']
    assert len(nodes) == 1 and not nodes[0].decorator_list
    namespace = {'np': np, 'math': math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<finite-location-helper>', 'exec'), namespace)
    return namespace['propose_restart']


@pytest.fixture(scope='module')
def functions():
    return helper(proposal.PARENT_BODY), helper(proposal.BODY)


def locations(values):
    # Numeric, unlabeled geometry: signed zero must not manufacture activity.
    return Counter((Fraction.from_float(float(values[2*i])),
                    Fraction.from_float(float(values[2*i+1]))) for i in range(26))


def active(before, after):
    return after is not None and locations(before) != locations(after)


def choose(values, index):
    """Separate scalar ranking, with filtering before median and tuple ties."""
    if index <= 0 or index % 16:
        return None
    i = (index//16-1) % 26
    radii = list(map(float, values[52:]))
    centers = [(float(values[2*k]), float(values[2*k+1])) for k in range(26)]
    options = []
    for j in range(26):
        if i == j or radii[i] == radii[j] or centers[i] == centers[j]:
            continue
        if not all(radii[a] <= x <= 1.-radii[a]
                   for a,b in ((i,j),(j,i)) for x in centers[b]):
            continue
        residuals = [0.]
        for a,b in ((i,j),(j,i)):
            for k in range(26):
                if k in (i,j):
                    continue
                dx,dy = centers[b][0]-centers[k][0], centers[b][1]-centers[k][1]
                residuals.append(radii[a]+radii[k]-math.sqrt(dx*dx+dy*dy))
        options.append((max(residuals),abs(radii[i]-radii[j]),j))
    if not options:
        return None
    gaps = sorted(row[1] for row in options)
    m = len(gaps)//2
    threshold = gaps[m] if len(gaps)%2 else (gaps[m-1]+gaps[m])*.5
    return i, min((row for row in options if row[1] >= threshold),
                  key=lambda row: (row[0],-row[1],row[2]))[2]


def invoke(function, best, noise, index, initial=None, ordinary=None):
    inputs = [best.copy(), np.zeros(78) if initial is None else initial.copy(),
              noise.copy(), best.copy() if ordinary is None else ordinary.copy()]
    copies = [v.copy() for v in inputs]
    result = function(*inputs,index)
    assert all(a.tobytes()==b.tobytes() for a,b in zip(inputs,copies,strict=True))
    if result is not None:
        assert result.shape == (78,) and result.dtype == np.float64
        assert all(not np.shares_memory(result,v) for v in inputs)
    return result


def expected(best, noise, selected):
    if selected is None:
        return None
    i,j = selected
    result = best.copy()
    for destination,origin in ((i,j),(j,i)):
        r = float(best[52+destination])
        for axis in (0,1):
            result[2*destination+axis] = min(1.-r,max(r,
                float(best[2*origin+axis])+float(noise[2*destination+axis])))
    return result


def vectors(radii):
    v = np.zeros(78)
    v[:2] = [.25,.5]; v[52] = .125
    for j,(radius,center) in enumerate(zip(radii,[(.75,.25),(.75,.75),(.25,.75),(.5,.25)]),1):
        v[2*j:2*j+2] = center; v[52+j] = radius
    return v


def assert_untouched(before,after,selected):
    i,j = selected
    assert before[52:].tobytes()==after[52:].tobytes()
    for k in range(26):
        if k not in (i,j):
            assert before[2*k:2*k+2].tobytes()==after[2*k:2*k+2].tobytes()


@pytest.mark.parametrize('radii,partner', [
    ([.0625],1), ([.0625,.03125],2), ([.09375,.0625,.03125],3),
    ([.09375,.0625,.03125,.015625],4), ([.0625,.03125,.03125],2),
])
def test_same_even_odd_tie_selection_then_receiving_index_noise(functions,radii,partner):
    best = vectors(radii)
    # Four distinct signed dyadic components identify the receiving-index map.
    noise = np.zeros(78)
    noise[:2] = [.015625,-.03125]; noise[2*partner:2*partner+2] = [-.0078125,.00390625]
    selected = choose(best,16)
    assert selected == (0,partner)
    parent = invoke(functions[0],best,noise,16)
    result = invoke(functions[1],best,noise,16)
    assert not active(best,parent) and active(best,result)
    assert result.tobytes()==expected(best,noise,selected).tobytes()
    assert_untouched(best,result,selected)


def test_wall_filter_before_median_and_noise_cannot_change_selection(functions):
    best = vectors([.0625,.03125,.015625]); best[6] = .0625
    assert choose(best,16)==(0,2)
    noise = np.full(78,.015625)
    result = invoke(functions[1],best,noise,16)
    changed_noise = noise.copy()
    changed_noise[4:6] = noise[4:6]
    for k in range(78):
        if k not in (0,1,4,5):
            changed_noise[k] = -.046875
    assert invoke(functions[1],best,changed_noise,16).tobytes()==result.tobytes()
    assert result.tobytes()==expected(best,noise,(0,2)).tobytes()


def test_clips_to_each_receiving_radius_not_donor_radius(functions):
    # Arithmetic stress: fixed noise beyond production range isolates all walls.
    best = vectors([.0625]); noise = np.zeros(78)
    noise[:4] = [-2.,2.,2.,-2.]
    result = invoke(functions[1],best,noise,16)
    assert result[:4].tolist()==[.125,.875,.9375,.0625]
    assert result.tobytes()==expected(best,noise,(0,1)).tobytes()
    assert_untouched(best,result,(0,1))


def test_zero_noise_and_fully_erased_clipping_are_explicitly_inactive(functions):
    best = vectors([.0625]); best[:2]=[.5,.5]; best[2:4]=[.125,.125]
    zeros = np.zeros(78)
    parent = invoke(functions[0],best,zeros,16)
    assert not active(best,invoke(functions[1],best,zeros,16))
    noise = zeros.copy(); noise[:2]=[-.03125,-.015625]
    result = invoke(functions[1],best,noise,16)
    assert result.tobytes()==parent.tobytes()  # accepted is not active
    assert not active(best,result)
    noise[1]=.015625
    assert active(best,invoke(functions[1],best,noise,16))


def test_none_gates_and_signed_zero_do_not_fabricate_activity(functions):
    best=vectors([.125,.125,.125]); noise=np.full(78,.015625)
    for index in (0,4,8,12,17,16):
        assert invoke(functions[1],best,noise,index) is None
    assert not active(best,None)
    other=best.copy(); other[6]=-0.
    best[6]=0.
    assert locations(best)==locations(other)


@pytest.mark.parametrize('index,pair', [(16,(0,17)),(32,(1,17)),(48,(2,17))])
def test_authored_inputs_change_location_multiset_without_solver(functions,index,pair):
    best = np.zeros(78)
    i, j = pair
    best[2*i:2*i+2], best[52+i] = [.25, .5], .125
    best[2*j:2*j+2], best[52+j] = [.75, .25], .0625
    noise = np.full(78, .015625)
    assert choose(best,index)==pair
    parent=invoke(functions[0],best,noise,index)
    swapped=best.copy()
    swapped[2*i:2*i+2]=best[2*j:2*j+2]
    swapped[2*j:2*j+2]=best[2*i:2*i+2]
    assert parent.tobytes()==swapped.tobytes()
    result=invoke(functions[1],best,noise,index)
    assert result.tobytes()==expected(best,noise,pair).tobytes()
    assert locations(best)==locations(parent) and active(best,result)
    assert_untouched(best,result,pair)
    assert any(Fraction.from_float(float(a))!=Fraction.from_float(float(b))
               for a,b in zip(parent[:52],result[:52],strict=True))


def test_only_four_body_assignments_change_same_profile_scaffold_and_default():
    assert sha(proposal.PARENT_BODY.encode())==proposal.PARENT_BODY_SHA256
    assert sha(proposal.BODY.encode())==BODY_SHA
    assert proposal.BODY.replace(proposal._PERTURB,proposal._SWAP,1)==proposal.PARENT_BODY
    source=proposal.source_for(); parent=builder.source_for(proposal.PARENT_BODY)
    built=proposal.apply(parent)
    assert built['code']==source and builder.body_of(source)==proposal.BODY
    proof=built['proof']; a,b=proof['before']['byte_span']; c,d=proof['after']['byte_span']
    assert parent.encode()[:a]==source.encode()[:c] and parent.encode()[b:]==source.encode()[d:]
    assert proof['execution_profile']==proposal.EXECUTION_PROFILE=='paired_restart_trace/2'
    assert sha(builder.source_for().encode())=='93bc6ed5658a2a299c660de06d5dc4bf0e623f1d8a258aa2504b2a8a5ad98cae'
    tree=ast.parse('def f():\n'+''.join('    '+line+'\n' for line in proposal.BODY.splitlines()))
    assert not any(isinstance(n,ast.Attribute) and n.attr=='random' for n in ast.walk(tree))


@pytest.mark.parametrize('radius_zero',[0.,-0.])
def test_exact_clip_boundary_preserves_signed_zero_bits(functions,radius_zero):
    best=np.zeros(78); best[:4]=[.03125,.03125,.75,.25]
    best[52:54]=[.015625,radius_zero]
    noise=np.zeros(78); noise[2:4]=[-.03125,-.03125]
    result=invoke(functions[1],best,noise,16)
    assert choose(best,16)==(0,1)
    assert result[2:4].tobytes()==struct.pack('<2d',0.,0.)
    assert result[53].tobytes()==struct.pack('<d',radius_zero)
    assert active(best,result)
