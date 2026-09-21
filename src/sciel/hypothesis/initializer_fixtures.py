"""Fixed, source-independent initializer witness input recipes."""
from __future__ import annotations

import numpy as np

def _initial():
    # Exact pinned initial_parameters recipe, without importing its solver module.
    centers = [[(col+.5)/5, (row+.5)/5] for row in range(5) for col in range(5)] + [[0.,0.]]
    return np.array([v for xy in centers for v in xy] + [.01]*25 + [0.], dtype=np.float64)


def _legacy_fixture_bundle():
    """Fixed source-independent cases; no output-adaptive or random test generation."""
    from . import circle_initializer_witness as _public
    initial = _initial()
    dyadic = np.array([v for k in range(26) for v in ((k%6+1)/8,(k//6+1)/8)] + [1/64]*26)
    specs = [(0,'zero'), (0,'positive'), (0,'alternating'), (0,'edge'),
             (4,'zero'), (8,'positive'), (12,'negative'), (16,'alternating'),
             (20,'edge'), (32,'positive'), (416,'negative'), (432,'alternating'),
             (448,'positive'), (16,'wall'), (32,'tie'), (16,'tolerance_overlap')]
    cases = []
    for number,(index,kind) in enumerate(specs):
        best = initial.copy() if index == 0 else dyadic.copy()
        noise = np.full(78, 1/64, dtype=np.float64)
        if kind == 'zero': noise[:] = 0
        elif kind == 'negative': noise[:] = -1/64
        elif kind == 'alternating': noise[1::2] *= -1
        elif kind == 'edge': noise[:] = [0.05 if i%2 else -0.05 for i in range(78)]
        elif kind == 'wall': best[:2] = (1/64, 1/64); noise[:] = -1/64
        elif kind == 'tie': noise[1::2] = -1/64
        elif kind == 'tolerance_overlap':
            best[:4] = (1/16,15/16,3/32-2**-22,15/16)
        ordinary = best + noise
        ordinary[52:] = np.maximum(ordinary[52:], .001)
        ordinary[:52] = np.clip(ordinary[:52], .001, .999)
        values = dict(zip(_public.ALIASES,(best,initial.copy(),noise,ordinary)))
        cases.append({'id':f'w{number:02d}', 'restart_index':index, 'fixture_kind':kind,
            'domain':'scaffold_interface_constraints; reachability_not_proven',
            'out_of_runtime_domain':False,
            'inputs':{key:_public._array(array) for key,array in values.items()}})
    body = {'schema':_public.FIXTURE_SCHEMA, 'cases':cases,
            'scope':'Fixed synthetic inputs, not sampled search states. Index0 initial==best by value; separate storage. Positive incumbents satisfy tolerance geometry; no claim of trajectory reachability.'}
    return {**body, 'sha256':_public._hash(body)}


FIXTURE_PROFILES = ('legacy_v1', 'unequal_radii_v1')


def _unequal_fixture_bundle():
    """Source-independent radius/location contrasts, fixed before helper output."""
    from . import circle_initializer_witness as _public
    initial = _initial()
    base = np.array([v for k in range(26) for v in ((k%6+1)/8,(k//6+1)/8)] + [1/64]*26)
    specs = [(0,'initial_zero'), (0,'initial_alternating'), (4,'positive'),
             (8,'negative'), (12,'alternating'), (16,'zero'), (16,'positive'),
             (16,'negative'), (32,'alternating'), (416,'positive'),
             (432,'negative'), (448,'alternating'), (16,'equal_radii'),
             (16,'wall_filtered_tie'), (16,'wall_no_partner'),
             (16,'coincident_tolerance')]
    cases = []
    for number,(index,kind) in enumerate(specs):
        best = initial.copy() if index == 0 else base.copy()
        if index:
            best[53:56] = (1/32,3/64,1/32)
        noise = np.full(78,1/64,dtype=np.float64)
        if kind in ('zero','initial_zero'): noise[:] = 0
        elif kind == 'negative': noise[:] = -1/64
        elif kind in ('alternating','initial_alternating'): noise[1::2] *= -1
        if kind == 'equal_radii': best[52:] = 1/64
        elif kind == 'wall_filtered_tie':
            best[:2] = (1/64,1/8)
            best[55:57] = 1/128
        elif kind == 'wall_no_partner':
            best[:2] = (1/64,1/8)
            best[53:] = 1/32
        elif kind == 'coincident_tolerance':
            # Valid only at the pinned 1e-6 tolerance, including the wall.
            best[:4] = 0
            best[52:54] = (0.,2**-22)
        ordinary = best + noise
        ordinary[52:] = np.maximum(ordinary[52:],.001)
        ordinary[:52] = np.clip(ordinary[:52],.001,.999)
        values = dict(zip(_public.ALIASES,(best,initial.copy(),noise,ordinary)))
        cases.append({'id':f'u{number:02d}','restart_index':index,'fixture_kind':kind,
            'domain':'scaffold_interface_constraints; reachability_not_proven',
            'out_of_runtime_domain':False,
            'inputs':{key:_public._array(array) for key,array in values.items()}})
    body = {'schema':_public.FIXTURE_SCHEMA,'profile':'unequal_radii_v1','cases':cases,
        'scope':'Fixed source-independent unequal-radius/location inputs, not sampled search states. '
                'Index0 initial==best by value with separate storage. Positive incumbents satisfy '
                'the pinned tolerance geometry; coincident_tolerance is not exactly feasible. '
                'No claim of trajectory reachability or an expected helper output.'}
    return {**body,'sha256':_public._hash(body)}


def make_fixture_bundle(profile='legacy_v1'):
    """Return one registered fixed recipe; the default legacy bytes are unchanged."""
    from . import circle_initializer_witness as _public
    _public._require(type(profile) is str and profile in FIXTURE_PROFILES,'fixture_profile')
    return _legacy_fixture_bundle() if profile=='legacy_v1' else _unequal_fixture_bundle()


def fixture_bundle_by_sha256(digest):
    """Resolve historical fixed data without current runtime/source validation."""
    from . import circle_initializer_witness as _public
    _public._require(type(digest) is str,'fixture_digest')
    for profile in FIXTURE_PROFILES:
        bundle = make_fixture_bundle(profile)
        if bundle['sha256']==digest:
            return bundle
    raise ValueError('unsupported fixed fixture version')


def _registered_fixture(bundle):
    from . import circle_initializer_witness as _public
    _public._require(type(bundle) is dict,'fixture_bundle_not_fixed')
    fixed = fixture_bundle_by_sha256(bundle.get('sha256'))
    _public._require(_public._canonical(bundle)==_public._canonical(fixed),'fixture_bundle_not_fixed')
    return fixed


