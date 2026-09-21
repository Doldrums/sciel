"""Authored dense restart traces for exact context-budget admission regressions.

Only primitive arrays and collector phases are evaluated. No candidate or solver
runs. Distinct input/progress states prevent all six views sharing one payload.
"""
import math

import numpy as np

from sciel.eval import packing_geometry as geometry
from sciel.hypothesis import restart_trace as codec
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.paired_comparisons import build_comparison
from test_paired_comparisons import LIMITS


def _receipt(source, seed, *, variant=None):
    adapter = dict(schema='sciel-repeat-adapter/1', runtime={'authored': 'offline'})
    adapter['identity_sha256'] = canonical_sha(adapter)
    identity = dict(source_sha256=source, seed=seed,
        runtime_sha256=canonical_sha(adapter['runtime']), scaffold_sha256='d'*64,
        invocation_id=canonical_sha([source, seed])[:32])
    collector = codec.RestartTrace(identity)
    if variant is None:
        variant = seed + (0.5 if source.startswith('b') else 0.)
    centers = [[.125 + .15625*(i % 5), .125 + .15625*(i // 5)] for i in range(26)]
    best = np.array([value for center in centers for value in center] + [.02]*26)
    initial = best.copy()
    quality, improvements = math.fsum(best[52:]), 0
    for index in range(61):
        collector.begin_start(index, incumbent_sum=quality, improvements=improvements)
        if index % 4 == 0:
            noise = np.full(78, (variant+1)*2.**-30)
            ordinary = best + noise
            inputs = dict(best=best.copy(), initial=initial.copy(), noise=noise, ordinary=ordinary)
            collector.begin_helper(inputs)
            best = best.copy()
            best[52:] += (variant+index+1)*2.**-31
            quality, improvements = math.fsum(best[52:]), improvements+1
            collector.helper_end(raw_kind='value', raw_type='ndarray', raw_value=ordinary,
                post_inputs=inputs, effective=ordinary, origin='accepted_return',
                incumbent_sum=quality, improvements=improvements)
        best = best.copy()
        best[52:] += (variant+index+1)*2.**-33
        quality, improvements = math.fsum(best[52:]), improvements+1
        collector.optimizer_enter(incumbent_sum=quality, improvements=improvements)
        if index % 4 == 0:
            best = best.copy()
            best[52:] += (variant+index+1)*2.**-30
            best[:52] += (variant+index+1)*2.**-40
            quality, improvements = math.fsum(best[52:]), improvements+1
        collector.optimizer_exit(incumbent_sum=quality, improvements=improvements, result=best.copy())
        collector.end_start('completed', incumbent_sum=quality, improvements=improvements)
    trace = collector.finish('restart_limit', incumbent_sum=quality, improvements=improvements)
    capture = geometry.validate_packing(dict(
        centers=best[:52].reshape(26, 2).tolist(), radii=best[52:].tolist()))
    certificate = geometry.certify_packing(capture)
    assert capture['valid'] and certificate['status'] == 'certified'
    return dict(schema='sciel-repeat-draw/1', source_sha256=source, seed=seed,
        adapter_identity=adapter, limits=LIMITS, status='valid', capture=capture,
        certificate=certificate, raw_sum=capture['sum_radii'], certified_sum=certificate['sum_exact'],
        seed_applied=dict(python_random=seed, numpy_legacy=seed, python_hash_seed=str(seed),
            before_candidate_import=True, determinism_proven=False),
        started_at='2026-01-01T00:00:00+00:00', completed_at='2026-01-01T00:00:01+00:00',
        elapsed_seconds=1., candidate_elapsed_seconds=.5,
        certification_elapsed_seconds=.5, identity_elapsed_seconds=0.,
        restart_trace=dict(status='captured', reason=None, trace=trace,
            sidecar_sha256=canonical_sha(trace)))


def dense_group():
    pairs = [dict(seed=seed, comparator=_receipt('a'*64, seed),
                  challenger=_receipt('b'*64, seed)) for seed in (11, 12, 13)]
    first = pairs[0]['comparator']
    return build_comparison(namespace='fixture', batch_id='fixture', branch='p1',
        origin_experiment_id=None, comparator_source_sha256='a'*64,
        challenger_source_sha256='b'*64, adapter_identity=first['adapter_identity'],
        limits=LIMITS, pairs=pairs, provenance={'synthetic': True})
