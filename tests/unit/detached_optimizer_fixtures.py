"""Authored inert traces for matched-input and numerical outcome separation."""
import math
from types import SimpleNamespace

import numpy as np

from sciel.eval import packing_geometry as geometry
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.hypothesis import restart_optimizer_trace as codec
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.paired_comparisons import build_comparison
from test_optimizer_restart_observations import receipt


MATCHED_SEED = 3084817015
CONTROL_GAIN = 0.002881161755933004
CHALLENGER_GAIN = 0.002881170166179725


def measured_draw(source, seed, *, challenger):
    """Record literal arrays and phase values; never execute a numerical solver."""
    saved = receipt(source=source, seed=seed)
    identity = dict(source_sha256=source, seed=seed,
                    runtime_sha256=canonical_sha(saved['adapter_identity']['runtime']),
                    scaffold_sha256=builder.identity()['common_scaffold_sha256'],
                    invocation_id=canonical_sha([source, seed])[:32])
    collector = codec.RestartTrace(identity)
    best = np.array([coordinate for i in range(26)
                     for coordinate in (.125+(i % 5)*.15625, .125+(i // 5)*.15625)]
                    + [.03125]*26)
    best[0] = best[52]
    initial = best.copy()
    total, improvements = math.fsum(best[52:]), 0
    for index in range(25):
        quality = dict(incumbent_sum=total, improvements=improvements)
        collector.begin_start(index, **quality)
        effective = best.copy()
        if index % 4 == 0:
            inputs = dict(best=best.copy(), initial=initial.copy(),
                          noise=np.zeros(78), ordinary=best.copy())
            collector.begin_helper(inputs)
            collector.helper_end(raw_kind='value' if challenger else 'none',
                raw_type='ndarray' if challenger else 'NoneType',
                raw_value=effective if challenger else None, post_inputs=inputs,
                effective=effective,
                origin='accepted_return' if challenger else 'saved_ordinary_fallback',
                **quality)
        collector.optimizer_enter(**quality)
        collector.optimizer_exit(**quality)
        collector.optimizer_outcome(True, SimpleNamespace(status=0))
        # One recorded interval improves both arms. Future inputs then diverge.
        if seed == MATCHED_SEED and index == 8:
            best[52] += CHALLENGER_GAIN if challenger else CONTROL_GAIN
            best[0] = best[52]
            total, improvements = math.fsum(best[52:]), 1
        result = best.copy()
        result[0] -= 2.**-30  # Tolerance-valid, strictly outside one wall.
        if challenger and index // 4 < (4 if seed == MATCHED_SEED else 5):
            result[2] += .001
        collector.optimizer_result(result)
        collector.end_start('completed', incumbent_sum=total, improvements=improvements)
    trace = collector.finish('restart_limit', incumbent_sum=total, improvements=improvements)
    capture = geometry.validate_packing(dict(
        centers=best[:52].reshape(26, 2).tolist(), radii=best[52:].tolist()))
    certificate = geometry.certify_packing(capture)
    assert capture['valid'] and certificate['status'] == 'certified'
    saved.update(capture=capture, certificate=certificate, raw_sum=capture['sum_radii'],
                 certified_sum=certificate['sum_exact'],
                 restart_trace=dict(status='captured', reason=None, trace=trace,
                                    sidecar_sha256=canonical_sha(trace)))
    return saved


def matched_group():
    pairs = [dict(seed=seed,
                  comparator=measured_draw('a'*64, seed, challenger=False),
                  challenger=measured_draw('c'*64, seed, challenger=True))
             for seed in (17, MATCHED_SEED)]
    pairs.append(dict(seed=31, comparator=None, challenger=None))
    first = pairs[0]['comparator']
    return build_comparison(namespace='authored_optimizer_facts', batch_id='comparison-1',
        branch='inert_fixture', origin_experiment_id=None,
        comparator_source_sha256='a'*64, challenger_source_sha256='c'*64,
        adapter_identity=first['adapter_identity'], limits=first['limits'], pairs=pairs,
        provenance={'synthetic_only': True, 'candidate_optimizer_provider_calls': 0})
