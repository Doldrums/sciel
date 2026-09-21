"""Inert full-prefix traces for Research memory and checkpoint delivery tests."""
import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sciel.eval import packing_geometry as geometry
from sciel.eval.registry import load_task
from sciel.hypothesis import restart_motion as motion
from sciel.hypothesis import restart_observations as observations
from sciel.hypothesis import restart_optimizer_trace as current
from sciel.hypothesis import restart_trace as legacy
from sciel.hypothesis.measurement import metric_context
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.paired_comparisons import UNKNOWN_RECEIPT, build_comparison
from test_optimizer_restart_observations import receipt


BRANCH = 'authored_memory'
FIRST_SEED = 3102356664
LOCATION_SEED = 53186339
CENTER_DRIFTS = {16: 2.**-39, 32: 2.**-22, 48: 3*2.**-23}
LEGACY_CENTER_DRIFT = 2.**-36
LEGACY_RADIUS_DRIFT = 2.**-37


def task_metric():
    task = Path(__file__).resolve().parents[2] / 'tasks/oe/circle_packing'
    return metric_context(load_task(str(task)))


def source(name):
    return f'# Inert source label: {name}; never executed.\ndef run_packing():\n    return None\n'


def draw(code, seed, *, kind, challenger=False):
    digest = hashlib.sha256(code.encode()).hexdigest()
    saved = receipt(source=digest, seed=seed)
    identity = dict(source_sha256=digest, seed=seed,
        runtime_sha256=canonical_sha(saved['adapter_identity']['runtime']),
        scaffold_sha256='b'*64, invocation_id=canonical_sha([digest, seed])[:32])
    collector = (current if kind == 'location' else legacy).RestartTrace(identity)
    best = np.array([value for i in range(26)
                     for value in (.125+(i % 5)*.15625, .125+(i // 5)*.15625)] + [.03125]*26)
    if kind == 'location':
        best[0] = best[52] if challenger else best[52] - 2.**-23
        if challenger:
            best[52] -= 2.**-25
    quality = dict(incumbent_sum=math.fsum(best[52:]), improvements=0)
    for index in range(61):
        collector.begin_start(index, **quality)
        effective = best.copy()
        accepted = kind == 'legacy' or challenger and index in CENTER_DRIFTS
        if accepted and kind == 'location':
            effective[2] += .0078125
        if index % 4 == 0:
            inputs = dict(best=best.copy(), initial=best.copy(),
                          noise=np.zeros(78), ordinary=best.copy())
            collector.begin_helper(inputs)
            collector.helper_end(raw_kind='value' if accepted else 'none',
                raw_type='ndarray' if accepted else 'NoneType',
                raw_value=effective if accepted else None,
                post_inputs=inputs, effective=effective,
                origin='accepted_return' if accepted else 'saved_ordinary_fallback', **quality)
        collector.optimizer_enter(**quality)
        result = best.copy()
        if kind == 'legacy':
            result[2] += LEGACY_CENTER_DRIFT
            result[53] += LEGACY_RADIUS_DRIFT
            collector.optimizer_exit(result=result, **quality)
        else:
            if accepted:
                result[2] += CENTER_DRIFTS[index]
            collector.optimizer_exit(**quality)
            collector.optimizer_outcome(True, SimpleNamespace(status=0))
            collector.optimizer_result(result)
        collector.end_start('completed', **quality)
    trace = collector.finish('restart_limit', **quality)
    capture = geometry.validate_packing(dict(
        centers=best[:52].reshape(26, 2).tolist(), radii=best[52:].tolist()))
    certificate = geometry.certify_packing(capture)
    assert capture['valid'] and certificate['status'] == 'certified'
    saved.update(capture=capture, certificate=certificate, raw_sum=capture['sum_radii'],
                 certified_sum=certificate['sum_exact'], restart_trace=dict(
                     status='captured', reason=None, trace=trace, sidecar_sha256=canonical_sha(trace)))
    return saved


def incomplete_group(kind, *, comparator_name='control', unknown=False):
    first_seed = LOCATION_SEED if kind == 'location' else FIRST_SEED
    codes = dict(comparator=source(comparator_name), challenger=source(kind + '_challenger'))
    pair = dict(seed=first_seed, **{role: draw(code, first_seed, kind=kind,
                challenger=role == 'challenger') for role, code in codes.items()})
    pairs = [pair, dict(seed=17, comparator=None, challenger=None),
             dict(seed=1657835957, comparator=None,
                  challenger=copy.deepcopy(UNKNOWN_RECEIPT) if unknown else None)]
    first = pair['comparator']
    return build_comparison(namespace='authored_memory', batch_id=kind + ':' + comparator_name,
        branch=BRANCH, origin_experiment_id=None,
        comparator_source_sha256=first['source_sha256'],
        challenger_source_sha256=pair['challenger']['source_sha256'],
        adapter_identity=first['adapter_identity'], limits=first['limits'], pairs=pairs,
        source_pair=codes, provenance={'synthetic_only': True, 'candidate_optimizer_provider_calls': 0})


def view(saved):
    observed = observations.build_observation(saved, source_sha256=saved['source_sha256'],
        seed=saved['seed'], adapter_identity=saved['adapter_identity'])
    return observations.observation_view(observed)


def legacy_dataset():
    groups, views = {}, {}
    for name, control in [('pair_vs_control', 'control'), ('pair_vs_single', 'single')]:
        group = incomplete_group('legacy', comparator_name=control, unknown=True)
        groups[name] = group
        for role in ('comparator', 'challenger'):
            views[name, FIRST_SEED, role] = view(group['pairs'][0][role])
    return dict(metric=task_metric(), groups=groups, views=views)


def location_dataset():
    group = incomplete_group('location')
    views, motions = {}, {}
    for role in ('comparator', 'challenger'):
        saved = group['pairs'][0][role]
        views[role] = view(saved)
        motions[role] = motion.build_motion(saved['restart_trace']['trace'])
    pair = group['derived']['pairs'][0]
    assert pair['raw_delta'] < 0 < pair['certified_delta']['numerator']
    return dict(metric=task_metric(), group=group, views=views, motions=motions,
                first_seed=LOCATION_SEED)
