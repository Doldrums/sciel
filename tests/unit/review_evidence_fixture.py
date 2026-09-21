"""Authored mixed raw/exact measurements and forecast disagreement witnesses."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from sciel.hypothesis.restart_optimizer_trace import RestartTrace

from sciel.eval import packing_geometry as geometry
from sciel.hypothesis import grounded_review
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.base import _snapshot
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.paired_comparisons import build_comparison
from test_restart_observations import synthetic_receipt, synthetic_adapter

SEEDS = (541090050, 3513246259, 4186712104)
NAMESPACE = 'authored-review-evidence'
BATCH = 'mixed-directions'
GROUP = 'comparison:' + canonical_sha([NAMESPACE, BATCH])
WRITER = dict(phase='reflection', reference=1, raw_response_sha256='d'*64)


def _trace(receipt, index):
    """Valid changed geometry and an explicitly missing second result."""
    centers = [[.125+.15625*(i % 5), .125+.15625*(i // 5)] for i in range(26)]
    best = np.array([v for pair in centers for v in pair] + [receipt['raw_sum']] + [0.]*25)
    effective = best.copy()
    effective[0] += .01
    inputs = dict(best=best, initial=best.copy(), noise=effective-best, ordinary=effective.copy())
    identity = dict(source_sha256=receipt['source_sha256'], seed=receipt['seed'],
        runtime_sha256=canonical_sha(receipt['adapter_identity']['runtime']),
        scaffold_sha256='b'*64, invocation_id='c'*32)
    collector = RestartTrace(identity)
    gain = .001 if index == 0 else 0.
    prior = receipt['raw_sum']-gain
    collector.begin_start(0, incumbent_sum=prior, improvements=0)
    collector.begin_helper(inputs)
    collector.helper_end(raw_kind='value', raw_type='ndarray', raw_value=effective,
        post_inputs=inputs, effective=effective, origin='accepted_return',
        incumbent_sum=prior, improvements=0)
    collector.optimizer_enter(incumbent_sum=prior, improvements=0)
    result = effective.copy()
    result[0] += .01
    collector.optimizer_exit(incumbent_sum=prior, improvements=0)
    collector.optimizer_outcome(True, SimpleNamespace(status=0))
    collector.optimizer_result(None if index == 1 else result)
    collector.end_start('completed', incumbent_sum=receipt['raw_sum'], improvements=int(gain > 0))
    trace = collector.finish('restart_limit', incumbent_sum=receipt['raw_sum'], improvements=int(gain > 0))
    return dict(status='captured', reason=None, trace=trace, sidecar_sha256=canonical_sha(trace))


def mixed_group(*, origin='r000002a1', challenger='b'*64, comparator='a'*64, metric=None, batch=BATCH):
    adapter = synthetic_adapter()
    pairs = []
    # The first two seeds disagree across metric axes; the last regresses on both.
    radii = ((.10000005, .10000002), (.10000002, .10000005), (.10000004, .10000001))
    for index, (seed, (left, right)) in enumerate(zip(SEEDS, radii)):
        pair = dict(seed=seed)
        for role, source, radius in (('comparator', comparator, left), ('challenger', challenger, right)):
            value = synthetic_receipt(source, seed, adapter_identity=adapter,
                mode='accepted', result='missing' if index == 1 else 'changed',
                raw_sum=radius, gain=.001 if index == 0 else 0.)
            # A tiny tolerance-valid boundary violation shrinks the strict certificate.
            if (index, role) in ((0, 'comparator'), (1, 'challenger')):
                capture = geometry.validate_packing(dict(
                    centers=[[.1, .5]] + [[0., 0.]]*25, radii=[radius]+[0.]*25))
                certificate = geometry.certify_packing(capture)
                value.update(capture=capture, certificate=certificate,
                    raw_sum=capture['sum_radii'], certified_sum=certificate['sum_exact'])
            value['restart_trace'] = _trace(value, index)
            pair[role] = value
        pairs.append(pair)
    first = pairs[0]['comparator']
    return build_comparison(namespace=NAMESPACE, batch_id=batch, branch='p1',
        origin_experiment_id=origin, comparator_source_sha256=comparator,
        challenger_source_sha256=challenger, adapter_identity=adapter, limits=first['limits'],
        pairs=pairs, provenance={'authored_fixture': True}, task_metric=metric)


def grounded_example():
    group = mixed_group()
    knowledge = KnowledgeBase()
    record = dict(id='r000002a1', branch='p1', status='evaluated',
        parent_score=.5, score=.4, actual_delta=.4-.5, source_sha256='b'*64,
        proposed_sha256='b'*64, parent_source_sha256='a'*64,
        hypothesis=dict(text='Explicit synthetic forecast.', reasoning='No causal assertion.',
            predicted_low=-.2, predicted_high=0., confidence=.5,
            restart_predictions=dict(schema='sciel-restart-predictions/1', expected=dict(
                effective_locations_changed=True, result_changed=False, accepted_interval_gain=False))))
    record['restart_prediction_binding'] = dict(schema='sciel-restart-prediction-binding/1',
        spec_sha256=canonical_sha(record['hypothesis']['restart_predictions']),
        response_sha256='c'*64, proposed_sha256='b'*64, source_sha256='b'*64)
    knowledge.record(record)
    records = {row['id']: _snapshot(row) for row in knowledge.experiments}
    seen = deepcopy(knowledge._seen)
    legacy = dict(comparison_id=group['id'], comparison_sha256=canonical_sha(group),
        interpretation='The signs differ between raw and exact measurements for all three seeds.',
        limitations='Synthetic false prose to test separation from measured facts.',
        next_hypothesis=dict(change='Try an alternative order.',
            expected_observation='A different result.', falsification='No change.'))
    return SimpleNamespace(group=group, records=records, seen=seen,
        writer=deepcopy(WRITER), old_proposal=legacy)


def reasoned_example():
    value = grounded_example()
    catalog = grounded_review.source_catalog(value.group, records=value.records, seen=value.seen)
    claims = [dict(fact_id=key, value=fact) for key, fact in sorted(catalog['facts'].items(), key=lambda item: item[0] != 'quality:all_valid')[:10]]
    proposal = dict(comparison_id=value.group['id'], comparison_sha256=canonical_sha(value.group),
        claims=claims, conjecture='Authored unverified explanation.',
        limitations='This fixture establishes no causal effect.', next_hypothesis=None)
    card = grounded_review.build_card(proposal, value.group, WRITER, records=value.records, seen=value.seen)
    return SimpleNamespace(group=value.group, groups=[value.group,
        mixed_group(batch='other-one'), mixed_group(batch='other-two')], records=value.records,
        seen=value.seen, proposal=proposal, old_card=card)
