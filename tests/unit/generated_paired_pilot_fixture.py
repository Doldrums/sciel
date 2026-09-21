"""Authored off/on/control observations; no candidate, model or draw execution."""
from copy import deepcopy
import hashlib
import json
import numpy as np
from types import SimpleNamespace

from sciel.eval import packing_geometry as geometry
from sciel.hypothesis import restart_optimizer_trace as codec
from sciel.hypothesis.comparison_targets import _quality
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.paired_comparisons import build_comparison

from evaluated_study_fixture import adapter_identity, receipt, LIMITS
from test_generated_paired_history import proposal

SEEDS = (763606949, 1167415771, 23)
OFF_BODY = 'if restart_index % 8 == 0:\n    return np.copy(best_params)\nreturn np.copy(ordinary_params)\n'
ON_BODY = 'if restart_index % 12 == 0:\n    return np.copy(initial_params)\nreturn np.copy(ordinary_params)\n'


def _sha(source):
    return hashlib.sha256(source.encode()).hexdigest()


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def _terminal(source, adapter, scaffold, *, control=False, missing_result=False):
    value = receipt(_sha(source), SEEDS[0], adapter, scaffold, radius_delta=-1e-7, gain=0.)
    if control:
        # Raw tolerance accepts a tiny overlap. Its common strict decrement
        # exceeds the raw score advantage over the separated challenger.
        centers = [[.125 + .15625 * (i % 5), .125 + .15625 * (i // 5)] for i in range(26)]
        centers[1] = [centers[0][0] + .04 - 2e-7, centers[0][1]]
        capture = geometry.validate_packing(dict(centers=centers, radii=[.02] * 26))
        certificate = geometry.certify_packing(capture)
        assert capture['valid'] and certificate['status'] == 'certified'
        value.update(capture=capture, certificate=certificate,
                     raw_sum=capture['sum_radii'], certified_sum=certificate['sum_exact'])
    trace = _trace(value, scaffold, missing_result=missing_result)
    value['restart_trace'] = dict(status='captured', reason=None, trace=trace,
                                 sidecar_sha256=canonical_sha(trace))
    return value


def _trace(value, scaffold, *, missing_result):
    """Author 17 scheduled observations; retain 16, including seven accepts.

    These are collector inputs, not outputs from invoking the candidate. The
    unselected optimizer array differs from best; final incumbent stays fixed.
    """
    identity = dict(source_sha256=value['source_sha256'], seed=value['seed'],
        runtime_sha256=canonical_sha(value['adapter_identity']['runtime']),
        scaffold_sha256=scaffold,
        invocation_id=canonical_sha([value['source_sha256'], value['seed'], 'pilot'])[:32])
    collector = codec.RestartTrace(identity)
    centers = [[.125 + .15625 * (i % 5), .125 + .15625 * (i // 5)] for i in range(26)]
    initial = np.array([x for pair in centers for x in pair] + [.02] * 26)
    effective, result = initial.copy(), initial.copy()
    effective[0] += 1 / 128
    result[0] += 1 / 256
    quality = value['raw_sum']
    for index in range(65):
        collector.begin_start(index, incumbent_sum=quality, improvements=0)
        if index % 4 == 0:
            accepted = index < 28
            inputs = dict(best=initial.copy(), initial=initial.copy(),
                          noise=np.zeros(78), ordinary=initial.copy())
            collector.begin_helper(inputs)
            collector.helper_end(raw_kind='value' if accepted else 'none',
                raw_value=effective if accepted else None,
                raw_type='ndarray' if accepted else None, post_inputs=inputs,
                effective=effective if accepted else initial,
                origin='accepted_return' if accepted else 'saved_ordinary_fallback',
                incumbent_sum=quality, improvements=0)
        collector.optimizer_enter(incumbent_sum=quality, improvements=0)
        collector.optimizer_exit(incumbent_sum=quality, improvements=0)
        collector.optimizer_outcome(True, SimpleNamespace(status=0))
        collector.optimizer_result(None if missing_result else result)
        collector.end_start('completed', incumbent_sum=quality, improvements=0)
    return collector.finish('restart_limit', incumbent_sum=quality, improvements=0)


def authored_pilot(*, task_metric=None):
    from test_research_measurement_state import circle_metric
    from sciel.hypothesis import circle_optimizer_traced_body as builder

    metric = circle_metric() if task_metric is None else task_metric
    parent_source = builder.source_for()
    parent = dict(id='p1', code=parent_source, score=.5, cases={}, island=0,
                  origin='seed', parent_id=None)
    bundle = dict(parent=parent, arms={arm: {'research_id': 'r000003a1'} for arm in ('off', 'on')})
    manifest = dict(run_id='authored-paired-pilot', scope='Inert fixture, no physical execution claim.')
    generation = {'manifest.json': _wire(manifest), 'frozen/bundle.json': _wire(bundle)}
    proposals = {}
    for arm, body in (('off', OFF_BODY), ('on', ON_BODY)):
        item = proposal(arm, body, metric, parent_source)
        raw = json.loads(item['response']['text'])
        raw['hypothesis']['restart_predictions']['expected']['accepted_interval_gain'] = arm == 'on'
        item['response']['text'] = json.dumps(raw)
        item['provider_response']['choices'][0]['message']['content'] = item['response']['text']
        proposals[arm] = item
        generation.update({
            arm + '/response.json': _wire(item['response']),
            arm + '/provider_request.json': _wire(item['request']),
            arm + '/provider_return.json': _wire(item['provider_response']),
            arm + '/candidate.py': item['source'].encode(),
            arm + '/choice.json': _wire(dict(proof=item['proof'])),
            arm + '/preparation.jsonl': _wire(dict(phase='prepared', fixture=True)) + b'\n',
        })
    adapter = adapter_identity()
    scaffold = builder.identity()['common_scaffold_sha256']
    sources = dict(control=parent_source, **{key: row['source'] for key, row in proposals.items()})
    terminals = {key: _terminal(source, adapter, scaffold, control=key == 'control',
                               missing_result=key == 'on') for key, source in sources.items()}
    groups = {}
    for name, left, right in (('on_off', 'off', 'on'), ('on_control', 'control', 'on'),
                              ('off_control', 'control', 'off')):
        groups[name] = build_comparison(namespace='authored-paired-pilot', batch_id=name,
            branch=name, origin_experiment_id=None,
            comparator_source_sha256=_sha(sources[left]), challenger_source_sha256=_sha(sources[right]),
            adapter_identity=adapter, limits=LIMITS,
            pairs=[dict(seed=seed, comparator=deepcopy(terminals[left]) if index == 0 else None,
                        challenger=deepcopy(terminals[right]) if index == 0 else None)
                   for index, seed in enumerate(SEEDS)],
            source_pair=dict(comparator=sources[left], challenger=sources[right]), task_metric=metric,
            provenance={'scope': 'Authored primitive test observations; no dispatch or payment assertion.'})
    numeric = {'comparisons/' + name + '.json': _wire(group) for name, group in groups.items()}
    numeric['summary.json'] = _wire(dict(issued_draw_intents=3,
        primary_on_minus_off=_quality(groups['on_off']),
        secondary_on_minus_control=_quality(groups['on_control'])))
    def read_generation(name):
        # Pure projection consumers never construct a controller. Only mixed
        # history integration asks for these genuine inert G/C/R journals.
        if name in ('frozen/capsule.json', 'frozen/config.json') and name not in generation:
            from research_history_fixture import build_history
            ordinary, final = build_history(task_metric=metric)
            assert ordinary['root_parent'] == dict(id=parent['id'], source_sha256=_sha(parent_source))
            generation['frozen/capsule.json'] = _wire(ordinary)
            generation['frozen/config.json'] = _wire(final['config'])
        return json.loads(generation[name])

    return SimpleNamespace(generation=generation, numeric=numeric, bundle=bundle, groups=groups,
        metric=metric, gj=read_generation, nj=lambda name: json.loads(numeric[name]))
