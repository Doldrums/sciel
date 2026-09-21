"""Authored offline study evidence: no archives, runtime probes or numerical search.

Literal geometry and collector calls exercise production sealing/validation. These
synthetic observations are test data, never historical measurements or new results.
"""
import copy
import hashlib
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from sciel.eval import packing_geometry as geometry
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.hypothesis import restart_optimizer_trace as codec
from sciel.hypothesis.comparison_targets import _quality
from sciel.knowledge_base import evaluated_detached_study as core
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.generated_paired_history import certificate_policy
from sciel.knowledge_base.paired_comparisons import build_comparison

SEEDS = (3084817015, 17, 23)
COMPARATOR_GAIN = 2.**-12
CHALLENGER_GAIN = 2.**-11
RATIO = dict(numerator=219306857579, denominator=72057594037927936)
LIMITS = dict(candidate_wall_seconds=30., certification_wall_seconds=10.,
    candidate_cpu_seconds=35, certification_cpu_seconds=35,
    candidate_mem_bytes=2**31, certification_mem_bytes=2**31)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def text(value):
    return canonical_json(value) + '\n'


def adapter_identity(*, runtime_config=None):
    policy = certificate_policy()
    value = dict(schema='sciel-repeat-adapter/1', runtime={'kind': 'offline fixture'},
        runtime_config=copy.deepcopy(runtime_config) if runtime_config is not None else {'kind': 'offline fixture'},
        certificate_policy=policy,
        certificate_policy_sha256=canonical_sha(policy))
    return dict(value, identity_sha256=canonical_sha(value))


def metric():
    from test_research_measurement_state import circle_metric
    return circle_metric()


def receipt(source, seed, adapter, scaffold, *, radius_delta=0., gain=0., calls=3):
    """Record fixed arrays and progress values; do not run source or an optimizer."""
    centers = [[.125 + .15625*(i % 5), .125 + .15625*(i // 5)] for i in range(26)]
    initial = np.array([x for pair in centers for x in pair] + [.02]*26)
    result = initial.copy()
    result[52] = float(Fraction(.02) + Fraction(radius_delta))
    capture = geometry.validate_packing(dict(centers=centers, radii=result[52:].tolist()))
    certificate = geometry.certify_packing(capture)
    assert capture['valid'] and certificate['status'] == 'certified'
    identity = dict(source_sha256=source, seed=seed, runtime_sha256=canonical_sha(adapter['runtime']),
        scaffold_sha256=scaffold, invocation_id=canonical_sha([source, seed, 'fixture'])[:32])
    collector = codec.RestartTrace(identity)
    quality, improvements = capture['sum_radii'] - gain, 0
    for index in range(4*(calls-1)+1):
        collector.begin_start(index, incumbent_sum=quality, improvements=improvements)
        if index % 4 == 0:
            inputs = dict(best=initial.copy(), initial=initial.copy(), noise=np.zeros(78), ordinary=initial.copy())
            collector.begin_helper(inputs)
            collector.helper_end(raw_kind='value', raw_type='ndarray', raw_value=initial,
                post_inputs=inputs, effective=initial, origin='accepted_return',
                incumbent_sum=quality, improvements=improvements)
        collector.optimizer_enter(incumbent_sum=quality, improvements=improvements)
        if index == 8 and gain:
            quality += gain
            improvements += 1
        collector.optimizer_exit(incumbent_sum=quality, improvements=improvements)
        collector.optimizer_outcome(True, SimpleNamespace(status=0))
        collector.optimizer_result(result)
        collector.end_start('completed', incumbent_sum=quality, improvements=improvements)
    trace = collector.finish('restart_limit', incumbent_sum=quality, improvements=improvements)
    return dict(schema='sciel-repeat-draw/1', status='valid', source_sha256=source, seed=seed,
        adapter_identity=adapter, limits=LIMITS, started_at='2026-01-01T00:00:00+00:00',
        completed_at='2026-01-01T00:00:01+00:00', elapsed_seconds=1.,
        candidate_elapsed_seconds=.5, certification_elapsed_seconds=.5, identity_elapsed_seconds=0.,
        seed_applied=dict(python_random=seed, numpy_legacy=seed, python_hash_seed=str(seed),
            before_candidate_import=True, determinism_proven=False),
        raw_sum=capture['sum_radii'], certified_sum=certificate['sum_exact'],
        capture=capture, certificate=certificate, restart_trace=dict(status='captured', reason=None,
            trace=trace, sidecar_sha256=canonical_sha(trace)))


def authoring():
    control = builder.source_for()
    base = builder.source_for('if restart_index % 8 == 0:\n    return np.copy(best_params)\nreturn None\n')
    body = 'if restart_index % 8 == 0:\n    return np.copy(ordinary_params)\nreturn None\n'
    built = builder.apply(base, body)
    sources = {digest(source): source for source in (control, base, built['code'])}
    binding = dict(edit_parent_source_sha256=digest(base),
        assembled_candidate_source_sha256=digest(built['code']),
        primary_control_source_sha256=digest(control), body_sha256=digest(body))
    design = text(dict(schema=core.HOST_DESIGN_SCHEMA, authorship='Explicit offline fixture',
        hypothesis=dict(mechanism='Synthetic contrast: no scientific claim.'), bindings=binding))
    dependencies = dict(kind='HOST', design_text=design, proof_text=text(built['proof']))
    author = dict(kind='HOST', artifact_sha256=digest(design),
        base_source_sha256=digest(base), candidate_source_sha256=digest(built['code']),
        comparator_source_sha256=digest(control))
    return author, sources, dependencies, built['proof']


def build(*, task_metric=None):
    author, sources, dependencies, proof = authoring()
    adapter = adapter_identity()
    task_metric = metric() if task_metric is None else task_metric
    producer = core.HOST_PRODUCERS[0]
    roles = dict(comparator='control', challenger='candidate')
    source_map = dict(control=author['comparator_source_sha256'], candidate=author['candidate_source_sha256'])
    schedule = [dict(index=2*i+j, seed=seed, role=role) for i, seed in enumerate(SEEDS)
        for j, role in enumerate(('control', 'candidate') if i % 2 == 0 else ('candidate', 'control'))]
    addendum = dict(schema=producer, schedule=schedule, limits=LIMITS, runtime=adapter['runtime_config'],
        execution_profile=core.PROFILE, restart_trace=True, no_resume=True, no_retry=True,
        max_candidate_calls=6, max_certifier_calls=6, quality_gate=core.QUALITY_GATE, automatic_promotion=False)
    manifest = dict(schema=producer, source_sha256=source_map, adapter_identity=adapter,
        numeric_addendum=addendum, source_proof_sha256=canonical_sha(proof),
        design_sha256=author['artifact_sha256'], files={
            **{'candidates/'+role+'.py': sha for role, sha in source_map.items()},
            'design.json': digest(dependencies['design_text']),
            'source_proof.json': digest(dependencies['proof_text'])})
    manifest['payload_sha256'] = canonical_sha(manifest)
    manifest_text = text(manifest)
    manifest_sha = digest(manifest_text)
    started_text = text(dict(manifest_sha256=manifest_sha, no_resume=True, timestamp=1))
    receipts, slots, summary_slots = {}, [], []
    artifacts = {'manifest.json': manifest_sha, 'started.json': digest(started_text),
        **{'frozen/'+name: sha for name, sha in manifest['files'].items()}}
    for item in schedule:
        slot = dict(item, **dict.fromkeys(('intent_text', 'raw_receipt_text', 'validated_text', 'unknown_text')))
        if item['index'] < 4:
            is_challenger = item['role'] == 'candidate'
            delta = (float(Fraction(**RATIO)) if item['seed'] == SEEDS[0] else -.001) if is_challenger else 0.
            value = receipt(source_map[item['role']], item['seed'], adapter, proof['common_scaffold_sha256'],
                radius_delta=delta, gain=CHALLENGER_GAIN if is_challenger else COMPARATOR_GAIN,
                calls=17 if item['index'] == 0 else 3)
            receipts[(item['seed'], item['role'])] = value
            intent = dict(item, schema=producer, manifest_sha256=manifest_sha,
                source_sha256=source_map[item['role']], adapter_identity_sha256=adapter['identity_sha256'],
                limits=LIMITS, runtime=adapter['runtime_config'], restart_trace=True)
            raw = text(value)
            slot.update(intent_text=text(intent), raw_receipt_text=raw,
                validated_text=text(dict(status='valid', raw_receipt_sha256=digest(raw))))
            summary_slots.append(dict(item, status='valid', intent_sha256=digest(slot['intent_text']),
                raw_receipt_sha256=digest(raw)))
            prefix = f"draws/{item['index']:02d}_{item['seed']}_{item['role']}/"
            for key, name in (('intent_text', 'intent.json'), ('raw_receipt_text', 'raw_receipt.json'),
                              ('validated_text', 'validated.json')):
                artifacts[prefix+name] = digest(slot[key])
        slots.append(slot)
    group = build_comparison(namespace='synthetic-host:'+manifest_sha, batch_id='host_control', branch='host_control',
        origin_experiment_id=None, comparator_source_sha256=source_map['control'],
        challenger_source_sha256=source_map['candidate'], adapter_identity=adapter, limits=LIMITS,
        pairs=[dict(seed=seed, comparator=receipts.get((seed, 'control')),
                    challenger=receipts.get((seed, 'candidate'))) for seed in SEEDS],
        provenance=dict(schema=producer, numeric_manifest_sha256=manifest_sha, schedule=schedule, detached=True,
            authorship='HOST', numeric_addendum_sha256=canonical_sha(addendum), source_proof_sha256=canonical_sha(proof)),
        task_metric=task_metric, source_pair=dict(comparator=sources[source_map['control']],
            challenger=sources[source_map['candidate']]))
    summary = dict(schema=producer, manifest_sha256=manifest_sha, status='early_stopped_known_host_loss',
        errors=[], evidence_integrity_verified=True, slots=summary_slots, artifacts=artifacts,
        planned_candidate_allocations=6, issued_draw_intents=4, validated_terminal_receipts=4,
        remaining_unissued_slots=schedule[4:], quality=_quality(group), group_sha256=canonical_sha(group),
        comparison_sha256={group['batch_id']: canonical_sha(group)}, quality_gate_passed=False,
        automatic_promotion=False, resume_allowed=False, provider_calls=0, new_api_cost_usd='0')
    capsule = dict(schema=core.SCHEMA, profile=core.PROFILE, metric=task_metric,
        receiving_root=dict(id='p1', source_sha256=source_map['control']), validation_policy=core.identity(),
        author=author, source_registry=sources, comparison=group,
        study=dict(manifest_text=manifest_text, summary_text=text(summary), started_text=started_text,
            role_map=roles, slots=slots))
    authority = dict(schema=core.AUTHORITY_SCHEMA, kind='HOST', dependencies_sha256=canonical_sha(dependencies),
        producer_manifest_sha256=manifest_sha, producer_summary_sha256=digest(text(summary)),
        author_artifact_sha256=author['artifact_sha256'], generation_config_sha256=None,
        host_decision_sha256=None, issuer='Authored offline test fixture')
    return dict(capsule=capsule, author_dependencies=dependencies, author_authority=authority,
        capsule_sha256=canonical_sha(capsule), author_authority_sha256=canonical_sha(authority))
