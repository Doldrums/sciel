"""Validate detached-study authority and retain authenticated study ancestry."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from .catalogs import canonical_sha
from .generated_paired_history import _bounded, certificate_policy
from .paired_comparisons import UNKNOWN_RECEIPT, validate_comparison
from ..hypothesis import circle_optimizer_traced_body as builder
from ..hypothesis import restart_motion_observations as observations
from ..hypothesis import restart_predictions as predictions
from ..hypothesis.comparison_targets import QUALITY_GATE, _quality, evaluate_target
from ..hypothesis.measurement import validate_metric_context
from .study_lifecycle import (
    _lifecycle,
)
from .study_validation import (
    AUTHORITY_SCHEMA,
    EVALUATED_G_PRODUCER,
    FACTS_SCHEMA,
    G_PRODUCER,
    HOST_DESIGN_SCHEMA,
    HOST_PRODUCERS,
    MAX_EVALUATED_ANCESTRY,
    PROFILE,
    SCHEMA,
    SCOPE,
    _AUTHORITY_FIELDS,
    _AUTHOR_FIELDS,
    _SLOT_FIELDS,
    _VALIDATION_SESSION,
    _digest,
    _object,
    _require,
    _same,
    _sha,
)


def identity():
    from ..engine.external_research_v2 import policy_identity
    return dict(schema=SCHEMA, authority_schema=AUTHORITY_SCHEMA, profile=PROFILE,
        production_policy_sha256=canonical_sha(policy_identity()), builder=builder.identity(),
        certificate_policy=certificate_policy(), maximum_studies=1, maximum_slots=6,
        maximum_evaluated_ancestry=MAX_EVALUATED_ANCESTRY, scope=SCOPE)

@contextmanager
def _validation_scope():
    """Share successful immutable validation only within one public call chain."""
    if _VALIDATION_SESSION.get() is not None:
        yield
        return
    token = _VALIDATION_SESSION.set(dict(cache={}, active=set()))
    try:
        yield
    finally:
        _VALIDATION_SESSION.reset(token)

def _ancestry_bound(deps):
    depth = 0
    while type(deps) is dict and deps.get('kind') == 'evaluated_target_G':
        depth += 1
        _require(depth <= MAX_EVALUATED_ANCESTRY, 'evaluated generation ancestry limit')
        bundle = deps.get('source_study_bundle')
        _require(type(bundle) is dict and set(bundle) == {'capsule','author_dependencies','author_authority'},
                 'evaluated generation source bundle fields')
        deps = bundle['author_dependencies']
    return depth

def validate_authority(value):
    """Shape only. The caller's frozen configuration, not this object, authorizes it."""
    _bounded(value)
    _require(type(value) is dict and set(value) == _AUTHORITY_FIELDS and value['schema'] == AUTHORITY_SCHEMA,
             'authority fields/schema')
    _require(value['kind'] in ('HOST', 'component_target_G', 'evaluated_target_G'), 'authority kind')
    for key in ('dependencies_sha256', 'producer_manifest_sha256', 'producer_summary_sha256', 'author_artifact_sha256'):
        _require(_sha(value[key]), 'authority ' + key)
    for key in ('generation_config_sha256', 'host_decision_sha256'):
        _require(value[key] is None if value['kind'] == 'HOST' else _sha(value[key]), 'authority ' + key)
    _require(type(value['issuer']) is str and 0 < len(value['issuer']) <= 128 and value['issuer'].strip(),
             'authority issuer')
    return copy.deepcopy(value)

def _author(capsule, deps, authority, manifest):
    author = capsule['author']
    _require(type(author) is dict and set(author) == _AUTHOR_FIELDS and author['kind'] == authority['kind'],
             'author fields/kind')
    _require(all(_sha(author[k]) for k in _AUTHOR_FIELDS - {'kind'}), 'author source/artifact digest')
    _require(type(deps) is dict and deps.get('kind') == author['kind']
             and canonical_sha(deps) == authority['dependencies_sha256'], 'external dependency authority differs')
    sources = capsule['source_registry']
    needed = {capsule['receiving_root']['source_sha256'], *(author[k] for k in
              ('base_source_sha256', 'candidate_source_sha256', 'comparator_source_sha256'))}
    _require(type(sources) is dict and set(sources) == needed, 'exact source registry required')
    for digest, source in sources.items():
        _require(_sha(digest) and type(source) is str and len(source.encode('utf-8')) <= 131072
                 and _digest(source) == digest, 'source registry binding differs')
        _require(builder.source_for(builder.body_of(source)) == source, 'source outside exact body family')
    base, candidate = (sources[author[key]] for key in ('base_source_sha256', 'candidate_source_sha256'))
    # Preserve the authored body bytes (including absence of a final LF) in the
    # edit proof; body_of is only the canonical inverse of the compiled source.
    proposed_body = (_object(deps['proof_text'], 'HOST proof').get('target_body')
                     if author['kind'] == 'HOST' else deps['generation_result'].get('body'))
    built = builder.apply(base, proposed_body)
    _require(built['code'] == candidate, 'candidate derivation differs')
    _require(author['candidate_source_sha256'] != author['comparator_source_sha256'], 'duplicate contrast source')
    if author['kind'] == 'HOST':
        _require(set(deps) == {'kind', 'design_text', 'proof_text'}, 'HOST dependency fields')
        design = _object(deps['design_text'], 'HOST design')
        proof = _object(deps['proof_text'], 'HOST proof')
        _require(author['artifact_sha256'] == authority['author_artifact_sha256'] == _digest(deps['design_text']),
                 'HOST design authority differs')
        _require(_same(proof, built['proof']), 'HOST source proof differs')
        binding = design.get('bindings')
        _require(type(binding) is dict and design.get('schema') == HOST_DESIGN_SCHEMA
            and type(design.get('authorship')) is str and bool(design['authorship'].strip())
            and binding.get('edit_parent_source_sha256') == author['base_source_sha256']
            and binding.get('assembled_candidate_source_sha256') == author['candidate_source_sha256']
            and binding.get('primary_control_source_sha256') == author['comparator_source_sha256']
            and binding.get('body_sha256') == _digest(proposed_body), 'HOST declared source bindings differ')
        files = manifest.get('files', {})
        _require(manifest.get('design_sha256') == author['artifact_sha256']
            and files.get('design.json') == _digest(deps['design_text'])
            and files.get('source_proof.json') == _digest(deps['proof_text'])
            and manifest.get('source_proof_sha256') == canonical_sha(proof), 'HOST frozen author files differ')
        return dict(**copy.deepcopy(author), generation_id=None,
            unverified_hypothesis=copy.deepcopy(design.get('hypothesis')), native_forecast=None), None, proof
    evaluated = author['kind'] == 'evaluated_target_G'
    expected_fields = ({'kind','generation_result','source_study_bundle','reflection_config','review_sidecar',
        'generation_config','host_decision','generation_acceptance','numeric_protocol'} if evaluated else
        {'kind','generation_result','source_component','review_sidecar','generation_config'})
    _require(set(deps) == expected_fields, 'target G dependency fields')
    result = deps['generation_result']
    _require(author['artifact_sha256'] == authority['author_artifact_sha256'] == canonical_sha(result)
        and canonical_sha(deps['generation_config']) == authority['generation_config_sha256'],
        'external generation/result configuration authority differs')
    if evaluated:
        from ..engine.evaluated_target_generation import validate_result
        result = validate_result(result, bundle=deps['source_study_bundle'],
            reflection_config=deps['reflection_config'], review_sidecar=deps['review_sidecar'],
            expected_sha256=authority['author_artifact_sha256'], config=deps['generation_config'])
        decision = deps['host_decision']
        _require(type(decision) is dict and decision.get('decision') == 'accepted'
            and canonical_sha(decision) == authority['host_decision_sha256']
            and any(_same(decision, saved) for saved in
                deps['generation_config']['comparison_review_text_policy']['decisions']),
            'separate accepted reflection decision differs')
        _require(_same(deps['source_study_bundle']['capsule']['metric'], capsule['metric']),
                 'source study metric differs')
        from ..hypothesis.evaluated_generation_acceptance import validate_decision
        validate_decision(deps['generation_acceptance'], generation_result=result,
            numeric_protocol=deps['numeric_protocol'], expected_sha256=canonical_sha(deps['generation_acceptance']))
        _require(_same(manifest.get('numeric_addendum'), deps['numeric_protocol']),
                 'accepted numeric protocol differs from executed addendum')
        for field, key in (('source_study_bundle_sha256','source_study_bundle'),
                ('reflection_config_sha256','reflection_config'), ('review_sidecar_sha256','review_sidecar'),
                ('generation_acceptance_sha256','generation_acceptance'), ('numeric_protocol_sha256','numeric_protocol')):
            _require(manifest.get(field) == canonical_sha(deps[key]), 'numeric evaluated G '+field+' differs')
    else:
        from ..engine.component_target_generation import validate_result
        result = validate_result(result, component=deps['source_component'], review_sidecar=deps['review_sidecar'],
            expected_sha256=authority['author_artifact_sha256'], config=deps['generation_config'])
    _require(result['eligible'] is True and result['status'] == 'eligible'
        and result['plan']['host_decision_sha256'] == authority['host_decision_sha256'], 'accepted generation required')
    target = result['plan']['target']
    _require(result['source_sha256'] == author['candidate_source_sha256'] and result['code'] == candidate
        and target['base_source_sha256'] == author['base_source_sha256']
        and target['comparator_source_sha256'] == author['comparator_source_sha256']
        and _same(target['receiving_root'], capsule['receiving_root']) and _same(result['proof'], built['proof']),
        'generation target/source/root differs')
    _require(manifest.get('generation_result_sha256') == author['artifact_sha256']
        and manifest.get('generation_config_sha256') == authority['generation_config_sha256']
        and manifest.get('host_decision_sha256') == authority['host_decision_sha256']
        and manifest.get('source_proof_sha256') == canonical_sha(result['proof']), 'numeric G authorship differs')
    return dict(**copy.deepcopy(author), generation_id=result['generation_id'],
        unverified_hypothesis=copy.deepcopy(result['hypothesis']), native_forecast=None), result, built['proof']

def _validate(capsule, expected_sha256, metric, root, deps, authority):
    _bounded([capsule, deps, authority])
    _ancestry_bound(deps)
    with _validation_scope():
        session = _VALIDATION_SESSION.get()
        key = canonical_sha(dict(capsule=capsule, expected_sha256=expected_sha256,
            metric=metric, root=root, dependencies=deps, authority=authority, policy=identity()))
        _require(key not in session['active'], 'cyclic evaluated generation validation')
        if key in session['cache']:
            return copy.deepcopy(session['cache'][key])
        session['active'].add(key)
        try:
            result = _validate_uncached(capsule, expected_sha256, metric, root, deps, authority)
            session['cache'][key] = copy.deepcopy(result)
            return copy.deepcopy(result)
        finally:
            session['active'].remove(key)

def _validate_uncached(capsule, expected_sha256, metric, root, deps, authority):
    _require(_sha(expected_sha256) and canonical_sha(capsule) == expected_sha256, 'configured study digest differs')
    _require(type(capsule) is dict and set(capsule) == {'schema','profile','metric','receiving_root',
        'validation_policy','author','source_registry','comparison','study'} and capsule['schema'] == SCHEMA,
        'capsule fields/schema')
    _require(capsule['profile'] == PROFILE and _same(capsule['validation_policy'], identity()), 'receiving policy/profile differs')
    checked_metric = validate_metric_context(metric)
    _require(_same(capsule['metric'], checked_metric) and checked_metric['task_name'] == 'oe/circle_packing'
             and checked_metric['primary_metric'] == 'combined_score', 'metric differs')
    receiving = capsule['receiving_root']
    _require(type(receiving) is dict and set(receiving) == {'id','source_sha256'}
        and type(receiving['id']) is str and 0 < len(receiving['id']) <= 128 and receiving['id'].strip()
        and receiving['source_sha256'] == _digest(builder.source_for())
        and (root is None or _same(receiving, root)), 'receiving root differs')
    authority = validate_authority(authority)
    study = capsule['study']
    _require(type(study) is dict and set(study) == {'manifest_text','summary_text','started_text','role_map','slots'},
             'study fields')
    manifest, summary, started = [_object(study[k+'_text'], k) for k in ('manifest','summary','started')]
    _require(_digest(study['manifest_text']) == authority['producer_manifest_sha256']
        and _digest(study['summary_text']) == authority['producer_summary_sha256'], 'external producer authority differs')
    author, generated, proof = _author(capsule, deps, authority, manifest)
    group = validate_comparison(capsule['comparison'])
    _require(group['origin_experiment_id'] is None and _same(group['task_metric'], checked_metric)
        and group['comparator_source_sha256'] == author['comparator_source_sha256']
        and group['challenger_source_sha256'] == author['candidate_source_sha256']
        and _same(group.get('source_pair'), dict(comparator=capsule['source_registry'][author['comparator_source_sha256']],
            challenger=capsule['source_registry'][author['candidate_source_sha256']])), 'comparison source/origin differs')
    adapter = group['adapter_identity']; certificate = certificate_policy()
    _require(adapter.get('schema') == 'sciel-repeat-adapter/1'
        and adapter.get('identity_sha256') == canonical_sha({k:v for k,v in adapter.items() if k != 'identity_sha256'})
        and _same(adapter.get('certificate_policy'), certificate)
        and adapter.get('certificate_policy_sha256') == canonical_sha(certificate), 'trusted certificate policy differs')
    coverage = _lifecycle(capsule, authority, manifest, summary, started)
    return author, generated, proof, coverage

def validate_capsule(capsule, *, expected_sha256, metric, root=None, author_dependencies, author_authority):
    try:
        _validate(capsule, expected_sha256, metric, root, author_dependencies, author_authority)
        return copy.deepcopy(capsule)
    except (KeyError, TypeError, AttributeError, OverflowError, UnicodeError, RecursionError, SyntaxError) as exc:
        raise ValueError('malformed evaluated detached study') from exc

def _retained_ancestry(bundle):
    """Collect consumed history only after complete bundle authentication.

    Internal callers must first validate this exact bundle. The registry here
    does not authorize edit bases, targets, measurements or provider freshness;
    it only prevents reuse of sources and response IDs actually retained in the
    authenticated dependencies. Older R sidecars without SDK IDs add no ID.
    """
    _ancestry_bound(bundle['author_dependencies'])
    sources, provider_ids = {}, set()
    while True:
        sources.update(bundle['capsule']['source_registry'])
        deps = bundle['author_dependencies']
        if deps['kind'] == 'HOST':
            break
        provider_ids.add(deps['generation_result']['provider_response']['id'])
        if deps['kind'] == 'evaluated_target_G':
            provider_ids.add(deps['review_sidecar']['provider_response']['id'])
            bundle = deps['source_study_bundle']
            continue
        # generated-paired history retains complete parent/generated sources
        # and SDK IDs for each G. Its legacy R sidecar retains no SDK ID.
        for proposal in deps['source_component']['proposals']:
            for source in (proposal['parent_source'], proposal['source']):
                sources[_digest(source)] = source
            provider_ids.add(proposal['provider_response']['id'])
        break
    return dict(source_registry=copy.deepcopy(dict(sorted(sources.items()))),
                provider_response_ids=sorted(provider_ids))

def retained_ancestry(bundle):
    """Authenticate one bounded bundle and return its detached consumed history.

    Authentication establishes consistency with the supplied authority. The
    receiving caller must independently pin that authority, as for this module's
    other public validators. No persistent validation or authority cache exists.
    """
    _bounded(bundle)
    _require(type(bundle) is dict and set(bundle) ==
        {'capsule', 'author_dependencies', 'author_authority'}, 'portable bundle fields')
    capsule = bundle['capsule']
    with _validation_scope():
        validate_capsule(capsule, expected_sha256=canonical_sha(capsule), metric=capsule['metric'],
            author_dependencies=bundle['author_dependencies'], author_authority=bundle['author_authority'])
        return _retained_ancestry(bundle)

def derive_facts(capsule, *, author_dependencies, author_authority):
    """Rich saved observations; context owners compact, never reinterpret, them."""
    author, generated, proof, coverage = _validate(capsule, canonical_sha(capsule), capsule['metric'], None,
                                                  author_dependencies, author_authority)
    group = capsule['comparison']; quality = _quality(group)
    paired, spec = None, None
    if generated is not None:
        hypothesis = generated['hypothesis']; spec = hypothesis['restart_predictions']
        forecast = hypothesis['paired_forecast']
        observed = (False if quality['verdict'] == 'REFUTED' else
                    True if quality['verdict'] == 'CONFIRMED' and coverage['producer_integrity_verified'] else None)
        paired = dict(spec=copy.deepcopy(forecast), spec_sha256=canonical_sha(forecast),
            target_sha256=forecast['target_sha256'], comparison_sha256=canonical_sha(group),
            expected=forecast['expected_gate_pass'], observed=observed,
            verdict='INCONCLUSIVE' if observed is None else 'CONFIRMED' if observed is forecast['expected_gate_pass'] else 'REFUTED',
            quality_verdict=quality['verdict'], producer_integrity_verified=coverage['producer_integrity_verified'])
        target = generated['plan']['target']
        legacy = dict(schema='sciel-comparison-target/1', **{k:copy.deepcopy(target[k]) for k in
            ('base_source_sha256','comparator_source_sha256','change','quality_gate','diagnostics')})
        saved = evaluate_target(legacy, group, candidate_source_sha256=author['candidate_source_sha256'])
        # Preserve the existing saved-data evaluator exactly. Its positive
        # arithmetic agreement alone cannot clear a failed producer lifecycle.
        verdict = saved['combined_verdict']
        if verdict == 'CONFIRMED' and not coverage['producer_integrity_verified']:
            verdict = 'INCONCLUSIVE'
        paired['target_evaluation'] = dict(schema='sciel-evaluated-detached-target/1',
            original_target_sha256=canonical_sha(target),
            projected_legacy_target_sha256=saved['target_sha256'],
            producer_integrity_verified=coverage['producer_integrity_verified'],
            verdict=verdict, saved_data=saved)
    restart = []
    for role in ('comparator','challenger'):
        source = group[role+'_source_sha256']; role_spec = spec if role == 'challenger' else None
        rows = []
        for pair in group['pairs']:
            receipt = pair[role]; observation = prediction = None
            reason = 'not_dispatched' if receipt is None else 'dispatch_unknown' if _same(receipt, UNKNOWN_RECEIPT) else None
            if reason is None:
                failure_reason = 'diagnostic_observation_invalid_or_unsupported'
                try:
                    observation = observations.build_observation(receipt, source_sha256=source, seed=pair['seed'],
                        adapter_identity=group['adapter_identity'], scaffold_sha256=proof['common_scaffold_sha256'])
                    if role_spec is not None:
                        failure_reason = 'diagnostic_prediction_invalid_or_unsupported'
                        prediction = predictions.evaluate(role_spec, observation)
                    if observation['status'] != 'available':
                        reason = observation['reason'] or 'diagnostic_unavailable'
                except (ValueError, KeyError, TypeError, OverflowError):
                    reason = failure_reason
            rows.append(dict(seed=pair['seed'], receipt_sha256=canonical_sha(receipt) if receipt is not None else None,
                observation=observation, prediction=prediction, unavailable_reason=reason))
        restart.append(dict(role=role, source_sha256=source, forecast=copy.deepcopy(role_spec), rows=rows))
    return dict(schema=FACTS_SCHEMA, study_sha256=canonical_sha(capsule), receiving_root=copy.deepcopy(capsule['receiving_root']),
        author=author, source_registry=copy.deepcopy(capsule['source_registry']), comparison=copy.deepcopy(group),
        quality=quality, coverage=coverage, native=dict(observation_status='UNMEASURED', forecast_status='NOT_DECLARED',
            score=None, ordinary_origin=None), paired_forecast=paired, restart=restart, scope=SCOPE)
