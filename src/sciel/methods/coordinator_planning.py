"""Authorize method drafts and freeze assigned experiment plans."""
from __future__ import annotations

import math

from .coordinator_contracts import Refusal, _experiment_shape, _serialized
from .engine_bridge import EngineCascadeBridge
from .models import content_hash, detached, validate_manifest, validate_value


@_serialized
def register_draft(coordinator, lab_id, proposal_id, files, manifest):
    def register():
        validate_manifest(manifest,files)
        if coordinator.registry.lab(lab_id)['mode'] == 'A':
            raise Refusal('fixed-toolbox mode allows new solutions, not permanent methods')
        if manifest.get('author_lab') != lab_id:
            raise Refusal('method author does not match the proposing lab')
        _, revision = coordinator._revision(lab_id)
        dependencies = coordinator._bundle_closure(manifest.get('dependencies', []))
        for dependency, bundle in dependencies.items():
            # An immutable dependency declaration is explicit selection for
            # a bounded C adaptation trial, never persistent adoption.
            coordinator._allowed(lab_id, revision, dependency, purpose='transfer', selected=list(dependencies))
            if bundle['manifest']['runtime_lock'] != revision['runtime_lock']:
                raise Refusal('dependency runtime differs from the lab lock')
            if any(bundle['manifest']['resources'][key] > bound for key, bound in coordinator.resources.items()):
                raise Refusal('dependency exceeds the permitted resource contract')
        for parent in manifest.get('parents', []):
            if (coordinator.registry.get_method(parent)['manifest']['author_lab']!=lab_id
                    and parent not in revision['active_methods']+revision['active_adapters']):
                raise Refusal('parent version is outside this lab history')
        if manifest.get('runtime_lock') != revision['runtime_lock']:
            raise Refusal('method runtime differs from the lab lock')
        for key, bound in coordinator.resources.items():
            if manifest.get('resources', {}).get(key, math.inf) > bound:
                raise Refusal('method exceeds the permitted resource contract')
        policy = coordinator._funding()
        parents = manifest.get('parents', [])
        own_parents = [p for p in parents if coordinator.registry.get_method(p)['manifest']['author_lab'] == lab_id]
        bound_parents = [p for p in own_parents if coordinator.lineage_for_version(p,policy=policy) is not None]
        ident = coordinator.registry.register(files, manifest)
        existing = coordinator.lineage_for_version(ident,policy=policy)
        # Immutable manifest order resolves multi-parent accounting. It is
        # neither a new grant nor an assertion of scientific attribution.
        lineage = existing or (coordinator.lineage_for_version(bound_parents[0],policy=policy) if bound_parents
                               else coordinator._active_funding_lineage(lab_id,revision))
        policy.create_lineage(lineage)
        if existing is None:
            policy.bind_version(ident,lineage,parent_versions=bound_parents)
        coordinator._save_funding(policy)
        return ident
    return coordinator._host('draft:'+proposal_id, lab_id, 'method_creation',
                      dict(files=files, manifest=manifest), register)


@_serialized
def plan(coordinator, lab_id, proposal_id, experiment, *, revision_id=None, purpose='local',
         selected=(), source_plan=None, input_origin='author_proposed', history_id=None,
         heldout_protocol_id=None, heldout_assignment_id=None):
    if purpose not in ('local', 'replay', 'transfer', 'heldout', 'solution'):
        raise Refusal('unsupported experiment purpose')
    revision_id, revision = coordinator._revision(lab_id, revision_id)
    def freeze():
        _experiment_shape(experiment)
        if purpose=='heldout' and (heldout_protocol_id is None or heldout_assignment_id is None):
            raise Refusal('held-out trials require a preregistered complete cohort assignment')
        if purpose!='heldout' and (heldout_protocol_id is not None or heldout_assignment_id is not None):
            raise Refusal('holdout assignments cannot be attached to another trial purpose')
        if set(experiment) != {'question', 'treatment', 'control', 'inputs', 'criterion', 'scope'}:
            raise Refusal('experiment fields differ from the protocol')
        if not isinstance(experiment['question'], str) or not experiment['question'].strip():
            raise Refusal('an experiment requires a question')
        if not isinstance(experiment['inputs'], list) or not 1 <= len(experiment['inputs']) <= coordinator.max_cases:
            raise Refusal('invalid assigned case count')
        criterion = experiment['criterion']
        if (set(criterion) != {'metric', 'direction', 'minimum_effect'}
                or criterion['metric'] != coordinator.task.metric or criterion['direction'] != coordinator.task.direction
                or type(criterion['minimum_effect']) not in (int, float)
                or not math.isfinite(criterion['minimum_effect']) or criterion['minimum_effect'] < 0):
            raise Refusal('criterion must use the trusted task metric/direction and nonnegative threshold')
        trial_dependencies = set()
        for arm in ('treatment', 'control'):
            pipeline = experiment[arm]
            if not isinstance(pipeline, list) or len(pipeline) > 8 or any(not isinstance(x, str) for x in pipeline):
                raise Refusal('invalid method pipeline')
            if len(set(pipeline)) != len(pipeline):
                raise Refusal('duplicate pipeline component requires a separately versioned protocol')
            closure = coordinator._bundle_closure(pipeline)
            selected_closure = list(coordinator._bundle_closure(selected))
            declared = set()
            if purpose == 'local':
                for root in pipeline:
                    if closure[root]['manifest']['author_lab'] == lab_id:
                        declared.update(coordinator._bundle_closure(closure[root]['manifest']['dependencies']))
                # A top-level foreign invocation still needs its own explicit
                # transfer selection; another wrapper cannot authorize it.
                declared.difference_update(pipeline)
            for ident, bundle in closure.items():
                dependency_trial = ident in declared
                coordinator._allowed(lab_id, revision, ident,
                              purpose='transfer' if dependency_trial else 'replay' if purpose=='heldout' else purpose,
                              selected=list(declared) if dependency_trial else selected_closure)
                if dependency_trial and bundle['manifest']['author_lab'] != lab_id:
                    trial_dependencies.add(ident)
                if bundle['manifest']['runtime_lock'] != revision['runtime_lock']:
                    raise Refusal('incompatible runtime lock')
                if any(bundle['manifest']['resources'][key]>bound for key,bound in coordinator.resources.items()):
                    raise Refusal('method or dependency exceeds the frozen resource envelope')
            if pipeline:
                for value in experiment['inputs']:
                    validate_value(value, closure[pipeline[0]]['manifest']['input_schema'])
                for left, right in zip(pipeline, pipeline[1:]):
                    if closure[left]['manifest']['output_schema'] != closure[right]['manifest']['input_schema']:
                        raise Refusal('pipeline schemas require an explicit versioned adapter')
                    lp,rp=closure[left]['manifest']['protocol'],closure[right]['manifest']['protocol']
                    left_units=lp.get('output_units') if isinstance(lp,dict) else None
                    right_units=rp.get('input_units') if isinstance(rp,dict) else None
                    if left_units!=right_units:
                        raise Refusal('declared units require an explicit versioned conversion adapter')
        if not experiment['treatment']:
            raise Refusal('treatment requires an actual invocation')
        measurement_protocols = {arm: coordinator.task.identity for arm in ('treatment', 'control')}
        if purpose == 'solution':
            measurement_protocols['treatment'] = coordinator.solution_task.identity
            if isinstance(coordinator.solution_task, EngineCascadeBridge) and (
                    len(experiment['treatment']) != 1 or len(experiment['inputs']) != 1):
                raise Refusal('canonical cascade solution requires one bundle and one input; confirmation is internal')
        if purpose == 'replay':
            original = coordinator.registry.get_artifact('experiment_plan', source_plan)
            measurement_protocols = detached(original['measurement_protocols'])
            if original['lab_id'] == lab_id or original['history_id'] == (history_id or lab_id):
                raise Refusal('independent replay requires a distinct lab and history')
            for key in ('treatment', 'control', 'inputs', 'criterion', 'scope'):
                if experiment[key] != original[key]:
                    raise Refusal('replay changes the original frozen protocol')
        body=dict(schema='sciel-method-plan/1',
            lab_id=lab_id, lab_revision_id=revision_id, proposal_id=proposal_id,
            purpose=purpose, selected=list(selected), trial_dependencies=sorted(trial_dependencies), source_plan=source_plan,
            history_id=history_id or lab_id, input_origin=input_origin,
            task=coordinator.task.identity, measurement_protocols=measurement_protocols,
            runtime=coordinator.executor.identity, resources=coordinator.resources,
            funding_lineage=coordinator._plan_funding_lineage(lab_id,revision,experiment['treatment']),
            adoption_rule='all treatment outputs valid and paired mean effect >= minimum_effect',
            replay_rule='exact assigned input and task-measured observables; runtime recorded separately',
            **detached(experiment))
        if purpose=='heldout':
            from .evidence import EvidenceAssessor
            body.update(EvidenceAssessor(coordinator.registry).validate_heldout_plan(
                heldout_protocol_id,heldout_assignment_id,body))
        return coordinator.registry.put_artifact('experiment_plan',body)
    request = dict(revision_id=revision_id, proposal_id=proposal_id, experiment=experiment,
        purpose=purpose, selected=list(selected), source_plan=source_plan,
        input_origin=input_origin, history_id=history_id,
        heldout_protocol_id=heldout_protocol_id,heldout_assignment_id=heldout_assignment_id)
    return coordinator._host('plan:'+content_hash(request), lab_id, 'planning', request, freeze)
