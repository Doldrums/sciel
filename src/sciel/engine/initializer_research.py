"""One-shot behavioral diagnostics after ordinary outcomes have been committed.

This module never evaluates a packing, selects a candidate or changes its score.
The existing research journal/checkpoint boundary forbids replay after a crash.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re

from ..campaign.budget import CampaignStopped
from ..knowledge_base.catalogs import canonical_json

SCHEMA = 'sciel-initializer-research/1'
MAX_CONTEXT_CHARS = 16000


def fixture_profile(config):
    """An omitted profile keeps the original fixed-input recipe."""
    profile = config.get('initializer_witness_profile', 'legacy_v1')
    if type(profile) is not str or profile not in ('legacy_v1', 'unequal_radii_v1'):
        raise ValueError('invalid initializer_witness_profile')
    return profile


def identity(profile='legacy_v1'):
    from ..hypothesis.circle_initializer_witness import identity as witness_identity, make_fixture_bundle
    fixtures = make_fixture_bundle(profile=profile)
    from .source_identity import package_source_hashes
    return {'schema': SCHEMA, 'witness': witness_identity(),
            'fixture_profile': profile, 'fixture_sha256': fixtures['sha256'],
            'source_sha256': package_source_hashes(files=('config.py', 'config_validation.py')),
            'policy': 'after_ordinary_outcome_before_conclusion; parent_and_candidate; no_retry'}


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _sources(record):
    from ..hypothesis import circle_restart_body as builder
    proof = record.get('controlled_change')
    if type(proof) is not dict or proof.get('operation') != builder.OPERATION:
        return None
    parent = builder.source_for(proof['parent_body'])
    candidate = builder.source_for(proof['target_body'])
    exact = builder.apply(parent, proof['target_body'])['proof']
    if (canonical_json(exact) != canonical_json(proof)
            or hashlib.sha256(parent.encode()).hexdigest() != record['parent_source_sha256']
            or hashlib.sha256(candidate.encode()).hexdigest() != record['source_sha256']
            or record['source_sha256'] != record['proposed_sha256'] or record['source_repaired']):
        raise ValueError('initializer witness source attribution differs')
    return {'parent': parent, 'candidate': candidate}


def collect(controller, record):
    """Preserve ordinary outcomes even when diagnostics fail; never retry a slot."""
    if not controller.config.get('initializer_witness', False):
        return
    ident = record['id']
    attempts = controller._initializer_attempts
    if ident in attempts:
        if attempts[ident]['status'] == 'started':
            raise CampaignStopped('unfinished initializer witness cannot be replayed')
        return
    # Resolve must already have committed precisely this ordinary observation.
    stored = next((r for r in controller.knowledge.experiments if r['id'] == ident), None)
    def normalized(value):
        # KB normalizes numeric scores to float; bool remains a distinct value.
        return float(value) if type(value) in (int, float) else value
    if stored is None or any(canonical_json(normalized(stored.get(k))) != canonical_json(normalized(record.get(k))) for k in
            ('source_sha256', 'parent_source_sha256', 'status', 'score', 'actual_delta', 'controlled_change')):
        raise ValueError('initializer witness requires a committed ordinary outcome')
    attempts[ident] = {'status': 'started', 'source_sha256': record['source_sha256'],
        'parent_source_sha256': record['parent_source_sha256'], 'plan_sha256': None,
        'evidence_sha256': None, 'reason': None}
    try:
        if ident in controller.knowledge._initializer_seen:
            raise ValueError('initializer registry has no matching consumed attempt')
        profile = fixture_profile(controller.config)
        if canonical_json(identity(profile)) != canonical_json(controller._initializer_identity):
            raise ValueError('initializer research identity changed')
        sources = _sources(record)
        if sources is None:
            attempts[ident].update(status='unavailable', reason='unsupported_source_contract')
            controller.emit('initializer_witness_unavailable', experiment_id=ident,
                            attempt=copy.deepcopy(attempts[ident]), quality_changed=False)
            return
        from ..hypothesis import circle_initializer_witness as witness
        from ..hypothesis.initializer_evidence import build_evidence
        pins = {k: hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()}
        fixtures = witness.make_fixture_bundle(profile=profile)
        plan = witness.prepare_witness(sources, fixtures, expected_source_sha256=pins)
        attempts[ident]['plan_sha256'] = plan['plan_sha256']
        # Journal intent precedes every helper process. A later unknown failure
        # leaves this consumed slot and stops auxiliary work, not the measurement.
        controller.emit('initializer_witness_intent', experiment_id=ident, plan=plan,
                        source_roles={'candidate': 'candidate', 'parent': 'parent'},
                        helper_invocation_limit=32, quality_changed=False)
        receipt = witness.witness_sources(sources, plan['fixtures'], expected_source_sha256=pins,
                                          expected_plan_sha256=plan['plan_sha256'])
        controller.emit('initializer_witness_result', experiment_id=ident,
                        receipt=receipt, quality_changed=False)
        if type(receipt) is not dict or canonical_json(receipt.get('plan')) != canonical_json(plan):
            raise ValueError('initializer receipt differs from prepared plan')
        evidence = build_evidence(receipt, roles={'candidate': 'candidate', 'parent': 'parent'})
        controller.knowledge.register_initializer_evidence(ident, evidence)
        attempts[ident].update(status='completed', evidence_sha256=evidence['evidence_sha256'])
        controller.emit('initializer_witness_registered', experiment_id=ident,
                        attempt=copy.deepcopy(attempts[ident]), quality_changed=False)
    except Exception as exc:
        # Do not convert a previously recorded evaluation to eval_failed or
        # return a successful missing witness after an internal integrity error.
        controller.stopped_reason = 'Initializer diagnostics stopped: ' + type(exc).__name__
        try:
            controller.emit('initializer_witness_stopped', experiment_id=ident,
                            error_type=type(exc).__name__, attempt=copy.deepcopy(attempts[ident]),
                            quality_changed=False)
        except Exception:
            pass
        raise CampaignStopped(controller.stopped_reason) from None


def state(controller):
    return {'schema': SCHEMA, 'identity': copy.deepcopy(controller._initializer_identity),
            'attempts': copy.deepcopy(controller._initializer_attempts)}


def validate_state(raw, controller, knowledge, outcomes):
    if (type(raw) is not dict or set(raw) != {'schema', 'identity', 'attempts'}
            or raw['schema'] != SCHEMA
            or canonical_json(raw['identity']) != canonical_json(controller._initializer_identity)
            or type(raw['attempts']) is not dict or len(raw['attempts']) > outcomes):
        raise ValueError('invalid initializer research checkpoint')
    fields = {'status', 'source_sha256', 'parent_source_sha256', 'plan_sha256', 'evidence_sha256', 'reason'}
    retained = knowledge._evidence()
    owners = knowledge._initializer_observations()
    # Resolve queues only these outcomes. Nonreviewable generation failures do
    # not get a diagnostic slot. Every retained eligible owner must keep one,
    # even if both its registry extension and attempt were removed together.
    # Pruned ordinary IDs cannot be resurrected through KnowledgeBase.record;
    # collect also requires a current, already committed experiment.
    for ident, record in owners.items():
        if (record.get('hypothesis') and record['status'] in ('evaluated', 'eval_failed')
                and ident not in raw['attempts']):
            raise ValueError('retained reviewable observation has no consumed initializer attempt')
    for ident, attempt in raw['attempts'].items():
        if (ident not in knowledge._seen or type(attempt) is not dict or set(attempt) != fields
                or attempt['status'] not in ('completed', 'unavailable')
                or any(attempt[k] is not None and not _hash(attempt[k])
                       for k in ('source_sha256', 'parent_source_sha256'))):
            raise ValueError('invalid or unfinished initializer attempt')
        if attempt['status'] == 'completed':
            if (any(not _hash(attempt[k]) for k in
                    ('source_sha256', 'parent_source_sha256', 'plan_sha256', 'evidence_sha256'))
                    or attempt['reason'] is not None):
                raise ValueError('invalid completed initializer attempt')
        elif (attempt['reason'] != 'unsupported_source_contract'
              or attempt['plan_sha256'] is not None or attempt['evidence_sha256'] is not None):
            raise ValueError('invalid unavailable initializer attempt')
        if ident in retained and any(attempt[k] != retained[ident].get(k)
                                     for k in ('source_sha256', 'parent_source_sha256')):
            raise ValueError('initializer attempt source binding differs')
        binding = knowledge.initializer_evidence_for(ident)
        if (binding is not None and binding['evidence']['fixture_sha256'] !=
                controller._initializer_identity['fixture_sha256']):
            raise ValueError('initializer evidence fixture profile differs')
        if binding is not None and (attempt['status'] != 'completed'
                or binding['evidence']['evidence_sha256'] != attempt['evidence_sha256']
                or binding['evidence']['plan_sha256'] != attempt['plan_sha256']):
            raise ValueError('initializer attempt/registry binding differs')
        if (attempt['status'] == 'completed' and ident in owners
                and ident not in knowledge._initializer_seen):
            raise ValueError('retained initializer attempt has no registry or aging tombstone')
    for ident in knowledge._initializer_seen:
        if ident not in raw['attempts'] or raw['attempts'][ident]['status'] != 'completed':
            raise ValueError('initializer registry has no completed consumed attempt')
    return copy.deepcopy(raw['attempts'])
