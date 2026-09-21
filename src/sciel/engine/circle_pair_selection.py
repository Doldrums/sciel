"""Pre-evaluation assessment of both Circle alternatives, without consumption.

The immutable snapshot and per-entry validation receipts are journal inputs to
restore. Replaying never calls an encoder/provider or executes a candidate: the
anchored novelty receipt is reused while source and all local gates are rebuilt.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from ..hypothesis.circle_source_family import builder_for_profile, builder_for_source
from ..hypothesis import circle_proposal_pair as codec
from ..hypothesis import circle_baseline_equivalence as equivalence
from .db import normalized_hash
from .prompt.sources import source_digest
from .validation import validate_candidate

POLICY = 'first_eligible_of_two/1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def snapshot(parent_source, code_hashes, consumed_sources, required_entrypoints, *, novelty_enabled,
             restart_predictions=False):
    if type(restart_predictions) is not bool:
        raise ValueError('restart_predictions must be boolean')
    builder = builder_for_source(parent_source)
    profile = getattr(builder, 'EXECUTION_PROFILE', None)
    return {'parent_body': builder.body_of(parent_source),
            'parent_source_sha256': source_digest(parent_source),
            'code_hashes': sorted(code_hashes), 'consumed_sources': sorted(consumed_sources),
            'required_entrypoints': list(required_entrypoints),
            'novelty_enabled': novelty_enabled,
            **({'restart_predictions': True} if restart_predictions else {}),
            **({'execution_profile': profile} if profile is not None else {})}


def _check_snapshot(value):
    fields = {
        'parent_body', 'parent_source_sha256', 'code_hashes', 'consumed_sources',
        'required_entrypoints', 'novelty_enabled'}
    if (type(value) is not dict or not fields <= set(value)
            or set(value) - fields - {'execution_profile', 'restart_predictions'}):
        raise ValueError('invalid Circle pair selection snapshot')
    if 'restart_predictions' in value and (value['restart_predictions'] is not True
            or value.get('execution_profile') != 'paired_restart_trace/2'):
        raise ValueError('invalid Circle pair restart predictions mode')
    if 'execution_profile' in value and type(value['execution_profile']) is not str:
        raise ValueError('invalid Circle pair execution profile')
    builder = builder_for_profile(value.get('execution_profile'))
    parent = builder.source_for(value['parent_body'])
    if source_digest(parent) != value['parent_source_sha256'] or type(value['novelty_enabled']) is not bool:
        raise ValueError('Circle pair parent or novelty snapshot differs')
    for key in ('code_hashes', 'consumed_sources'):
        values = value[key]
        if (type(values) is not list or any(type(s) is not str or len(s) != 64
                or any(c not in '0123456789abcdef' for c in s) for s in values)
                or values != sorted(set(values))):
            raise ValueError('invalid Circle pair source-history snapshot')
    names = value['required_entrypoints']
    if type(names) is not list or any(type(n) is not str or not n for n in names):
        raise ValueError('invalid Circle pair entrypoint snapshot')
    return parent


def _receipt(validation):
    return {'violation': ({'kind': validation.violation.kind, 'detail': validation.violation.detail}
                          if validation.violation is not None else None),
            'grafted': copy.deepcopy(validation.grafted), 'novelty': copy.deepcopy(validation.novelty),
            'decisions': copy.deepcopy(validation.decisions), 'profile': validation.profile}


def assess(raw, selection_snapshot, *, provider_error=None, novelty_check=None, replay_evidence=None):
    """Assess both slots against one snapshot; never mark either source consumed.

    Returns gates/evidence plus selected source/proof/validation (or None).
    Unexpected validator/compiler/I/O failures propagate and stop the request.
    During restore only a saved novelty receipt can substitute for its callback;
    every rebuilt receipt must match the journal exactly.
    """
    from ..hypothesis.circle_proposal_group import gate

    parent = _check_snapshot(selection_snapshot)
    builder = builder_for_profile(selection_snapshot.get('execution_profile'))
    replay = replay_evidence is not None
    if replay and novelty_check is not None:
        raise ValueError('Circle pair replay cannot dispatch novelty checks')
    if not replay and bool(novelty_check is not None) != selection_snapshot['novelty_enabled']:
        raise ValueError('Circle pair novelty callback differs from snapshot')
    decoded = codec.decode_response(raw)
    if replay and (type(replay_evidence) is not list or len(replay_evidence) != len(decoded['entries'])):
        raise ValueError('Circle pair replay entry count differs')
    gates, evidence, candidates = [], [], []
    for entry in decoded['entries']:
        index = entry['index']
        item = {'index': index, 'source_sha256': None, 'normalized_sha256': None,
                'match_kind': None, 'initializer_equivalence_proof': None, 'validation': None}
        proposal = entry['proposal']
        status, reason, diagnostic, candidate = 'not_applicable', None, None, None
        missing_prediction = (proposal is not None and selection_snapshot.get('restart_predictions', False)
                              and proposal.hypothesis.restart_predictions is None)
        if missing_prediction and provider_error is None:
            status, reason = 'rejected', 'missing_restart_predictions'
            diagnostic = 'This mode requires the three pre-evaluation restart predictions; candidate unmeasured.'
        elif proposal is not None and provider_error is None:
            source = builder.source_for(proposal.body)
            exact, normalized = source_digest(source), normalized_hash(source)
            item.update(source_sha256=exact, normalized_sha256=normalized)
            match = ('current_parent' if source == parent else
                     'previous_source' if exact in selection_snapshot['consumed_sources'] else
                     'known_normalized_source' if normalized in selection_snapshot['code_hashes'] else None)
            # The legacy algebraic proof is bound to the exact untraced
            # default parent. It is not evidence for a traced source family.
            if match is None and 'execution_profile' not in selection_snapshot:
                result = equivalence.verify(proposal.body, parent,
                    builder_source=Path(builder.__file__).read_bytes(),
                    proposal_source=Path(builder.proposals.__file__).read_bytes())
                if result['verdict'] == 'equivalent':
                    match = 'baseline_initializer'
                    item['initializer_equivalence_proof'] = result['proof']
            item['match_kind'] = match
            if match is not None:
                status, reason = 'rejected', match
                diagnostic = ('Initializer values match the current control; quality unmeasured.'
                              if match == 'baseline_initializer' else
                              'This source is already the parent or in recorded source history; quality unmeasured.')
            else:
                callback = novelty_check
                if replay and selection_snapshot['novelty_enabled']:
                    saved = replay_evidence[index]
                    saved_novelty = (copy.deepcopy(saved['validation'].get('novelty'))
                                     if type(saved) is dict and type(saved.get('validation')) is dict else None)

                    def callback(code, expected=source, value=saved_novelty):
                        if code != expected or type(value) is not dict:
                            raise ValueError('Circle pair replay lacks matching novelty receipt')
                        return copy.deepcopy(value)
                validation = validate_candidate(source, parent_code=parent,
                    required_entrypoints=selection_snapshot['required_entrypoints'],
                    code_hashes=set(selection_snapshot['code_hashes']), novelty_check=callback)
                if validation.code != source or validation.grafted:
                    raise ValueError('Circle pair source cannot be repaired or grafted')
                if validation.novelty is not None and (type(validation.novelty) is not dict
                        or type(validation.novelty.get('accept')) is not bool):
                    raise ValueError('invalid Circle pair novelty receipt')
                item['validation'] = _receipt(validation)
                if validation.violation is not None:
                    status, reason = 'rejected', 'gate_rejected'
                    diagnostic = str(validation.violation.kind) + ': ' + str(validation.violation.detail)
                else:
                    status = 'passed'
                    candidate = {'source': source, 'body': proposal.body, 'validation': validation}
        if replay and digest(item) != digest(replay_evidence[index]):
            raise ValueError('Circle pair gate receipt differs from replay')
        evidence.append(item)
        gates.append(gate(status, reason=reason, diagnostic=diagnostic, evidence_sha256=digest(item)))
        candidates.append(candidate)
    chosen = next((i for i, c in enumerate(candidates) if c is not None), None)
    selected = candidates[chosen] if chosen is not None else None
    if selected is not None:
        selected = {**selected, 'index': chosen, 'built': builder.apply(parent, selected['body'])}
        if selected['built']['code'] != selected['source']:
            raise ValueError('Circle pair selected source assembly differs')
    return {'gates': gates, 'evidence': evidence, 'selected': selected}
