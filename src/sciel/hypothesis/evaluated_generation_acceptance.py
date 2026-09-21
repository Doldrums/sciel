"""Host approval for numerical work on one exact, still-unmeasured G result.

This module checks externally pinned bindings, not prose truth, paid dispatch,
generation authenticity or permission inferred from a model response. Callers
must authenticate the G result separately and supply the trusted decision SHA.
The numerical protocol is the complete producer addendum, not a second overlay.
"""
from __future__ import annotations

import copy
import hashlib
import re

from ..knowledge_base.catalogs import canonical_json, canonical_sha
from ..knowledge_base.generated_paired_history import _bounded

SCHEMA = 'sciel-evaluated-G-numeric-decision/1'
PROTOCOL_SCHEMA = 'sciel-evaluated-target-numeric/1'
RESULT_SCHEMA = 'sciel-evaluated-target-generation-result/1'
QUALITY_GATE = 'paired_raw_exact_zero_loss_strict_v1'
RUNTIME = {'backend': 'docker', 'image': 'sha256:0989ff53d1ca9fad39bd6caf2d230ebabed2b6dc8f160932af05b5938201c029'}
LIMITS = dict(candidate_wall_seconds=30., certification_wall_seconds=10.,
    candidate_cpu_seconds=35, certification_cpu_seconds=35,
    candidate_mem_bytes=2147483648, certification_mem_bytes=2147483648)
_FIELDS = {'schema','decision','generation_result_sha256','generated_source_sha256',
    'source_proof_sha256','target_sha256','policy_sha256','audit_sha256','criteria_sha256',
    'numeric_protocol_sha256','issuer'}


def _require(ok, message):
    if not ok:
        raise ValueError('evaluated G numeric acceptance: ' + message)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def make_protocol(study_id, seeds):
    """Construct a prospective protocol; this is not an acceptance or execution."""
    _require(type(study_id) is str and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', study_id),
             'bounded study ID required')
    _require(type(seeds) is list and len(seeds) == 3
        and all(type(seed) is int and 0 <= seed <= 2**32-1 for seed in seeds)
        and len(set(seeds)) == 3, 'three distinct uint32 seeds required')
    orders = (('comparator','challenger'),('challenger','comparator'),('comparator','challenger'))
    schedule = [dict(index=2*i+j, seed=seed, role=role)
                for i,(seed,roles) in enumerate(zip(seeds,orders)) for j,role in enumerate(roles)]
    return dict(schema=PROTOCOL_SCHEMA, study_id=study_id, seeds=copy.deepcopy(seeds),
        authorship='evaluated_target_G', schedule=schedule, limits=copy.deepcopy(LIMITS),
        runtime=copy.deepcopy(RUNTIME), execution_profile='paired_restart_trace/2', restart_trace=True,
        max_candidate_calls=6, max_certifier_calls=6, provider_calls=0, provider_budget_usd='0',
        no_resume=True, no_retry=True,
        early_stop='After each complete pair, any known challenger minus comparator raw or exact loss stops.',
        infrastructure_stop='Immediate; retain raw and unknown intents, never replace a slot.',
        outer_series_wall_cap_seconds=360., computational_cutoff_seconds=348.,
        cleanup_reserve_seconds=12., pair_admission_seconds=136.,
        timing_scope='Monotonic elapsed starts before execution verification and frozen re-exec. '
            'Whole-pair admission reserves 128 seconds for unchanged 30/10 phase caps plus adapter cleanup, '
            'and 8 seconds for verification. One-shot BaseException cutoff at 348 seconds leaves 12 seconds '
            'for mandatory adapter finally cleanup within 360. Terminal integrity/report/archive I/O is separate. '
            'No real-time guarantee under OS/filesystem stalls. A cleanup interruption is never certified clean.',
        quality_gate=QUALITY_GATE, automatic_promotion=False,
        geometry_diagnostics='Separate saved-data audit; forecast agreement, motion and local gain cannot replace quality.',
        native_observation='UNMEASURED', native_forecast='NOT_DECLARED',
        knowledge_benefit_established=False)


def validate_protocol(value):
    _bounded(value)
    _require(type(value) is dict, 'literal numerical protocol required')
    expected = make_protocol(value.get('study_id'), value.get('seeds'))
    _require(canonical_json(value) == canonical_json(expected), 'fixed numerical protocol differs')
    return copy.deepcopy(expected)


def validate_decision(decision, *, generation_result, numeric_protocol, expected_sha256):
    _bounded([decision, generation_result, numeric_protocol])
    _require(type(decision) is dict and set(decision) == _FIELDS
        and decision['schema'] == SCHEMA and decision['decision'] == 'accepted',
        'exact accepted host decision required')
    _require(_sha(expected_sha256) and canonical_sha(decision) == expected_sha256,
             'external decision digest differs')
    _require(all(_sha(decision[key]) for key in _FIELDS-{'schema','decision','issuer'}),
             'decision digest field invalid')
    _require(type(decision['issuer']) is str and 0 < len(decision['issuer']) <= 128
        and bool(decision['issuer'].strip()), 'bounded host issuer required')
    protocol = validate_protocol(numeric_protocol)
    result = generation_result
    _require(type(result) is dict and result.get('schema') == RESULT_SCHEMA
        and result.get('status') == 'eligible' and result.get('eligible') is True
        and type(result.get('code')) is str and type(result.get('proof')) is dict
        and type(result.get('plan')) is dict and type(result['plan'].get('target')) is dict,
        'eligible evaluated target G required; caller must also replay authenticity')
    bindings = dict(generation_result_sha256=canonical_sha(result),
        generated_source_sha256=hashlib.sha256(result['code'].encode('utf-8')).hexdigest(),
        source_proof_sha256=canonical_sha(result['proof']), target_sha256=canonical_sha(result['plan']['target']),
        policy_sha256=result['plan'].get('policy_sha256'), numeric_protocol_sha256=canonical_sha(protocol))
    _require(result.get('source_sha256') == bindings['generated_source_sha256']
        and _sha(bindings['policy_sha256']) and all(decision[k] == v for k,v in bindings.items()),
        'host decision differs from exact G/source/proof/target/policy/protocol')
    return copy.deepcopy(decision)
