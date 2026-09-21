"""Authenticate declared study slots, receipts and lifecycle accounting."""
from __future__ import annotations

from .catalogs import canonical_sha
from .paired_comparisons import UNKNOWN_RECEIPT
from ..hypothesis.comparison_targets import _quality, QUALITY_GATE
from .study_validation import (
    EVALUATED_G_PRODUCER,
    G_PRODUCER,
    HOST_PRODUCERS,
    PROFILE,
    _SLOT_FIELDS,
    _digest,
    _object,
    _require,
    _same,
    _sha,
)


def _lifecycle(capsule, authority, manifest, summary, started):
    study, group, author = capsule['study'], capsule['comparison'], capsule['author']
    manifest_sha = authority['producer_manifest_sha256']
    producer = manifest.get('schema')
    _require(producer in HOST_PRODUCERS if author['kind'] == 'HOST' else
        producer == (EVALUATED_G_PRODUCER if author['kind'] == 'evaluated_target_G' else G_PRODUCER),
             'unsupported producer family')
    _require(manifest.get('payload_sha256') == canonical_sha({k:v for k,v in manifest.items() if k != 'payload_sha256'}),
             'producer manifest payload differs')
    _require(summary.get('schema') == producer and summary.get('manifest_sha256') == manifest_sha
        and started.get('manifest_sha256') == manifest_sha and started.get('no_resume') is True,
        'producer lifecycle binding differs')
    roles = study['role_map']
    _require(type(roles) is dict and set(roles) == {'comparator', 'challenger'}
        and all(type(v) is str and 0 < len(v) <= 128 and v.strip() for v in roles.values())
        and roles['comparator'] != roles['challenger'], 'role map differs')
    inverse = {v:k for k,v in roles.items()}
    source_map = {roles[role]:author['comparator_source_sha256' if role == 'comparator' else 'candidate_source_sha256']
                  for role in roles}
    _require(_same(manifest.get('source_sha256'), source_map)
        and _same(manifest.get('adapter_identity'), group['adapter_identity']), 'manifest source/adapter differs')
    files = manifest.get('files')
    _require(type(files) is dict and all(files.get('candidates/'+role+'.py') == digest
        for role,digest in source_map.items()), 'frozen candidate file binding differs')
    addendum = manifest.get('numeric_addendum')
    _require(type(addendum) is dict and addendum.get('schema') == producer
        and addendum.get('execution_profile') == PROFILE and addendum.get('restart_trace') is True
        and addendum.get('no_resume') is True and addendum.get('no_retry') is True
        and type(addendum.get('max_candidate_calls')) is int and addendum['max_candidate_calls'] == 6
        and type(addendum.get('max_certifier_calls')) is int and addendum['max_certifier_calls'] == 6
        and addendum.get('quality_gate') == QUALITY_GATE and addendum.get('automatic_promotion') is False
        and _same(addendum.get('limits'), group['limits'])
        and _same(addendum.get('runtime'), group['adapter_identity'].get('runtime_config')),
        'frozen numeric contract differs')
    schedule = addendum.get('schedule')
    _require(type(schedule) is list and len(schedule) == 6, 'six-slot schedule required')
    for i, item in enumerate(schedule):
        _require(type(item) is dict and set(item) == {'index', 'seed', 'role'}
            and type(item['index']) is int and item['index'] == i
            and type(item['seed']) is int and item['seed'] == group['pairs'][i//2]['seed']
            and item['role'] in inverse, 'schedule source/seed/index differs')
    _require(all({schedule[i]['role'], schedule[i+1]['role']} == set(inverse) for i in (0,2,4)),
             'each planned pair must contain both roles exactly once')
    provenance = group['provenance']
    _require(provenance.get('schema') == producer and provenance.get('numeric_manifest_sha256') == manifest_sha
        and _same(provenance.get('schedule'), schedule) and provenance.get('detached') is True
        and provenance.get('authorship') == author['kind']
        and provenance.get('numeric_addendum_sha256') == canonical_sha(addendum)
        and provenance.get('source_proof_sha256') == manifest.get('source_proof_sha256'),
        'comparison producer provenance differs')
    slots = study['slots']
    _require(type(slots) is list and len(slots) == 6, 'six retained slot states required')
    summary_slots, artifacts = summary.get('slots'), summary.get('artifacts')
    _require(type(summary_slots) is list and len(summary_slots) <= 6 and type(artifacts) is dict,
             'terminal slot/artifact inventory required')
    errors = summary.get('errors')
    integrity = summary.get('evidence_integrity_verified')
    _require(type(errors) is list and all(type(error) is dict for error in errors)
        and type(integrity) is bool and integrity == (not errors), 'producer error/integrity state differs')
    _require(artifacts.get('started.json') == _digest(study['started_text'])
        and artifacts.get('manifest.json') == manifest_sha
        and all(_sha(digest) and artifacts.get('frozen/'+name) == digest for name,digest in files.items()),
        'recorded frozen/start inventory differs')
    coverage, issued_rows, expected_artifacts = [], [], {}
    ended = False
    for i, slot in enumerate(slots):
        _require(type(slot) is dict and set(slot) == _SLOT_FIELDS
            and _same({key:slot[key] for key in ('index','seed','role')}, schedule[i]), 'slot fields/order differ')
        role = inverse[slot['role']]
        receipt = group['pairs'][i//2][role]
        directory = f"draws/{i:02d}_{slot['seed']}_{slot['role']}/"
        texts = ('intent_text', 'raw_receipt_text', 'validated_text', 'unknown_text')
        for key, filename in zip(texts, ('intent.json', 'raw_receipt.json', 'validated.json', 'unknown.json')):
            raw = slot[key]
            _require(raw is None or type(raw) is str, 'slot text must be original JSON or null')
            if raw is not None:
                _object(raw, filename)
                expected_artifacts[directory+filename] = _digest(raw)
        if slot['intent_text'] is None:
            ended = True
            _require(all(slot[key] is None for key in texts) and receipt is None, 'unissued slot has retained work')
            status = 'not_dispatched'
        else:
            _require(not ended, 'issued slots must form one unretried prefix')
            intent = _object(slot['intent_text'], 'intent')
            _require(_same({key:intent.get(key) for key in ('index','seed','role')}, schedule[i])
                and intent.get('schema') == producer and intent.get('manifest_sha256') == manifest_sha
                and intent.get('source_sha256') == source_map[slot['role']]
                and intent.get('adapter_identity_sha256') == group['adapter_identity']['identity_sha256']
                and intent.get('restart_trace') is True and _same(intent.get('limits'), group['limits'])
                and _same(intent.get('runtime'), addendum['runtime']), 'intent binding differs')
            if _same(receipt, UNKNOWN_RECEIPT):
                retrospective = any(error.get('stage') == 'terminal_receipt_integrity'
                    and type(error.get('slot')) is int and error['slot'] == i for error in errors)
                _require(not integrity and (slot['unknown_text'] is not None or retrospective),
                         'unknown dispatch requires an explicit producer failure')
                if slot['unknown_text'] is not None:
                    unknown = _object(slot['unknown_text'], 'unknown')
                    _require(_same(unknown.get('marker'), UNKNOWN_RECEIPT) and unknown.get('cleanup_confirmed') is False,
                             'unknown dispatch marker differs')
                # Terminal integrity failures can downgrade an earlier slot
                # after subsequent calls. Its leftover raw/validated files are
                # retained evidence of uncertainty, never numeric observations.
                status = 'unknown'
            else:
                _require(slot['raw_receipt_text'] is not None and slot['validated_text'] is not None
                    and slot['unknown_text'] is None and receipt is not None, 'terminal receipt data missing')
                raw = _object(slot['raw_receipt_text'], 'raw receipt')
                validated = _object(slot['validated_text'], 'validated receipt')
                _require(_same(raw, receipt) and _same(validated, dict(status=receipt['status'],
                    raw_receipt_sha256=_digest(slot['raw_receipt_text']))), 'raw/validated/group receipt differs')
                status = receipt['status']
            row = dict(**schedule[i], status=status, intent_sha256=_digest(slot['intent_text']))
            if slot['raw_receipt_text'] is not None and status != 'unknown':
                row['raw_receipt_sha256'] = _digest(slot['raw_receipt_text'])
            _require(i < len(summary_slots) and all(_same(summary_slots[i].get(k),v) for k,v in row.items()),
                     'summary dropped or changed issued slot')
            issued_rows.append(row)
            if status == 'infrastructure' or status == 'unknown' and not retrospective:
                ended = True
        coverage.append(dict(**schedule[i], source_sha256=source_map[slot['role']], status=status,
                             receipt_sha256=canonical_sha(receipt) if receipt is not None else None))
    _require(len(summary_slots) == len(issued_rows), 'summary has extra issued slots')
    actual_draw_artifacts = {k:v for k,v in artifacts.items() if k.startswith('draws/')}
    _require(_same(actual_draw_artifacts, expected_artifacts), 'terminal inventory has omitted or extra draw data')
    quality = _quality(group)
    count = len(issued_rows)
    _require(_same(summary.get('issued_draw_intents'), count)
        and _same(summary.get('validated_terminal_receipts'), sum(r['status'] != 'unknown' for r in issued_rows))
        and _same(summary.get('planned_candidate_allocations'), 6)
        and _same(summary.get('remaining_unissued_slots'), schedule[count:])
        and _same(summary.get('quality'), quality) and summary.get('group_sha256') == canonical_sha(group)
        and summary.get('quality_gate_passed') is (integrity and quality['verdict'] == 'CONFIRMED')
        and summary.get('automatic_promotion') is False and summary.get('resume_allowed') is False
        and type(summary.get('provider_calls')) is int and summary['provider_calls'] == 0
        and summary.get('new_api_cost_usd') in ('0','0.0'), 'terminal accounting/quality differs')
    status = summary.get('status')
    _require(status in ('completed', 'early_stopped_known_host_loss', 'early_stopped_known_target_loss',
        'stopped_pair_time_budget', 'stopped_outer_deadline', 'stopped_unknown', 'stopped_infrastructure'),
        'unsupported terminal disposition')
    for pair_index, row in enumerate(quality['pairs']):
        if row['no_loss'] is False:
            _require(count <= 2*(pair_index+1), 'work continued after a complete known-loss pair')
            _require(status in ('early_stopped_known_host_loss', 'early_stopped_known_target_loss')
                or not integrity and status in ('stopped_infrastructure','stopped_unknown','stopped_outer_deadline'),
                     'known loss was not retained as stop reason')
            break
    if status.startswith('early_stopped_known_'):
        _require(quality['reason'] == 'known_paired_loss' and count % 2 == 0, 'unfounded known-loss stop')
    if status == 'completed':
        _require(integrity and count == 6 and all(r['status'] not in ('unknown','infrastructure') for r in issued_rows),
                 'incomplete study called completed')
    return dict(planned=6, issued=count, terminal=sum(r['status'] not in ('unknown','not_dispatched') for r in coverage),
        unknown=sum(r['status'] == 'unknown' for r in coverage), unissued=6-count,
        slots=coverage, stop_reason=status, recorded_lifecycle_consistent=True, execution_verified=False,
        producer_integrity_verified=integrity, producer_error_count=len(errors), producer_errors_sha256=canonical_sha(errors),
        producer_quality_gate_passed=summary['quality_gate_passed'])
