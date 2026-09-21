"""One immutable G-only/paired study, independent of ordinary experiments.

This validates saved literal data and recompiles typed bodies without executing
them. The caller's configured digest supplies admission authority. Matching
transport fields are recorded provenance, not proof of provider dispatch,
payment, preregistration, truthful prose, causal benefit or permission to run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

from .catalogs import canonical_json, canonical_sha
from .paired_comparisons import UNKNOWN_RECEIPT, validate_comparison
from ..eval import packing_geometry as geometry
from ..hypothesis import circle_optimizer_traced_body as builder
from ..hypothesis import restart_motion_observations as observations
from ..hypothesis import restart_predictions as predictions
from ..hypothesis.comparison_targets import _quality
from ..hypothesis.measurement import validate_metric_context
from ..hypothesis.rendering import decode_evidence

SCHEMA = 'sciel-generated-paired-history/1'
PROFILE = 'paired_restart_trace/2'
MAX_BYTES = 8 * 1024**2
MAX_PROPOSALS = 8
MAX_COMPARISONS = 16
MAX_SOURCE_BYTES = 131072
MAX_REQUEST_BYTES = 262144
MAX_RESPONSE_BYTES = 131072
SCOPE = ('One source-bound paired study; generated proposals are not ordinary outcomes, lessons, '
    'advice exposures or population members. Model hypothesis prose is unverified. Native forecasts '
    'remain unmeasured. Paired quality uses raw binary64 and exact certified radii separately; a '
    'known loss refutes even with missing pairs. None means not dispatched; the exact unknown '
    'marker means an intent without a trusted terminal receipt. Repeated source/seed slots across '
    'groups are the same saved draw, not independent replications. Restart facts concern retained '
    'scheduled-call prefixes, not omitted calls or causal mechanisms. Saved prompt parent scores/cases '
    'are recorded forecast context, not newly authenticated native measurements. No execution or promotion authority.')


def _require(value, message):
    if not value:
        raise ValueError('generated paired history: ' + message)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _same(a, b):
    return canonical_json(a) == canonical_json(b)


def _text(value, name, limit=256):
    _require(type(value) is str and 0 < len(value.encode('utf-8')) <= limit, 'invalid ' + name)
    return value


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bounded(value):
    remaining, active = MAX_BYTES, set()
    def visit(item, depth):
        nonlocal remaining
        _require(depth < 64, 'capsule nesting limit')
        remaining -= 1
        _require(remaining >= 0, 'capsule byte limit')
        if type(item) is str:
            remaining -= len(item.encode('utf-8'))
        elif item is None or type(item) is bool:
            pass
        elif type(item) is int:
            _require(item.bit_length() <= 8192, 'integer limit')
        elif type(item) is float:
            _require(math.isfinite(item), 'nonfinite number')
        else:
            _require(type(item) in (dict, list) and id(item) not in active, 'literal acyclic JSON required')
            active.add(id(item))
            if type(item) is dict:
                for key, child in item.items():
                    _require(type(key) is str, 'object keys must be strings')
                    visit(key, depth + 1)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
            active.remove(id(item))
        _require(remaining >= 0, 'capsule byte limit')
    visit(value, 0)
    _require(len(canonical_json(value).encode('utf-8')) <= MAX_BYTES, 'capsule byte limit')


def identity():
    """Conservative installed production policy pin; no provider/runtime probe."""
    root = Path(__file__).resolve().parents[1]
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob('*.py')) if path.is_file()}
    return dict(schema=SCHEMA, profile=PROFILE, limits=dict(max_bytes=MAX_BYTES,
        max_proposals=MAX_PROPOSALS, min_proposals=2, max_comparisons=MAX_COMPARISONS,
        max_source_bytes=MAX_SOURCE_BYTES, max_request_bytes=MAX_REQUEST_BYTES,
        max_response_bytes=MAX_RESPONSE_BYTES), production_python_sha256=canonical_sha(files),
        builder=builder.identity(), scope=SCOPE)


def proposal_id(provenance, response):
    """Foreign prepared research IDs never become ordinary experiment IDs."""
    _require(type(provenance) is dict and set(provenance) ==
             {'manifest_sha256', 'run_id', 'request_key', 'research_id'}, 'proposal provenance fields')
    _require(_sha(provenance['manifest_sha256']), 'generation manifest digest')
    for field in ('run_id', 'request_key', 'research_id'):
        _text(provenance[field], field, 128)
    _require(type(response) is dict, 'response must be an object')
    raw = _text(response.get('text'), 'raw response', MAX_RESPONSE_BYTES)
    return 'generated:' + canonical_sha(dict(manifest_sha256=provenance['manifest_sha256'],
        run_id=provenance['run_id'], request_key=provenance['request_key'], response_sha256=_digest(raw)))


def response_format():
    wire = copy.deepcopy(builder.response_format())
    hypothesis = wire['json_schema']['schema']['properties']['hypothesis']
    hypothesis['required'].append('restart_predictions')
    hypothesis['properties']['restart_predictions'] = predictions.json_schema()
    wire['json_schema']['name'] += '_restart_predictions_v1'
    return wire


def certificate_policy():
    return dict(name=geometry.CERTIFICATE_POLICY,
        geometry_sha256=hashlib.sha256(Path(geometry.__file__).read_bytes()).hexdigest(),
        raw_tolerance=geometry.TOLERANCE, raw_target=geometry.TARGET_VALUE,
        strict_method='common absolute decrement; clip at zero; round each radius down; exact recheck')


def _request_context(prompt, parent, metric):
    marker = 'RESEARCH_CONTEXT_JSON:\n'
    _require(prompt.count(marker) == 1, 'one complete original research context required')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate research context key')
            result[key] = value
        return result
    value, _ = json.JSONDecoder(object_pairs_hook=unique).raw_decode(prompt.split(marker, 1)[1])
    context = decode_evidence(value)
    _require(type(context) is dict and context.get('parent_source_sha256') == _digest(parent)
             and _same(validate_metric_context(context.get('metric')), metric),
             'request parent/task context differs')
    _require(type(context.get('source_scope')) is dict
             and context['source_scope'].get('contract') == 'circle_restart_body/1',
             'request research source scope differs')
    _require('parent_score' in context and (context['parent_score'] is None or _number(context['parent_score']))
             and 'parent_cases' in context and (context['parent_cases'] is None or type(context['parent_cases']) is dict),
             'recorded parent forecast context differs')
    return context


def _proposal(value, known_sources, known_normalized, metric):
    _require(type(value) is dict and set(value) == {'id', 'provenance', 'request', 'response',
             'provider_response', 'parent_source', 'source', 'proof'}, 'proposal fields')
    request, response, provider = value['request'], value['response'], value['provider_response']
    ident = proposal_id(value['provenance'], response)
    _require(value['id'] == ident, 'namespaced proposal ID differs')
    _require(type(request) is dict and {'model', 'messages', 'temperature', 'max_tokens', 'response_format'} <= set(request)
             and set(request) <= {'model', 'messages', 'temperature', 'max_tokens', 'response_format', 'extra_body'},
             'unsupported recorded chat request')
    _require(len(canonical_json(request).encode()) <= MAX_REQUEST_BYTES, 'request byte limit')
    _text(request['model'], 'request model', 256)
    _require(type(request['messages']) is list and len(request['messages']) == 1
             and type(request['messages'][0]) is dict and set(request['messages'][0]) == {'role', 'content'}
             and request['messages'][0]['role'] == 'user', 'single-user request required')
    _text(request['messages'][0]['content'], 'complete request prompt', MAX_REQUEST_BYTES)
    _require(_number(request['temperature']) and 0 <= request['temperature'] <= 2
             and type(request['max_tokens']) is int and 0 < request['max_tokens'] <= 131072,
             'request sampling fields')
    _require('extra_body' not in request or type(request['extra_body']) is dict, 'extra_body must be an object')
    _require(_same(request['response_format'], response_format()), 'structured request schema differs')
    _require(type(response) is dict and set(response) == {'text', 'model', 'cost', 'prompt_tokens',
             'completion_tokens', 'finish_reason', 'refusal'}, 'returned response fields')
    _require(response['model'] == request['model'] and response['finish_reason'] == 'stop'
             and response['refusal'] in (None, '') and _number(response['cost']) and response['cost'] >= 0
             and all(type(response[k]) is int and response[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')),
             'incomplete returned response')
    _require(type(provider) is dict and len(canonical_json(provider).encode()) <= MAX_REQUEST_BYTES,
             'provider response byte limit')
    _text(provider.get('id'), 'recorded provider response ID', 256)
    _require(provider.get('model') == request['model'] and type(provider.get('choices')) is list
             and len(provider['choices']) == 1, 'recorded provider model/choices differ')
    choice, usage = provider['choices'][0], provider.get('usage')
    _require(type(choice) is dict and choice.get('finish_reason') == 'stop'
             and type(choice.get('index')) is int and choice['index'] == 0
             and type(choice.get('message')) is dict, 'provider choice incomplete')
    message = choice['message']
    _require(message.get('role') == 'assistant' and message.get('content') == response['text']
             and message.get('refusal') == response['refusal'], 'provider text/refusal differs')
    _require(type(usage) is dict and all(_same(usage.get(k), response[k])
             for k in ('cost', 'prompt_tokens', 'completion_tokens')), 'provider usage differs')
    decoded = builder.decode_response(response['text'])
    _require(decoded['proposal'] is not None and decoded['hypothesis'] is not None
             and decoded['hypothesis'].restart_predictions is not None, 'valid typed hypothesis/body required')
    parent = _text(value['parent_source'], 'parent source', MAX_SOURCE_BYTES)
    source = _text(value['source'], 'generated source', MAX_SOURCE_BYTES)
    _require(known_sources.get(_digest(parent)) == parent, 'parent lacks a preceding bound source')
    context = _request_context(request['messages'][0]['content'], parent, metric)
    built = builder.apply(parent, decoded['proposal'].body)
    _require(source == built['code'] and _same(value['proof'], built['proof']), 'source/proof differs from raw response')
    # Parse/compile only. Never import or evaluate a generated module.
    compile(source, '<saved-generated-proposal>', 'exec')
    from ..engine.db import normalized_hash
    _require(_digest(source) not in known_sources and normalized_hash(source) not in known_normalized,
             'generated source repeats a consumed source')
    return dict(id=ident, provenance=copy.deepcopy(value['provenance']),
        hypothesis=decoded['hypothesis'].to_dict(), body=decoded['proposal'].body,
        source=source, source_sha256=_digest(source), parent_source=parent, proof=copy.deepcopy(built['proof']),
        request_sha256=canonical_sha(request), response_sha256=_digest(response['text']),
        request_context_sha256=canonical_sha(context),
        provider_response_sha256=canonical_sha(provider),
        native_observation=dict(score=None, actual_delta=None, evaluation=None,
            forecast_verdict='INCONCLUSIVE', forecast_scored=False, reason='native target unmeasured'))


def _validate(capsule, *, expected_sha256, metric, root):
    _bounded(capsule)
    _require(_sha(expected_sha256) and canonical_sha(capsule) == expected_sha256, 'configured capsule digest differs')
    _require(type(capsule) is dict and set(capsule) == {'schema', 'profile', 'metric', 'root_parent',
             'policy', 'proposals', 'comparisons'} and capsule['schema'] == SCHEMA, 'capsule fields')
    _require(capsule['profile'] == PROFILE and _same(capsule['policy'], identity()), 'profile/policy differs')
    checked_metric = validate_metric_context(metric)
    _require(_same(capsule['metric'], checked_metric) and checked_metric['task_name'] == 'oe/circle_packing'
             and checked_metric['primary_metric'] == 'combined_score', 'receiving task metric differs')
    declared_root = capsule['root_parent']
    root_source = builder.source_for()
    _require(type(declared_root) is dict and set(declared_root) == {'id', 'source_sha256'}
             and declared_root['source_sha256'] == _digest(root_source), 'root source differs')
    _text(declared_root['id'], 'root parent ID', 128)
    _require(root is None or _same(root, declared_root), 'receiving root differs')
    _require(type(capsule['proposals']) is list and 2 <= len(capsule['proposals']) <= MAX_PROPOSALS,
             'proposal capacity')
    _require(type(capsule['comparisons']) is list and 1 <= len(capsule['comparisons']) <= MAX_COMPARISONS,
             'comparison capacity')
    from ..engine.db import normalized_hash
    known = {_digest(root_source): root_source}
    normalized = {normalized_hash(root_source)}
    records, requests, provider_ids = {}, set(), set()
    for proposal in capsule['proposals']:
        record = _proposal(proposal, known, normalized, metric)
        p = record['provenance']
        request_id = (p['manifest_sha256'], p['run_id'], p['request_key'])
        provider_id = proposal['provider_response']['id']
        _require(record['id'] not in records and request_id not in requests and provider_id not in provider_ids,
                 'duplicate proposal/request/provider identity')
        requests.add(request_id); provider_ids.add(provider_id)
        records[record['id']] = record
        known[record['source_sha256']] = record['source']
        normalized.add(normalized_hash(record['source']))
    ids, used, slots = set(), set(), {}
    shared = None
    for raw_group in capsule['comparisons']:
        group = validate_comparison(raw_group)
        _require(group['id'] not in ids and group['origin_experiment_id'] is None
                 and _same(group['task_metric'], metric), 'comparison identity/origin/metric differs')
        ids.add(group['id'])
        adapter = group['adapter_identity']
        policy = certificate_policy()
        _require(adapter.get('schema') == 'sciel-repeat-adapter/1'
                 and adapter.get('identity_sha256') == canonical_sha({k: v for k, v in adapter.items() if k != 'identity_sha256'})
                 and _same(adapter.get('certificate_policy'), policy)
                 and adapter.get('certificate_policy_sha256') == canonical_sha(policy),
                 'trusted certificate policy differs')
        _require(type(group.get('source_pair')) is dict, 'full comparison source pair required')
        for role in ('comparator', 'challenger'):
            source = group[role + '_source_sha256']
            _require(source in known and group['source_pair'][role] == known[source], 'comparison source differs')
            used.add(source)
        setting = dict(adapter_identity=group['adapter_identity'], limits=group['limits'],
                       seeds=[p['seed'] for p in group['pairs']])
        _require(shared is None or _same(shared, setting), 'comparisons mix different paired studies')
        shared = setting
        for pair in group['pairs']:
            for role in ('comparator', 'challenger'):
                key = (group[role + '_source_sha256'], pair['seed'])
                _require(key not in slots or _same(slots[key], pair[role]), 'shared draw receipt differs')
                slots[key] = pair[role]
    _require(all(r['source_sha256'] in used for r in records.values()), 'proposal has no planned paired comparison')
    return copy.deepcopy(capsule), records


def validate_capsule(capsule, *, expected_sha256, metric, root=None):
    """Validate everything before returning a detached byte-faithful JSON tree."""
    try:
        return _validate(capsule, expected_sha256=expected_sha256, metric=metric, root=root)[0]
    except (KeyError, TypeError, AttributeError, OverflowError, UnicodeError, RecursionError, SyntaxError) as exc:
        raise ValueError('malformed generated paired history') from exc


def proposal_records(capsule):
    """Re-derived foreign proposals for dedup, never ordinary record matching."""
    try:
        return copy.deepcopy(_validate(capsule, expected_sha256=canonical_sha(capsule),
                                       metric=capsule['metric'], root=None)[1])
    except (KeyError, TypeError, AttributeError, OverflowError, UnicodeError, RecursionError, SyntaxError) as exc:
        raise ValueError('malformed generated paired history') from exc


def derive_facts(capsule):
    """Source-bound hypotheses, strict paired gates and same-draw trace feedback."""
    records = proposal_records(capsule)
    comparisons = []
    for group in capsule['comparisons']:
        feedback = []
        for record in records.values():
            for role in ('comparator', 'challenger'):
                if group[role + '_source_sha256'] != record['source_sha256']:
                    continue
                rows = []
                for pair in group['pairs']:
                    receipt = pair[role]
                    result = None
                    reason = ('receipt_not_dispatched' if receipt is None else
                              'dispatch_unknown' if _same(receipt, UNKNOWN_RECEIPT) else None)
                    if reason is None:
                        try:
                            observation = observations.build_observation(receipt,
                                source_sha256=record['source_sha256'], seed=pair['seed'],
                                adapter_identity=group['adapter_identity'],
                                scaffold_sha256=record['proof']['common_scaffold_sha256'])
                            result = predictions.evaluate(record['hypothesis']['restart_predictions'], observation)
                            if result['status'] != 'available':
                                reason = result['reason'] or 'diagnostic_unavailable'
                        except (ValueError, KeyError, TypeError, OverflowError):
                            reason = 'diagnostic_invalid_or_unsupported'
                    rows.append(dict(seed=pair['seed'], receipt_sha256=canonical_sha(receipt) if receipt is not None else None,
                                     result=result, unavailable_reason=reason))
                feedback.append(dict(proposal_id=record['id'], role=role, rows=rows))
        comparisons.append(dict(group=copy.deepcopy(group), quality=_quality(group), restart_predictions=feedback))
    return dict(root_parent=copy.deepcopy(capsule['root_parent']), proposals=list(records.values()),
                comparisons=comparisons, scope=SCOPE)
