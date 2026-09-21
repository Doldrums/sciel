"""Pure, raw-bound authoring of one review of a detached evaluated study.

This object neither dispatches nor grants text or experiment authority. Its
portable sidecar is reconstructed from a separately pinned receiving R config,
complete literal provider captures and the exact prepared request.
"""
from __future__ import annotations

import copy
import json

from .component_target_generation import _transport, _receipt_error
from .evaluated_study_import import validate_bundle, configured_digests
from .external_reflection import ExternalReflectionRejected, _digest, _event, _hash, _id, _journal
from .external_research_v2 import policy_identity, _review_text_policy
from ..knowledge_base.catalogs import canonical_json, canonical_sha
from ..knowledge_base.generated_paired_history import _bounded

SCHEMA = 'sciel-evaluated-study-reflection-job/1'
PLAN_SCHEMA = 'sciel-evaluated-study-reflection-plan/1'
SIDECAR_SCHEMA = 'sciel-evaluated-study-review-sidecar/1'
JOURNAL_CONTRACT = 'sciel-evaluated-study-reflection/1'
MAX_PROMPT_BYTES = 96000
MAX_RESPONSE_BYTES = 32768
MAX_CAPTURE_BYTES = 512000
INSTRUCTION = (
    'Review the one complete detached evaluated study below. Return one bare JSON object '
    'matching the supplied static schema. No markers, code fences or surrounding prose. '
    'Use only exact fact IDs from the supplied host catalog; metadata names are not fact IDs. '
    'The host resolves references and compiles source/root/policy bindings separately; do not '
    'invent or copy extra provenance fields into your response. Selected references do not '
    'verify the truth of your prose. Every conjecture, limitation, decision reason and target '
    'remains unverified until separate semantic review.\n'
    'Read raw and exact-certified quality separately. A known loss refutes the strict quality '
    'gate even if some planned pairs were not dispatched; missing evidence is unknown, never a tie. '
    'Exact certification concerns a different packing and is not optimizer recovery of raw quality. '
    'HOST authorship carries no invented model forecast. Restart forecast agreement, retained '
    'optimizer movement, local interval gain, optimizer success and final paired quality are '
    'distinct. Accepted helper returns do not prove internal optimizer-step acceptance. Read '
    'matched control evidence before attributing any gain to an intervention. Omitted prefix '
    'geometry is unknown; full-run aggregates cannot reconstruct it. Movement is not basin proof.\n'
    'A negative, tied or incomplete past study may motivate a concrete prospective falsifiable '
    'test. No prior gain or completion of stopped measurements is required to propose a new '
    'test; neither does proposing a test show that it will pass. Choose available source hashes '
    'and one implementable change, tie it to catalog facts, and state uncertainty without causal '
    'or quality guarantees. If no sufficiently grounded test can be identified, return '
    'next_hypothesis=null with a concrete truthful reason. Do not invent a target to avoid '
    'abstaining. Do not repeat stopped experiments or supply a presumed desired algorithm.\n'
    'This is review-only: no lesson, retirement, ordinary outcome, scheduler progress, source '
    'execution or promotion is created. The whole raw response is audited; partial favorable '
    'text cannot be selected for acceptance. Structural acceptance does not authorize later G.\n\n'
)


def _require(ok, reason):
    if not ok:
        raise ValueError('evaluated study reflection: ' + reason)


def _same(a, b):
    return canonical_json(a) == canonical_json(b)


def _append(rows, plan, phase, **fields):
    _event(rows, plan['run_id'], plan['segment_id'], 'evaluated_study_reflection',
           contract=JOURNAL_CONTRACT, phase=phase, request_key=plan['request_key'], **fields)


class EvaluatedStudyReflectionJob:
    def __init__(self, bundle, *, expected_sha256, expected_authority_sha256, config):
        for value in (bundle, config):
            _bounded(value)
        _require(type(config) is dict and all(_hash(v) for v in
            (expected_sha256, expected_authority_sha256)), 'explicit input and authority digests required')
        _require(configured_digests(config) == (expected_sha256, expected_authority_sha256),
                 'independent receiving study/authority pins differ')
        _require(not any(key in config for key in ('external_history_sha256', 'external_component_sha256',
            'external_component_review_sha256', 'generated_paired_review_version',
            'external_evaluated_review_sha256', 'source_reflection_config_sha256')),
            'one standalone study without old history or self-referential review pins required')
        _require(all(config.get(key) is True for key in ('restart_predictions', 'structured_auxiliary',
            'comparison_reviews', 'inject_lessons', 'batch_only_lessons')),
            'explicit structured review-only study configuration required')
        _require(_review_text_policy(config) is not None, 'explicit host text eligibility policy required')
        _require(type(config.get('max_aux_calls')) is int and config['max_aux_calls'] == 1,
                 'exactly one auxiliary request must be configured')
        model = config.get('model')
        _require(type(model) is str and 0 < len(model) <= 256 and bool(model.strip()), 'explicit model required')
        _require(type(bundle) is dict and type(bundle.get('capsule')) is dict, 'portable study bundle required')
        checked = validate_bundle(bundle, config, metric=bundle['capsule'].get('metric'))
        self._bundle, self._config = checked, copy.deepcopy(config)
        self._policy = policy_identity()
        self._bindings = dict(source_study_sha256=expected_sha256,
            author_authority_sha256=expected_authority_sha256,
            author_dependencies_sha256=canonical_sha(checked['author_dependencies']),
            config_sha256=canonical_sha(config), policy_sha256=canonical_sha(self._policy))

    def _unchanged(self):
        _require(_same(policy_identity(), self._policy), 'production policy changed since preparation')
        _require(canonical_sha(self._bundle['capsule']) == self._bindings['source_study_sha256']
            and canonical_sha(self._bundle['author_authority']) == self._bindings['author_authority_sha256']
            and canonical_sha(self._bundle['author_dependencies']) == self._bindings['author_dependencies_sha256']
            and canonical_sha(self._config) == self._bindings['config_sha256'], 'frozen input changed')

    def prepare(self, *, run_id, segment_id, transport_settings, request_key='study_r1', max_tokens=4096):
        self._unchanged()
        _require(all(_id(v) for v in (run_id, segment_id, request_key)), 'bounded request identities required')
        producer = json.loads(self._bundle['capsule']['study']['manifest_text'])
        _require(run_id != producer.get('run_id'), 'fresh reflection run identity required')
        _require(type(max_tokens) is int and 1 <= max_tokens <= 8192, 'bounded token cap required')
        transport = _transport(transport_settings)
        from ..hypothesis import evaluated_study_review as wire
        from ..hypothesis.evaluated_study_targets import instruction as target_instruction
        catalog = wire.source_catalog(self._bundle)
        source_text = wire._source_block(catalog)
        authoring_text = wire._authoring_block(self._bundle, catalog)
        prompt = INSTRUCTION + target_instruction() + '\n\n' + authoring_text + source_text
        _require(len(prompt.encode('utf-8')) <= MAX_PROMPT_BYTES, 'complete reflection prompt exceeds byte cap')
        response_format = wire.response_format()
        request = dict(model=self._config['model'], messages=[dict(role='user', content=prompt)],
            temperature=.2, max_tokens=max_tokens, response_format=response_format, extra_body=transport)
        plan = dict(schema=PLAN_SCHEMA, **copy.deepcopy(self._bindings), run_id=run_id,
            segment_id=segment_id, request_key=request_key, max_tokens=max_tokens,
            transport_settings=transport, request=request, request_sha256=canonical_sha(request),
            model=self._config['model'], temperature=.2, prompt=prompt, prompt_sha256=_digest(prompt),
            prompt_bytes=len(prompt.encode('utf-8')), response_format=response_format,
            source_catalog_sha256=canonical_sha(catalog), authoring_block_sha256=_digest(authoring_text),
            source_block_sha256=_digest(source_text),
            new_ordinary_outcomes=0, new_search_attempts=0)
        rows = []
        _append(rows, plan, 'prepared', **self._bindings, request_sha256=plan['request_sha256'],
            prompt_sha256=plan['prompt_sha256'], response_format=response_format,
            source_catalog_sha256=plan['source_catalog_sha256'],
            new_ordinary_outcomes=0, new_search_attempts=0)
        prefix = _journal(rows)
        plan.update(journal_prefix=prefix, journal_prefix_sha256=_digest(prefix))
        return copy.deepcopy(plan)

    def admit(self, plan, response, *, manifest_sha256, journal_prefix, provider_request, provider_response):
        _bounded(plan)
        _require(type(plan) is dict and _hash(manifest_sha256), 'prepared plan and manifest digest required')
        expected = self.prepare(run_id=plan.get('run_id'), segment_id=plan.get('segment_id'),
            request_key=plan.get('request_key'), max_tokens=plan.get('max_tokens'),
            transport_settings=plan.get('transport_settings'))
        _require(_same(plan, expected), 'prepared request changed')
        _require(type(journal_prefix) is str and journal_prefix == expected['journal_prefix'],
                 'persisted request prefix differs')
        rows = [json.loads(line) for line in journal_prefix.splitlines()]
        raw = response.get('text') if type(response) is dict else None
        raw_sha = None
        try:
            raw_sha = _digest(raw) if type(raw) is str else None
            for value in (response, provider_request, provider_response):
                _bounded(value)
                _require(len(canonical_json(value).encode('utf-8')) <= MAX_CAPTURE_BYTES, 'capture byte limit')
        except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
            _append(rows, expected, 'invalid_capture', response_sha256=raw_sha,
                response=raw[:MAX_RESPONSE_BYTES] if type(raw) is str and raw_sha is not None else None,
                response_truncated=True, error=dict(code='capture_bound', path='response'))
            raise ExternalReflectionRejected('bounded literal response/capture required', _journal(rows)) from None
        too_large = type(raw) is str and len(raw.encode('utf-8')) > MAX_RESPONSE_BYTES
        _append(rows, expected, 'returned', response_sha256=raw_sha,
            response=raw[:MAX_RESPONSE_BYTES] if type(raw) is str else None,
            response_truncated=too_large, response_metadata_sha256=canonical_sha(response),
            provider_request_sha256=canonical_sha(provider_request), provider_response_sha256=canonical_sha(provider_response))
        error = _receipt_error(response, provider_request, provider_response, expected)
        if error is None:
            # self._bundle was fully authenticated at construction and its
            # immutable bindings were rechecked by prepare above.
            from ..knowledge_base.evaluated_detached_study import _retained_ancestry
            if provider_response['id'] in _retained_ancestry(self._bundle)['provider_response_ids']:
                error = dict(code='historical_provider_replay', path='provider_response.id')
        if error is None and too_large:
            error = dict(code='response_bound', path='response.text')
        from ..hypothesis import evaluated_study_review as wire
        card = None
        if error is None:
            decoded = wire.decode_response(raw)
            error = decoded['error']
            if error is None:
                try:
                    card = wire.build_card(decoded['review'], self._bundle,
                        dict(phase='reflection', reference=1, raw_response_sha256=raw_sha))
                except (ValueError, KeyError, TypeError, OverflowError) as exc:
                    error = dict(code='review_admission', path='review', detail=str(exc)[:1024])
        _append(rows, expected, 'admission', status='accepted' if error is None else 'invalid', error=error,
            review_card_sha256=canonical_sha(card) if card is not None else None,
            new_ordinary_outcomes=0, new_search_attempts=0, new_lesson_ids=[], retired_lesson_ids=[],
            text_eligibility='pending_host_decision', generation_allowed=False)
        journal = _journal(rows)
        if error is not None:
            raise ExternalReflectionRejected('returned response rejected: ' + error['code'], journal)
        sidecar = dict(schema=SIDECAR_SCHEMA, **copy.deepcopy(self._bindings), policy=copy.deepcopy(self._policy),
            config=copy.deepcopy(self._config), plan=copy.deepcopy(expected), response=copy.deepcopy(response),
            provider_request=copy.deepcopy(provider_request), provider_response=copy.deepcopy(provider_response),
            manifest_sha256=manifest_sha256, journal=journal, review_card=card)
        _bounded(sidecar)
        return dict(schema=SCHEMA, status='accepted', journal=journal, journal_sha256=_digest(journal),
            extended_capsule=sidecar, extended_capsule_sha256=canonical_sha(sidecar),
            accepted_review_ids=[card['comparison_id']], review_card_sha256=canonical_sha(card),
            known_cost=float(response['cost']), recorded_provider_responses=1,
            provider_dispatch_verified=False, generation_allowed=False, execution_performed=False,
            new_ordinary_outcomes=0, new_search_attempts=0, new_lesson_ids=[], retired_lesson_ids=[])


def validate_sidecar(sidecar, *, bundle, expected_sha256, config):
    """Replay recorded authoring with external config; not a text or G approval."""
    _bounded(sidecar)
    _require(_hash(expected_sha256) and canonical_sha(sidecar) == expected_sha256
        and type(sidecar) is dict and sidecar.get('schema') == SIDECAR_SCHEMA, 'declared sidecar differs')
    _require(_same(sidecar.get('config'), config), 'independent reflection configuration differs')
    expected, authority = configured_digests(config)
    job = EvaluatedStudyReflectionJob(bundle, expected_sha256=expected,
        expected_authority_sha256=authority, config=config)
    plan = sidecar.get('plan')
    _require(type(plan) is dict, 'stored plan required')
    rebuilt = job.admit(plan, sidecar.get('response'), manifest_sha256=sidecar.get('manifest_sha256'),
        journal_prefix=plan.get('journal_prefix'), provider_request=sidecar.get('provider_request'),
        provider_response=sidecar.get('provider_response'))['extended_capsule']
    _require(_same(sidecar, rebuilt), 'sidecar differs from raw request, response, card or journal')
    return copy.deepcopy(rebuilt)
