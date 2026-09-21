"""One pure, source-bound reflection over generation-only paired evidence.

The result is a separate immutable review sidecar. It neither appends an
ordinary outcome nor advances a scheduler watermark. Provider dispatch,
durability, cost reservations and semantic text eligibility remain explicit
responsibilities of the transport owner and receiving host.
"""
from __future__ import annotations

import copy
import math

from .external_reflection import (ExternalReflectionRejected, _digest, _event,
    _hash, _id, _journal, _research, _RESPONSE_FIELDS, MAX_RETAINED_RESPONSE_CHARS)
from .external_research_v2 import policy_identity, _review_text_policy
from ..hypothesis.auxiliary_schema import response_format as auxiliary_format
from ..hypothesis.parsing import decode_reflection
from ..hypothesis.review_eligibility import require_review_only
from ..knowledge_base.catalogs import canonical_json, canonical_sha
from ..knowledge_base.generated_paired_history import validate_capsule, _bounded
from ..knowledge_base.generated_paired_context import build_context, block

SCHEMA = 'sciel-generated-paired-reflection-job/2'
PLAN_SCHEMA = 'sciel-generated-paired-reflection-plan/2'
SIDECAR_SCHEMA = 'sciel-generated-paired-review-sidecar/2'
SCOPED_SCHEMA = 'sciel-generated-paired-reflection-job/3'
SCOPED_PLAN_SCHEMA = 'sciel-generated-paired-reflection-plan/3'
SCOPED_SIDECAR_SCHEMA = 'sciel-generated-paired-review-sidecar/3'
MAX_PROMPT_CHARS = 90000
INSTRUCTION = (
    'Review the complete saved generation-only paired study below. Return one bare JSON '
    'object matching the supplied schema; no markers, fences or surrounding prose. '
    'lessons and retire_ids MUST both be empty. Write zero to four reasoned comparison '
    'reviews using the exact group IDs, SHA256 bindings and named facts supplied below. '
    'claims.fact_id must be a key in that group catalog\'s facts object, with its exact '
    'string value. comparison_id, comparison_sha256 and other metadata are bindings, '
    'not claim fact IDs. The schema enum combines groups; membership and value in the '
    'particular reviewed group are still checked locally. '
    'The generated-paired catalogs are authoritative for this request: forecasts refer '
    'to the explicitly named comparator or challenger author, not an invented ordinary '
    'origin. Keep stored group classification separate from the strict no-loss verdict. '
    'Describe the actual failed or unknown predictions, identify what the traces can '
    'and cannot explain. Use the source registry and role mapping for EACH group; '
    'the same author/source can change roles across groups. Read each expected, '
    'observed, forecast_verdict triple together. REFUTED means a disagreement with '
    'the forecast, not observed=false. Unknown is not false. '
    'Mixed, tied or refuted evidence may motivate a concrete falsifiable next test '
    'without a previous gain or completion of stopped measurements. Name one '
    'implementable change to an available source, the motivating evidence and the '
    'uncertainty to test; proposing it does not demonstrate or guarantee that the quality gate will pass. '
    'Both target source hashes must belong to the selected comparison, and may coincide. '
    'If you cannot identify such a test, return null with a concrete reason for your '
    'abstention. Absence of a previous gain is not proof that no useful future test '
    'exists. Do not invent a target to avoid abstaining. Keep next_hypothesis_reason '
    'concise and within 1024 characters for either decision. Source changes, conjectures and explanations remain '
    'unverified. Do not treat conditional explanations as demonstrated causes. '
    'A proposed target is not permission to execute, repeat stopped runs or promote a '
    'candidate. Native candidate forecasts remain unmeasured. No ordinary outcome, '
    'lesson support, exposure credit or scheduler progress is created by this review.\n\n'
)


def _require(ok, message):
    if not ok:
        raise ValueError('generated paired reflection: ' + message)


def response_format(component=None, *, version=1):
    from ..hypothesis.generated_paired_review import RULES, fact_vocabulary
    wire = copy.deepcopy(auxiliary_format('reflection', comparison_reviews=True,
                                           comparison_review_version=3))
    wire['json_schema']['name'] = 'sciel_generated_paired_reflection_v1'
    properties = wire['json_schema']['schema']['properties']
    properties['comparison_reviews']['description'] = (
        'Zero to four distinct group reviews; all reasoned proposal bounds are enforced '
        'locally. This request uses generated-paired source catalogs. ' + RULES)
    for key in ('lessons', 'retire_ids'):
        properties[key]['description'] = 'Must be empty for this review-only component; enforced locally.'
    _require(type(version) is int and version in (1, 2), 'unsupported generated review version')
    if version == 2:
        _require(component is not None, 'scoped wire requires its authenticated component')
        from ..hypothesis import scoped_paired_review
        wire['json_schema']['name'] = 'sciel_generated_paired_reflection_v3'
        properties['comparison_reviews'] = scoped_paired_review.json_schema(component)
        return wire
    if component is not None:
        vocabulary = fact_vocabulary(component)
        wire['json_schema']['name'] = 'sciel_generated_paired_reflection_v2'
        proposal = properties['comparison_reviews']['items']['properties']
        proposal['comparison_id']['enum'] = vocabulary['comparison_ids']
        proposal['comparison_sha256']['enum'] = vocabulary['comparison_sha256']
        claim = proposal['claims']['items']['properties']
        claim['fact_id']['enum'] = vocabulary['fact_ids']
        claim['fact_id']['description'] = (
            'Exact key of facts in the selected group catalog; never comparison_id, '
            'comparison_sha256 or another metadata key. This enum is the union across '
            'groups; the selected group and exact value are checked locally.')
        for key in ('base_source_sha256', 'comparator_source_sha256'):
            proposal['next_hypothesis']['properties'][key]['enum'] = vocabulary['source_sha256']
            proposal['next_hypothesis']['properties'][key]['description'] += (
                ' This enum combines groups. Both target sources must belong to the '
                'selected comparison, and may coincide; checked locally.')
        proposal['next_hypothesis_reason']['description'] += (
            ' A mixed, tied or refuted result may motivate a prospective test without '
            'previous gain. State motivating evidence and uncertainty, not a guarantee. '
            'A null target is a permitted abstention, not proof that no useful test exists.')
        properties['comparison_reviews']['description'] += (
            ' At most 4 reviews and 16384 canonical JSON characters including reasons. '
            'Prior gain is not required to propose a future test; justified null is allowed.')
    return wire


def _decode_scoped(response):
    """Separate bare-object transport; never upgrade a legacy reply in place."""
    import json
    from ..hypothesis.models import Reflection
    from ..hypothesis.parsing import _unique_object, _reject_constant
    from ..hypothesis.reasoned_review import ReasonedReviewError
    from ..hypothesis.scoped_paired_review import validate_proposals
    try:
        value = json.loads(response, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        _require(type(value) is dict and set(value) == {'lessons', 'retire_ids', 'comparison_reviews'},
                 'invalid scoped reflection fields')
        _require(type(value['lessons']) is list and not value['lessons']
            and type(value['retire_ids']) is list and not value['retire_ids'],
            'scoped reflection is review-only')
        proposals = validate_proposals(value['comparison_reviews'])
        return dict(reflection=Reflection([], [], proposals), error=None)
    except ReasonedReviewError as exc:
        return dict(reflection=None, error=exc.diagnostic)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return dict(reflection=None, error=dict(code='invalid_scoped_reflection', path='comparison_reviews'))


class GeneratedPairedReflectionJob:
    def __init__(self, component, *, expected_sha256, config):
        _bounded(component)
        _bounded(config)
        _require(type(config) is dict and _hash(expected_sha256)
                 and config.get('external_component_sha256') == expected_sha256,
                 'explicit source component digest required')
        _require(not any(key in config for key in
            ('external_history_sha256', 'external_component_review_sha256')),
            'this job reviews exactly one component, without another history or review')
        _require(all(config.get(key) is True for key in
            ('restart_predictions', 'structured_auxiliary', 'comparison_reviews', 'inject_lessons'))
            and type(config.get('comparison_review_version')) is int
            and config['comparison_review_version'] == 3,
            'explicit structured reasoned review configuration required')
        _require(type(config.get('max_aux_calls')) is int and config['max_aux_calls'] >= 1,
                 'one auxiliary request must be explicitly allowed')
        _require(type(config.get('model')) is str and 0 < len(config['model']) <= 256
                 and bool(config['model'].strip()),
                 'explicit model required')
        _require(_review_text_policy(config) is not None,
                 'explicit host text policy required; model cannot issue its own acceptance')
        from .generated_paired_review_import import configured_version
        self._review_version = configured_version(config)
        self._job_schema = SCOPED_SCHEMA if self._review_version == 2 else SCHEMA
        self._plan_schema = SCOPED_PLAN_SCHEMA if self._review_version == 2 else PLAN_SCHEMA
        self._sidecar_schema = SCOPED_SIDECAR_SCHEMA if self._review_version == 2 else SIDECAR_SCHEMA
        self._component, self._config = copy.deepcopy(component), copy.deepcopy(config)
        self._expected_sha256, self._policy = expected_sha256, policy_identity()
        self._validated()

    def _validated(self):
        _require(canonical_json(policy_identity()) == canonical_json(self._policy),
                 'production policy changed since preparation')
        return validate_capsule(self._component, expected_sha256=self._expected_sha256,
                                metric=self._component['metric'])

    def prepare(self, *, run_id, segment_id, interval=1):
        _require(_id(run_id) and _id(segment_id), 'bounded run and segment identities required')
        _require(type(interval) is int and interval == 1,
                 'one component reflection has no ordinary interval or watermark')
        component = self._validated()
        _require(run_id not in {p['provenance']['run_id'] for p in component['proposals']},
                 'fresh reflection run identity required')
        from ..hypothesis.generated_paired_review import source_block, authoring_block
        from ..hypothesis.comparison_targets import instruction as target_instruction
        projection = build_context([component], branch=component['root_parent']['id'],
                                   max_chars=60000, max_items=1)
        _require(projection is not None and projection['presented_count'] == 1,
                 'complete study must fit the reflection prompt')
        sources = source_block(component)
        authoring = authoring_block(component)
        instruction = INSTRUCTION
        target_rules = target_instruction()
        if self._review_version == 2:
            from ..hypothesis.scoped_paired_review import RULES as scoped_rules
            instruction = instruction.replace(
                'Both target source hashes must belong to the selected comparison, and may coincide. ',
                'The reviewed comparison supplies motivating evidence. Both next-target source hashes '
                'may come from the complete authenticated study and may coincide. ')
            # The new target changes source scope, not quality or diagnostic
            # semantics. Preserve the shared contract's complete scientific rule.
            target_rules = (scoped_rules + target_rules[target_rules.index('The caller must verify'):]
                + '\nSCOPED_TARGET_BINDINGS_JSON:\n' + canonical_json(dict(
                component_sha256=self._expected_sha256, receiving_root=component['root_parent'],
                policy_sha256=canonical_sha(component['policy']))))
        prompt = instruction + target_rules + '\n\n' + authoring + block(projection) + sources
        _require(len(prompt) <= MAX_PROMPT_CHARS, 'complete reflection prompt exceeds limit')
        wire = response_format(component, version=self._review_version)
        wire_identity = dict(schema=('sciel-generated-paired-auxiliary-schema/3'
                                     if self._review_version == 2 else 'sciel-generated-paired-auxiliary-schema/2'),
                             response_format_sha256=canonical_sha(wire))
        rows = []
        _event(rows, run_id, segment_id, 'run_start', schema_version=2, resumed=False,
               scope='new generation-only paired reflection', job_schema=self._job_schema)
        _research(rows, run_id, segment_id, 'generated_paired_reflection_start',
            source_component_sha256=self._expected_sha256,
            policy_sha256=canonical_sha(self._policy),
            author_sequence=1, new_ordinary_outcomes=0, new_search_attempts=0,
            scope='Author sequence identifies this R only; it is not a native outcome watermark.')
        _research(rows, run_id, segment_id, 'generated_paired_reflection_context',
            generated_paired_history=projection, source_catalog_block_sha256=_digest(sources),
            authoring_block_sha256=_digest(authoring),
            source_comparison_ids=[g['id'] for g in component['comparisons']],
            prompt_sha256=_digest(prompt))
        _research(rows, run_id, segment_id, 'aux_start', kind='reflection', reference=1,
            model=self._config['model'], call_index=1, prompt_sha256=_digest(prompt),
            response_format=wire, auxiliary_schema=wire_identity)
        prefix = _journal(rows)
        return dict(schema=self._plan_schema, source_component_sha256=self._expected_sha256,
            config_sha256=canonical_sha(self._config), job_policy=copy.deepcopy(self._policy),
            run_id=run_id, segment_id=segment_id, interval=1, author_sequence=1,
            model=self._config['model'], temperature=.2, prompt=prompt,
            prompt_sha256=_digest(prompt), prompt_bytes=len(prompt.encode('utf-8')),
            response_format=wire, auxiliary_schema=wire_identity,
            journal_prefix=prefix, journal_prefix_sha256=_digest(prefix),
            new_ordinary_outcomes=0, new_search_attempts=0)

    def _result(self, plan, response, *, manifest_sha256, journal_prefix):
        _require(type(plan) is dict and _hash(manifest_sha256), 'prepared plan and manifest required')
        expected = self.prepare(run_id=plan.get('run_id'), segment_id=plan.get('segment_id'),
                                interval=plan.get('interval'))
        _require(canonical_json(plan) == canonical_json(expected), 'prepared request changed')
        _require(type(journal_prefix) is str and journal_prefix == expected['journal_prefix'],
                 'persisted request prefix differs')
        _require(type(response) is dict and set(response) == _RESPONSE_FIELDS,
                 'complete returned response metadata required')
        raw, cost = response['text'], response['cost']
        finite_cost = (type(cost) is int and 0 <= cost <= 10**100
                       or type(cost) is float and math.isfinite(cost) and cost >= 0)
        _require(type(raw) is str and response['model'] == expected['model'] and finite_cost,
                 'returned model, text or known cost differs')
        _require(all(type(response[key]) is int and 0 <= response[key] <= 10**9
                     for key in ('prompt_tokens', 'completion_tokens')), 'invalid token metadata')
        for key in ('finish_reason', 'refusal'):
            _require(response[key] is None or type(response[key]) is str, 'invalid response ' + key)
        try:
            raw_sha = _digest(raw)
        except UnicodeError as exc:
            raise ValueError('generated paired reflection: invalid response Unicode') from exc
        import json
        rows = list(map(json.loads, journal_prefix.splitlines()))
        run_id, segment_id = expected['run_id'], expected['segment_id']
        usable = response['finish_reason'] == 'stop' and response['refusal'] in (None, '')
        truncated = len(raw) > MAX_RETAINED_RESPONSE_CHARS
        _research(rows, run_id, segment_id, 'aux_result', kind='reflection', reference=1,
            model=response['model'], call_index=1, cost=float(cost),
            prompt_tokens=response['prompt_tokens'], completion_tokens=response['completion_tokens'],
            response=raw[:MAX_RETAINED_RESPONSE_CHARS], response_sha256=raw_sha,
            response_truncated=truncated,
            finish_reason=(response['finish_reason'] if type(response['finish_reason']) is str
                           and len(response['finish_reason']) <= 128 else None),
            refusal=response['refusal'][:2048] if type(response['refusal']) is str else None,
            refusal_truncated=type(response['refusal']) is str and len(response['refusal']) > 2048,
            structured_response_usable=usable)
        if not usable or truncated:
            _research(rows, run_id, segment_id, 'aux_stopped', kind='reflection', reference=1,
                call_index=1, stage='response', generation_returned=True, cost=float(cost),
                current_call_cost_unknown=False, error_type='UnusableGeneratedPairedReflectionResponse')
            raise ExternalReflectionRejected('complete untruncated non-refused R required', _journal(rows))
        from ..hypothesis.generated_paired_review import build_cards
        if self._review_version == 2:
            from ..hypothesis.scoped_paired_review import build_cards
        # Preserve a rejected legacy envelope verbatim; this structured wire
        # promises a bare object and never repairs or strips response framing.
        bare = raw.strip().startswith('{') and raw.strip().endswith('}')
        decoded = ((_decode_scoped(raw) if self._review_version == 2 else
                    decode_reflection(raw, comparison_reviews=True, comparison_review_version=3))
                   if bare else dict(reflection=None,
                       error=dict(code='bare_object_required', path='response')))
        parsed = decoded['reflection']
        cards, error = {}, None
        try:
            _require(parsed is not None,
                     'invalid reasoned reflection response: ' + canonical_json(decoded['error']))
            require_review_only(parsed)
            cards = build_cards(parsed.comparison_reviews, self._component,
                dict(phase='reflection', reference=1, raw_response_sha256=raw_sha))
        except (ValueError, TypeError, KeyError) as exc:
            error = exc
        _research(rows, run_id, segment_id, 'generated_paired_reflection', author_sequence=1,
            status='parsed' if parsed is not None else 'invalid',
            source_component_sha256=self._expected_sha256,
            comparison_review_ids=list(cards) if error is None else [],
            error=type(error).__name__ if error else None,
            error_detail=str(error)[:1024] if error else None,
            error_diagnostic=copy.deepcopy(decoded['error']),
            new_ordinary_outcomes=0, new_search_attempts=0, new_lesson_ids=[], retired_lesson_ids=[])
        journal = _journal(rows)
        if error is not None:
            raise ExternalReflectionRejected('reply or admission is invalid: ' + str(error), journal)
        sidecar = dict(schema=self._sidecar_schema, source_component_sha256=self._expected_sha256,
            policy=copy.deepcopy(self._policy), config=copy.deepcopy(self._config),
            plan=copy.deepcopy(plan), response=copy.deepcopy(response),
            manifest_sha256=manifest_sha256, journal=journal, review_cards=cards)
        return dict(schema=self._job_schema, status='accepted', journal=journal,
            journal_sha256=_digest(journal), extended_capsule=sidecar,
            extended_capsule_sha256=canonical_sha(sidecar),
            accepted_review_ids=list(cards), new_lesson_ids=[], retired_lesson_ids=[],
            known_cost=float(cost), recorded_provider_responses=1, provider_dispatch_verified=False,
            new_ordinary_outcomes=0, new_search_attempts=0)

    def admit(self, plan, response, *, manifest_sha256, journal_prefix):
        result = self._result(plan, response, manifest_sha256=manifest_sha256,
                              journal_prefix=journal_prefix)
        validate_sidecar(result['extended_capsule'], component=self._component,
                         expected_sha256=result['extended_capsule_sha256'])
        return result


def validate_sidecar(sidecar, *, component, expected_sha256):
    """Rebuild the exact prepared prompt, raw writer, cards and whole journal."""
    _bounded(sidecar)
    _require(_hash(expected_sha256) and canonical_sha(sidecar) == expected_sha256,
             'configured review digest differs')
    _require(type(sidecar) is dict and set(sidecar) == {'schema', 'source_component_sha256',
        'policy', 'config', 'plan', 'response', 'manifest_sha256', 'journal', 'review_cards'}
        and sidecar['schema'] in (SIDECAR_SCHEMA, SCOPED_SIDECAR_SCHEMA), 'invalid sidecar fields')
    _require(sidecar['source_component_sha256'] == canonical_sha(component), 'source component differs')
    _require(canonical_json(sidecar['policy']) == canonical_json(policy_identity()), 'review policy differs')
    job = GeneratedPairedReflectionJob(component,
        expected_sha256=sidecar['source_component_sha256'], config=sidecar['config'])
    _require(sidecar['schema'] == job._sidecar_schema, 'sidecar version differs from explicit configuration')
    plan = sidecar['plan']
    _require(type(plan) is dict, 'prepared plan required')
    result = job._result(plan, sidecar['response'], manifest_sha256=sidecar['manifest_sha256'],
                         journal_prefix=plan.get('journal_prefix'))
    _require(canonical_json(result['extended_capsule']) == canonical_json(sidecar),
             'sidecar differs from its raw writer and authoritative source')
    return copy.deepcopy(sidecar)
