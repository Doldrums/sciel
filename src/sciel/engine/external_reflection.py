"""Pure planning and admission for one new reflection over foreign evidence.

This API owns no transport, file, Engine, scheduler or local research counters.
The caller persists the prepared journal prefix before a single paid request and
retains the original provider/ledger receipts separately. A supplied response
authenticates no provider by itself; the resulting capsule binds recorded bytes
and admissions, not the truth or causal value of the model's interpretation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

from ..hypothesis import CONTRACT
from ..hypothesis.comparison_review import build_cards, MAX_STORED
from ..hypothesis.parsing import parse_reflection
from ..knowledge_base.catalogs import canonical_json, canonical_sha
from ..knowledge_base.paired_comparisons import SCHEMA as PAIRED_SCHEMA
from .external_research_v2 import SCHEMA as HISTORY_SCHEMA, policy_identity, validate
from .reflection_context import prepare_reflection_context

SCHEMA = 'sciel-external-reflection-job/1'
PLAN_SCHEMA = 'sciel-external-reflection-plan/1'
MAX_RETAINED_RESPONSE_CHARS = 32768
_RESPONSE_FIELDS = {'text', 'model', 'cost', 'prompt_tokens', 'completion_tokens', 'finish_reason', 'refusal'}


def _require(ok, message):
    if not ok:
        raise ValueError('external reflection: ' + message)


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _id(value):
    return type(value) is str and 0 < len(value) <= 128 and not any(c.isspace() or ord(c) < 32 for c in value)


def _event(rows, run_id, segment_id, event, **fields):
    rows.append(dict(event=event, **fields, run_id=run_id, segment_id=segment_id, event_seq=len(rows) + 1))


def _research(rows, run_id, segment_id, phase, **fields):
    _event(rows, run_id, segment_id, 'research', phase=phase, contract=CONTRACT, **fields)


def _journal(rows):
    return ''.join(canonical_json(row) + '\n' for row in rows)


class ExternalReflectionRejected(ValueError):
    """A returned reply was recorded but cannot produce an accepted capsule."""

    def __init__(self, message, journal):
        super().__init__('external reflection: ' + message)
        self.journal = journal
        self.journal_sha256 = _digest(journal)


class ExternalReflectionJob:
    """One immutable input history; preparing/admitting never invokes a model.

    Configuration is the effective research-only mapping. It must explicitly
    bind the exact current-policy v2 capsule. A run ID and segment ID are fixed
    before prepare; constructing a new job is not permission to repeat a paid
    request. One-shot execution and unknown-outcome holds belong to the caller's
    durable ledger, not to this re-entrant pure validation object.
    """

    def __init__(self, capsule, *, expected_sha256, config):
        _require(type(config) is dict and _hash(expected_sha256)
                 and config.get('external_history_sha256') == expected_sha256,
                 'explicit capsule digest required')
        _require(type(capsule) is dict and capsule.get('schema') == HISTORY_SCHEMA,
                 'current v2 history required')
        _require(all(config.get(key) is True for key in
                     ('restart_predictions', 'structured_auxiliary', 'comparison_reviews')),
                 'structured traced comparison-review policy required')
        _require(type(config.get('max_aux_calls')) is int and config['max_aux_calls'] >= 1,
                 'one auxiliary request must be explicitly allowed')
        _require(type(config.get('model')) is str and bool(config['model'].strip()), 'explicit model required')
        self._capsule = copy.deepcopy(capsule)
        self._config = copy.deepcopy(config)
        self._expected_sha256 = expected_sha256
        self._policy = policy_identity()
        self._validated()

    def _validated(self):
        _require(policy_identity() == self._policy, 'production policy changed since preparation')
        # The reducer enforces bounds before the capsule is canonicalized here.
        restored = validate(self._capsule, config=self._config, metric=self._capsule['metric'])
        _require(canonical_sha(self._capsule) == self._expected_sha256,
                 'capsule differs from its declared digest')
        return restored

    def _preparation(self, *, interval):
        _require(type(interval) is int and 1 <= interval <= 256, 'interval must be a bounded positive integer')
        source = self._validated()
        historical_through = len(source['knowledge'].state()['seen_experiments'])
        prepared = prepare_reflection_context(source['knowledge'], source['conclusions'],
            config=self._config, interval=interval, through=historical_through)
        _require(prepared['response_format'] is not None and prepared['comparison_review_groups'] is not None,
                 'structured reflection preparation is unavailable')
        return source, historical_through, prepared

    def prepare(self, *, run_id, segment_id, interval=1):
        """Return exact request bytes and a prefix to persist before dispatch."""
        _require(_id(run_id) and _id(segment_id), 'bounded run and segment identities required')
        foreign = [json.loads(block['journal'].splitlines()[0]) for block in self._capsule['journals']]
        _require(run_id not in {row['run_id'] for row in foreign}
                 and segment_id not in {row['segment_id'] for row in foreign}, 'fresh journal identities required')
        _, through, prepared = self._preparation(interval=interval)
        rows = []
        _event(rows, run_id, segment_id, 'run_start', schema_version=2, resumed=False,
               scope='new external-history reflection job', job_schema=SCHEMA)
        _research(rows, run_id, segment_id, 'external_reflection_start',
            source_capsule_sha256=self._expected_sha256, policy_sha256=canonical_sha(self._policy),
            historical_through=through, interval=interval, new_ordinary_outcomes=0,
            new_search_attempts=0, auxiliary_call_index=1,
            scope='Historical watermark identifies source evidence, never local scheduler outcomes.')
        for event in prepared['presentation_events']:
            _research(rows, run_id, segment_id, **event)
        _research(rows, run_id, segment_id, 'reflection_context', **prepared['context_event'])
        prompt = prepared['prompt']
        _research(rows, run_id, segment_id, 'aux_start', kind='reflection', reference=through,
            model=self._config['model'], call_index=1, prompt_sha256=_digest(prompt),
            response_format=prepared['response_format'], auxiliary_schema=prepared['auxiliary_schema'])
        prefix = _journal(rows)
        return dict(schema=PLAN_SCHEMA, source_capsule_sha256=self._expected_sha256,
            config_sha256=canonical_sha(self._config), job_policy=copy.deepcopy(self._policy),
            run_id=run_id, segment_id=segment_id, interval=interval, historical_through=through,
            model=self._config['model'], temperature=.2,
            prompt=prompt, prompt_sha256=_digest(prompt), prompt_bytes=len(prompt.encode('utf-8')),
            response_format=copy.deepcopy(prepared['response_format']),
            auxiliary_schema=copy.deepcopy(prepared['auxiliary_schema']),
            journal_prefix=prefix, journal_prefix_sha256=_digest(prefix),
            new_ordinary_outcomes=0, new_search_attempts=0)

    def admit(self, plan, response, *, manifest_sha256, journal_prefix):
        """Bind one complete reply and return a separately replayed new capsule.

        ``journal_prefix`` is the exact persisted execute-time prefix, supplied
        by the transport owner. Equality cannot prove that it was fsynced or
        that a provider was called; the runner's receipt/ledger does that.
        """
        _require(type(plan) is dict and _hash(manifest_sha256), 'prepared plan and manifest digest required')
        expected = self.prepare(run_id=plan.get('run_id'), segment_id=plan.get('segment_id'),
                                interval=plan.get('interval'))
        _require(canonical_json(plan) == canonical_json(expected), 'prepared request changed')
        _require(type(journal_prefix) is str and journal_prefix == expected['journal_prefix'],
                 'persisted request prefix differs')
        _require(type(response) is dict and set(response) == _RESPONSE_FIELDS,
                 'complete explicit returned-response fields required')
        raw, model, cost = response['text'], response['model'], response['cost']
        _require(type(raw) is str and type(model) is str and model == expected['model']
                 and type(cost) in (int, float) and math.isfinite(cost) and cost >= 0,
                 'returned response identity or known cost differs')
        _require(all(type(response[key]) is int and 0 <= response[key] <= 10**9
                     for key in ('prompt_tokens', 'completion_tokens')), 'invalid returned token metadata')
        _require(response['finish_reason'] is None or type(response['finish_reason']) is str,
                 'invalid finish metadata')
        _require(response['refusal'] is None or type(response['refusal']) is str, 'invalid refusal metadata')
        # Reject malformed Unicode before constructing a claimed raw digest.
        try:
            raw_sha = _digest(raw)
        except UnicodeError as exc:
            raise ValueError('external reflection: response is not valid UTF-8 text') from exc
        run_id, segment_id, through = expected['run_id'], expected['segment_id'], expected['historical_through']
        rows = list(map(json.loads, journal_prefix.splitlines()))
        usable = response['finish_reason'] == 'stop' and response['refusal'] in (None, '')
        truncated = len(raw) > MAX_RETAINED_RESPONSE_CHARS
        finish = response['finish_reason']
        refusal = response['refusal']
        _research(rows, run_id, segment_id, 'aux_result', kind='reflection', reference=through,
            model=model, call_index=1, cost=float(cost), prompt_tokens=response['prompt_tokens'],
            completion_tokens=response['completion_tokens'], response=raw[:MAX_RETAINED_RESPONSE_CHARS],
            response_sha256=raw_sha, response_truncated=truncated,
            finish_reason=finish if type(finish) is str and len(finish) <= 128 else None,
            refusal=refusal[:2048] if type(refusal) is str else None,
            refusal_truncated=type(refusal) is str and len(refusal) > 2048, structured_response_usable=usable)
        if not usable or truncated:
            _research(rows, run_id, segment_id, 'aux_stopped', kind='reflection', reference=through,
                call_index=1, stage='response', generation_returned=True, cost=float(cost),
                current_call_cost_unknown=False, error_type='UnusableExternalReflectionResponse')
            raise ExternalReflectionRejected('complete untruncated non-refused reflection required', _journal(rows))
        source, _, prepared = self._preparation(interval=expected['interval'])
        kb, conclusions = source['knowledge'], source['conclusions']
        reviews = copy.deepcopy(source['comparison_reviews'])
        result = parse_reflection(raw, comparison_reviews=True,
            comparison_review_version=self._config.get('comparison_review_version', 1))
        applied, changed, review_changed, additions = None, False, False, {}
        error = None
        before = kb.lessons
        if result is not None:
            try:
                if 'comparison_review_text_policy' in self._config:
                    from ..hypothesis.review_eligibility import require_review_only
                    require_review_only(result)
                _require(set(result.retire_ids) <= {lesson['id'] for lesson in prepared['lessons']
                                                  if lesson.get('retirement_eligible') is True},
                         'retirement requires eligible presented lessons')
                writer = dict(phase='reflection', reference=through, raw_response_sha256=raw_sha)
                additions = build_cards(result.comparison_reviews, prepared['comparison_review_groups'], writer,
                    version=self._config.get('comparison_review_version', 1), records=kb._evidence(), seen=kb._seen)
                updated = copy.deepcopy(reviews)
                for ident, card in additions.items():
                    updated.pop(ident, None)
                    updated[ident] = card
                updated = dict(list(updated.items())[-MAX_STORED:])
                applied = kb.reflect(result.lessons, result.retire_ids,
                    allowed_evidence_ids=[record['id'] for record in prepared['records']], writer_context=writer)
                changed, review_changed = kb.lessons != before, updated != reviews
                reviews = updated
            except (ValueError, TypeError, KeyError) as exc:
                error = exc
        _research(rows, run_id, segment_id, 'reflection', through=through, applied=applied,
            changed=changed, error=type(error).__name__ if error else None,
            error_detail=str(error)[:1024] if error else None,
            comparison_review_ids=list(additions) if error is None else [],
            comparison_reviews_changed=review_changed, status='parsed' if result is not None else 'invalid')
        journal = _journal(rows)
        if result is None or error is not None:
            raise ExternalReflectionRejected('reply or admission is invalid', journal)
        retained = {ident for ident, record in kb._evidence().items() if record.get('schema') == PAIRED_SCHEMA}
        reviews = {ident: card for ident, card in reviews.items() if ident in retained}
        ordinary_ids = {record['id'] for record in kb.experiments}
        conclusions = {ident: review for ident, review in conclusions.items() if ident in ordinary_ids}
        terminal = dict(knowledge=kb.state(), conclusions=conclusions, comparison_reviews=reviews)
        extended = copy.deepcopy(self._capsule)
        index = len(extended['journals'])
        extended['journals'].append(dict(manifest_sha256=manifest_sha256, journal=journal))
        extended['operations'].append(dict(kind='reflection', journal=index, event_seq=rows[-1]['event_seq'],
                                           interval=expected['interval']))
        extended['terminal'] = {key + '_sha256': canonical_sha(value) for key, value in terminal.items()}
        # Independent reconstruction is mandatory; this API cannot weaken the
        # capsule's raw-response, contextual or authoring admission checks.
        replay = validate(extended, config=self._config, metric=extended['metric'])
        _require(replay['knowledge'].state() == terminal['knowledge']
                 and replay['conclusions'] == conclusions and replay['comparison_reviews'] == reviews,
                 'extended history did not reconstruct exactly')
        return dict(schema=SCHEMA, status='accepted', journal=journal, journal_sha256=_digest(journal),
            extended_capsule=extended, extended_capsule_sha256=canonical_sha(extended), terminal=terminal,
            accepted_review_ids=list(additions), new_lesson_ids=applied['lesson_ids'],
            retired_lesson_ids=applied['retired_ids'], known_cost=float(cost),
            recorded_provider_responses=1, provider_dispatch_verified=False,
            new_ordinary_outcomes=0, new_search_attempts=0)
