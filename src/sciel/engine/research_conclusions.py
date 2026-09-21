"""Conclusion validation, bounded repair and lesson completion."""
from __future__ import annotations

import copy
from ..hypothesis import conclusion_prompt, parse_reflection
from ..hypothesis.rendering import serialize_record
from .reflection_context import prepare_reflection_context
from ..hypothesis.observed_facts import observed_comparison
from ..hypothesis.conclusion_feedback import LESSON_ADMISSION_SCHEMA
from ..hypothesis.parsing import decode_conclusion
from ..hypothesis.mechanism_review import validate_binding as validate_mechanism_binding
from ..hypothesis.prompts import initializer_block
from ..knowledge_base.catalogs import canonical_json
from .research_protocol import (_FrozenPayload, digest)


class ResearchConclusions:

    def _retire_conclusions(self):
        keep = {r['id'] for r in self.knowledge.experiments}
        self.conclusions = {k: v for k, v in self.conclusions.items() if k in keep}

    def decode_conclusion(self, job, text, attempt=0):
        """Return (parsed conclusion, correction prompt), without transport.

        attempt=0 is the original complete response; at most attempt=1 may be
        requested by the configured correction policy. None means skipped, and
        an invalid complete original remains the terminal text if repair skips.
        """
        if type(attempt) is not int or not 0 <= attempt <= self.config.get('conclusion_repair_attempts', 0):
            raise ValueError('invalid conclusion correction attempt')
        from ..hypothesis.conclusion_source import validate_job
        validate_job(job)
        if text is None:
            return None, None
        from ..hypothesis import conclusion_refs
        evidence, prompt = job['evidence'], job['prompt']
        references = self.config.get('conclusion_source_refs', False)
        typed = self.config.get('structured_auxiliary', False)
        decoded = (conclusion_refs.decode(text, evidence) if references else
            decode_conclusion(text, require_mechanism=True, require_typed=typed))
        result = decoded['conclusion']
        diagnostic = decoded['error']
        reason, detail = 'invalid_conclusion_structure', None
        if result is not None and not references:
            try:
                validate_mechanism_binding(result.mechanism_review, evidence,
                                           require_typed=typed)
            except ValueError as exc:
                reason, detail = 'mechanism_binding_invalid', str(exc)[:512]
                diagnostic = getattr(exc, 'diagnostic',
                    {'code': reason, 'path': 'mechanism_review'})
                result = None
        if result is not None:
            if references:
                self.emit('conclusion_reference_resolution', experiment_id=evidence['id'],
                    **decoded['resolution'])
            return result, None
        self.emit('conclusion_rejected', experiment_id=evidence['id'], reason=reason,
            diagnostic=diagnostic, **({'detail': detail} if detail is not None else {}))
        if attempt >= self.config.get('conclusion_repair_attempts', 0):
            return None, None
        self.emit('conclusion_repair', experiment_id=evidence['id'], attempt=attempt + 1,
            diagnostic=diagnostic, previous_response_sha256=digest(text))
        return None, conclusion_refs.repair_prompt(prompt, text, diagnostic)

    def _request_conclusion(self, job):
        """One measured experiment, at most one known-complete format correction."""
        text = self._aux('conclusion', job['prompt'], job['evidence']['id'],
            conclusion_evidence=job['evidence'] if self.config.get('conclusion_source_refs', False) else None)
        for attempt in range(1 + self.config.get('conclusion_repair_attempts', 0)):
            result, repair_prompt = self.decode_conclusion(job, text, attempt)
            if repair_prompt is None:
                return result, text
            repaired = self._aux('conclusion', repair_prompt, job['evidence']['id'],
                repair_evidence=job['evidence'] if self.config.get('conclusion_source_refs', False) else None)
            if repaired is None:
                return None, text
            text = repaired
        return None, text

    def prepare_conclusions(self, *, limit=None):
        """Detach source-bound jobs in queue order without any provider calls.

        limit=1 keeps the serial path's event/correction/cap ordering. Wave
        coordinators may prepare all jobs, but must keep their evidence retained
        until ordered admission and durably own every detached job/receipt.
        """
        from . import research as _public
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError('conclusion preparation limit must be positive')
        self._feedback_delivery = None
        self._review_text_options()
        from .generated_paired_import import check as check_component
        check_component(self, require_present=True)
        from .evaluated_study_import import check as check_study
        check_study(self, require_present=True)
        from .evaluated_review_import import check as check_study_review
        check_study_review(self, require_present=True)
        jobs = []
        # Pop before dispatch: a stopped request is never retried on resume.
        while self.queue and (limit is None or len(jobs) < limit):
            record = copy.deepcopy(self.queue.pop(0))
            if self.config.get('initializer_witness', False):
                from .initializer_research import collect
                collect(self, record)
            evidence = {k: v for k, v in record.items() if k != 'code_fragment'}
            from ..hypothesis import conclusion_source
            from ..hypothesis.prompts import initializer_block
            from ..hypothesis.conclusion_refs import instruction
            references = self.config.get('conclusion_source_refs', False)
            structured = self.config.get('structured_auxiliary', False)
            pair = self._conclusion_sources.pop(record['id'], None)
            source_binding = None

            def prepare_view(projection=None):
                from . import research as _public
                view = dict(evidence)
                if projection is not None:
                    view['conclusion_source'] = projection
                source_rules = conclusion_source.RULES if projection is not None else ''
                if projection is not None and not references:
                    source_rules += 'Additional allowed quote fields: ' + ', '.join(conclusion_source.fields(projection)) + '.\n'
                used = (len(serialize_record(view)) + len(canonical_json(observed_comparison(view)))
                        + (len(instruction(view)) if references else 0) + len(source_rules))
                if used > _public.MAX_CONTEXT_CHARS:
                    return None
                initializer = self._initializer_context([record['id']], _public.MAX_CONTEXT_CHARS-used)
                if used + len(initializer_block(initializer)) > _public.MAX_CONTEXT_CHARS:
                    return None
                prompt = conclusion_prompt(view, initializer_context=initializer,
                    structured=structured, source_refs=references)
                return view, initializer, prompt

            if pair is None:
                prepared = prepare_view()
            else:
                projection = conclusion_source.project(*pair)
                prepared = prepare_view(projection)
                # Full evidence first. When genuinely oversized, stratify over
                # the entire diff, never a prefix or only the first few hunks.
                count, width = conclusion_source.MAX_WINDOWS, max(1, projection['full_diff_chars']//conclusion_source.MAX_WINDOWS)
                while prepared is None:
                    projection = conclusion_source.project(*pair, window_chars=width, window_count=count)
                    prepared = prepare_view(projection)
                    if prepared is not None:
                        break
                    if width > 1:
                        width = max(1, width//2)
                    elif count > 1:
                        count = max(1, count//2)
                    else:
                        raise ValueError('research evidence cannot fit source projection metadata')
                source_binding = conclusion_source.bind(record, *pair, projection)
            if prepared is None:
                raise ValueError('research evidence exceeds context character limit')
            evidence, initializer, prompt = prepared
            self._initializer_presentation('conclusion', [record['id']], initializer, prompt)
            if self.config.get('conclusion_source_refs', False):
                from ..hypothesis import conclusion_refs
                catalog = conclusion_refs.catalog(evidence)
                self.emit('conclusion_reference_catalog', experiment_id=evidence['id'],
                    catalog=catalog, catalog_sha256=digest(canonical_json(catalog)))
            jobs.append(_FrozenPayload.from_dict(dict(record=record, evidence=evidence, prompt=prompt,
                **({'source_evidence': source_binding} if source_binding is not None else {}))))
        return jobs

    def apply_conclusion(self, job, result, text):
        """Admit one terminal conclusion on the coordinator, in dispatch order."""
        from ..hypothesis.conclusion_source import validate_job
        validate_job(job)
        record = job['record']
        accepted, error, error_detail, admission = [], None, None, None
        if result is not None:
            if self.config.get('batch_only_lessons', False):
                # Keep the complete parsed review and its proposals. This
                # policy does not attempt admission, create writer/support
                # credit, or classify suppression as an admission failure.
                admission = {'schema': LESSON_ADMISSION_SCHEMA,
                    'policy': 'batch_only_lessons', 'status': 'suppressed',
                    'proposed_count': len(result.lessons)}
            else:
                try:
                    accepted = self.knowledge.add_lessons(result.lessons,
                        allowed_evidence_ids=[record['id']], writer_context={
                            'phase': 'conclusion', 'reference': record['id'],
                            'raw_response_sha256': digest(text)})
                except (ValueError, TypeError, KeyError) as exc:
                    error = type(exc).__name__
                    error_detail = str(exc)[:1024]
        self.conclusions[record['id']] = {
            'status': 'parsed' if result is not None else ('skipped' if text is None else 'invalid'),
            'llm': result.to_dict() if result is not None else None,
            'observed_forecast_verdict': record['forecast_verdict'],
            # Persist the same normalized KB view that resume checks. A
            # supported integer evaluator score remains an int in the raw
            # queue/journal, but is stored as a float in KnowledgeBase.
            # A queued observation already evicted from KB remains in the
            # journal only; its conclusion is pruned at the end of this step.
            'observed_comparison': observed_comparison(self.knowledge._evidence().get(record['id'], record)),
            'verdict_matches_observation': (result.verdict == record['forecast_verdict']) if result else None,
            'accepted_lesson_ids': accepted, 'lesson_error': error,
            'lesson_error_detail': error_detail}
        if 'source_evidence' in job:
            self.conclusions[record['id']]['source_evidence'] = copy.deepcopy(job['source_evidence'])
        if admission is not None:
            self.conclusions[record['id']]['lesson_admission'] = admission
        self.emit('conclusion', experiment_id=record['id'], **self.conclusions[record['id']])

    def finish_conclusions(self, *, reflect=True):
        """Run one due R only after all wave conclusions have been committed."""
        if type(reflect) is not bool:
            raise ValueError('reflect must be a boolean')
        if self._aux_requests or self.queue:
            raise ValueError('cannot reflect before queued research requests finish')
        self._feedback_delivery = None
        interval = self.config['reflection_interval']
        if reflect and interval and self.outcomes - self.reflected_through >= interval:
            preparation = prepare_reflection_context(self.knowledge, self.conclusions,
                config=self.config, interval=interval, through=self.outcomes,
                selection_epoch=self.reflection_attempts)
            records, lessons = preparation['records'], preparation['lessons']
            comparison_review_groups = preparation['comparison_review_groups']
            prompt = preparation['prompt']
            for event in preparation['presentation_events']:
                self.emit(**event)
            self.reflected_through = self.outcomes
            # Advance for every scheduled R, including invalid or budget-skipped
            # responses. Outcome cadence and accepted proposals cannot skip turns.
            self.reflection_attempts += 1
            self.emit('reflection_context', **preparation['context_event'])
            text = self._aux('reflection', prompt, self.outcomes)
            result = parse_reflection(text, comparison_reviews=self.config.get('comparison_reviews', False),
                comparison_review_version=self.config.get('comparison_review_version', 1)) if text is not None else None
            applied, error, error_detail, changed = None, None, None, False
            review_ids, review_changed = [], False
            if result is not None:
                try:
                    if self._review_text_options():
                        from ..hypothesis.review_eligibility import require_review_only
                        require_review_only(result)
                    if not set(result.retire_ids) <= {lesson['id'] for lesson in lessons}:
                        raise ValueError('retirement requires a lesson supplied in the reflection context')
                    if not set(result.retire_ids) <= {lesson['id'] for lesson in lessons
                                                     if lesson.get('retirement_eligible') is True}:
                        raise ValueError('retirement exceeds the replacement evidence limit')
                    before_lessons = self.knowledge.lessons
                    updated_reviews = None
                    if self.config.get('comparison_reviews', False):
                        from ..hypothesis.comparison_review import build_cards, MAX_STORED
                        additions = build_cards(result.comparison_reviews, comparison_review_groups,
                            {'phase': 'reflection', 'reference': self.outcomes, 'raw_response_sha256': digest(text)},
                            version=self.config.get('comparison_review_version', 1),
                            records=self.knowledge._evidence(), seen=self.knowledge._seen)
                        updated_reviews = self._retained_comparison_reviews()
                        for key, card in additions.items():
                            updated_reviews.pop(key, None)
                            updated_reviews[key] = card
                        updated_reviews = dict(list(updated_reviews.items())[-MAX_STORED:])
                        review_ids = list(additions)
                    applied = self.knowledge.reflect(result.lessons, result.retire_ids,
                        allowed_evidence_ids=[r['id'] for r in records], writer_context={
                            'phase': 'reflection', 'reference': self.outcomes,
                            'raw_response_sha256': digest(text)})
                    if updated_reviews is not None:
                        review_changed = updated_reviews != self._comparison_reviews
                        self._comparison_reviews = updated_reviews
                    self.reflections += 1
                    changed = self.knowledge.lessons != before_lessons
                    self.reflection_changes += int(changed)
                except (ValueError, TypeError, KeyError) as exc:
                    error = type(exc).__name__
                    error_detail = str(exc)[:1024]
            self.emit('reflection', through=self.outcomes, applied=applied, changed=changed, error=error,
                      error_detail=error_detail,
                      **({'comparison_review_ids': review_ids if error is None else [],
                          'comparison_reviews_changed': review_changed}
                         if self.config.get('comparison_reviews', False) else {}),
                      status='parsed' if result else ('skipped' if text is None else 'invalid'))
        # Full immutable experiment evidence lives in the journal; checkpoint
        # keeps only the bounded recent conclusion window.
        self._retire_conclusions()
        self._comparison_reviews = self._retained_comparison_reviews()

    def complete_step(self, *, reflect=True):
        """Serial G/C/R behavior using the same separated coordinator stages.

        Repeat orchestration may defer reflection until its terminal comparison
        is imported, without consuming the ordinary-outcome watermark.
        """
        if type(reflect) is not bool:
            raise ValueError('reflect must be a boolean')
        # Also perform component/presentation checks when there are no C jobs.
        jobs = self.prepare_conclusions(limit=1)
        while jobs:
            job = jobs[0]
            result, text = self._request_conclusion(job)
            self.apply_conclusion(job, result, text)
            jobs = self.prepare_conclusions(limit=1)
        self.finish_conclusions(reflect=reflect)
