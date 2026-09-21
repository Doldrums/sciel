"""Research observation, comparison and presentation views."""
from __future__ import annotations

import copy
from ..knowledge_base.paired_comparisons import (SCHEMA as PAIRED_SCHEMA,
                                                validate_comparison)
from ..knowledge_base.catalogs import canonical_json, canonical_sha
from .research_protocol import (digest)


class ResearchViews:

    def _initializer_context(self, ids, available):
        from .reflection_context import initializer_context
        return initializer_context(self.knowledge, self.config, ids, available)

    def _evaluated_study_context(self, branch, *, prior_chars=0, root_source_sha256=None):
        from . import research as _public
        from .evaluated_study_import import check
        bundle = check(self, require_present=True)
        if bundle is None:
            return None
        if branch != bundle['capsule']['receiving_root']['id']:
            raise ValueError('evaluated study requires its imported root branch')
        if (root_source_sha256 is not None
                and root_source_sha256 != bundle['capsule']['receiving_root']['source_sha256']):
            raise ValueError('evaluated study receiving root source differs')
        from ..knowledge_base.evaluated_study_context import build_context
        return build_context(bundle['capsule'], author_dependencies=bundle['author_dependencies'],
            author_authority=bundle['author_authority'], branch=branch,
            max_chars=max(0, min(16000, _public.MAX_MEMORY_CHARS - prior_chars - 2048)),
            max_items=1, enabled=self.config.get('inject_lessons', True))

    def _evaluated_review_context(self, branch, study, *, prior_chars=0):
        from . import research as _public
        from .evaluated_review_import import check
        retained = check(self, require_present=True)
        if retained is None:
            return None
        from ..knowledge_base.evaluated_review_context import build_context
        return build_context(self.knowledge.evaluated_detached_study, retained['sidecar']['review_card'],
            branch=branch, max_chars=max(0, min(8000, _public.MAX_MEMORY_CHARS - prior_chars - 2048)),
            text_enabled=self.config['comparison_review_text'], enabled=self.config.get('inject_lessons', True),
            source_facts_presented=study is not None and study['presented_count'] == 1,
            **self._review_text_options())

    def _component_context(self, branch):
        from .generated_paired_import import check
        capsule = check(self, require_present=True)
        if capsule is None:
            return None
        if branch != capsule['root_parent']['id']:
            raise ValueError('generated paired evidence requires its imported root branch')
        from ..knowledge_base.generated_paired_context import build_context
        return build_context([capsule], branch=branch, max_chars=16000, max_items=1,
                             enabled=self.config.get('inject_lessons', True))

    def _component_review_context(self, branch):
        from .generated_paired_import import check
        component = check(self, require_present=True)
        sidecar = self.knowledge.generated_paired_review
        if sidecar is None:
            return None
        from ..knowledge_base.generated_paired_review_context import build_context
        return build_context(component, sidecar['review_cards'], branch=branch,
            text_enabled=self.config['comparison_review_text'],
            enabled=self.config.get('inject_lessons', True), **self._review_text_options())

    def _initializer_presentation(self, role, ids, context, prompt, *, experiment_id=None):
        if self.config.get('initializer_witness', False):
            self.emit('initializer_witness_presentation', role=role,
                      selected_experiment_ids=ids, prompt_sha256=digest(prompt),
                      experiment_id=experiment_id,
                      context=context, context_sha256=digest(canonical_json(context)) if context else None,
                      quality_changed=False, delivery='prompt_prepared_not_model_use')

    @staticmethod
    def _restart_context(groups, available):
        from .reflection_context import restart_context
        return restart_context(groups, available)

    def _restart_presentation(self, role, ids, context, prompt, *, experiment_id=None):
        if ids:
            self.emit('restart_observation_presentation', role=role,
                selected_comparison_ids=ids, experiment_id=experiment_id,
                context=context, context_sha256=digest(canonical_json(context)) if context else None,
                omission='presentation_budget' if context is None else None,
                prompt_sha256=digest(prompt), quality_changed=False,
                delivery='prompt_prepared_not_model_use')

    def _prediction_records(self):
        records = self.knowledge._evidence()
        records.update((r['id'], r) for r in self.knowledge.experiments)
        return records

    def _comparison_fact_records(self, branch, comparisons):
        """Use only the actual branch-selected groups and their measured origins."""
        from ..knowledge_base.generated_paired_context import block as component_block
        component_chars = len(component_block(self._component_context(branch)))
        review = self._component_review_context(branch)
        component_chars += review['reserved_chars'] if review is not None else 0
        from ..knowledge_base.evaluated_study_context import block as study_block
        study = self._evaluated_study_context(branch, prior_chars=component_chars)
        component_chars += len(study_block(study))
        study_review = self._evaluated_review_context(branch, study, prior_chars=component_chars)
        component_chars += study_review['reserved_chars'] if study_review is not None else 0
        expected = self.knowledge.comparison_context(branch, max_items=3,
            max_chars=self._comparison_char_budget(component_chars))
        ids = [card['id'] for card in comparisons['cards']]
        if canonical_json(comparisons) != canonical_json(expected):
            raise ValueError('comparison facts require the selected branch comparisons')
        catalog = self._prediction_records()
        records, seen = [], set()
        for ident in ids:
            group = catalog[ident]
            for key in (ident, group.get('origin_experiment_id')):
                if key in catalog and key not in seen:
                    records.append(catalog[key])
                    seen.add(key)
        return records

    @staticmethod
    def _comparison_char_budget(component_chars):
        from . import research as _public
        if component_chars == 0:
            return 16000
        return min(16000, _public.MAX_MEMORY_CHARS - component_chars - 2048)

    def _prediction_context(self, groups, available):
        from .reflection_context import prediction_context
        return prediction_context(self.knowledge, self.config, groups, available)

    def _check_comparison_binding(self, group, knowledge):
        # A repeat comparison has its own physical units. task_metric identifies
        # the original task only; it must never replace the forecast score metric.
        from .repeat_support import KNOWN_CIRCLE_TASK_HASHES
        if (self.metric['task_name'] != 'oe/circle_packing'
                or self.metric['task_hash'] not in KNOWN_CIRCLE_TASK_HASHES
                or self.metric['primary_metric'] != 'combined_score'
                or group['task_metric'] != self.metric):
            raise ValueError('paired comparison requires the same canonical Circle task metric')
        origin_id = group['origin_experiment_id']
        if origin_id is not None and origin_id not in knowledge._seen:
            raise ValueError('paired comparison origin was never recorded')
        origin = knowledge._evidence().get(origin_id)
        if origin is not None and (origin.get('status') != 'evaluated'
                                   or origin.get('source_sha256') != group['challenger_source_sha256']
                                   or origin.get('branch') != group['branch']):
            raise ValueError('paired comparison origin source/branch differs')

    def import_comparison(self, group):
        """Import one immutable external group, without inventing a search step.

        The H15 repeat journal durably owns the receipts; the next safe checkpoint
        owns this KB projection. No research event is appended between those
        checkpoints: that would become an uncheckpointed research tail and block
        legitimate repeat recovery. Identical re-imports are idempotent, and this
        path never increments attempts, outcomes, model calls or advice exposures.
        """
        group = validate_comparison(group)
        self._check_comparison_binding(group, self.knowledge)
        result = self.knowledge.record_comparison(group)
        self._comparison_reviews = self._retained_comparison_reviews()
        return result

    def _review_groups(self, knowledge=None):
        knowledge = self.knowledge if knowledge is None else knowledge
        return [g for g in knowledge._evidence().values() if g.get('schema') == PAIRED_SCHEMA]

    def _retained_comparison_reviews(self):
        # Evidence windows own lifetime. A review cannot keep an evicted group
        # alive or manufacture a new observation to support its interpretation.
        if not self._comparison_reviews:
            return {}
        ids = {g['id'] for g in self._review_groups()}
        return {key: copy.deepcopy(card) for key, card in self._comparison_reviews.items() if key in ids}

    def _review_text_options(self):
        """The host-configured registry, never a grant taken from model/checkpoint.

        Changing a decision requires a fresh explicitly configured stage. A
        running controller cannot silently acquire or drop this boundary after
        preparing a prompt; checkpoint loading retains the exact configuration.
        """
        from .external_research_v2 import _review_text_policy, policy_identity
        current = _review_text_policy(self.config)
        if canonical_json(current) != canonical_json(self._configured_review_text_policy):
            raise ValueError('review text policy changed during this run')
        if current is None:
            return {}
        if self.config['comparison_review_text'] is not self._review_text_policy_identity['text_enabled']:
            raise ValueError('review text policy ablation mode changed during this run')
        policy_sha = canonical_sha(policy_identity())
        if policy_sha != self._review_text_policy_identity['policy_sha256']:
            raise ValueError('review text policy source changed during this run')
        return dict(text_policy=copy.deepcopy(self._configured_review_text_policy),
                    policy_sha256=policy_sha)
