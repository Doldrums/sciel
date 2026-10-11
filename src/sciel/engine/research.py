"""The hypothesis/evidence/knowledge loop, independent of population retention.

Generation hypotheses share the mutation request. Conclusions and reflection use
the same LLM object (and therefore the same campaign ledger), with a separate cap
on auxiliary calls. An LLM verdict never overwrites observed fitness or the
deterministic forecast check.
"""

from __future__ import annotations
import copy
import difflib
import hashlib
import json
import math
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from ..campaign.budget import CampaignStopped, RequestNotDispatched
from .llm.client import (llm_error_metadata, accounted_cost, response_accounting,
                         response_cost_provenance, response_model_identity, response_resolved_model)
from ..hypothesis import (CONTRACT, hypothesis_instruction, parse_hypothesis,
                          conclusion_prompt,
                          parse_reflection)
from ..knowledge_base import KnowledgeBase
from ..knowledge_base.retrieval import SourceDiverseSelection, selection_identity
from ..knowledge_base.presentation import (observation_card,
                                            MAX_CARD_CHARS, MAX_MEMORY_CHARS,
                                            COMPARISON_CARD_SCHEMA, LEGACY_COMPARISON_CARD_SCHEMA)
from ..hypothesis.rendering import (MAX_CONTEXT_CHARS, presentation_identity, reference_rules,
    serialize_card, serialize_context, serialize_record)
from .reflection_context import ALLOCATION_POLICY, prepare_reflection_context
from .prompt.sources import (SCHEMA as SOURCE_SCHEMA, fragment_reference_views,
                             source_digest)
from ..evidence.measurement import (metric_context, validate_metric_context,
                                      validate_cases, validate_evaluation)
from ..hypothesis.observed_facts import observed_comparison
from ..hypothesis.controlled_changes import validate_controlled_change
from ..hypothesis.conclusion_feedback import (conclusion_feedback, validate_conclusions,
    LESSON_ADMISSION_SCHEMA)
from ..hypothesis.parsing import decode_conclusion
from ..hypothesis.mechanism_review import validate_binding as validate_mechanism_binding
from ..hypothesis.prompts import (COMPARISON_RULES, PAIRED_COMPARISON_RULES,
                                 PRIOR_EXPERIMENT_RULES, initializer_block)
from ..knowledge_base.paired_comparisons import (SCHEMA as PAIRED_SCHEMA,
                                                validate_comparison)
from ..knowledge_base.catalogs import canonical_json, canonical_sha
from .research_protocol import (
    _FrozenPayload,
    _OUTCOME_CAPTURES,
    _fragment_views,
    assert_no_uncheckpointed_research,
    digest,
)

from .research_views import ResearchViews
from .research_context import ResearchContext
from .research_episodes import ResearchEpisodes
from .research_evidence import ResearchEvidence
from .research_auxiliary import ResearchAuxiliary
from .research_conclusions import ResearchConclusions
from .research_state import ResearchState


class ResearchController(
    ResearchViews,
    ResearchContext,
    ResearchEpisodes,
    ResearchEvidence,
    ResearchAuxiliary,
    ResearchConclusions,
    ResearchState,
):

    def __init__(self, config, llm, log, charge, *, metric=None):
        self.config = copy.deepcopy(config)
        if type(self.config.get('batch_only_lessons', False)) is not bool:
            raise ValueError('batch_only_lessons must be boolean')
        if type(self.config.get('restart_predictions', False)) is not bool:
            raise ValueError('restart_predictions must be boolean')
        if type(self.config.get('structured_auxiliary', False)) is not bool:
            raise ValueError('structured_auxiliary must be boolean')
        for option in ('compact_lesson_memory', 'conclusion_source_refs'):
            if type(self.config.get(option, False)) is not bool:
                raise ValueError(f'{option} must be boolean')
        repairs = self.config.get('conclusion_repair_attempts', 0)
        if type(repairs) is not int or not 0 <= repairs <= 1:
            raise ValueError('conclusion_repair_attempts must be an integer from 0 to 1')
        conclusion_tokens = self.config.get('conclusion_max_tokens')
        if conclusion_tokens is not None and (type(conclusion_tokens) is not int or conclusion_tokens <= 0):
            raise ValueError('conclusion_max_tokens must be a positive integer or null')
        if self.config.get('conclusion_source_refs', False) and not self.config.get('structured_auxiliary', False):
            raise ValueError('conclusion_source_refs requires structured_auxiliary')
        if repairs and not self.config.get('structured_auxiliary', False):
            raise ValueError('conclusion repairs require structured_auxiliary')
        if type(self.config.get('comparison_reviews', False)) is not bool:
            raise ValueError('comparison_reviews must be boolean')
        if self.config.get('comparison_reviews', False) and not self.config.get('structured_auxiliary', False):
            raise ValueError('comparison_reviews requires structured_auxiliary')
        if (type(self.config.get('comparison_review_version', 1)) is not int
                or self.config.get('comparison_review_version', 1) not in (1, 2, 3)
                or ('comparison_review_version' in self.config and not self.config.get('comparison_reviews', False))):
            raise ValueError('comparison_review_version requires enabled reviews and version 1, 2 or 3')
        if 'comparison_review_text' in self.config:
            if type(self.config['comparison_review_text']) is not bool:
                raise ValueError('comparison_review_text must be boolean')
            if not self.config.get('comparison_reviews', False):
                raise ValueError('comparison_review_text requires comparison_reviews')
        from .external_research_v2 import _review_text_policy, policy_identity
        self._configured_review_text_policy = _review_text_policy(self.config)
        self._review_text_policy_identity = None
        if self._configured_review_text_policy is not None:
            from ..hypothesis.review_eligibility import identity
            self._review_text_policy_identity = dict(schema='sciel-review-text-delivery/1',
                contract=identity(), policy_sha256=canonical_sha(policy_identity()),
                decisions_sha256=canonical_sha(self._configured_review_text_policy),
                text_enabled=self.config['comparison_review_text'])
        self._external_history = None
        self._external_history_identity = None
        from .evaluated_study_import import configured_digests, identity as study_identity
        self._external_evaluated_study_digests = configured_digests(self.config)
        self._external_evaluated_study_identity = (study_identity()
            if self._external_evaluated_study_digests[0] is not None else None)
        from .evaluated_review_import import configured_digests as review_digests, identity as evaluated_review_identity
        self._external_evaluated_review_digests = review_digests(self.config)
        self._external_evaluated_review_identity = (evaluated_review_identity()
            if self._external_evaluated_review_digests[0] is not None else None)
        from .generated_paired_import import configured_digest, identity as component_identity
        self._external_component_digest = configured_digest(self.config)
        self._external_component_identity = (component_identity()
            if self._external_component_digest is not None else None)
        from .generated_paired_review_import import configured_digest as review_digest, identity as review_identity
        self._external_component_review_digest = review_digest(self.config)
        self._external_component_review_identity = (review_identity(self.config)
            if self._external_component_review_digest is not None else None)
        if 'external_history_sha256' in self.config:
            value = self.config['external_history_sha256']
            if (type(value) is not str or len(value) != 64
                    or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('external_history_sha256 must be a lowercase SHA256')
            from .external_research import identity
            self._external_history_identity = identity()
        self._auxiliary_schema_identity = None
        if self.config.get('structured_auxiliary', False):
            self._auxiliary_schema_identity = self._aux_schema_identity()
        selection_identity(self.config)  # Also validate direct, non-factory construction.
        self.metric = metric_context() if metric is None else validate_metric_context(metric)
        if config['model'] is not None and config['model'] not in llm.models:
            raise ValueError('research model must be in the configured ensemble')
        if getattr(llm, 'max_retries', 1) != 1:
            raise ValueError('research requires an LLM without hidden retries')
        self.llm, self.log, self.charge = llm, log, charge
        self.knowledge = KnowledgeBase(max_lessons=config['max_lessons'],
            max_experiments=config['max_experiments'], max_fragments=config['max_fragments'])
        self.pending = {}
        self._parents = {}
        # Pending pairs are owned by queued jobs, then detached jobs/conclusions.
        # They never enter ordinary KB records or copied lesson evidence.
        self._conclusion_sources = {}
        # Only the opt-in body operator uses these transient raw-response
        # bindings. Pending work is never replayable from a checkpoint.
        self._structured_responses = {}
        self.queue = []
        self.conclusions = {}
        # Each cooperative job's retries reuse one immutable Snapshot. Keep its
        # already validated feedback until the G wave drains: a sibling's begin
        # must not replace this binding when old source evidence is evicted.
        self._feedback_delivery = None
        self._comparison_reviews = {}
        # Coordinator-owned issued requests. Workers receive immutable copies
        # and never book cost, mutate knowledge, or assign journal sequence IDs.
        self._aux_requests = {}
        self.aux_calls = 0
        self.aux_cost = 0.0
        self.attempts = 0
        self.outcomes = 0
        self.valid_hypotheses = 0
        self.reflected_through = 0
        self.reflection_attempts = 0
        self.reflections = 0
        self.reflection_changes = 0
        self.stopped_reason = None
        # Preserve cancellation ancestry between cooperative auxiliary siblings.
        # Exception objects are transient; checkpoints retain only the reason.
        self._stopped_cause = None
        if 'initializer_witness' in self.config and type(self.config['initializer_witness']) is not bool:
            raise ValueError('initializer_witness must be boolean')
        from .initializer_research import fixture_profile
        initializer_profile = fixture_profile(self.config)
        self._initializer_attempts = {}
        self._initializer_identity = None
        if self.config.get('initializer_witness', False):
            from .initializer_research import identity
            self._initializer_identity = identity(initializer_profile)

    def emit(self, phase, **fields):
        if (phase in ('outcome', 'conclusion', 'reflection')
                and self._external_history is not None
                and self._external_history['schema'] == 'sciel-external-research-registry/2'):
            from .external_research_resume import transition_binding
            fields['external_history_transition'] = transition_binding(self)
        self.log.research(phase=phase, contract=CONTRACT, **fields)
