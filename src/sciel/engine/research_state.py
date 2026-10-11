"""Research state serialization and exact restored-evidence checks."""
from __future__ import annotations

import copy
import math
from ..hypothesis import CONTRACT
from ..knowledge_base import KnowledgeBase
from ..knowledge_base.retrieval import selection_identity
from ..hypothesis.rendering import presentation_identity
from .reflection_context import ALLOCATION_POLICY
from ..evidence.measurement import validate_metric_context
from ..hypothesis.conclusion_feedback import validate_conclusions
from ..knowledge_base.paired_comparisons import SCHEMA as PAIRED_SCHEMA
from ..knowledge_base.catalogs import canonical_json


class ResearchState:

    def summary(self):
        self._review_text_options()
        from .generated_paired_import import check as check_component
        check_component(self)
        from .evaluated_study_import import check as check_study
        check_study(self)
        from .evaluated_review_import import check as check_study_review
        check_study_review(self)
        summary = {'contract': CONTRACT, 'presentation': presentation_identity(
                       batch_only_lessons=self.config.get('batch_only_lessons', False),
                       restart_predictions=self.config.get('restart_predictions', False),
                       comparison_reviews=self.config.get('comparison_reviews', False),
                       compact_lesson_memory=self.config.get('compact_lesson_memory', False)),
                'attempts': self.attempts,
                'outcomes': self.outcomes, 'valid_hypotheses': self.valid_hypotheses,
                'aux_calls': self.aux_calls, 'aux_cost': self.aux_cost,
                'reflection_allocation': ALLOCATION_POLICY,
                'reflection_attempts': self.reflection_attempts,
                'reflections': self.reflections, 'reflection_changes': self.reflection_changes,
                'lessons': len(self.knowledge.lessons),
                'stopped_reason': self.stopped_reason}
        if self._auxiliary_schema_identity is not None:
            summary['auxiliary_schema'] = copy.deepcopy(self._auxiliary_schema_identity)
        if self._external_history_identity is not None:
            summary['external_history_identity'] = copy.deepcopy(self._external_history_identity)
        if self._external_evaluated_study_identity is not None:
            summary['external_evaluated_study_identity'] = copy.deepcopy(self._external_evaluated_study_identity)
        if self._external_evaluated_review_identity is not None:
            summary['external_evaluated_review_identity'] = copy.deepcopy(self._external_evaluated_review_identity)
        if self._external_component_identity is not None:
            summary['external_component_identity'] = copy.deepcopy(self._external_component_identity)
        if self._external_component_review_identity is not None:
            summary['external_component_review_identity'] = copy.deepcopy(self._external_component_review_identity)
        if self._review_text_policy_identity is not None:
            summary['review_text_policy'] = copy.deepcopy(self._review_text_policy_identity)
        selection = selection_identity(self.config)
        if selection is not None:
            summary['lesson_selection'] = selection
        if self.knowledge.comparisons:
            summary['comparison_count'] = len(self.knowledge.comparisons)
        if self.config.get('comparison_reviews', False):
            summary['comparison_review_count'] = len(self._retained_comparison_reviews())
        return summary

    def state(self):
        initializer = {}
        if self._external_history is not None:
            from .external_research import state
            initializer['external_history'] = state(self)
        if self.config.get('initializer_witness', False):
            from .initializer_research import state
            initializer['initializer_witness'] = state(self)
        if self.config.get('comparison_reviews', False):
            initializer['comparison_reviews'] = self._retained_comparison_reviews()
        return {'schema': CONTRACT, 'config': copy.deepcopy(self.config),
                'metric': copy.deepcopy(self.metric),
                'knowledge': self.knowledge.state(), 'pending': copy.deepcopy(self.pending),
                'queue': copy.deepcopy(self.queue), 'conclusions': copy.deepcopy(self.conclusions),
                **self.summary(), 'reflected_through': self.reflected_through, **initializer}

    def load_state(self, state):
        self._review_text_options()
        if state.get('schema') != CONTRACT or state.get('config') != self.config:
            raise ValueError('research checkpoint contract/configuration mismatch')
        # Pre-policy research snapshots remain inspectable only before any R.
        # Historical nonzero watermarks cannot reveal failed/skipped attempts.
        if 'reflection_allocation' not in state and 'reflection_attempts' not in state:
            if state.get('reflected_through') != 0 or state.get('reflections') != 0:
                raise ValueError('research checkpoint is missing reflection allocation history')
            state = dict(state, reflection_allocation=ALLOCATION_POLICY, reflection_attempts=0)
        if state.get('reflection_allocation') != ALLOCATION_POLICY:
            raise ValueError('research checkpoint reflection allocation mismatch')
        if (canonical_json(state.get('review_text_policy')) != canonical_json(self._review_text_policy_identity)
                or (self._review_text_policy_identity is None and 'review_text_policy' in state)):
            raise ValueError('research checkpoint review text policy mismatch')
        # Dictionary equality alone treats a tampered bool weight as numeric 0/1.
        selection_identity(state['config'])
        if type(state['config'].get('batch_only_lessons', False)) is not bool:
            raise ValueError('batch_only_lessons must be boolean')
        if type(state['config'].get('restart_predictions', False)) is not bool:
            raise ValueError('restart_predictions must be boolean')
        if type(state['config'].get('structured_auxiliary', False)) is not bool:
            raise ValueError('structured_auxiliary must be boolean')
        if type(state['config'].get('comparison_reviews', False)) is not bool:
            raise ValueError('comparison_reviews must be boolean')
        if (type(state['config'].get('comparison_review_version', 1)) is not int
                or state['config'].get('comparison_review_version', 1) not in (1, 2, 3)):
            raise ValueError('unsupported comparison_review_version')
        if ('comparison_review_text' in state['config']
                and type(state['config']['comparison_review_text']) is not bool):
            raise ValueError('comparison_review_text must be boolean')
        if (state.get('auxiliary_schema') != self._auxiliary_schema_identity or
                (self._auxiliary_schema_identity is None and 'auxiliary_schema' in state)):
            raise ValueError('research checkpoint auxiliary schema mismatch')
        if (state.get('external_history_identity') != self._external_history_identity
                or (self._external_history_identity is None and 'external_history_identity' in state)):
            raise ValueError('research checkpoint external history identity mismatch')
        if self._external_history_identity is not None:
            from .external_research import identity
            if identity() != self._external_history_identity:
                raise ValueError('external history policy changed during this run')
        if state.get('presentation') != presentation_identity(
                batch_only_lessons=self.config.get('batch_only_lessons', False),
                restart_predictions=self.config.get('restart_predictions', False),
                comparison_reviews=self.config.get('comparison_reviews', False),
                compact_lesson_memory=self.config.get('compact_lesson_memory', False)):
            raise ValueError('research checkpoint presentation mismatch')
        selection = selection_identity(self.config)
        saved_selection = state.get('lesson_selection')
        if ((selection is None and 'lesson_selection' in state)
                or (selection is not None and (type(saved_selection) is not dict
                    or saved_selection != selection
                    or type(saved_selection.get('diversity_weight')) is not float))):
            raise ValueError('research checkpoint lesson selection mismatch')
        if validate_metric_context(state.get('metric')) != self.metric:
            raise ValueError('research checkpoint metric mismatch')
        if state.get('pending') or state.get('queue'):
            raise ValueError('research checkpoint contains unfinished work; refusing replay')
        for key in ('attempts', 'outcomes', 'valid_hypotheses', 'aux_calls',
                    'reflections', 'reflection_changes', 'reflection_attempts', 'reflected_through'):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError('invalid research checkpoint counter')
        if (not isinstance(state.get('aux_cost'), (int, float))
                or isinstance(state['aux_cost'], bool) or not math.isfinite(state['aux_cost'])
                or state['aux_cost'] < 0
                or (self.config['max_aux_calls'] is not None
                    and state['aux_calls'] > self.config['max_aux_calls'])
                or state['valid_hypotheses'] > state['attempts']
                or state['reflection_changes'] > state['reflections']
                or state['reflections'] > state['reflection_attempts']
                or state['reflection_attempts'] > (state['reflected_through'] //
                    self.config['reflection_interval'] if self.config['reflection_interval'] else 0)
                or bool(state['reflection_attempts']) != bool(state['reflected_through'])
                or state['outcomes'] != state['attempts']
                or state['reflected_through'] > state['outcomes']):
            raise ValueError('inconsistent research checkpoint accounting')
        knowledge = KnowledgeBase(max_lessons=self.config['max_lessons'],
            max_experiments=self.config['max_experiments'], max_fragments=self.config['max_fragments'])
        knowledge.load_state(state['knowledge'])
        evidence = knowledge._evidence()
        ordinary = {ident: record for ident, record in evidence.items()
                    if record.get('schema') != PAIRED_SCHEMA}
        if any(e.get('metric') != self.metric for e in ordinary.values()):
            raise ValueError('research checkpoint contains evidence from a different metric')
        for group in evidence.values():
            if group.get('schema') == PAIRED_SCHEMA:
                self._check_comparison_binding(group, knowledge)
        if ((knowledge.comparisons and (type(state.get('comparison_count')) is not int
                or state['comparison_count'] != len(knowledge.comparisons)))
                or (not knowledge.comparisons and 'comparison_count' in state)):
            raise ValueError('research checkpoint comparison count differs')
        evidence = ordinary
        conclusions = validate_conclusions(state.get('conclusions'), evidence,
            batch_only_lessons=self.config.get('batch_only_lessons', False))
        comparison_reviews = {}
        if self.config.get('comparison_reviews', False):
            from ..hypothesis.comparison_review import validate_cards
            comparison_reviews = validate_cards(state.get('comparison_reviews'), self._review_groups(knowledge),
                seen=knowledge._seen)
            card_versions = {'sciel-comparison-review-card/1': 1,
                             'sciel-comparison-review-card/2': 2,
                             'sciel-comparison-review-card/3': 3}
            required_version = max((card_versions[card['schema']]
                                    for card in comparison_reviews.values()), default=1)
            if self.config.get('comparison_review_version', 1) < required_version:
                raise ValueError('grounded comparison review checkpoint requires receiving version '
                                 + str(required_version))
            if (type(state.get('comparison_review_count')) is not int
                    or state['comparison_review_count'] != len(comparison_reviews)):
                raise ValueError('comparison review checkpoint accounting differs')
        elif 'comparison_reviews' in state or 'comparison_review_count' in state:
            raise ValueError('comparison review checkpoint requires enabled policy')
        external_history = None
        if 'external_history' in state:
            if self._external_history_identity is None:
                raise ValueError('external history requires an explicit configured digest')
            from .external_research import validate_loaded
            external_history = validate_loaded(state['external_history'], config=self.config,
                metric=self.metric, knowledge=knowledge, conclusions=conclusions,
                comparison_reviews=comparison_reviews)
        elif self._external_history_identity is not None:
            # A fresh pre-seed checkpoint can precede the explicit import.
            # Once any work or evidence exists, removing its registry is invalid.
            if (any(state[key] for key in ('attempts', 'outcomes', 'aux_calls', 'aux_cost',
                                          'reflections', 'reflected_through'))
                    or knowledge._seen or knowledge._evidence() or conclusions):
                raise ValueError('research checkpoint is missing its declared external history')
        foreign_reviews = external_history.get('comparison_reviews', {}) if external_history else {}
        local_reviews = [card for ident, card in comparison_reviews.items()
                         if card != foreign_reviews.get(ident)]
        if ((local_reviews and state['reflections'] == 0)
                or any(card['writer']['reference'] > state['reflected_through'] for card in local_reviews)):
            raise ValueError('comparison review checkpoint accounting differs')
        initializer_attempts = {}
        if self.config.get('initializer_witness', False):
            from .initializer_research import validate_state
            initializer_attempts = validate_state(state.get('initializer_witness'), self, knowledge, state['outcomes'])
        elif 'initializer_witness' in state:
            raise ValueError('initializer research state requires explicit enabled policy')
        from .generated_paired_import import validate_loaded as validate_component
        validate_component(self, knowledge, state)
        from .evaluated_study_import import validate_loaded as validate_study
        validate_study(self, knowledge, state)
        from .evaluated_review_import import validate_loaded as validate_study_review
        validate_study_review(self, knowledge, state)
        # Validate retained snapshots before replacing any live controller state.
        self.knowledge = knowledge
        for key in ('attempts', 'outcomes', 'valid_hypotheses', 'aux_calls',
                    'reflections', 'reflection_changes', 'reflection_attempts', 'reflected_through', 'aux_cost', 'stopped_reason'):
            setattr(self, key, state[key])
        self.conclusions = conclusions
        self._conclusion_sources = {}
        self._stopped_cause = None
        self._feedback_delivery = None
        self._comparison_reviews = comparison_reviews
        self._external_history = external_history
        self._initializer_attempts = initializer_attempts
