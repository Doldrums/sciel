"""One structured model-written restart body through the real Engine lifecycle.

The builder validates and compiles source text only. Execution remains the
ordinary bounded evaluator; source receipts never establish numerical quality.
"""

from __future__ import annotations
from ..eval.facade import EvaluationInfrastructureError
from .llm.client import response_accounting, response_cost_provenance
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from ..campaign.budget import CampaignStopped, RequestNotDispatched
from ..hypothesis import circle_restart_body as builder
from ..hypothesis import circle_baseline_equivalence
from ..hypothesis.circle_source_family import builder_for_profile
from ..evidence.measurement import evaluation_observation
from .db import normalized_hash
from .prompt.sources import source_digest
from .types import NoChild, Result
from .validation import TRACE_SCHEMA, trace_attempt, validate_candidate
from .circle_generation_contracts import (
    CONTEXT_SCOPE,
    CONTRACT,
    _COUNTERS,
    _PAIR_COUNTERS,
    _finite,
    _integration_identity,
    _journal_known_cost,
    _json,
)

from .circle_generation_run import CircleGenerationRun
from .circle_generation_state import CircleGenerationState


class CircleGeneration(
    CircleGenerationRun,
    CircleGenerationState,
):

    def __init__(self, research, *, proposal_policy=None, execution_profile=None):
        if research is None:
            raise ValueError('Circle generation requires research')
        if proposal_policy not in (None, 'first_eligible_of_two'):
            raise ValueError('unknown Circle proposal policy')
        self.research = research
        self.execution_profile = execution_profile
        if research.config.get('restart_predictions', False) and execution_profile != 'paired_restart_trace/2':
            raise ValueError('restart predictions require paired_restart_trace/2')
        self.builder = builder_for_profile(execution_profile)
        self.pair_mode = proposal_policy is not None
        self._identity = {'schema': CONTRACT, 'builder': self.builder.identity(),
            'response_format': self._response_format(), 'context_scope': CONTEXT_SCOPE,
            'integration_sha256': _integration_identity()}
        if execution_profile is not None:
            self._identity['execution_profile'] = execution_profile
        if self.pair_mode:
            self._identity['proposal_pair'] = self._pair_identity()
            self.pair_counts = dict.fromkeys(_PAIR_COUNTERS, 0)
            self.pair_history = []
        for name in _COUNTERS:
            setattr(self, name, 0)
        self.known_cost = 0.0
        self.seen_proofs = []
        self.pending = None
        self.stopped_reason = None

    def _response_format(self):
        if self.pair_mode:
            from ..hypothesis import circle_proposal_pair
            value = circle_proposal_pair.response_format()
        else:
            value = builder.response_format()
        if self.research.config.get('restart_predictions', False):
            from ..hypothesis.restart_predictions import json_schema
            value = copy.deepcopy(value)
            schema = value['json_schema']['schema']
            if self.pair_mode:
                schema = schema['properties']['candidates']['items']
            hypothesis = schema['properties']['hypothesis']
            hypothesis['required'].append('restart_predictions')
            hypothesis['properties']['restart_predictions'] = json_schema()
            value['json_schema']['name'] += '_restart_predictions_v1'
        return value

    @staticmethod
    def _pair_identity():
        from ..hypothesis import circle_proposal_pair
        from .circle_pair_selection import POLICY
        from .source_identity import package_source_hashes
        return {'policy': POLICY, 'codec': circle_proposal_pair.identity(),
                'integration_sha256': package_source_hashes(files=('components/code_novelty.py',))}

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def import_external_history(self, capsule, *, expected_sha256, parent):
        """Import declared historical evidence without replaying its requests."""
        from .external_research import import_history
        return import_history(self, capsule, expected_sha256=expected_sha256, parent=parent)

    def import_external_evaluated_study(self, capsule, dependencies, authority, *, expected_sha256, parent):
        from .evaluated_study_import import import_study
        return import_study(self, capsule, dependencies, authority,
                            expected_sha256=expected_sha256, parent=parent)

    def import_external_evaluated_review(self, sidecar, reflection_config, *, expected_sha256):
        from .evaluated_review_import import import_review
        return import_review(self, sidecar, reflection_config, expected_sha256=expected_sha256)

    def import_external_component(self, capsule, *, expected_sha256, parent):
        from .generated_paired_import import import_component
        return import_component(self, capsule, expected_sha256=expected_sha256, parent=parent)

    def import_external_component_review(self, sidecar, *, expected_sha256):
        from .generated_paired_review_import import import_review
        return import_review(self, sidecar, expected_sha256=expected_sha256)

    def _emit(self, action, **fields):
        self.research.emit('circle_generation', generation_contract=CONTRACT, action=action,
            experiment_id=self.pending.get('research_id') if self.pending else None, **fields)

    def prepare_prompt(self, snapshot):
        ctx = snapshot.research_context
        if (type(ctx) is not dict or ctx.get('parent_code') != snapshot.parent_code
                or ctx.get('parent_source_sha256') != source_digest(snapshot.parent_code)
                or ctx.get('parent_id') != snapshot.parent_id
                or ctx.get('parent_score') != snapshot.parent_score):
            raise ValueError('Circle generation requires the exact measured parent')
        if (_json(self.builder.identity()) != _json(self._identity['builder'])
                or _json(self._response_format()) != _json(self._identity['response_format'])
                or _integration_identity() != self._identity['integration_sha256']
                or (self.pair_mode and _json(self._pair_identity()) != _json(self._identity['proposal_pair']))):
            raise ValueError('Circle generation source or wire identity changed')
        body = self.builder.body_of(snapshot.parent_code)
        current = {'source_sha256': source_digest(snapshot.parent_code), 'body': body,
            'interface_reference': {'contract': self.builder.CONTRACT,
                'interface_sha256': self._identity['builder']['interface_sha256'],
                'location': 'Complete typed array interface in the response instruction.'},
            'scope': 'Only this body is editable. Parent score and cases are measured facts in RESEARCH_CONTEXT_JSON.'}
        if self.execution_profile is not None:
            current['execution_profile'] = self.execution_profile
        # Do not expose the enclosing solver's Incumbent internals as names that
        # the body can use. Historical paired source changes remain quoted data.
        opening = ('Generate two independent numerical restart alternatives in preferred order. '
                   'Both use the same measured parent; at most one will be evaluated. '
                   if self.pair_mode else 'Generate one new numerical restart body using only the documented array API. ')
        return (opening +
            'The immutable scaffold owns random draws, deadlines, SLSQP and incumbent retention. '
            'Array parameters are values, not solver objects. Historical paired sources, if shown, '
            'are read-only evidence; their names and helpers are not available in this body. '
            'Forecast score(child) minus the actual measured parent_score, never a historical '
            'comparator or an absolute score. Do not claim causal or guaranteed improvement. '
            'Do not resubmit an unchanged body from prior evidence. Changing only its forecast '
            'does not create a new candidate; repeated sources are rejected without evaluation. '
            'Compare the proposed initializer with the CURRENT parent, not a rejected historical child. '
            + ('A recognized rewrite of the current default initializer is also rejected without evaluation; '
               'different variable names or using initial_params at index 0 do not introduce a new '
               'initializer when the resulting values are unchanged. This is not a quality measurement.\n'
               if self.execution_profile is None else
               'The legacy symbolic initializer-equivalence checker is unavailable for this execution profile; '
               'exact and normalized source repetition checks still apply.\n') +
            'CURRENT_RESTART_BODY_JSON:\n' + _json(current) + '\n'
            'TASK:\n' + snapshot.task_prompt + '\nDIRECTIVE:\n' + snapshot.directive + '\n'
            'LESSONS (unverified interpretations):\n' + '\n'.join(snapshot.kb_lessons))
