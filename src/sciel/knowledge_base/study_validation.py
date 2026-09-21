"""Literal detached-study contracts and bounded validation primitives."""
from __future__ import annotations

from contextvars import ContextVar
import hashlib
import json
from .catalogs import canonical_json
from ..hypothesis.parsing import _unique_object, _reject_constant


SCHEMA = 'sciel-evaluated-detached-study/1'

FACTS_SCHEMA = 'sciel-evaluated-detached-facts/1'

AUTHORITY_SCHEMA = 'sciel-evaluated-study-authority/1'

PROFILE = 'paired_restart_trace/2'

HOST_PRODUCERS = ('sciel-host-composed-rotation-numeric/1',)

HOST_DESIGN_SCHEMA = 'sciel-host-ordinary-composed-rotation-design/1'

G_PRODUCER = 'sciel-component-target-numeric/1'

EVALUATED_G_PRODUCER = 'sciel-evaluated-target-numeric/1'

MAX_EVALUATED_ANCESTRY = 8

_VALIDATION_SESSION = ContextVar('evaluated_detached_validation_session', default=None)

SCOPE = ('One recorded detached three-seed contrast. HOST authorship is not model generation or memory benefit. '
    'Native measurements remain UNMEASURED and native forecasts are NOT_DECLARED. Raw and exact certified '
    'quality, paired-forecast agreement, retained-prefix restart observations and prediction agreement are '
    'separate. Unknown is not false; unissued is not an unknown dispatch. No ordinary outcome, lesson, '
    'exposure, causal credit, promotion, new cost, or execution authority is created. Recorded bindings '
    'are not independent proof of execution, payment or preregistration.')

_AUTHOR_FIELDS = {'kind', 'artifact_sha256', 'base_source_sha256', 'candidate_source_sha256', 'comparator_source_sha256'}

_AUTHORITY_FIELDS = {'schema', 'kind', 'dependencies_sha256', 'producer_manifest_sha256',
    'producer_summary_sha256', 'author_artifact_sha256', 'generation_config_sha256', 'host_decision_sha256', 'issuer'}

_SLOT_FIELDS = {'index', 'seed', 'role', 'intent_text', 'raw_receipt_text', 'validated_text', 'unknown_text'}

def _require(ok, message):
    if not ok:
        raise ValueError('evaluated detached study: ' + message)

def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def _same(a, b):
    return canonical_json(a) == canonical_json(b)

def _object(text, name):
    _require(type(text) is str and 0 < len(text.encode('utf-8')) <= 8*1024**2, name + ' text bound')
    value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    _require(type(value) is dict, name + ' must be a literal object')
    return value
