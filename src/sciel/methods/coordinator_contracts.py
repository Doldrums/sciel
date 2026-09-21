"""Coordinator refusals, transaction boundary and experiment shape checks."""
from __future__ import annotations

from functools import wraps

from .budget import BudgetExceeded


class Refusal(ValueError):
    """A proposal has no authority to perform the requested operation."""


class StopRequested(RuntimeError):
    """Operator stop observed before an external dispatch, never a code result."""


class TrialQuotaExceeded(BudgetExceeded):
    """A local research-policy quota; shared resource ceilings are unchanged."""


def _serialized(function):
    @wraps(function)
    def call(self, *args, **kwargs):
        with self._locked():
            return function(self, *args, **kwargs)
    return call


def _experiment_shape(value):
    if not isinstance(value,dict) or set(value)!={'question','treatment','control','inputs','criterion','scope'}:
        raise Refusal('experiment requires question, treatment, control, inputs, criterion and scope')
    if not isinstance(value['question'],str) or not value['question'].strip():
        raise Refusal('experiment question must be nonempty text')
    for arm in ('treatment','control'):
        if not isinstance(value[arm],list) or any(not isinstance(i,str) or not i for i in value[arm]):
            raise Refusal('experiment pipelines must be arrays of version IDs')
    if not isinstance(value['inputs'],list):
        raise Refusal('experiment inputs must be an array')
    if not isinstance(value['criterion'],dict) or set(value['criterion'])!={'metric','direction','minimum_effect'}:
        raise Refusal('experiment criterion fields are incomplete or unsupported')
    if not isinstance(value['criterion']['metric'],str) or not isinstance(value['criterion']['direction'],str):
        raise Refusal('criterion metric and direction must be text')
