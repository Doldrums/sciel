"""Resolve exact Circle source families without executing any candidate.

The model response remains the legacy typed-body wire format. An execution
profile selects different complete source bytes, never a relabeled measurement.
"""
from __future__ import annotations

TRACED_PROFILE = 'paired_restart_trace/1'
OPTIMIZER_TRACED_PROFILE = 'paired_restart_trace/2'


def builder_for_profile(profile=None):
    if profile is None:
        from . import circle_restart_body
        return circle_restart_body
    if type(profile) is str and profile == TRACED_PROFILE:
        from . import circle_traced_body
        return circle_traced_body
    if type(profile) is str and profile == OPTIMIZER_TRACED_PROFILE:
        from . import circle_optimizer_traced_body
        return circle_optimizer_traced_body
    raise ValueError('unknown Circle execution profile')


def builder_for_source(source):
    """Recognize a complete source only through its strict inverse/roundtrip."""
    if type(source) is not str:
        raise ValueError('Circle source must be text')
    for profile in (None, TRACED_PROFILE, OPTIMIZER_TRACED_PROFILE):
        builder = builder_for_profile(profile)
        try:
            body = builder.body_of(source)
        except ValueError:
            continue
        if builder.source_for(body) != source:
            raise ValueError('Circle source family roundtrip differs')
        return builder
    raise ValueError('source is outside the registered Circle families')


def builder_for_operation(operation):
    if type(operation) is not str:
        raise ValueError('unknown controlled change operation')
    if operation == 'circle_restart_body/1':
        return builder_for_profile()
    if operation == 'circle_restart_body_traced/1':
        return builder_for_profile(TRACED_PROFILE)
    if operation == 'circle_restart_body_traced/2':
        return builder_for_profile(OPTIMIZER_TRACED_PROFILE)
    if operation == 'circle_restart_reference/1':
        from . import circle_interventions
        return circle_interventions
    if operation == 'circle_restart_scale/1':
        from . import circle_restart_scales
        return circle_restart_scales
    raise ValueError('unknown controlled change operation')
