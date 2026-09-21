"""Explicitly new trace-enabled scaffold; historical source families stay exact.

Assembly and proof generation execute no helper or solver. Recording changes
wall-clock trajectories, so comparisons require this scaffold on both arms.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from . import circle_proposals as proposals
from . import circle_restart_body as body_builder

CONTRACT = 'sciel-circle-traced-restart/2'
_PLACEHOLDER = '0' * 64

_SAFE_RECORDER = '''
class _TraceRecorder:
    """Recorder failure invalidates diagnostics, never a numerical observation."""
    def __init__(self, identity):
        self.failed = None
        self.trace = None
        try:
            self.trace = RestartTrace(identity)
        except Exception as exc:
            self.failed = type(exc).__name__[:80]

    def __getattr__(self, name):
        def call(*args, **kwargs):
            if self.failed is None:
                try:
                    return getattr(self.trace, name)(*args, **kwargs)
                except Exception as exc:
                    self.failed = type(exc).__name__[:80]
            if name == "finish":
                return {"schema": "sciel-restart-trace-unavailable/1",
                        "reason": "recorder_error", "error_type": self.failed}
            return None
        return call

'''


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


_INVOKE = '''def _proposal_invoke(incumbent, initial, noise, default_guess, restart_index, stats,
                     trace, quality):
    started = time.monotonic()
    stats["proposal_attempts"] += 1
    args = None
    raw_kind, raw_array, raw_type = "exception", None, None
    effective, origin, error, stage = None, None, None, "helper"
    try:
        args = {"best": incumbent.copy(), "initial": initial.copy(),
                "noise": noise.copy(), "ordinary": default_guess.copy()}
        trace.begin_helper(args)
        value = propose_restart(args["best"], args["initial"], args["noise"],
                                args["ordinary"], restart_index)
        raw_type = type(value).__name__
        raw_kind = "none" if value is None else "value"
        stage = "normalization"
        if value is None:
            stats["proposal_none"] += 1
            effective = default_guess.copy()
            origin = "saved_ordinary_fallback"
            return effective
        array = np.asarray(value)
        raw_array = array
        if array.dtype.kind not in "fiu" or array.shape != (78,):
            raise ValueError("restart proposal must be a real numeric vector of shape (78,)")
        array = np.asarray(array, dtype=float).copy()
        if not np.isfinite(array).all():
            raise ValueError("restart proposal must contain finite values")
        stats["proposal_accepted"] += 1
        effective, origin = array, "accepted_return"
        return effective
    except Exception as exc:
        stats["proposal_errors"] += 1
        error = type(exc).__name__
        raise
    finally:
        if args is not None:
            trace.helper_end(raw_kind=raw_kind, raw_value=raw_array, raw_type=raw_type,
                post_inputs=args, effective=effective, origin=origin, error=error,
                error_stage=stage if error is not None else None,
                incumbent_sum=quality[0], improvements=quality[1])
        stats["proposal_time_seconds"] += max(0.0, time.monotonic() - started)
'''


def _patches(codec):
    runtime_start = proposals._RUNTIME.index('def _proposal_invoke(')
    runtime_end = proposals._RUNTIME.index('# END trusted generated-restart runtime')
    before_invoke = proposals._RUNTIME[runtime_start:runtime_end]
    embedded = ('\n# BEGIN bounded restart trace collector\n' +
        codec.replace('from __future__ import annotations\n', '') +
        '\n# END bounded restart trace collector\n' + _SAFE_RECORDER +
        'SCIEL_RESTART_TRACE_SCAFFOLD_SHA256 = ' + repr(_PLACEHOLDER) + '\n' +
        'SCIEL_RESTART_TRACE_BINDING = None\nSCIEL_RESTART_TRACE = None\n\n')
    return [
        ('collector', '# BEGIN trusted generated-restart runtime\n',
         embedded + '# BEGIN trusted generated-restart runtime\n'),
        ('single_helper_invocation', before_invoke, _INVOKE),
        ('solver_binding', '    started_at = time.monotonic()\n',
         '    global SCIEL_RESTART_TRACE\n'
         '    if not isinstance(SCIEL_RESTART_TRACE_BINDING, dict):\n'
         '        raise ValueError("trace scaffold requires an explicit invocation binding")\n'
         '    identity = {key: SCIEL_RESTART_TRACE_BINDING[key] for key in\n'
         '                ("source_sha256", "seed", "runtime_sha256", "invocation_id")}\n'
         '    identity["scaffold_sha256"] = SCIEL_RESTART_TRACE_SCAFFOLD_SHA256\n'
         '    trace = _TraceRecorder(identity)\n'
         '    SCIEL_RESTART_TRACE = None\n'
         '    started_at = time.monotonic()\n'),
        ('start_boundary', '        for _ in range(max_restarts):\n            check_deadline()\n',
         '        for _ in range(max_restarts):\n'
         '            trace.begin_start(_, incumbent_sum=incumbent._sum, improvements=incumbent.improvements)\n'
         '            check_deadline()\n'),
        ('helper_boundary', 'guess = _proposal_invoke(incumbent._params, initial, noise, guess, _, stats)',
         'guess = _proposal_invoke(incumbent._params, initial, noise, guess, _, stats,\n'
         '                    trace, (incumbent._sum, incumbent.improvements))'),
        ('optimizer_entry', '            result = minimize(timed(objective), guess, method="SLSQP",\n',
         '            trace.optimizer_enter(incumbent_sum=incumbent._sum, improvements=incumbent.improvements)\n'
         '            result = minimize(timed(objective), guess, method="SLSQP",\n'),
        ('optimizer_return', '            stats["completed_starts"] += 1\n',
         '            trace.optimizer_exit(incumbent_sum=incumbent._sum, improvements=incumbent.improvements)\n'
         '            stats["completed_starts"] += 1\n'),
        ('optimizer_result', '            incumbent.consider(result.x)\n',
         '            returned_params = result.x\n'
         '            trace.optimizer_result(returned_params)\n'
         '            incumbent.consider(returned_params)\n'),
        ('start_complete', '            finish_proposal_role()\n',
         '            finish_proposal_role()\n'
         '            trace.end_start("completed", incumbent_sum=incumbent._sum, improvements=incumbent.improvements)\n'),
        ('trace_export', '        LAST_STATS.update(stats)\n',
         '        LAST_STATS.update(stats)\n'
         '        SCIEL_RESTART_TRACE = trace.finish(stats["stop_reason"],\n'
         '            incumbent_sum=incumbent._sum, improvements=incumbent.improvements)\n'),
    ]


def _assemble(original, codec):
    source, patches = original, []
    for name, before, after in _patches(codec):
        if source.count(before) != 1:
            raise ValueError('trace scaffold anchor is not unique: ' + name)
        start = len(source[:source.index(before)].encode('utf-8'))
        source = source.replace(before, after, 1)
        patches.append(dict(name=name, before=before, after=after,
            sequential_byte_start=start, before_sha256=_sha(before), after_sha256=_sha(after)))
    return source, patches


def build(body=body_builder.DEFAULT_BODY):
    """Return traced source and reversible, byte-bound instrumentation proof."""
    from .restart_trace import collector_source
    codec = collector_source()
    old_control = body_builder.source_for()
    common, _ = _assemble(old_control, codec)
    scaffold_sha = _sha(common)
    original = body_builder.source_for(body)
    source, patches = _assemble(original, codec)
    before = 'SCIEL_RESTART_TRACE_SCAFFOLD_SHA256 = ' + repr(_PLACEHOLDER)
    after = 'SCIEL_RESTART_TRACE_SCAFFOLD_SHA256 = ' + repr(scaffold_sha)
    if source.count(before) != 1:
        raise ValueError('trace scaffold identity anchor differs')
    patches.append(dict(name='scaffold_identity', before=before, after=after,
        sequential_byte_start=len(source[:source.index(before)].encode('utf-8')),
        before_sha256=_sha(before), after_sha256=_sha(after)))
    source = source.replace(before, after, 1)
    return {'code': source, 'proof': {'schema': CONTRACT,
        'untraced_source_sha256': _sha(original), 'traced_source_sha256': _sha(source),
        'untraced_control_sha256': _sha(old_control), 'scaffold_sha256': scaffold_sha,
        'collector_sha256': _sha(codec), 'body_sha256': _sha(body), 'patches': patches,
        'scope': 'Same-invocation bounded diagnostics; instrumentation may alter wall-clock trajectories. '
                 'Use this shared scaffold on both arms of future quality comparisons.'}}


def source_for(body=body_builder.DEFAULT_BODY):
    return build(body)['code']
