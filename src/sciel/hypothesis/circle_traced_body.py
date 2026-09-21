"""Exact typed-body family with optional, same-source restart tracing.

Assembly executes no generated helper or solver. The trace-enabled scaffold is
a distinct execution profile: comparisons must use it on both arms, including
the default control. Ordinary evaluation has an explicit no-op recorder.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import circle_restart_body as legacy
from . import circle_traced_restart as traced

CONTRACT = OPERATION = 'circle_restart_body_traced/1'
EXECUTION_PROFILE = 'paired_restart_trace/1'
WIRE_CONTRACT = legacy.CONTRACT
DEFAULT_BODY = legacy.DEFAULT_BODY
STATS_MARKER = legacy.STATS_MARKER
MAX_BODY_BYTES = legacy.MAX_BODY_BYTES
MAX_BODY_LINES = legacy.MAX_BODY_LINES
MAX_RESPONSE_BYTES = legacy.MAX_RESPONSE_BYTES
BodyProposal = legacy.BodyProposal
validate_body = legacy.validate_body
decode_response = legacy.decode_response
parse_response = legacy.parse_response
parse_hypothesis_response = legacy.parse_hypothesis_response
parse_response_hypothesis = legacy.parse_response_hypothesis
response_format = legacy.response_format

_NOOP = '''class _NoopTraceRecorder:
    """No trace was requested; no diagnostic state or failure is fabricated."""
    def _ignore(self, *args, **kwargs):
        return None

    begin_start = begin_helper = helper_end = optimizer_enter = _ignore
    optimizer_exit = optimizer_result = end_start = finish = _ignore


'''
_BINDING = '''    global SCIEL_RESTART_TRACE
    SCIEL_RESTART_TRACE = None
    if SCIEL_RESTART_TRACE_BINDING is None:
        trace = _NoopTraceRecorder()
    else:
        if type(SCIEL_RESTART_TRACE_BINDING) is not dict:
            raise ValueError("invalid explicit restart trace binding")
        required = {"source_sha256", "seed", "runtime_sha256", "invocation_id"}
        if set(SCIEL_RESTART_TRACE_BINDING) not in (required, required | {"adapter_identity_sha256"}):
            raise ValueError("invalid explicit restart trace binding fields")
        if ("adapter_identity_sha256" in SCIEL_RESTART_TRACE_BINDING and
                not _rt_hex(SCIEL_RESTART_TRACE_BINDING["adapter_identity_sha256"])):
            raise ValueError("invalid explicit restart trace adapter hash")
        identity = {key: SCIEL_RESTART_TRACE_BINDING[key] for key in required}
        identity["scaffold_sha256"] = SCIEL_RESTART_TRACE_SCAFFOLD_SHA256
        _rt_identity(identity)
        trace = _TraceRecorder(identity)
    started_at = time.monotonic()
'''


def _sha(value):
    return hashlib.sha256(value if type(value) is bytes else value.encode('utf-8')).hexdigest()


def _require(ok, reason):
    if not ok:
        raise ValueError('traced body: ' + reason)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _family():
    """Derive one exact common template from the existing traced /2 builder."""
    base = traced.build(DEFAULT_BODY)
    source = base['code']
    anchor = 'class _TraceRecorder:\n'
    _require(source.count(anchor) == 1, 'recorder anchor differs')
    source = source.replace(anchor, _NOOP + anchor, 1)
    binding = next(p['after'] for p in base['proof']['patches'] if p['name'] == 'solver_binding')
    _require(source.count(binding) == 1, 'binding anchor differs')
    source = source.replace(binding, _BINDING, 1)
    stamp = 'SCIEL_RESTART_TRACE_SCAFFOLD_SHA256 = ' + repr(base['proof']['scaffold_sha256'])
    placeholder = 'SCIEL_RESTART_TRACE_SCAFFOLD_SHA256 = ' + repr('0'*64)
    _require(source.count(stamp) == 1, 'scaffold identity anchor differs')
    normalized = source.replace(stamp, placeholder, 1)
    scaffold_sha = _sha(normalized)
    source = normalized.replace(placeholder,
        'SCIEL_RESTART_TRACE_SCAFFOLD_SHA256 = ' + repr(scaffold_sha), 1)
    helper = legacy._HEADER + legacy._indented(DEFAULT_BODY)
    _require(source.count(helper) == 1, 'typed helper anchor differs')
    prefix, _, suffix = source.partition(helper)
    dependencies = dict(typed_body_builder_sha256=_sha(Path(legacy.__file__).read_bytes()),
        traced_builder_sha256=_sha(Path(traced.__file__).read_bytes()),
        collector_sha256=base['proof']['collector_sha256'],
        proposal_builder_sha256=legacy.PINNED_PROPOSALS_SHA256)
    return prefix, suffix, dict(common_scaffold_sha256=scaffold_sha,
        control_source_sha256=_sha(source), dependencies=dependencies)


def source_for(body=DEFAULT_BODY):
    """Compile the existing strict body surface into the new traced family."""
    validate_body(body)
    prefix, suffix, _ = _family()
    return prefix + legacy._HEADER + legacy._indented(body) + suffix


def body_of(source):
    """Exact family inverse; reject legacy, bound-only and damaged scaffolds."""
    _require(type(source) is str, 'source must be text')
    try:
        source.encode('utf-8')
    except UnicodeError as exc:
        raise ValueError('traced body: source must be strict UTF8') from exc
    prefix, suffix, _ = _family()
    begin = prefix + legacy._HEADER
    _require(source.startswith(begin) and source.endswith(suffix) and
        len(source) >= len(begin) + len(suffix), 'source is outside the exact traced body family')
    middle = source[len(begin):len(source)-len(suffix)]
    _require(middle.endswith('\n'), 'compiled body lacks structural LF')
    lines = middle.split('\n')[:-1]
    _require(bool(lines) and all(line.startswith('    ') for line in lines), 'body indentation differs')
    body = ''.join(line[4:] + '\n' for line in lines)
    validate_body(body)
    _require(source_for(body) == source, 'source differs outside exact body assembly')
    return body


def apply(parent_source, target_body):
    """One byte-bound helper edit, with no semantic or quality certification."""
    parent_body = body_of(parent_source)
    child = source_for(target_body)
    _require(child != parent_source, 'body edit leaves source unchanged')
    prefix, suffix, metadata = _family()
    before = legacy._HEADER + legacy._indented(parent_body)
    after = legacy._HEADER + legacy._indented(target_body)
    p, s, a, b = [value.encode('utf-8') for value in (prefix, suffix, before, after)]
    parent_raw, child_raw = parent_source.encode('utf-8'), child.encode('utf-8')
    _require(parent_raw == p+a+s and child_raw == p+b+s, 'body source assembly differs')
    proof = dict(schema='source-bound-edit/1', contract=CONTRACT, operation=OPERATION,
        execution_profile=EXECUTION_PROFILE, wire_contract=WIRE_CONTRACT,
        baseline_source_sha256=metadata['control_source_sha256'],
        common_scaffold_sha256=metadata['common_scaffold_sha256'],
        dependencies=metadata['dependencies'], parent_source_sha256=_sha(parent_raw),
        child_source_sha256=_sha(child_raw), parent_body=parent_body, target_body=target_body,
        parent_body_sha256=_sha(parent_body), target_body_sha256=_sha(target_body),
        assembly=dict(indent='    ', append_final_lf=not target_body.endswith('\n'),
            line_endings='LF/CRLF preserved; lone CR rejected', parent_body_form='canonical final-LF source inverse'),
        before=dict(text=before, sha256=_sha(a), byte_span=[len(p),len(p)+len(a)]),
        after=dict(text=after, sha256=_sha(b), byte_span=[len(p),len(p)+len(b)]),
        unchanged_prefix=dict(bytes=len(p), sha256=_sha(p)),
        unchanged_suffix=dict(bytes=len(s), sha256=_sha(s)), unchanged_outside_helper=True,
        scope='Exact traced source/body assembly only; no semantic, causal, safety, runtime or quality proof. '
              'No trace binding selects an explicit no-op recorder. Both comparison arms require this execution profile.')
    proof['plan_key'] = OPERATION + ':' + _sha(parent_raw) + ':' + _sha(child_raw)
    return dict(code=child, proof=proof)


def interface_description():
    result = legacy.interface_description()
    _, _, metadata = _family()
    result.update(contract=CONTRACT, wire_contract=WIRE_CONTRACT, execution_profile=EXECUTION_PROFILE)
    result['default_transformation'].update(control_source_sha256=metadata['control_source_sha256'],
        common_scaffold_sha256=metadata['common_scaffold_sha256'])
    result['tracing'] = dict(schema='sciel-restart-trace/2',
        unbound='Ordinary evaluation uses a no-op recorder and exports trace=None.',
        bound='An explicit validated repeat binding records same-invocation bounded diagnostics; invalid non-null bindings fail.',
        comparison='Both arms use this same traced scaffold; recording may alter wall-clock trajectories. No runtime or quality benefit is implied.',
        dependencies=metadata['dependencies'])
    return result


def response_instruction():
    instruction = legacy.response_instruction()
    before = _canonical(legacy.interface_description())
    _require(instruction.endswith(before), 'response interface suffix differs')
    return instruction[:-len(before)] + _canonical(interface_description())


def identity():
    _, _, metadata = _family()
    return dict(contract=CONTRACT, operation=OPERATION, execution_profile=EXECUTION_PROFILE,
        wire_contract=WIRE_CONTRACT, source_sha256=_sha(Path(__file__).read_bytes()),
        **metadata, baseline_source_sha256=metadata['control_source_sha256'],
        default_body_sha256=_sha(DEFAULT_BODY), interface_sha256=_sha(_canonical(interface_description())),
        response_format_sha256=_sha(_canonical(response_format())), stats_marker=STATS_MARKER,
        scope='Distinct traced control/scaffold identity; no legacy metric relabeling or unchanged-timing claim.')
