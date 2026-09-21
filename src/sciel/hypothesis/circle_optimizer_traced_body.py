"""Distinct typed-body profile with same-invocation optimizer outcome capture.

Historical builders/codecs are byte-frozen. Assembly executes no source: only a
later candidate import defines two isolated, pinned codec namespaces. Ordinary
evaluation uses the same source with a no-op recorder. Timing may change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import circle_restart_body as legacy
from . import circle_traced_body as previous
from . import circle_traced_restart as previous_trace
from . import restart_trace as legacy_codec
from . import restart_optimizer_trace as codec

CONTRACT = OPERATION = 'circle_restart_body_traced/2'
EXECUTION_PROFILE = 'paired_restart_trace/2'
WIRE_CONTRACT = legacy.CONTRACT
DEFAULT_BODY = legacy.DEFAULT_BODY
STATS_MARKER = legacy.STATS_MARKER
MAX_BODY_BYTES, MAX_BODY_LINES, MAX_RESPONSE_BYTES = legacy.MAX_BODY_BYTES, legacy.MAX_BODY_LINES, legacy.MAX_RESPONSE_BYTES
BodyProposal = legacy.BodyProposal
validate_body = legacy.validate_body
decode_response = legacy.decode_response
parse_response = legacy.parse_response
parse_hypothesis_response = legacy.parse_hypothesis_response
parse_response_hypothesis = legacy.parse_response_hypothesis
response_format = legacy.response_format
PINNED_LEGACY = {
    'collector_sha256': '4239a6990d7fc99ff6eaa41fd9a05762fdd4f1bbea5e5c683e3379d51717864c',
    'body_builder_sha256': '9331f145000ff1a4faa435b6eceaa357977e62e3eef7a1af4d7d0e52b5ad319d',
    'traced_builder_sha256': '9f3a8fa640fd375d9ffda33ff1a19755e369e7cfdb9ae3b117e06b5a7ba43f8c',
}


def _sha(value):
    return hashlib.sha256(value if type(value) is bytes else value.encode('utf-8')).hexdigest()


def _require(ok, reason):
    if not ok:
        raise ValueError('optimizer traced body: ' + reason)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _replace(source, before, after):
    _require(source.count(before) == 1, 'nonunique source anchor')
    return source.replace(before, after, 1)


def _family():
    paths = {'body_builder_sha256': Path(previous.__file__),
             'traced_builder_sha256': Path(previous_trace.__file__)}
    old_raw = {key: path.read_bytes() for key, path in paths.items()}
    old_raw['collector_sha256'] = legacy_codec.collector_source().encode('utf-8')
    _require({key: _sha(raw) for key, raw in old_raw.items()} == PINNED_LEGACY, 'historical source dependency differs')
    legacy_text = old_raw['collector_sha256'].decode('utf-8')
    extension_raw = Path(codec.__file__).read_bytes()
    extension = extension_raw.decode('utf-8')
    import_line = 'from . import restart_trace as _legacy\n'
    _require(extension.count(import_line) == 1, 'codec import anchor differs')
    extension = extension.replace(import_line, '', 1)
    # Separate globals are essential: old methods must keep their /2 constants
    # and validator even while the new subclass exports a /3 trace.
    embedded = ('\n# BEGIN bounded optimizer restart trace collector\n'
        'import types as _optimizer_trace_types\n'
        '_optimizer_trace_legacy = _optimizer_trace_types.ModuleType("_sciel_restart_trace_v2")\n'
        'exec(compile(' + repr(legacy_text) + ', "<sciel-restart-trace/2>", "exec"), _optimizer_trace_legacy.__dict__)\n'
        '_optimizer_trace_namespace = {"__name__": "_sciel_restart_trace_v3", "_legacy": _optimizer_trace_legacy}\n'
        'exec(compile(' + repr(extension) + ', "<sciel-restart-trace/3>", "exec"), _optimizer_trace_namespace)\n'
        'RestartTrace = _optimizer_trace_namespace["RestartTrace"]\n'
        '_rt_identity = _optimizer_trace_legacy._rt_identity\n'
        '_rt_hex = _optimizer_trace_legacy._rt_hex\n'
        '# END bounded optimizer restart trace collector\n')
    source = previous.source_for()
    old_block = ('\n# BEGIN bounded restart trace collector\n' +
        legacy_text.replace('from __future__ import annotations\n', '') +
        '\n# END bounded restart trace collector\n')
    source = _replace(source, old_block, embedded)
    source = _replace(source, '    optimizer_exit = optimizer_result = end_start = finish = _ignore\n',
        '    optimizer_exit = optimizer_outcome = optimizer_result = end_start = finish = _ignore\n')
    source = _replace(source, '            stats["successful_starts"] += int(bool(result.success))\n',
        '            optimizer_success = bool(result.success)\n'
        '            stats["successful_starts"] += int(optimizer_success)\n'
        '            trace.optimizer_outcome(optimizer_success, result)\n')
    old_identity = previous.identity()
    stamp = 'SCIEL_RESTART_TRACE_SCAFFOLD_SHA256 = ' + repr(old_identity['common_scaffold_sha256'])
    placeholder = 'SCIEL_RESTART_TRACE_SCAFFOLD_SHA256 = ' + repr('0'*64)
    normalized = _replace(source, stamp, placeholder)
    scaffold_sha = _sha(normalized)
    source = _replace(normalized, placeholder, 'SCIEL_RESTART_TRACE_SCAFFOLD_SHA256 = ' + repr(scaffold_sha))
    helper = legacy._HEADER + legacy._indented(DEFAULT_BODY)
    _require(source.count(helper) == 1, 'typed helper anchor differs')
    prefix, _, suffix = source.partition(helper)
    dependencies = dict(typed_body_builder_sha256=_sha(Path(legacy.__file__).read_bytes()),
        previous_body_builder_sha256=PINNED_LEGACY['body_builder_sha256'],
        traced_builder_sha256=PINNED_LEGACY['traced_builder_sha256'],
        legacy_collector_sha256=PINNED_LEGACY['collector_sha256'], collector_sha256=_sha(extension_raw),
        proposal_builder_sha256=legacy.PINNED_PROPOSALS_SHA256,
        list_policy=legacy.list_policy_identity())
    return prefix, suffix, dict(common_scaffold_sha256=scaffold_sha,
        control_source_sha256=_sha(source), dependencies=dependencies)


def source_for(body=DEFAULT_BODY):
    validate_body(body)
    prefix, suffix, _ = _family()
    return prefix + legacy._HEADER + legacy._indented(body) + suffix


def body_of(source):
    _require(type(source) is str, 'source must be text')
    try:
        source.encode('utf-8')
    except UnicodeError as exc:
        raise ValueError('optimizer traced body: source must be strict UTF8') from exc
    prefix, suffix, _ = _family()
    begin = prefix + legacy._HEADER
    _require(source.startswith(begin) and source.endswith(suffix) and len(source) >= len(begin)+len(suffix),
        'source is outside the exact optimizer traced body family')
    middle = source[len(begin):len(source)-len(suffix)]
    _require(middle.endswith('\n'), 'compiled body lacks structural LF')
    lines = middle.split('\n')[:-1]
    _require(bool(lines) and all(line.startswith('    ') for line in lines), 'body indentation differs')
    body = ''.join(line[4:] + '\n' for line in lines)
    validate_body(body)
    _require(source_for(body) == source, 'source differs outside exact body assembly')
    return body


def apply(parent_source, target_body):
    parent_body = body_of(parent_source)
    child = source_for(target_body)
    _require(child != parent_source, 'body edit leaves source unchanged')
    prefix, suffix, metadata = _family()
    before, after = [legacy._HEADER + legacy._indented(body) for body in (parent_body, target_body)]
    p, s, a, b = [value.encode('utf-8') for value in (prefix, suffix, before, after)]
    parent_raw, child_raw = parent_source.encode('utf-8'), child.encode('utf-8')
    _require(parent_raw == p+a+s and child_raw == p+b+s, 'body source assembly differs')
    proof = dict(schema='source-bound-edit/1', contract=CONTRACT, operation=OPERATION,
        execution_profile=EXECUTION_PROFILE, wire_contract=WIRE_CONTRACT,
        baseline_source_sha256=metadata['control_source_sha256'], common_scaffold_sha256=metadata['common_scaffold_sha256'],
        dependencies=metadata['dependencies'], parent_source_sha256=_sha(parent_raw), child_source_sha256=_sha(child_raw),
        parent_body=parent_body, target_body=target_body, parent_body_sha256=_sha(parent_body), target_body_sha256=_sha(target_body),
        assembly=dict(indent='    ', append_final_lf=not target_body.endswith('\n'),
            line_endings='LF/CRLF preserved; lone CR rejected', parent_body_form='canonical final-LF source inverse'),
        before=dict(text=before, sha256=_sha(a), byte_span=[len(p),len(p)+len(a)]),
        after=dict(text=after, sha256=_sha(b), byte_span=[len(p),len(p)+len(b)]),
        unchanged_prefix=dict(bytes=len(p), sha256=_sha(p)), unchanged_suffix=dict(bytes=len(s), sha256=_sha(s)),
        unchanged_outside_helper=True,
        scope='Exact optimizer-traced source/body assembly only; no semantic, causal, safety or quality proof. '
              'No trace binding selects a no-op recorder. Both arms require this execution profile; timing may change.')
    proof['plan_key'] = OPERATION + ':' + _sha(parent_raw) + ':' + _sha(child_raw)
    return dict(code=child, proof=proof)


def interface_description():
    result = legacy.interface_description()
    _, _, metadata = _family()
    result.update(contract=CONTRACT, wire_contract=WIRE_CONTRACT, execution_profile=EXECUTION_PROFILE)
    result['default_transformation'].update(control_source_sha256=metadata['control_source_sha256'],
        common_scaffold_sha256=metadata['common_scaffold_sha256'])
    result['tracing'] = dict(schema=codec.TRACE_SCHEMA,
        unbound='Ordinary evaluation uses a no-op recorder and exports trace=None.',
        bound='Validated repeat bindings record bounded same-invocation diagnostics and observed optimizer outcomes; invalid non-null bindings fail.',
        optimizer='Success is the existing bool(result.success); integer status is observed only for retained calls. Neither implies feasible result.x or incumbent admission.',
        comparison='Both arms use this same scaffold; recording may alter wall-clock trajectories. No runtime or quality benefit is implied.',
        dependencies=metadata['dependencies'])
    return result


def response_instruction():
    instruction = legacy.response_instruction()
    before = _canonical(legacy.interface_description())
    _require(instruction.endswith(before), 'response interface suffix differs')
    return instruction[:-len(before)] + _canonical(interface_description())


def identity():
    _, _, metadata = _family()
    return dict(contract=CONTRACT, operation=OPERATION, execution_profile=EXECUTION_PROFILE, wire_contract=WIRE_CONTRACT,
        source_sha256=_sha(Path(__file__).read_bytes()), **metadata, baseline_source_sha256=metadata['control_source_sha256'],
        default_body_sha256=_sha(DEFAULT_BODY), interface_sha256=_sha(_canonical(interface_description())),
        response_format_sha256=_sha(_canonical(response_format())), stats_marker=STATS_MARKER,
        scope='Distinct optimizer-traced scaffold identity; historical source bytes remain unchanged. No unchanged-timing claim.')
