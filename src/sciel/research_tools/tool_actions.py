"""Validate model-proposed research actions before any tool dispatch."""
from __future__ import annotations

import json
from ..methods.models import canonical_bytes, require_hash
from .config import effective_limits
from .result_contract import REPLAY_POLICIES, validate_policy
from .research_steps import TEXT_FIELDS, proposal_fields
from . import experiment_protocol
from .observation_cards import _clip
from .tool_protocol import _DEPENDENCY_GUIDANCE

def _finite_json(text):
    def constant(_):
        raise ValueError('non-finite JSON is not supported')
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    value = json.loads(text, parse_constant=constant, object_pairs_hook=unique)
    canonical_bytes(value)
    return value


def _proposal_failure(error, response, artifact_id):
    """Explain recorded provider context without repairing or executing output."""
    reason = _clip(str(error), 2000)
    finish_reason = getattr(response, 'finish_reason', None)
    if isinstance(error, json.JSONDecodeError) and finish_reason in ('length', 'max_tokens'):
        reason = ('Provider finish_reason=' + finish_reason + '; invalid JSON. '
            'The partial proposal was not executed. Return a shorter complete JSON object '
            'with concise complete tool source. Do not append to the partial response. '
            'Parser: ' + reason)
        return dict(status='proposal_invalid', reason=reason, receipt_id=artifact_id,
                    finish_reason=finish_reason)
    if isinstance(error, json.JSONDecodeError) and finish_reason == 'error':
        body = 'empty response' if not response.text.strip() else 'invalid JSON response'
        reason = ('Provider finish_reason=error; ' + body + '. The proposal was not executed. '
            'Return a complete JSON object; do not continue the failed response. Parser: ' + reason)
        return dict(status='proposal_invalid', reason=reason, receipt_id=artifact_id,
                    finish_reason=finish_reason)
    return dict(status='proposal_invalid', reason=reason)


def response_format(background=False, *, visible_tool_ids=None, sequential=False, experimental_protocol=False, independent_research=False):
    strings = ('reason', 'name', 'description', 'code', 'tool_id', 'arguments_json')
    properties = {key: {'type': 'string'} for key in strings}
    properties['reason']['description'] = (
        'State the task/candidate uncertainty, supplied input/source, proposed check, '
        'and next decision its findings could inform; for skip, explain the limitation. '
        'Host trigger signals explain episode timing, not the research objective. '
        'Static checks and negative findings are useful; improvement is not required.')
    properties['code']['description'] = (
        'For create: complete research-tool Python module with one module-level synchronous '
        'def run(payload), callable with one positional JSON input. Candidate entrypoints '
        'or helper functions do not replace run. run returns a sciel-tool-result/1 object with '
        'exactly schema, status, observations, diagnostics, error; candidate task instructions '
        'are study-subject data, not this tool API. Never submit a standalone candidate solution.')
    properties['arguments_json']['description'] = (
        'Tool arguments encoded as a valid finite JSON object string. Use "{}" for no arguments; '
        'an empty string is not valid JSON. Do not encode null or an array.')
    properties['action'] = {'type': 'string', 'enum': ['create', 'use', 'skip']}
    if sequential:
        properties['action']['enum'] += ['finish', 'submit_candidate']
        properties['action']['description'] = (
            'Choose the action to execute now. finish closes the question; a worthwhile feasible '
            'unperformed check calls for create/use, or continue for an eligible saved job. '
            'Static/no-need decisions and limited negative/inconclusive findings can close without a new tool.')
        properties['reason']['description'] += (
            ' For finish, explain the evidence, what remains unknown and why further trials are '
            'not warranted or are blocked now; do not merely propose an unperformed experiment.')
        for key in TEXT_FIELDS:
            properties[key] = {'type': 'string', 'pattern': r'\S'}
        for key in ('input_receipt_id', 'candidate_receipt_id'):
            properties[key] = {'type': 'string'}
        properties['question']['description'] = 'A concrete, testable question about an observed search limitation.'
        properties['decision_if_supported']['description'] = 'Which next experiment or search change follows a supporting measurement?'
        properties['decision_if_refuted']['description'] = 'Which different next experiment or search change follows a negative measurement?'
        properties['next_decision']['description'] = (
            'Advice for a later decision, not automatically scheduled work. If a worthwhile feasible '
            'check is still required, perform it with the current create/use/eligible-continue action.')
        if independent_research:
            properties['next_decision']['description'] = ('Concrete next step for an open question, saved for later research opportunities. '
                'finish closes the question: its advice is not automatically queued. Keep feasible required work open with create/use/eligible continue.')
        properties['input_receipt_id']['description'] = 'Exact successful receipt ID from this cell to pass saved observations/state directly to the next tool; empty means no stage input.'
        properties['candidate_receipt_id']['description'] = 'For submit_candidate: exact successful receipt ID containing observations.candidate_source; no model rewriting.'
        properties['conclusion'] = {'type':'string', 'enum':['', 'supported', 'refuted', 'inconclusive']}
    if experimental_protocol:
        from .experiment_protocol import response_property
        properties['experimental_protocol'] = response_property()
    properties['replay_policy'] = {'type': 'string', 'enum': list(REPLAY_POLICIES)}
    if background:
        properties['action']['enum'] += ['poll', 'continue', 'cancel']
        properties['execution'] = {'type': 'string', 'enum': ['foreground', 'background']}
        properties['job_id'] = {'type': 'string'}
        for key in ('cpu_seconds', 'wall_seconds'):
            properties[key] = {'type': ['number', 'null']}
    for key in ('parents', 'dependencies'):
        purpose = ('existing tools this new version derives from (tool lineage)' if key == 'parents'
                   else 'existing tools imported by this code via m_<tool_id>')
        description = ('Exact visible TOOL-version IDs for ' + purpose + '. '
            'Never program, candidate, lab, cell, source-code or input SHA IDs. '
            'Use [] for a new independent tool; at most 16 IDs, enforced locally.')
        if key == 'dependencies':
            description += ' ' + _DEPENDENCY_GUIDANCE.strip()
        items = {'type': 'string', 'description': 'An exact visible immutable TOOL-version ID.'}
        if visible_tool_ids is not None:
            if visible_tool_ids:
                items['enum'] = list(visible_tool_ids)
            else:
                description += ' No tool versions are visible: this array must be [].'
        properties[key] = {'type': 'array', 'description': description, 'items': items}
    properties['tool_id']['description'] = 'For use: an exact visible TOOL-version ID; unused for create/skip.'
    return {'type': 'json_schema', 'json_schema': {'name': 'sciel_research_tool_action',
        'strict': True, 'schema': {'type': 'object', 'additionalProperties': False,
            'properties': properties, 'required': list(properties)}}}


def _tool_reference_error(field, values, available, problem):
    """Explain identity semantics on the first format or membership rejection."""
    shown = json.dumps(values, ensure_ascii=True, sort_keys=True)
    visible = ('not supplied to this parser' if available is None else
               json.dumps(list(available[:8]), ensure_ascii=True, sort_keys=True)
               + (f' ({len(available)-8} more in the response schema)' if len(available) > 8 else ''))
    return (field + ': ' + problem + '; received ' + _clip(shown, 320)
        + '. Expected exact visible TOOL-version IDs, never program/candidate/lab/cell IDs '
          'or source/input SHA256. Use [] for a new independent tool. Visible TOOL-version IDs: '
        + visible + '.')


def parse_action(text, config, *, visible_tool_ids=None):
    if type(text) is not str or len(text) > config['max_response_chars']:
        raise ValueError('tool response is missing or exceeds the response bound')
    stripped = text.strip()
    if stripped.startswith('```') and stripped.endswith('```'):
        stripped = stripped.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
    action = _finite_json(stripped)
    sequential = config['max_scientific_steps'] > 1
    allowed = ('create', 'use', 'skip', 'poll', 'continue', 'cancel') if config['jobs']['enabled'] else ('create', 'use', 'skip')
    if sequential:
        allowed += ('finish', 'submit_candidate')
    if type(action) is not dict or action.get('action') not in allowed:
        raise ValueError('expected a JSON create, use, or skip action')
    research = proposal_fields(action) if sequential else {}
    if config.get('experimental_protocol'):
        research['experimental_protocol'] = experiment_protocol.parse_protocol(action.get('experimental_protocol'))
    if action['action'] in ('skip', 'finish'):
        if action['action'] == 'finish' and type(action.get('reason', '')) is not str:
            raise ValueError('finish.reason must be model interpretation text')
        return {'action': action['action'], 'reason': _clip(action.get('reason', ''), 2000), **research}
    if action['action'] == 'submit_candidate':
        return {'action':'submit_candidate', 'reason':_clip(action.get('reason', ''), 2000),
                'candidate_receipt_id':require_hash(action.get('candidate_receipt_id'), 'candidate_receipt_id'),
                **research}
    raw = action.get('arguments_json', '{}')
    if type(raw) is not str or len(raw) > 65536:
        raise ValueError('arguments_json must be a bounded JSON object encoded as text')
    try:
        arguments = _finite_json(raw)
    except (ValueError, RecursionError) as error:
        # This is a nested field failure in an already parsed outer response.
        # Preserve that distinction when constructing provider/repair feedback.
        raise ValueError('arguments_json must encode a valid finite JSON object; '
                         'use "{}" for no arguments. Field parser: ' + _clip(str(error), 400)) from None
    if type(arguments) is not dict:
        raise ValueError('arguments_json must encode an object')
    result = {'action': action['action'], 'arguments': arguments,
              'reason': _clip(action.get('reason', ''), 2000), **research}
    if sequential:
        value = action.get('input_receipt_id', '')
        result['input_receipt_id'] = require_hash(value, 'input_receipt_id') if value else None
    if config['jobs']['enabled']:
        result['execution'] = action.get('execution', 'foreground')
        if result['execution'] not in ('foreground', 'background'):
            raise ValueError('execution must be foreground or background')
        result['limits'] = effective_limits(config, {key: action[key] for key in ('cpu_seconds', 'wall_seconds')
                                                   if action.get(key) is not None})
        if action['action'] in ('poll', 'continue', 'cancel'):
            result['job_id'] = require_hash(action.get('job_id'), 'job_id')
            return result
    if action['action'] == 'use':
        result['tool_id'] = require_hash(action.get('tool_id'), 'tool_id')
        return result
    code = action.get('code')
    if type(code) is not str or not code.strip() or len(code) > config['max_source_chars']:
        raise ValueError('create requires nonempty Python code within max_source_chars')
    for name in ('name', 'description'):
        if type(action.get(name)) is not str or not action[name].strip():
            raise ValueError('create requires a nonempty ' + name)
    result.update(code=code, name=_clip(action['name'], 120),
                  description=_clip(action['description'], 2000),
                  replay_policy=validate_policy(action.get('replay_policy', 'exact_observations')))
    for key in ('parents', 'dependencies'):
        refs = action.get(key, [])
        if type(refs) is not list or len(refs) > 16:
            raise ValueError(_tool_reference_error(key, refs, visible_tool_ids,
                'expected an array of at most 16 tool references'))
        bad = []
        for item in refs:
            try:
                require_hash(item, key)
            except ValueError:
                bad.append(item)
                continue
            if visible_tool_ids is not None and item not in visible_tool_ids:
                bad.append(item)
        if bad:
            raise ValueError(_tool_reference_error(key, bad, visible_tool_ids,
                'invalid or unavailable tool reference'))
        result[key] = list(dict.fromkeys(refs))
    return result
