"""Bounded JSON proposals. Only the external coordinator grants capabilities."""
from __future__ import annotations

import json
import copy
import math
import re

from .models import canonical_bytes, content_hash, detached
from ..campaign.budget import request_cost_bound, money


ACTION_SCHEMA = {
    'type': 'object',
    'oneOf': [
        {'properties': {'kind': {'const': 'method'}, 'files': {'type': 'object', 'additionalProperties': {'type': 'string'}},
                        'manifest': {'type': 'object'}, 'experiment': {'type': 'object'}},
         'required': ['kind', 'files', 'manifest'], 'additionalProperties': False},
        {'properties': {'kind': {'const': 'experiment'}, 'experiment': {'type': 'object'}},
         'required': ['kind', 'experiment'], 'additionalProperties': False},
        {'properties': {'kind': {'const': 'solution'}, 'code': {'type': 'string'}, 'experiment': {'type': 'object'}},
         'required': ['kind', 'code', 'experiment'], 'additionalProperties': False},
        {'properties': {'kind': {'const': 'stop'}, 'reason': {'type': 'string', 'minLength': 1}},
         'required': ['kind', 'reason'], 'additionalProperties': False},
    ],
}

PROCEDURE = {
    'schema': 'sciel-research-procedure/1',
    'name': 'evidence-conditioned-tool-development',
    'version': 2,
    'action_schema': ACTION_SCHEMA,
    'instructions': (
        'Develop reusable research methods in response to an observed difficulty. '
        'Return exactly one JSON object, without Markdown fences or surrounding text, '
        'matching ACTION_SCHEMA_JSON below. Never claim an experiment succeeded before '
        'its executor receipt. Use available method versions by exact hash. '
        'A failed draft can be corrected using a new immutable version with the old '
        'version as parent. Read the supplied evidence; do not repeat unsupported '
        'claims. A method and a complete task solution are different artifacts. '
        'The coordinator freezes plans and judges results. You cannot grant yourself '
        'budgets, release methods, alter the evaluator, or access another lab. '
        'The top-level kind is the action: "method", "experiment", "solution", or "stop". '
        'For a method proposal, files and manifest are siblings of kind:"method"; '
        'never wrap the proposal inside a "method" object. "tool" is a possible '
        'manifest.kind, never a top-level action kind. Each files value is complete '
        'source text. When feasible, include a bounded experiment with a new method '
        'to test its stated change. A saved draft alone is not experimental progress. '
        'Experiment fields: question, treatment (ordered method '
        'hashes, or ["$new"] for the proposed method), control (ordered hashes), '
        'inputs (JSON cases), criterion:{metric,direction,minimum_effect}, scope. '
        'Method entrypoints take one JSON value and return one JSON value. '
        'Declare dependency hashes in manifest.dependencies. Inside a method, import '
        'the pinned dependency as m_<hash> and call m_<hash>.invoke(JSONvalue), replacing '
        '<hash> with its exact version hash. Nested calls share the invocation budget '
        'and do not create independent use/replication receipts. '
        'Example dependency usage with a placeholder to replace: '
        'import m_<hash>\nresult = m_<hash>.invoke(value)\n'
        'Do not import another method bundle\'s private helper filenames seen in its '
        'source; bundles have isolated namespaces. Declaring a dependency does not '
        'make its helper modules importable. Any ordinary local helper import must '
        'refer to source included in your own files. Reuse a dependency through its '
        'pinned invoke interface and preserve its declared JSON input/output contract. '
        'Manifest fields: name, kind(tool/experiment/search/adapter), author_lab, '
        'entrypoint(path.py:function), input_schema, output_schema, protocol, '
        'runtime_lock, dependencies, parents, resources, scope, examples. '
        'Every listed manifest field is mandatory; use examples:[] and parents:[] '
        'when no examples or parent versions apply. Copy the full runtime_lock '
        'from the lab revision and declare all resource fields from context.resources. '
        'Use the runtime and resource contract in the supplied context. '
        'Adjacent pipeline output_schema and input_schema must be exactly equal, '
        'including optional fields and array order within schema definitions. Copy '
        'the exact declared contract when chaining, or implement the composition '
        'inside one method with declared dependencies and compatible JSON calls. '
        'Solution code is a standalone candidate with the task entrypoint; it may '
        'use sciel_method(hash, JSONvalue) for explicitly allowed active methods. '
        'A solution experiment must set treatment:["$new"] and have exactly one '
        'input matching context.solution_input_schema. Do not reuse tool-state '
        'inputs or controls if they require a different schema. Use control:[] '
        'when no compatible control exists; an empty control does not establish '
        'improvement over an existing method. Method experiments instead use '
        'inputs accepted by their first method\'s input_schema.'
    ),
}
PROCEDURE_ID = content_hash(PROCEDURE)


class ProposalFormatError(ValueError):
    """Safe, actionable model feedback without copying untrusted response text."""
    def __init__(self, code, message, *, field=None, line=None, column=None):
        super().__init__(message)
        self.diagnostic = dict(code=code, message=message)
        self.diagnostic.update({k: v for k, v in dict(field=field, line=line, column=column).items()
                                if v is not None})


def proposal_diagnostic(error):
    if isinstance(error, ProposalFormatError):
        return dict(error.diagnostic)
    # Never persist repr/str from arbitrary exceptions or interpolate model text.
    return dict(code='invalid_proposal_format',
                message='Return one JSON object matching ACTION_SCHEMA_JSON with a top-level action kind.')


def parse_proposal(text, *, max_bytes=262144):
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError('response byte limit must be a positive integer')
    if not isinstance(text, str):
        raise ProposalFormatError('response_not_text', 'Proposal response must be UTF-8 text containing one JSON object.')
    try:
        byte_count = len(text.encode('utf-8'))
    except UnicodeError:
        raise ProposalFormatError('invalid_text_encoding', 'Proposal response must contain valid UTF-8 text.') from None
    if byte_count > max_bytes:
        raise ProposalFormatError('response_too_large', 'Proposal exceeds the declared response byte limit; shorten source or split the research step.')

    payload = text.strip()
    if payload.startswith('```'):
        fence = re.fullmatch(r'```(?:json)?[ \t]*\r?\n(.*)\r?\n```', payload, re.DOTALL | re.IGNORECASE)
        if fence is None:
            raise ProposalFormatError('invalid_json_envelope',
                'Return one JSON object without surrounding text. Only a single whole-response JSON fence is accepted.')
        # Envelope decoding only: do not repair keys, values, source, or a draft.
        # The original response remains unchanged in its dispatch receipt.
        payload = fence.group(1)

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ProposalFormatError('duplicate_json_field', 'Duplicate JSON proposal fields are not allowed, including inside nested objects.')
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ProposalFormatError(
                               'nonfinite_json_value', 'JSON numbers must be finite; NaN and Infinity are not allowed.')))
        canonical_bytes(value)
    except ProposalFormatError:
        raise
    except json.JSONDecodeError as exc:
        raise ProposalFormatError('invalid_json',
            'Malformed JSON. Return one complete object, escape source strings, and remove surrounding text or additional objects.',
            line=exc.lineno, column=exc.colno) from None
    except RecursionError:
        raise ProposalFormatError('json_nesting_limit', 'JSON nesting is too deep; simplify the proposal structure.') from None
    except (ValueError, TypeError, OverflowError):
        raise ProposalFormatError('invalid_json_value', 'Proposal must contain only finite JSON values within parser limits.') from None
    if not isinstance(value, dict):
        raise ProposalFormatError('proposal_not_object', 'Proposal must be one JSON object, not an array or scalar.')
    fields = {'method': {'kind', 'files', 'manifest', 'experiment'},
              'experiment': {'kind', 'experiment'},
              'solution': {'kind', 'code', 'experiment'},
              'stop': {'kind', 'reason'}}
    kind = value.get('kind')
    if kind is None:
        raise ProposalFormatError('missing_action_kind',
            'Set the top-level kind to "method", "experiment", "solution", or "stop". '
            'For a method, files and manifest must be top-level siblings; do not wrap them inside a "method" object. '
            'The tool category belongs to manifest.kind.', field='kind')
    if not isinstance(kind, str) or kind not in fields:
        raise ProposalFormatError('unsupported_action_kind',
            'Top-level kind must be "method", "experiment", "solution", or "stop". '
            'Use "tool" only within manifest.kind.', field='kind')
    if set(value) - fields[kind]:
        raise ProposalFormatError('unsupported_action_field',
            'Allowed top-level fields for this action: ' + ', '.join(sorted(fields[kind])) + '.', field='root')
    required = fields[kind] - ({'experiment'} if kind == 'method' else set())
    if not required <= set(value):
        raise ProposalFormatError('missing_action_field',
            'Missing required top-level fields: ' + ', '.join(sorted(required - set(value))) + '.', field='root')
    if kind == 'stop' and (not isinstance(value['reason'], str) or not value['reason'].strip()):
        raise ProposalFormatError('invalid_stop_reason', 'Stop needs a nonempty textual reason.', field='reason')
    if kind == 'solution' and not isinstance(value['code'], str):
        raise ProposalFormatError('invalid_solution_code', 'Solution code must be source text in a JSON string.', field='code')
    if kind == 'method' and (not isinstance(value['manifest'], dict) or not isinstance(value['files'], dict)):
        raise ProposalFormatError('invalid_method_shape', 'Method files and manifest must both be JSON objects.', field='files/manifest')
    if 'experiment' in value and not isinstance(value['experiment'], dict):
        raise ProposalFormatError('invalid_experiment_shape', 'Experiment must be a JSON object with the documented bounded protocol fields.', field='experiment')
    return value


def prompt_for(context):
    return (PROCEDURE['instructions'] + '\nACTION_SCHEMA_JSON\n' + canonical_bytes(ACTION_SCHEMA).decode()
            + '\nCONTEXT_JSON\n' + canonical_bytes(context).decode())


class BudgetedProposer:
    """Use the existing priced, no-blind-retry transport; the outer ledger also
    records its charge as one research expense (the two ledgers are not summed).
    """
    def __init__(self, llm, model, *, temperature=.7, timeout=120):
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 3600
                or type(temperature) not in (int, float) or not math.isfinite(temperature)
                or not 0 <= temperature <= 2):
            raise ValueError('invalid proposer timeout or temperature')
        if not isinstance(model, str) or model not in llm.models or model not in llm.prices:
            raise ValueError('proposer requires a priced model in the frozen ensemble')
        if type(llm.max_tokens) is not int or not 0 < llm.max_tokens <= 1000000:
            raise ValueError('proposer requires a bounded completion token limit')
        # Clone configuration, retaining the existing durable money ledger. The
        # SDK client itself is cloned with concrete timeout/retry options: a
        # metadata-only timeout would provide no transport constraint at all.
        self.llm = copy.copy(llm)
        self.llm.models = list(llm.models)
        self.llm.prices = {name: {key: str(money(rate)) for key, rate in rates.items()}
                           for name, rates in llm.prices.items()}
        client = getattr(llm, 'client', None)
        if client is None or not callable(getattr(client, 'with_options', None)):
            raise ValueError('priced proposer needs an SDK client with enforceable timeout options')
        self.llm.client = client.with_options(timeout=float(timeout), max_retries=0)
        self.model, self.temperature, self.timeout = model, temperature, timeout
        self.identity = dict(kind='budgeted_llm', model=model, temperature=temperature,
            max_tokens=self.llm.max_tokens, pricing=detached(self.llm.prices[model]), timeout=timeout,
            timeout_policy='SDK per-IO timeout; observed total wall time charged separately',
            transport_retries=0, response_format='prompt_only_json_local_validation')
        # Validate prices before any dispatch, including required prompt/output.
        request_cost_bound('', self.llm.max_tokens, self.llm.prices[model])

    def bounds(self, prompt):
        return dict(dollars=str(request_cost_bound(prompt, self.llm.max_tokens,
            self.llm.prices[self.model])),
            wall_seconds=self.timeout + 10, cpu_seconds=10, gpu_seconds=0,
            storage_bytes=2_000_000, steps=1)

    def generate(self, prompt):
        return self.llm.generate(prompt, self.model, self.temperature,
                                 response_format=None)


def response_record(response):
    """Lossless response text and known usage; never infer missing charges."""
    if not isinstance(response.text, str) or not isinstance(response.model, str):
        raise ValueError('malformed model response')
    cost = response.cost
    known = type(cost) in (int, float) and math.isfinite(cost) and cost >= 0
    tokens = {}
    for key in ('prompt_tokens', 'completion_tokens'):
        value = getattr(response, key, None)
        tokens[key] = value if type(value) is int and 0 <= value <= 10**9 else None
    from ..engine.llm.client import response_resolved_model
    resolved = response_resolved_model(response)
    return dict(status='completed' if known else 'unknown', text=response.text, model=response.model,
                cost=float(cost) if known else None, **tokens,
                **({'resolved_model': resolved} if resolved is not None else {}),
                finish_reason=getattr(response, 'finish_reason', None),
                refusal=getattr(response, 'refusal', None))


class ScriptedProposer:
    """Explicit technical fixture, not a model or scientific experiment."""
    def __init__(self, responses):
        def encode(items):
            if not isinstance(items, list):
                raise ValueError('scripted responses require a list per lab')
            result = [json.dumps(value) if isinstance(value, dict) else value for value in items]
            if any(not isinstance(value, str) for value in result):
                raise ValueError('scripted response must be text or an object')
            return result
        self.responses = ({lab: encode(items) for lab, items in responses.items()}
                          if isinstance(responses, dict) else encode(responses))
        self.calls = []
        self.identity = dict(kind='scripted_fixture', responses_sha256=content_hash(self.responses))

    def bounds(self, prompt):
        return dict(dollars=0, wall_seconds=10, cpu_seconds=10, gpu_seconds=0,
                    storage_bytes=2_000_000, steps=1)

    def generate(self, prompt):
        from ..engine.llm.client import LLMResponse
        context = {}
        if isinstance(prompt, str) and '\nCONTEXT_JSON\n' in prompt:
            context = json.loads(prompt.rsplit('\nCONTEXT_JSON\n', 1)[1])
        if not isinstance(context, dict):
            raise ValueError('invalid scripted context')
        responses = self.responses
        if isinstance(responses, dict):
            lab = context.get('lab_id')
            if not isinstance(lab, str) or lab not in responses:
                raise ValueError('no scripted responses for this lab')
            responses = responses[lab]
        index = context.get('proposal_index', len(self.calls))
        if type(index) is not int or not 0 <= index < len(responses):
            raise ValueError('fixture exhausted')
        value = responses[index]
        self.calls.append(prompt)
        return LLMResponse(text=value, model='scripted_fixture', cost=0.,
                           prompt_tokens=0, completion_tokens=0)
