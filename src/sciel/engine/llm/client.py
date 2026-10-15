"""One observable provider dispatch per call, with explicit usage and failures."""

from __future__ import annotations

from dataclasses import dataclass
import json
import hashlib
import math
import re

from .errors import (LLMError as LLMError, llm_error_metadata as llm_error_metadata,
                     MAX_RETRY_AFTER_SECONDS as MAX_RETRY_AFTER_SECONDS,
                     _ERROR_CAUSES as _ERROR_CAUSES,
                     _retry_after as _retry_after)


def validate_model_capabilities(value, models):
    """Explicit per-model transport exceptions; unspecified models stay unchanged."""
    if type(value) is not dict:
        raise ValueError('llm.model_capabilities must be an object')
    if value and (type(models) not in (list, tuple) or not models
                  or any(type(model) is not str or not model.strip() for model in models)):
        raise ValueError('llm.model_capabilities requires configured model IDs')
    result = {}
    for model, entry in value.items():
        if type(model) is not str or model not in models:
            raise ValueError('llm.model_capabilities model must be in the configured ensemble')
        if (type(entry) is not dict or set(entry) != {'temperature'}
                or type(entry['temperature']) is not bool):
            raise ValueError('llm.model_capabilities entries require only a boolean temperature')
        result[model] = dict(entry)
    return result


def temperature_parameters(provider, model, requested):
    """Exact SDK temperature kwargs, shared with durable request projections."""
    capabilities = getattr(provider, 'model_capabilities', {})
    if capabilities.get(model, {}).get('temperature', True) is False:
        return {}
    return {'temperature': provider.temperature if requested is None else requested}


def transport_parameters(provider, model, requested):
    """Additive trace/WAL evidence; absence preserves the legacy request shape."""
    capabilities = getattr(provider, 'model_capabilities', {})
    if not capabilities:
        return None
    effective = temperature_parameters(provider, model, requested)
    return {'schema': 'sciel-llm-transport-parameters/1',
            'temperature': {'enabled': capabilities.get(model, {}).get('temperature', True),
                            'sent': 'temperature' in effective,
                            'value': effective.get('temperature')}}


@dataclass
class LLMResponse:
    text: str
    model: str
    cost: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None = None
    refusal: str | None = None
    response_error: dict | None = None
    # Additive diagnostics only; None means provenance was not recorded.
    cost_provenance: dict | None = None
    # Provider-reported identity only. `model` remains the requested selection
    # key; this optional observation never changes pricing or bandit semantics.
    resolved_model: str | None = None
    token_stats: dict | None = None
    reasoning: str | None = None


def _reasoning(choice) -> str | None:
    """Reasoning text returned beside the answer (vLLM/DeepSeek reasoning_content, OpenRouter reasoning)."""
    message = getattr(choice, 'message', None)
    extra = getattr(message, 'model_extra', None) or {}
    for name in ('reasoning_content', 'reasoning'):
        value = getattr(message, name, None) or extra.get(name)
        if type(value) is str and value:
            return value
    return None


def _token_stats(choice) -> dict | None:
    content = getattr(getattr(choice, 'logprobs', None), 'content', None) or []
    logprobs, entropies = [], []
    for token in content:
        logprobs.append(float(token.logprob))
        top = [math.exp(float(alt.logprob)) for alt in (token.top_logprobs or [])]
        rest = max(1e-12, 1.0 - sum(top))
        entropies.append(-sum(p * math.log(p) for p in top if p > 0) - rest * math.log(rest))
    if not logprobs:
        return None
    return {'tokens': len(logprobs), 'mean_logprob': sum(logprobs) / len(logprobs),
            'min_logprob': min(logprobs), 'mean_entropy': sum(entropies) / len(entropies)}


def validate_resolved_model(value):
    """A bounded literal model identifier, never an SDK object or inferred alias."""
    if (type(value) is not str or not value or len(value) > 256
            or not value.isprintable() or any(c.isspace() for c in value)):
        return None
    return value if len(value.encode('utf-8')) <= 256 else None


def response_resolved_model(response):
    """Optional malformed metadata must not hide a paid returned response."""
    try:
        return validate_resolved_model(getattr(response, 'resolved_model', None))
    except Exception:
        return None


def response_model_identity(response):
    """Typed evidence for comparisons; missing resolved identity stays unknown."""
    resolved = response_resolved_model(response)
    try:
        requested = validate_resolved_model(response.model)
    except Exception:
        requested = None
    if requested is None or resolved is None:
        return None
    return {'schema': 'sciel-llm-model-identity/1',
            'requested_model': requested, 'resolved_model': resolved}


def validate_response_format(value):
    """Detach a strict JSON-schema request; never parse or repair model output.

    This validates the transport envelope and finite literal JSON, not the
    provider's supported JSON Schema vocabulary or the truth of an answer.
    """
    if value is None:
        return None

    def literal(item, depth=0):
        if depth > 64:
            raise ValueError('response_format is too deeply nested')
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                literal(child, depth + 1)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                literal(child, depth + 1)
            return
        raise ValueError('response_format requires finite literal JSON')

    literal(value)
    if type(value) is not dict or set(value) != {'type', 'json_schema'} or value['type'] != 'json_schema':
        raise ValueError('response_format must use json_schema')
    spec = value['json_schema']
    if (type(spec) is not dict or not {'name', 'strict', 'schema'} <= set(spec)
            or set(spec) - {'name', 'strict', 'schema', 'description'}
            or type(spec['name']) is not str or re.fullmatch(r'[A-Za-z0-9_-]{1,64}', spec['name']) is None
            or spec['strict'] is not True or type(spec['schema']) is not dict or not spec['schema']
            or ('description' in spec and type(spec['description']) is not str)):
        raise ValueError('response_format requires a named strict JSON schema')
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError('response_format is not bounded UTF-8 JSON') from exc
    if len(encoded) > 131072:
        raise ValueError('response_format exceeds 131072 UTF-8 bytes')
    return json.loads(encoded)


def response_format_bytes(value):
    """Canonical bytes counted in a structured request's cost reservation."""
    value = validate_response_format(value)
    return (b'' if value is None else json.dumps(value, ensure_ascii=False,
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8'))


def _response_metadata(choice):
    # Optional provider fields are diagnostics. Never serialize arbitrary SDK
    # objects or turn malformed refusal metadata into apparent non-refusal.
    try:
        finish = getattr(choice, 'finish_reason', None)
    except Exception:
        finish = None
    try:
        refusal = getattr(choice.message, 'refusal', None)
    except Exception:
        refusal = '[unavailable refusal metadata]'
    return {'finish_reason': finish if type(finish) is str else None,
            'refusal': refusal if refusal is None or type(refusal) is str
                       else '[unsupported refusal metadata]'}


PRICE_PER_1K = {
    # Uncached token estimate, verified 2026-09-19 against
    # https://developers.openai.com/api/docs/models/gpt-4.1-mini
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gemini-2.5-flash-lite": (0.0001, 0.0004),
    # Fallback only. Verified against /api/v1/models on 2026-09-14;
    # provider usage.cost is authoritative whenever supplied.
    "gemini-2.5-flash": (0.0003, 0.0025),
    # Anthropic first-party list prices, verified 2026-10-15 against the claude-api model table.
    "claude-fable-5-1": (0.010, 0.050),
    "claude-opus-5-5": (0.004, 0.020),
    "claude-sonnet-5-5": (0.002, 0.010),
    "claude-haiku-4-5": (0.001, 0.005),
}
_DEFAULT_PRICE = (0.0002, 0.0008)


def known_cost(response) -> float:
    """Require a known charge before accepting a response or accumulating cost."""
    value = getattr(response, "cost", None)
    if type(value) not in (int, float):
        raise ValueError("LLM response cost is unknown or invalid")
    try:
        cost = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("LLM response cost is unknown or invalid") from exc
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("LLM response cost is unknown or invalid")
    return cost


def _tokens(value):
    # SDK token usage is integral. Missing, partial or malformed usage is not
    # evidence of a free call and must not enter the pricing fallback as zero.
    return value if type(value) is int and 0 <= value <= 10**9 else None


COST_PROVENANCE_SCHEMA = 'sciel-llm-cost-provenance/1'
SUBSCRIPTION_ENDPOINT_SHA256 = hashlib.sha256(
    b'https://chatgpt.com/backend-api/codex/responses').hexdigest()


def response_cost_provenance(response):
    """Detach only bounded pricing facts consistent with the returned amount.

    Missing/invalid optional metadata never changes accounting or invents a
    provider charge. In particular old receipts remain unannotated. The rates
    stored for an estimate are historical inputs, not today's price table.
    """
    try:
        value = getattr(response, 'cost_provenance', None)
        if type(value) is not dict or value.get('schema') != COST_PROVENANCE_SCHEMA:
            return None
        try:
            cost = known_cost(response)
        except ValueError:
            cost = None
        source = value.get('source')
        base = {'schema': COST_PROVENANCE_SCHEMA, 'source': source}
        if source == 'subscription_unpriced':
            # A completed, usable response from a subscription transport is not
            # an unknown dispatch and is never evidence of a zero dollar charge.
            endpoint = value.get('endpoint_sha256')
            if (set(value) != {*base, 'provider', 'response_status', 'endpoint_sha256'}
                    or getattr(response, 'cost', None) is not None
                    or value.get('provider') != 'codex_responses'
                    or value.get('response_status') != 'completed'
                    or getattr(response, 'finish_reason', None) != 'stop'
                    or getattr(response, 'response_error', None) is not None
                    or type(endpoint) is not str or endpoint != SUBSCRIPTION_ENDPOINT_SHA256):
                return None
            return {**base, 'provider': 'codex_responses', 'response_status': 'completed',
                    'endpoint_sha256': endpoint}
        if source == 'unavailable':
            return base if set(value) == set(base) and cost is None else None
        if source == 'provider_reported':
            amount = value.get('reported_cost')
            if (set(value) != {*base, 'reported_cost'} or type(amount) not in (int, float)
                    or not math.isfinite(amount) or amount < 0 or cost is None or amount != cost):
                return None
            return {**base, 'reported_cost': amount}
        if source != 'token_estimate' or set(value) != {
                *base, 'pricing_key', 'prompt_usd_per_1k', 'completion_usd_per_1k'}:
            return None
        key = value['pricing_key']
        if type(key) is not str or key not in {*PRICE_PER_1K, 'default'}:
            return None
        pin, pout = value['prompt_usd_per_1k'], value['completion_usd_per_1k']
        if any(type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0
               for rate in (pin, pout)):
            return None
        pt, ct = (_tokens(getattr(response, field, None))
                  for field in ('prompt_tokens', 'completion_tokens'))
        if pt is None or ct is None or cost is None or (pt / 1000.0) * pin + (ct / 1000.0) * pout != cost:
            return None
        return {**base, 'pricing_key': key, 'prompt_usd_per_1k': pin,
                'completion_usd_per_1k': pout}
    except Exception:
        return None


def response_accounting(response):
    """Classify a returned response; USD subtotal zero does not mean free.

    This is intentionally separate from known_cost: historical unknown prices,
    malformed numbers and incomplete physical outcomes remain fail-closed.
    """
    try:
        return {'known_usd': known_cost(response), 'unpriced_calls': 0}
    except ValueError:
        provenance = response_cost_provenance(response)
        if provenance is None or provenance.get('source') != 'subscription_unpriced':
            raise
        return {'known_usd': 0.0, 'unpriced_calls': 1}


def accounted_cost(response):
    """Known-USD contribution only; consumers retain explicit unpriced counts."""
    return response_accounting(response)['known_usd']


class EnsembleLLM:
    def __init__(self, models: list, weights: list, api_base: str, api_key: str,
                 temperature: float = 0.7, max_tokens: int = 16000,
                 timeout: float = 120.0, max_retries: int = 1, model_capabilities=None,
                 reasoning_effort: str | None = None, top_logprobs: int | None = None):
        if type(max_retries) is not int or max_retries != 1:
            raise ValueError('max_retries must be 1: unknown provider outcomes cannot be replayed implicitly')
        self.models = models
        self.model_capabilities = validate_model_capabilities(
            {} if model_capabilities is None else model_capabilities, models)
        self.weights = weights
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        if reasoning_effort not in (None, 'low', 'medium', 'high'):
            raise ValueError('llm.reasoning_effort must be low, medium, high or null')
        self.reasoning_effort = reasoning_effort
        if top_logprobs is not None and (type(top_logprobs) is not int or not 1 <= top_logprobs <= 20):
            raise ValueError('llm.top_logprobs must be an integer from 1 to 20 or null')
        self.top_logprobs = top_logprobs
        from openai import OpenAI
        # Recovery belongs to the durable caller; SDK retries hide paid attempts.
        self._client = OpenAI(base_url=api_base, api_key=api_key, timeout=timeout,
                              max_retries=0)

    def choose_model(self, rng) -> str:
        return rng.choices(self.models, weights=self.weights, k=1)[0]

    def generate(self, prompt: str, model: str, temperature: float | None = None,
                 *, response_format=None, max_tokens=None) -> LLMResponse:
        sampling = temperature_parameters(self, model, temperature)
        output_tokens = self.max_tokens if max_tokens is None else max_tokens
        if type(output_tokens) is not int or output_tokens <= 0:
            raise ValueError('max_tokens must be a positive integer or null')
        format_value = validate_response_format(response_format)
        options = {} if format_value is None else {'response_format': format_value}
        if getattr(self, 'reasoning_effort', None) is not None:
            options['reasoning_effort'] = self.reasoning_effort
        if getattr(self, 'top_logprobs', None) is not None:
            options.update(logprobs=True, top_logprobs=self.top_logprobs)
        endpoint = getattr(self, 'api_base', None)
        if (format_value is not None and type(endpoint) is str
                and endpoint.rstrip('/') == 'https://openrouter.ai/api/v1'):
            # These routing guarantees are OpenRouter-specific. Other compatible
            # endpoints receive the native schema parameter without vendor fields.
            options['extra_body'] = {'provider': {
                'require_parameters': True, 'allow_fallbacks': False}}

        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=output_tokens,
                **sampling,
                **options,
            )

        except Exception as e:
            raise LLMError(cause=e) from e
        # A returned malformed body can still carry authoritative paid usage.
        # Read it first; malformed content must not erase a known charge.
        def field(obj, name):
            try:
                return getattr(obj, name, None)
            except Exception:
                return None
        usage = field(resp, 'usage')
        resolved_model = validate_resolved_model(field(resp, 'model'))
        pt = _tokens(field(usage, 'prompt_tokens'))
        ct = _tokens(field(usage, 'completion_tokens'))
        try:
            cost = known_cost(usage)
            provenance = dict(schema=COST_PROVENANCE_SCHEMA,
                              source='provider_reported', reported_cost=cost)
        except Exception:
            cost = _cost(model, pt, ct) if pt is not None and ct is not None else None
            provenance = dict(schema=COST_PROVENANCE_SCHEMA, source='unavailable')
            if cost is not None:
                key, pin, pout = _price(model)
                provenance = dict(schema=COST_PROVENANCE_SCHEMA, source='token_estimate',
                    pricing_key=key, prompt_usd_per_1k=pin, completion_usd_per_1k=pout)
        error = None
        try:
            choices = resp.choices or []
            if not choices:
                raise LLMError('LLM response carried no choices',
                               stage='response', reason='empty_choices')
            text = choices[0].message.content
            if text is None:
                text = ''
            if type(text) is not str:
                raise TypeError('response content is not text')
        except LLMError as e:
            error = e
        except Exception as e:
            error = LLMError('Malformed LLM response', stage='response',
                             reason='malformed_response', cause=e)
        if error is not None:
            if cost is None:
                raise error from None
            return LLMResponse(text='', model=model, cost=cost,
                               prompt_tokens=pt, completion_tokens=ct,
                               response_error=llm_error_metadata(error), cost_provenance=provenance,
                               resolved_model=resolved_model)
        return LLMResponse(text=text, model=model, cost=cost,
                           prompt_tokens=pt, completion_tokens=ct,
                           cost_provenance=provenance,
                           resolved_model=resolved_model,
                           token_stats=_token_stats(choices[0]) if getattr(self, 'top_logprobs', None) else None,
                           reasoning=_reasoning(choices[0]),
                           **_response_metadata(choices[0]))


def _cost(model: str, pt: int, ct: int) -> float:
    _, pin, pout = _price(model)
    return (pt / 1000.0) * pin + (ct / 1000.0) * pout


def _price(model):
    key = next((k for k in PRICE_PER_1K if k in model), None)
    pin, pout = PRICE_PER_1K.get(key, _DEFAULT_PRICE)
    return key or 'default', pin, pout
