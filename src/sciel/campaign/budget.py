"""A shared, durable cost ledger for bounded experimental API calls."""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
import fcntl
import json
import math
import os
import re
from pathlib import Path
import time
import uuid


class CampaignStopped(RuntimeError):
    """Do not dispatch another chargeable request in this campaign."""


class RequestNotDispatched(CampaignStopped):
    """The current request was rejected before reservation/provider dispatch.

    This marker describes only this request. Earlier receipts may still be
    charged or unresolved. An ordinary CampaignStopped makes no such promise.
    """


def request_cost_bound(prompt, max_tokens, rates, *, response_format=None):
    """Same conservative bound as generation, including explicit schema bytes."""
    from ..engine.llm.client import response_format_bytes
    if type(prompt) is not str or type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError('request requires a text prompt and positive token cap')
    schema_bytes = response_format_bytes(response_format)
    return ((len(prompt.encode('utf-8')) + len(schema_bytes) + 1024) * money(rates['prompt'])
            + max_tokens * money(rates['completion'])
            + money(rates.get('request', 0))) * Decimal('1.10')


def _error_attr(value, name):
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _error_json(value):
    # Do not stringify SDK objects or read a streaming response to recover errors.
    if type(value) is dict:
        return value
    if type(value) not in (str, bytes) or len(value) > 16384:
        return None
    try:
        value = json.loads(value)
    except (ValueError, UnicodeError, RecursionError):
        return None
    return value if type(value) is dict else None


_PROVIDER_MESSAGE_HINTS = (
    ('schema_invalid', r'(?:invalid (?:json )?schema\b|(?:json )?schema is invalid\b)'),
    ('schema_complexity_limit', r'(?:(?:json )?schema (?:is )?too complex\b|'
        r'(?:json )?schema (?:has|contains) too many (?:properties|parameters|enum values)\b)'),
    ('parameter_unsupported', r'(?:unsupported (?:request )?parameter\b|'
        r'(?:unknown|unrecognized) (?:request )?(?:parameter|argument)\b)'),
    ('context_limit', r'(?:context (?:length|window) (?:limit )?exceeded\b|'
        r'(?:this model\'s )?maximum context length (?:is|exceeded)\b|'
        r'(?:input|prompt) (?:is )?too long for (?:the )?context (?:window|length)\b)'),
)


def _provider_message_hints(value):
    """Fixed lexical categories only, never an error quote or verified cause.

    A short schema literal such as ``object`` can correctly force whole-field
    redaction even in a useful provider message. Recognize a few narrow leading
    assertions before that redaction, without emitting matched text, numbers,
    parameter names or any other data from the message. Quoted/request-prefixed
    phrases do not match; an unmarked echo is still possible, hence unverified.
    Oversized and non-string fields receive no hint and are never stringified.
    """
    if type(value) is not str or not value or len(value) > 8192:
        return None
    plain = ' '.join(value.split())
    categories = [name for name, pattern in _PROVIDER_MESSAGE_HINTS
                  if re.match(pattern, plain, flags=re.IGNORECASE | re.ASCII)]
    if not categories:
        return None
    return dict(schema='sciel-provider-message-hints/1', categories=categories,
                source='bounded_provider_message_pattern', cause_verified=False)


def provider_error_metadata(exc, *, request=None):
    """Allowlisted, bounded diagnostics, never raw exceptions/headers/requests.

    Legacy callers without the actual request context retain exactly the old
    metadata shape. Details are provider assertions, not an inferred cause.
    Detected credential literals/patterns and normalized prompt excerpts redact
    the entire field. Prompt matching covers whole short inputs or >=16-character
    excerpts; it does not prove removal of arbitrary encodings or paraphrases.
    Fixed message hints may survive redaction, but never establish a root cause,
    whether a request was processed or charged, or permission to retry.
    """
    cause, status = _error_attr(exc, "__cause__"), _error_attr(exc, "status_code")
    result = {"error_type": type(exc).__name__,
              "error_status_code": status if type(status) is int and 100 <= status <= 599 else None,
              "error_cause_type": type(cause).__name__ if cause is not None else None}
    if type(request) is not dict:
        return result
    response = _error_attr(exc, 'response')
    body = _error_json(_error_attr(exc, 'body'))
    if body is None:
        body = _error_json(_error_attr(response, 'content'))
    if body is None:
        return result
    # Header values are used solely as redaction needles, never serialized.
    remote_request = _error_attr(exc, 'request')
    if remote_request is None:
        remote_request = _error_attr(response, 'request')
    headers = _error_attr(remote_request, 'headers')
    secrets = []
    for key in ('authorization', 'proxy-authorization', 'x-api-key', 'api-key', 'cookie'):
        try:
            value = headers.get(key) if headers is not None else None
        except Exception:
            value = None
        if type(value) is str and value:
            secrets.append(value)
            if value.lower().startswith(('bearer ', 'basic ')):
                secrets.append(value.split(' ', 1)[1])
    messages = request.get('messages')
    contents = [m.get('content') for m in messages if type(m) is dict] if type(messages) is list else []
    texts, pending, nodes = [], contents + [request.get('response_format')], 0
    while pending and nodes < 4096:
        value = pending.pop(); nodes += 1
        if type(value) is str and value:
            texts.append(value)
        elif type(value) is dict:
            pending.extend(list(value.values())[:4096])
        elif type(value) is list:
            pending.extend(value[:4096])
    # Fail closed if request coverage could not be completed within the bound.
    complete_context = not pending
    normalized = [' '.join(text.split()) for text in texts]
    def clean(value, limit):
        if type(value) is not str or not value:
            return None
        if len(value) > 8192:
            return '[omitted: oversized diagnostic]'
        decoded = re.sub(r'\\u([0-9a-fA-F]{4})', lambda m: chr(int(m[1], 16)), value)
        if (not complete_context or any(secret in decoded for secret in secrets)
                or re.search(r'(?i)(?:bearer|basic)\s+\S+|(?:api[_-]?key|authorization|password|secret|access[_-]?token)\s*[=:]|sk-[A-Za-z0-9_-]{8,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.', decoded)):
            return '[redacted]'
        plain = ' '.join(decoded.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ').split())
        # Whole short inputs and any 16-character request excerpt are sensitive.
        if any((text in plain if len(text) < 16 else
                any(plain[i:i+16] in text for i in range(max(0, len(plain)-15))))
               for text in normalized):
            return '[redacted]'
        value = ''.join(char if char.isprintable() else ' ' for char in value)
        return value if len(value) <= limit else value[:limit-3] + '...'
    def fields(node):
        if type(node) is not dict:
            return {}
        values = {}
        hints = _provider_message_hints(node.get('message'))
        if hints is not None:
            values['message_hints'] = hints
        for key, limit in (('code', 128), ('param', 128), ('message', 512), ('request_id', 128)):
            value = node.get(key)
            if key == 'code' and type(value) is int and abs(value) < 10**10:
                values[key] = value
            else:
                value = clean(value, limit)
                if value is not None:
                    values[key] = value
        return values
    error = body.get('error') if type(body.get('error')) is dict else body
    details = fields(error)
    request_id = clean(_error_attr(exc, 'request_id'), 128)
    if request_id is not None:
        details['request_id'] = request_id
    metadata = error.get('metadata')
    raw = _error_json(metadata.get('raw')) if type(metadata) is dict else None
    if raw is not None:
        nested = fields(raw.get('error') if type(raw.get('error')) is dict else raw)
        if nested:
            details['provider'] = nested
    if details:
        result['error_details'] = dict(schema='sciel-provider-error-details/1', **details)
    return result


def money(value):
    if isinstance(value, bool):
        raise ValueError("money must be a non-negative finite number")
    try:
        result = Decimal(str(value))
    except ArithmeticError as exc:
        raise ValueError("money must be a non-negative finite number") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("money must be a non-negative finite number")
    return result


def validate_llm_config(cfg, pricing):
    """Reject an unusable ensemble or ambiguous price snapshot before any I/O."""
    if not isinstance(cfg, dict) or not isinstance(pricing, dict):
        raise ValueError("LLM config and pricing must be objects")
    models, weights = cfg.get("models"), cfg.get("weights")
    if (not isinstance(models, list) or not models
            or any(not isinstance(m, str) or not m.strip() for m in models)
            or len(set(models)) != len(models)):
        raise ValueError("models must be a nonempty list of unique model IDs")
    from ..engine.llm.client import validate_model_capabilities
    validate_model_capabilities(cfg.get('model_capabilities', {}), models)
    if (not isinstance(weights, list) or len(weights) != len(models)
            or any(isinstance(w, bool) or not isinstance(w, (int, float))
                   or not math.isfinite(w) or w < 0 for w in weights)
            or not math.isfinite(sum(weights)) or sum(weights) <= 0):
        raise ValueError("weights must match models and have a positive finite sum")
    if type(cfg.get("max_tokens")) is not int or cfg["max_tokens"] <= 0:
        raise ValueError("max_tokens must be a positive integer")
    for key, default, upper in (("temperature", .7, 2), ("timeout", 120, None)):
        value = cfg.get(key, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (key == "timeout" and value == 0)
                or (upper is not None and value > upper)):
            raise ValueError(f"invalid LLM {key}")
    # This wrapper relies on OpenRouter's provider price caps. A generic compatible
    # endpoint may silently ignore those fields, invalidating the reservation.
    endpoint = cfg.get("api_base")
    if not isinstance(endpoint, str) or endpoint.rstrip("/") != "https://openrouter.ai/api/v1":
        raise ValueError("budgeted calls require the OpenRouter API endpoint")
    rows = pricing.get("models")
    if not isinstance(rows, list) or not rows:
        raise ValueError("pricing must contain a nonempty models list")
    prices = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not row["id"].strip() or row["id"] in prices):
            raise ValueError("pricing model IDs must be nonempty and unique")
        if row["id"] in models and "reasoning" in row:
            reasoning = row["reasoning"]
            if (type(reasoning) is not dict
                    or ("mandatory" in reasoning and type(reasoning["mandatory"]) is not bool)
                    or (reasoning.get("supported_efforts") is not None and
                        (type(reasoning["supported_efforts"]) is not list
                         or not reasoning["supported_efforts"]
                         or any(type(e) is not str or not e for e in reasoning["supported_efforts"])))):
                raise ValueError("malformed selected-model reasoning metadata")
            if (reasoning.get("mandatory") is True
                    or (reasoning.get("supported_efforts") is not None and "none" not in reasoning["supported_efforts"])):
                raise ValueError("selected model is incompatible with fixed reasoning effort none")
        rates = row.get("pricing")
        if not isinstance(rates, dict) or not {"prompt", "completion"} <= rates.keys():
            raise ValueError(f"missing text token prices for {row['id']}")
        prices[row["id"]] = {key: money(rates.get(key, 0))
                             for key in ("prompt", "completion", "request")}
    if any(model not in prices for model in models):
        raise ValueError("pinned model prices are required for every ensemble model")
    return prices


class CostLedger:
    """Reserve a conservative request bound before dispatch, including on restart.

    Unknown outcomes keep their full reservation: a network failure is not proof
    of a free request. A separate lock file makes atomic JSON replacement safe.
    """
    def __init__(self, path, limit_usd):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit = money(limit_usd)
        if self.limit <= 0:
            raise ValueError("campaign limit must be positive")
        with self._locked() as state:
            if not state:
                state.update(schema="sciel-cost-ledger/1", limit_usd=str(self.limit),
                             calls={}, halted=None)
            elif money(state["limit_usd"]) != self.limit:
                raise ValueError("existing campaign limit cannot change on resume")

    @contextmanager
    def _locked(self):
        with open(str(self.path) + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {}
            try:
                yield state
                temp = self.path.with_suffix(self.path.suffix + ".tmp")
                with open(temp, "w") as out:
                    json.dump(state, out, indent=2)
                    out.write("\n")
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temp, self.path)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def totals(state):
        known = sum((money(c["cost_usd"]) for c in state["calls"].values()
                     if c.get("cost_usd") is not None), Decimal(0))
        held = sum((money(c["reservation_usd"]) for c in state["calls"].values()
                    if c.get("cost_usd") is None), Decimal(0))
        return known, held

    def reserve(self, bound_usd, *, run_id, model):
        bound = money(bound_usd)
        with self._locked() as state:
            known, held = self.totals(state)
            if state.get("halted"):
                raise RequestNotDispatched(state["halted"])
            if known + held + bound > self.limit:
                raise RequestNotDispatched("campaign cost limit leaves no room for this request")
            ident = uuid.uuid4().hex
            state["calls"][ident] = dict(run_id=run_id, model=model,
                reservation_usd=str(bound), cost_usd=None, status="reserved",
                created_at=time.time())
        return ident

    def settle(self, ident, actual_cost=None, *, halt_reason=None, **metadata):
        overrun = False
        protected = {"run_id", "model", "reservation_usd", "cost_usd", "status",
                     "created_at", "completed_at"}
        if protected.intersection(metadata):
            raise ValueError("settlement metadata cannot replace ledger accounting fields")
        with self._locked() as state:
            call = state["calls"][ident]
            if call["status"] != "reserved":
                raise ValueError("request already settled")
            actual = None if actual_cost is None else money(actual_cost)
            call.update(metadata)
            call.update(status="unknown" if actual is None else "settled",
                        cost_usd=None if actual is None else str(actual),
                        completed_at=time.time())
            if halt_reason and not state.get("halted"):
                state["halted"] = str(halt_reason)
            if actual is not None and actual > money(call["reservation_usd"]):
                state["halted"] = "provider charge exceeded the reserved request bound"
                overrun = True
            halted = state.get("halted")
        if overrun:
            raise CampaignStopped("provider charge exceeded the reserved request bound")
        # An already-dispatched peer still settles its charge after a stop, but
        # must not resume search as if the shared campaign remained healthy.
        return halted

    def snapshot(self):
        with self._locked() as state:
            known, held = self.totals(state)
            return {"limit_usd": str(self.limit), "known_cost_usd": str(known),
                    "reserved_or_unknown_usd": str(held),
                    "remaining_usd": str(max(Decimal(0), self.limit - known - held)),
                    "calls": len(state["calls"]), "halted": state.get("halted")}

    def resume_local_pause(self, expected_reason, *, resolution):
        """Resume an audited local research pause without forgiving any charge.

        Provider failures, unknown outcomes and overruns are not resumable here.
        A local pause may be resolved only after every dispatched call settles.
        """
        prefix = "local research pause:"
        if (not isinstance(expected_reason, str) or not expected_reason.startswith(prefix)
                or not isinstance(resolution, str) or not resolution.strip()):
            raise ValueError("need the exact local pause reason and its resolution")
        with self._locked() as state:
            if state.get("halted") != expected_reason:
                raise CampaignStopped("the local pause reason changed; halt preserved")
            if any(c.get("status") != "settled" or c.get("cost_usd") is None
                   or money(c["cost_usd"]) > money(c["reservation_usd"])
                   for c in state["calls"].values()):
                raise CampaignStopped("unsettled, unknown or overrun charges; halt preserved")
            known, held = self.totals(state)
            if held or known > self.limit:
                raise CampaignStopped("invalid accounting balance; halt preserved")
            state.setdefault("local_pause_resolutions", []).append({
                "reason": expected_reason, "resolution": resolution,
                "resumed_at": time.time(), "known_cost_usd": str(known),
                "limit_usd": str(self.limit)})
            state["halted"] = None


class BudgetedLLM:
    """OpenRouter text calls with pinned price caps and no hidden client retries."""
    def __init__(self, cfg, pricing, ledger, run_id, api_key, client=None):
        self.prices = validate_llm_config(cfg, pricing)
        self.models = list(cfg["models"])
        from ..engine.llm.client import validate_model_capabilities
        self.model_capabilities = validate_model_capabilities(cfg.get('model_capabilities', {}), self.models)
        self.weights = list(cfg["weights"])
        self.temperature = cfg.get("temperature", .7)
        self.max_tokens = cfg["max_tokens"]
        self.ledger, self.run_id = ledger, run_id
        from openai import OpenAI
        self.client = client if client is not None else OpenAI(
            base_url=cfg["api_base"], api_key=api_key,
            timeout=cfg.get("timeout", 120), max_retries=0)

    def choose_model(self, rng):
        return rng.choices(self.models, weights=self.weights, k=1)[0]

    def generate(self, prompt, model, temperature=None, *, response_format=None, max_tokens=None):
        from ..engine.llm.client import (LLMResponse, validate_response_format, _response_metadata,
                                        COST_PROVENANCE_SCHEMA, validate_resolved_model)
        output_tokens = self.max_tokens if max_tokens is None else max_tokens
        if type(output_tokens) is not int or output_tokens <= 0:
            raise ValueError('max_tokens must be a positive integer or null')
        format_value = validate_response_format(response_format)
        options = {} if format_value is None else {'response_format': format_value}
        if model not in self.models:
            raise ValueError("request model is outside the frozen ensemble")
        temp = self.temperature if temperature is None else temperature
        if (isinstance(temp, bool) or not isinstance(temp, (int, float))
                or not math.isfinite(temp) or not 0 <= temp <= 2):
            raise ValueError("invalid LLM temperature")
        rates = self.prices[model]
        pin, pout = money(rates["prompt"]), money(rates["completion"])
        # UTF-8 bytes upper-bound text token count; leave ample message overhead.
        # Output cap includes reasoning tokens; metadata must permit fixed effort none.
        bound = request_cost_bound(prompt, output_tokens, rates, response_format=format_value)
        request = dict(
            model=model, messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=output_tokens,
                extra_body={"reasoning": {"effort": "none"}, "provider": {
                    "allow_fallbacks": False, "require_parameters": True,
                    "max_price": {"prompt": float(pin * 1_000_000),
                                  "completion": float(pout * 1_000_000),
                                  "request": float(money(rates.get("request", 0)))}}}, **options)
        from ..engine.llm.client import temperature_parameters
        if not temperature_parameters(self, model, temperature):
            del request['temperature']
        ident = self.ledger.reserve(bound, run_id=self.run_id, model=model)
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            halt_reason = "provider request outcome is unknown; reservation retained; campaign stopped"
            halted = self.ledger.settle(ident, halt_reason=halt_reason,
                                       **provider_error_metadata(exc, request=request))
            raise CampaignStopped(halted) from None
        usage = getattr(response, "usage", None)
        raw_cost = getattr(usage, "cost", None)
        try:
            cost = None if raw_cost is None else money(raw_cost)
        except (ValueError, ArithmeticError):
            cost = None
        pt, ct = getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0
        halt_reason = "provider did not return a usable actual charge; reservation retained"
        halted = self.ledger.settle(ident, cost, halt_reason=halt_reason if cost is None else None,
                                  generation_id=getattr(response, "id", None),
                                  prompt_tokens=pt, completion_tokens=ct)
        if halted:
            raise CampaignStopped(halted) from None
        choices = getattr(response, "choices", None) or []
        text = (choices[0].message.content or "") if choices else ""
        try:
            resolved_model = validate_resolved_model(getattr(response, 'model', None))
        except Exception:
            resolved_model = None
        return LLMResponse(text=text, model=model, cost=float(cost),
                           prompt_tokens=pt, completion_tokens=ct,
                           resolved_model=resolved_model,
                           cost_provenance=dict(schema=COST_PROVENANCE_SCHEMA,
                               source='provider_reported', reported_cost=float(cost)),
                           **(_response_metadata(choices[0]) if choices else {}))
