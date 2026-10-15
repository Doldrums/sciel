"""Claude through the official Anthropic SDK, behind the same generate() contract as the other providers.

The SDK never retries: recovery belongs to the durable caller, as for chat completions. Cost is the token
estimate from PRICE_PER_1K, so its provenance can be rechecked from the recorded usage.
"""

from __future__ import annotations

import os

from .client import (COST_PROVENANCE_SCHEMA, LLMError, LLMResponse, _price, _tokens, validate_response_format)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
EFFORTS = (None, "low", "medium", "high", "xhigh", "max")


def validate_config(llm: dict) -> dict:
    models = llm.get("models") or []
    if not models or any(type(m) is not str or not m.startswith("claude-") for m in models):
        raise ValueError("llm.models must name Claude models such as claude-sonnet-5-5")
    if llm.get("reasoning_effort") not in EFFORTS:
        raise ValueError(f"llm.reasoning_effort must be one of {EFFORTS[1:]} or null")
    return llm


class AnthropicLLM:
    def __init__(self, cfg: dict):
        import anthropic
        cfg = validate_config(cfg)
        self.models = list(cfg["models"])
        self.weights = cfg.get("weights") or [1.0] * len(self.models)
        self.max_tokens = cfg.get("max_tokens", 32000)
        self.temperature = None
        self.effort = cfg.get("reasoning_effort")
        # Current Claude models reject non-default sampling parameters.
        self.model_capabilities = {model: {"temperature": False} for model in self.models}
        key = os.environ.get(cfg.get("api_key_env", "ANTHROPIC_API_KEY")) or None
        self._client = anthropic.Anthropic(api_key=key, timeout=cfg.get("timeout", 600.0), max_retries=0)

    def choose_model(self, rng) -> str:
        return rng.choices(self.models, weights=self.weights, k=1)[0]

    def generate(self, prompt, model, temperature=None, *, response_format=None, max_tokens=None) -> LLMResponse:
        output_tokens = self.max_tokens if max_tokens is None else max_tokens
        if type(output_tokens) is not int or output_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer or null")
        output_config = {}
        if self.effort is not None:
            output_config["effort"] = self.effort
        schema = validate_response_format(response_format)
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema["json_schema"]["schema"]}
        params = {"model": model, "max_tokens": output_tokens, "messages": [{"role": "user", "content": prompt}],
                  "betas": [FALLBACK_BETA], "extra_body": {"fallbacks": "default"}}
        if output_config:
            params["output_config"] = output_config
        try:
            with self._client.beta.messages.stream(**params) as stream:
                message = stream.get_final_message()
        except Exception as e:
            raise LLMError(cause=e) from e
        served = getattr(message, "model", None) or model
        usage = getattr(message, "usage", None)
        pt, ct = _tokens(getattr(usage, "input_tokens", None)), _tokens(getattr(usage, "output_tokens", None))
        cost = provenance = None
        if pt is not None and ct is not None:
            key, pin, pout = _price(served)
            cost = (pt / 1000.0) * pin + (ct / 1000.0) * pout
            provenance = dict(schema=COST_PROVENANCE_SCHEMA, source="token_estimate", pricing_key=key,
                              prompt_usd_per_1k=pin, completion_usd_per_1k=pout)
        text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
        refusal = None
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            refusal = getattr(details, "explanation", None) or getattr(details, "category", None) or "refusal"
        return LLMResponse(text=text, model=model, cost=cost, prompt_tokens=pt, completion_tokens=ct,
                           finish_reason=message.stop_reason, refusal=refusal, cost_provenance=provenance,
                           resolved_model=served if served != model else None)
